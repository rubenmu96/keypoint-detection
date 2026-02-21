import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import os

from src.core import compute_loss, compute_accuracy
from src.utils import (
    keypoint_unscaler,
    extract_keypoints,
    get_file_type
)

class Trainer:  
    def __init__(
            self, cfg, model, optimizer, criterion, scheduler=None, scaler=torch.amp.GradScaler('cuda'), use_amp=False
        ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = cfg.device
        self.model_name = cfg.model_name
        self.folder = self.cfg.folder
        self.scaler = scaler
        self.use_amp = use_amp

    def save_model_fp32(self, model, path):
        """Save model in FP32 format"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'dtype': 'fp32',
            'model_name': self.model_name
        }, path)

    def save_model_fp16(self, model, path):
        """Save model in FP16 format to reduce file size"""
        try:
            model_fp16 = model.half() if self.device != 'cpu' else model
            
            torch.save({
                'model_state_dict': model_fp16.state_dict(),
                'dtype': 'fp16' if self.device != 'cpu' else 'fp32',
                'model_name': self.model_name
            }, path)
        finally:
            if self.device != 'cpu':
                model.float()

    def save_model(self, model, base_path):
        """Save model in both FP32 and FP16 formats"""
        base, ext = os.path.splitext(base_path)
        if not ext:
            ext = '.pth'

        fp32_path = f"{base}_fp32{ext}"
        fp16_path = f"{base}_fp16{ext}"
        
        self.save_model_fp32(model, fp32_path)
        if self.use_amp:
            self.save_model_fp16(model, fp16_path)
    
    def _train(self, train_data):
        total_loss = 0
        num_batches = 0
        
        times = {
            'total_time': 0,
            'data': 0,
            'forward': 0,
            'backward': 0,
            'optimizer': 0,
        }
        
        self.model.train()
        pbar = tqdm(train_data, desc="Training", leave=False)
        
        start = time.time()
        for img, kps in pbar:
            times["data"] += time.time() - start
            
            img = img.to(self.device, non_blocking=True)
            if self.model_name == "KeypointRCNN":
                kps = [{k: v.to(self.cfg.device) for k, v in t.items()} for t in kps]
            else:
                kps = kps.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            
            start = time.time()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                if self.model_name == "KeypointRCNN":
                    outputs = self.model(img, kps)
                    loss = sum(loss for loss in outputs.values())
                else:
                    outputs = self.model(img)
                    loss = compute_loss(self.cfg, self.criterion, outputs, kps)
            times["forward"] += time.time() - start
            
            start = time.time()
            self.scaler.scale(loss).backward()
            times["backward"] += time.time() - start
            
            start = time.time()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            times["optimizer"] += time.time() - start
            
            num_batches += 1
            total_loss += loss.item()

            if self.scheduler is not None:
                self.scheduler.step()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            start = time.time()
        
        times["total_time"] = sum(times.values())

        tracking = {
            "loss": total_loss / num_batches,
        }
        tracking.update(times)
        
        return tracking

    @torch.no_grad()
    def _evaluate(self, valid_data):        
        losses = []
        all_pck = []
        all_mpjpe = []
        
        times = {
            "total_time": 0,
            "data": 0,
            "forward": 0,
            "loss_compute": 0,
            "accuracy_calcs": 0
        }
        
        self.model.eval()
        pbar = tqdm(valid_data, desc="Validation", leave=False)
        
        start = time.time()
        for img, kps in pbar:
            times['data'] += time.time() - start
            
            img = img.to(self.device, non_blocking=True)
            if self.model_name == "KeypointRCNN":
                kps = [{k: v.to(self.cfg.device) for k, v in t.items()} for t in kps]
            else:
                kps = kps.to(self.device, non_blocking=True)
            
            start = time.time()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                outputs = self.model(img)
            times['forward'] += time.time() - start
            
            start = time.time()
            loss = compute_loss(self.cfg, self.criterion, outputs, kps)
            times['loss_compute'] += time.time() - start
            
            losses.append(loss)

            # TODO: add time for accuracy calculations
            start = time.time()
            scores = compute_accuracy(self.cfg, outputs, kps)
            all_pck.append(scores["pck@0.05"])
            all_mpjpe.append(scores["mpjpe"])
            times["accuracy_calcs"] += time.time() - start

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'PCK': f'{scores["pck@0.05"]:.3f}',
                'MPJPE': f'{scores["mpjpe"]:.3f}'
            })
            start = time.time()
        
        # Single synchronization at the end
        avg_loss = torch.stack(losses).mean().item()
        avg_pck = np.mean(all_pck)
        avg_mpjpe = np.mean(all_mpjpe)

        times["total_time"] = sum(times.values())

        tracking = {
            "loss": avg_loss,
            "pck@0.05": avg_pck,
            "mpjpe": avg_mpjpe
        }
        tracking.update(times)
        
        return tracking
    
    def _early_stopping(self, es, current_loss, best_loss, patience, greater_is_better=False):
        if patience <= 0:
            raise ValueError("Patience must be a positive integer.")
        
        if greater_is_better:
            is_better = current_loss > best_loss
        else:
            is_better = current_loss < best_loss

        if is_better:
            save_path = os.path.join(self.folder, self.cfg.save_path)
            self.save_model(self.model, save_path)
            if self.cfg.reset:
                es = 0
            best_loss = current_loss
        else:
            es += 1
        
        stop = es >= patience
        return es, best_loss, stop
    
    # # TODO: move testing functions outside?
    # @torch.no_grad()
    # def vis_testing(self, cfg, model, image_path, epoch, name, use_amp, folder_path="test-images/"):
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     def draw_keypoints(image, keypoints):
    #         if keypoints.ndim == 2:
    #             keypoints = keypoints.flatten()

    #         image = image.copy()
    #         for i in range(0, len(keypoints), 2):
    #             x = int(keypoints[i])
    #             y = int(keypoints[i+1])
    #             cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #             cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    #         plt.imshow(image)
    #         plt.savefig(f"{folder_path}/{name}_{epoch}.png")

    #     transform = A.Compose([
    #         A.Resize(width=cfg.width, height=cfg.height),
    #         A.Normalize(mean=cfg.mean, std=cfg.std),
    #         ToTensorV2(p=1.0),
    #     ])
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     original_h, original_w = image.shape[:2]
    #     image_tensor = transform(image=image)["image"].unsqueeze(0)

    #     image_tensor = image_tensor.to(cfg.device)

    #     model.eval()
    #     with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
    #         outputs = model(image_tensor)

    #     if cfg.model_name == "ResNetHeatmap":
    #         outputs = extract_keypoints(outputs)

    #     keypoints = outputs.cpu().numpy()[0]

    #     keypoints = keypoint_unscaler(
    #         cfg, keypoints, original_w, original_h
    #     )
    #     draw_keypoints(image, keypoints)

    # def testing(self, cfg, model, epoch, use_amp):
    #     for i, img_pth in enumerate(cfg.sample_image_path):
    #         if len(cfg.sample_image_path) == len(cfg.display_names):
    #             name = cfg.display_names[i]
    #         else:
    #             name = f"img_{i}"

    #         file_type = get_file_type(img_pth)
    #         if file_type == "image":
    #             self.vis_testing(cfg, model, img_pth, epoch + 1, name, use_amp)
    #         else:
    #             continue
    
    def train(self, train_data, valid_data):
        epoch_metrics = []
        es = 0
        greater_is_better = self.cfg.greater_is_better
        best_loss = -float("inf") if greater_is_better else float("inf")

        if self.cfg.patience is None:
            self.cfg.patience = self.cfg.epochs

        pbar = tqdm(range(self.cfg.epochs), desc="Epochs")
        
        for epoch in pbar:
            train_metrics = self._train(train_data=train_data)
            valid_metrics = self._evaluate(valid_data=valid_data)

            pbar.set_postfix({
                "train_loss": f'{train_metrics["loss"]:.4f}',
                "valid_loss": f'{valid_metrics["loss"]:.4f}',
                "valid_pck@0.05": f'{valid_metrics["pck@0.05"]:.4f}',
                "valid_mpjpe": f'{valid_metrics["mpjpe"]:.4f}',
            })
            
            es, best_loss, stop = self._early_stopping(
                es, valid_metrics["loss"], best_loss, 
                self.cfg.patience, greater_is_better
            )

            if self.cfg.display_examples:
                testing(self.cfg, self.model, epoch + 1, self.use_amp)

            if stop:
                print("Early stopping triggered.")
                break
            
            # Update metrics for epoch
            output_path = os.path.join(self.folder, "tracking.csv")
            metrics = {
                "training": train_metrics,
                "validation": valid_metrics,
                "epoch": epoch + 1
            }

            epoch_metrics.append(metrics)
            df = pd.DataFrame(epoch_metrics)
            df.to_csv(output_path, index=False)
        
        return epoch_metrics
    
@torch.no_grad()
def vis_testing(cfg, model, image_path, epoch, name, use_amp, folder_path="test-images/"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    def draw_keypoints(image, keypoints):
        if keypoints.ndim == 2:
            keypoints = keypoints.flatten()

        image = image.copy()
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        plt.imshow(image)
        plt.savefig(f"{folder_path}/{name}_{epoch}.png")

    transform = A.Compose([
        A.Resize(width=cfg.width, height=cfg.height),
        A.Normalize(mean=cfg.mean, std=cfg.std),
        ToTensorV2(p=1.0),
    ])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image.shape[:2]
    image_tensor = transform(image=image)["image"].unsqueeze(0)

    image_tensor = image_tensor.to(cfg.device)

    model.eval()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        outputs = model(image_tensor)

    if cfg.model_name == "ResNetHeatmap":
        outputs = extract_keypoints(outputs)

    keypoints = outputs.cpu().numpy()[0]

    keypoints = keypoint_unscaler(
        cfg, keypoints, original_w, original_h
    )
    draw_keypoints(image, keypoints)

def testing(cfg, model, epoch, use_amp):
    for i, img_pth in enumerate(cfg.sample_image_path):
        if len(cfg.sample_image_path) == len(cfg.display_names):
            name = cfg.display_names[i]
        else:
            name = f"img_{i}"

        file_type = get_file_type(img_pth)
        if file_type == "image":
            vis_testing(cfg, model, img_pth, epoch + 1, name, use_amp)
        else:
            continue