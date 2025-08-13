from src.core import compute_loss
import torch
from src.utils import keypoint_unscaler
from src.heatmap import extract_keypoints
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, cfg, model, optimizer, criterion, scheduler=None):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = cfg.device
        self.model_name = cfg.model_name
    
    def _train(self, train_data):
        total_loss = 0
        num_batches = 0
        
        self.model.train()
        for img, kps in train_data:
            img = img.to(self.device)

            self.optimizer.zero_grad()
            if self.model_name == "KeypointRCNN":
                kps = [{k: v.to(self.cfg.device) for k, v in t.items()} for t in kps]
                outputs = self.model(img, kps)
                loss = sum(loss for loss in outputs.values())
            else:
                kps = kps.to(self.device)
                outputs = self.model(img)
                loss = compute_loss(
                    self.cfg, self.criterion, outputs, kps
                )

            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            num_batches += 1

            if self.scheduler is not None:
                self.scheduler.step()
        
        return total_loss / num_batches
    
    def _evaluate(self, valid_data):
        total_loss = 0
        num_batches = 0

        self.model.eval()
        for i, (img, kps) in enumerate(valid_data):
            img = img.to(self.device)

            if self.model_name == "KeypointRCNN":
                kps = [{k: v.to(self.cfg.device) for k, v in t.items()} for t in kps]
            else:
                kps = kps.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img)

            loss = compute_loss(self.cfg, self.criterion, outputs, kps)

            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches
    
    def _early_stopping(self, es, current_loss, best_loss, patience):
        if patience <= 0:
            raise ValueError("Patience must be a positive integer.")
        
        if current_loss < best_loss:
            torch.save(self.model.state_dict(), self.cfg.save_path)

            if self.cfg.reset: es = 0
            best_loss = current_loss
        else:
            es += 1
        
        stop = es >= patience
        return es, best_loss, stop
    
    def display_sample_images(self, cfg, model, image_path, epoch, name):
        # need to create directory for sample_images
        def draw_keypoints(image, keypoints):
            image = image.copy()
            for i in range(0, len(keypoints), 2):
                x = int(keypoints[i])
                y = int(keypoints[i+1])
                cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            plt.imshow(image)
            plt.savefig(f"sample_images/{name}_{epoch}.png")

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
        with torch.no_grad():
            outputs = model(image_tensor)

        if cfg.model_name == "ResNetHeatmap":
            outputs = extract_keypoints(outputs)

        keypoints = outputs.cpu().numpy()[0]

        keypoints = keypoint_unscaler(
            cfg, keypoints, original_w, original_h
        )
        draw_keypoints(image, keypoints)
    
    def train(self, train_data, valid_data):
        epoch_metrics = []
        es, best_loss = 0, float("inf")

        if self.cfg.patience is None:
            self.cfg.patience = self.cfg.epochs

        for epoch in range(self.cfg.epochs):
            train_metrics = self._train(train_data=train_data)
            valid_metrics = self._evaluate(valid_data=valid_data)

            print(f'Epoch: {epoch + 1}. Training loss: {train_metrics}. Validation loss: {valid_metrics}')
            es, best_loss, stop = self._early_stopping(
                es, valid_metrics, best_loss, self.cfg.patience
            )

            if self.cfg.display_examples:
                for i, img_pth in enumerate(self.cfg.sample_image_path):
                    self.display_sample_images(self.cfg, self.model, img_pth, epoch + 1, self.cfg.display_names[i])

            if stop:
                print("Early stopping triggered.")
                break

            metrics = {
                "train": train_metrics,
                "valid": valid_metrics
            }
            epoch_metrics.append(metrics)
        return epoch_metrics
    
