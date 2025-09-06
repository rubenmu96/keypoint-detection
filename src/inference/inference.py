from src.utils import (
    get_model_and_config,
    keypoint_unscaler,
)
from src.utils import extract_keypoints
import torch
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np

class ImagePredictor:
    def __init__(self, cfg, model, load_model):
        self.model = model
        self.model.load_state_dict(torch.load(load_model, map_location='cpu'))
        self.model = self.model.to(cfg.device)
        self.model.eval()
        self.cfg = cfg
        self.width = cfg.width
        self.height = cfg.height
        
        self.transform = A.Compose([
            A.Resize(width=cfg.width, height=cfg.height),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.0),
        ])

    def predict(self, image_path):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        image_tensor = self.transform(image=image)["image"].unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.cfg.device))

        if self.cfg.model_name == "ResNetHeatmap":
            keypoints = extract_keypoints(outputs)
            keypoints = keypoints.squeeze().cpu().numpy()
        else:
            keypoints = outputs.squeeze().cpu().numpy()

        keypoints = keypoint_unscaler(
            self.cfg, keypoints, original_w, original_h
        )
        return keypoints

    def draw_keypoints(self, image_path, keypoints):
        if keypoints.ndim == 2:
            keypoints = keypoints.flatten()

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = image.copy()
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        plt.imshow(image)


class VideoPredictor:
    def __init__(self, cfg, model, load_model):
        self.model = model
        self.model.load_state_dict(torch.load(load_model, map_location='cpu'))
        self.model = self.model.to(cfg.device)
        self.model.eval()
        self.cfg = cfg
        self.width = cfg.width
        self.height = cfg.height
        
        self.transform = A.Compose([
            A.Resize(width=cfg.width, height=cfg.height),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.0),
        ])
    def draw_keypoints_on_frame(self, frame, keypoints):
        """Draw keypoints on a frame (BGR format)"""
        if keypoints.ndim == 2:
            keypoints = keypoints.flatten()

        frame_copy = frame.copy()
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            # Ensure coordinates are within frame bounds
            x = max(0, min(x, frame_copy.shape[1] - 1))
            y = max(0, min(y, frame_copy.shape[0] - 1))
            
            cv2.putText(frame_copy, str(i//2), (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
        return frame_copy

    def predict_batch(self, frames):
        batch_tensors = []
        original_shapes = []
        
        for frame in frames:
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
                
            original_h, original_w = frame_rgb.shape[:2]
            original_shapes.append((original_w, original_h))
            
            image_tensor = self.transform(image=frame_rgb)["image"]
            batch_tensors.append(image_tensor)
        
        batch_tensors = torch.stack(batch_tensors).to(self.cfg.device)
        
        with torch.no_grad():
            outputs = self.model(batch_tensors)
        
        keypoints_list = []
        for i, output in enumerate(outputs):
            if self.cfg.model_name == "ResNetHeatmap":
                keypoints = extract_keypoints(output.unsqueeze(0))
                keypoints = keypoints.squeeze().cpu().numpy()
            else:
                keypoints = output.squeeze().cpu().numpy()
            
            original_w, original_h = original_shapes[i]
            keypoints = keypoint_unscaler(self.cfg, keypoints, original_w, original_h)
            keypoints_list.append(keypoints)
        
        return keypoints_list
    
    def batch_video_prediction(self, detector, video_path, output_path, batch_size=16):
        cap = cv2.VideoCapture(video_path)
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path, fourcc, original_fps, (width, height)
        )
        
        frames_batch = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_batch.append(frame)
            
            if len(frames_batch) == batch_size:
                keypoints_batch = detector.predict_batch(frames_batch)
                for frame, keypoints in zip(frames_batch, keypoints_batch):
                    frame_with_kps = detector.draw_keypoints_on_frame(frame, keypoints)
                    writer.write(frame_with_kps)
                
                frames_batch = []
        
        if frames_batch:
            keypoints_batch = detector.predict_batch(frames_batch)
            for frame, keypoints in zip(frames_batch, keypoints_batch):
                frame_with_kps = detector.draw_keypoints_on_frame(frame, keypoints)
                writer.write(frame_with_kps)
        
        cap.release()
        writer.release()
        print(f"Video saved to: {output_path}")

    def predict_frame(self, frame):
        original_h, original_w = frame.shape[:2]
        image_tensor = self.transform(image=frame)["image"].unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.cfg.device))

        if self.cfg.model_name == "ResNetHeatmap":
            keypoints = extract_keypoints(outputs)
            keypoints = keypoints.squeeze().cpu().numpy()
        else:
            keypoints = outputs.squeeze().cpu().numpy()

        keypoints = keypoint_unscaler(
            self.cfg, keypoints, original_w, original_h
        )
        return keypoints

    def predict_video(self, detector, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path, fourcc, original_fps, (width, height)
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = detector.predict_frame(frame)
            frame_with_kps = detector.draw_keypoints_on_frame(frame, keypoints)
            writer.write(frame_with_kps)

            cv2.imshow("Keypoint detection", frame_with_kps)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()