from src.utils import keypoint_unscaler
from src.utils import extract_keypoints
import torch
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import time
import numpy as np

def overlapping_kps(keypoints, max_values, pixel_distance):
    """Remove overlapping keypoints"""    
    def remove_kps(pairs, max_values):
        remove = []
        for row, column in pairs:
            if max_values[row] > max_values[column]:
                remove.append(column)
            else:
                remove.append(row)
        return remove

    if keypoints.ndim == 1 or keypoints.shape[1] != 2:
        keypoints = keypoints.reshape(-1, 2)

    n = len(keypoints)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        dist_matrix[i] = np.sqrt(np.sum((keypoints[i] - keypoints) ** 2, axis=1))
    
    close_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i, j] < pixel_distance:
                close_pairs.append((i, j))

    remove_vec = [[rem] for rem in remove_kps(close_pairs, max_values)]
    keypoints[remove_vec] = -1
    return keypoints.flatten()


def filter_low_probabilities(keypoints, max_values, threshold):
    if keypoints.ndim == 1 or keypoints.shape[1] != 2:
        keypoints = keypoints.reshape(-1, 2)

    remove_kps = []
    for i, value in enumerate(max_values):
        if value < threshold:
            remove_kps.append(i)

    keypoints[remove_kps] = -1
    return keypoints.flatten()


def keypoints_fewer_n(keypoints):
    pass

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
            keypoints, max_values = extract_keypoints(outputs, return_max_values=True)
            keypoints = filter_low_probabilities(keypoints, max_values, threshold=-2)
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

        if isinstance(image, str):
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            image = image.copy()

        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        plt.imshow(image)


class VideoPredictor:
    # TODO: limit the fps if the fps of the model is faster than the video.
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

            x = max(0, min(x, frame_copy.shape[1] - 1))
            y = max(0, min(y, frame_copy.shape[0] - 1))
            
            cv2.putText(frame_copy, str(i//2), (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
        return frame_copy

    def predict_frame(self, frame):
        """Predict a single frame."""
        original_h, original_w = frame.shape[:2]
        image_tensor = self.transform(image=frame)["image"].unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.cfg.device))

        if self.cfg.model_name == "ResNetHeatmap":
            keypoints, max_values = extract_keypoints(outputs, return_max_values=True)
            keypoints = filter_low_probabilities(keypoints, max_values, threshold=-2)
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
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            current_fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            keypoints = detector.predict_frame(frame)
            frame_with_kps = detector.draw_keypoints_on_frame(frame, keypoints)
            writer.write(frame_with_kps)

            fps_text = f"FPS: {current_fps:.2f}"
            cv2.putText(frame_with_kps, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Keypoint detection", frame_with_kps)


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()