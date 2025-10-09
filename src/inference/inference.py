import time

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.utils import keypoint_unscaler
from src.utils import extract_keypoints

### TODO: Move the heatmap post-processing to another .py
def overlapping_kps(keypoints, max_values, pixel_distance):
    """
    Remove overlapping keypoints
    
    Args:
        keypoints: numpy array of shape (n_keypoints, 2) or (n_keypoints*2,)
        max_values: list or array of confidence values for each keypoint
        pixel_distance: minimum distance in pixels between keypoints
        image_width, image_height: dimensions for converting normalized coords to pixels
    """
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
    
    keypoints_copy = keypoints.copy()
    
    valid_mask = ~np.all(keypoints_copy == -1, axis=1)
    if not np.any(valid_mask):
        return keypoints_copy.flatten()
    
    valid_keypoints = keypoints_copy[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    n_valid = len(valid_keypoints)
    if n_valid < 2:
        return keypoints_copy.flatten()
    
    diff = valid_keypoints[:, np.newaxis, :] - valid_keypoints[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    
    close_pairs = []
    for i in range(n_valid):
        for j in range(i+1, n_valid):
            if dist_matrix[i, j] < pixel_distance:
                # Map back to original indices
                orig_i = valid_indices[i]
                orig_j = valid_indices[j]
                close_pairs.append((orig_i, orig_j))

    if close_pairs:
        remove_indices = remove_kps(close_pairs, max_values)
        keypoints_copy[remove_indices] = -1
    
    return keypoints_copy.flatten()

def filter_low_probabilities(keypoints, max_values, threshold):
    """Remove keypoints (set to (-1, -1)) if confidence is below threshold"""
    if keypoints.ndim == 1 or keypoints.shape[1] != 2:
        keypoints = keypoints.reshape(-1, 2)
    
    keypoints_copy = keypoints.copy()
    
    remove_kps = []
    for i, value in enumerate(max_values):
        if value < threshold:
            remove_kps.append(i)

    if remove_kps:
        keypoints_copy[remove_kps] = -1
    
    return keypoints_copy.flatten()


def num_kps_req(keypoints, num_kps=7):
    """Remove all keypoints if the number of valid keypoints is less than required"""
    # TODO: need some rework, maybe require that both center keypoints and certain amount of keypoints should be present
    if keypoints.ndim == 1 or keypoints.shape[1] != 2:
        keypoints = keypoints.reshape(-1, 2)
    
    valid_kps = np.sum(np.all(keypoints != -1, axis=1))
    
    if valid_kps < num_kps:
        keypoints_copy = keypoints.copy()
        keypoints_copy[:] = -1
        return keypoints_copy.flatten()
    
    return keypoints.flatten()


def kps_postprocessor(keypoints, max_values, threshold, pixel_distance, num_kps=7):
    """
    Apply all keypoint post-processing steps
    
    Args:
        keypoints: numpy array of keypoints
        max_values: confidence values for each keypoint
        threshold: minimum confidence threshold
        pixel_distance: minimum pixel distance between keypoints
        num_kps: minimum number of required keypoints
    
    Returns:
        processed keypoints as flattened array
    """
    # Step 1: Filter low probability keypoints
    keypoints = filter_low_probabilities(keypoints, max_values, threshold)
    
    # Step 2: Remove overlapping keypoints
    keypoints = overlapping_kps(
        keypoints, max_values, pixel_distance
    )
    
    # Step 3: Check if we have enough keypoints
    keypoints = num_kps_req(keypoints, num_kps)
    
    return keypoints

def process_heatmap_keypoints(
        cfg, keypoints, threshold=-2, pixel_distance=10, num_kps=7, image_width=None, image_height=None
    ):
    """
    Complete pipeline for processing a batch of heatmaps into filtered keypoints.
    """
    keypoints, max_values = extract_keypoints(keypoints, return_max_values=True)
    
    processed_keypoints = []
    
    for b in range(len(keypoints)):
        batch_kps = keypoints[b]
        batch_max_vals = max_values[b]
        
        scaled_kps = keypoint_unscaler(
            cfg, batch_kps, image_width, image_height
        )
        
        kps_np = scaled_kps.cpu().numpy() if torch.is_tensor(scaled_kps) else scaled_kps
        
        processed_kps = kps_postprocessor(
            kps_np, batch_max_vals, threshold, pixel_distance, num_kps
        )
        
        processed_keypoints.append(processed_kps)
    
    return np.array(processed_keypoints)


def load_fp16_model(model, checkpoint_path, device):
    """Load model from FP16 checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Loading FP16 model")
        state_dict = checkpoint['model_state_dict']
        dtype_info = checkpoint.get('dtype', 'fp32')
        model.load_state_dict(state_dict)
        
        if dtype_info == 'fp16' and device != 'cpu':
            model = model.half()
        print(f"Loaded model with dtype: {dtype_info}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model from legacy format (FP32)")
    
    return model


class ImagePredictor:
    def __init__(self, cfg, model, load_model, use_fp16=True):
        self.cfg = cfg
        self.use_fp16 = use_fp16 and cfg.device != 'cpu'
        
        # Load model with proper FP16 support
        self.model = load_fp16_model(model, load_model, cfg.device)
        self.model = self.model.to(cfg.device)
        
        # Ensure model is in FP16 if requested
        if self.use_fp16:
            self.model = self.model.half()
            print(f"Model dtype after conversion: {next(self.model.parameters()).dtype}")
        
        self.model.eval()
        
        self.width = cfg.width
        self.height = cfg.height
        
        self.transform = A.Compose([
            A.Resize(width=cfg.width, height=cfg.height),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.0),
        ])
        
        if cfg.device != 'cpu':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

    def predict(self, image_path):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        image_tensor = self.transform(image=image)["image"].unsqueeze(0)

        # Convert input to half precision if using FP16
        if self.use_fp16:
            image_tensor = image_tensor.half()
            
        image_tensor = image_tensor.to(self.cfg.device, non_blocking=True)

        self.model.eval()
        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    keypoints = self.model(image_tensor, True)
            else:
                keypoints = self.model(image_tensor)

        if self.cfg.model_name == "ResNetHeatmap":
            keypoints = process_heatmap_keypoints(
                cfg=self.cfg,
                keypoints=keypoints,
                threshold=-2,
                pixel_distance=10,
                image_width=original_w,
                image_height=original_h
            )
        else: # change to elif afer resnet pipeline
            # TODO: make post-processing pipeline for resnet, probably drop for rcnn
            keypoints = keypoints.squeeze().cpu().numpy()
            keypoints = keypoint_unscaler(
                self.cfg, keypoints, original_w, original_h
            )

        return keypoints

    def draw_keypoints(self, image, keypoints):
        if keypoints.ndim == 2:
            keypoints = keypoints.flatten()

        if isinstance(image, str):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            image = image.copy()

        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            if x > 0 and y > 0:  # Only draw valid keypoints
                cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        plt.imshow(image)
        plt.show()


class VideoPredictor:
    def __init__(self, cfg, model, load_model, use_fp16=True):
        self.cfg = cfg
        self.use_fp16 = use_fp16 and cfg.device != 'cpu'
        
        # Load model with proper FP16 support
        self.model = load_fp16_model(model, load_model, cfg.device)
        self.model = self.model.to(cfg.device)
        
        # Ensure model is in FP16 if requested
        if self.use_fp16:
            self.model = self.model.half()
            print(f"Model dtype after conversion: {next(self.model.parameters()).dtype}")
            
        self.model.eval()
        
        self.width = cfg.width
        self.height = cfg.height
        
        self.transform = A.Compose([
            A.Resize(width=cfg.width, height=cfg.height),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.0),
        ])

        # Pre-allocate tensors to avoid repeated allocation
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.input_tensor = torch.zeros((1, 3, cfg.height, cfg.width), 
                                       device=cfg.device, dtype=dtype)
        
        # Enable cudnn benchmarking for better performance
        if cfg.device != 'cpu':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            # Use tensor cores if available
            if self.use_fp16:
                torch.set_float32_matmul_precision('high')

    def draw_keypoints_on_frame(self, frame, keypoints):
        """Draw keypoints on a frame (BGR format)"""
        if keypoints.ndim == 2:
            keypoints = keypoints.flatten()

        frame_copy = frame.copy()
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])

            # Only draw valid keypoints (not filtered out with -1)
            if x > 0 and y > 0:
                x = max(0, min(x, frame_copy.shape[1] - 1))
                y = max(0, min(y, frame_copy.shape[0] - 1))
                
                cv2.putText(frame_copy, str(i//2), (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
        return frame_copy

    def predict_frame(self, frame):
        """Predict a single frame with optimized FP16."""
        original_h, original_w = frame.shape[:2]
        
        # Transform frame
        transformed = self.transform(image=frame)["image"]
        
        # Convert to appropriate dtype
        if self.use_fp16:
            transformed = transformed.half()
        
        # Copy to pre-allocated tensor to avoid repeated allocation
        self.input_tensor.copy_(transformed.unsqueeze(0))

        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    keypoints = self.model(self.input_tensor)
            # don't think this is needed?
            else:
                keypoints = self.model(self.input_tensor)

        if self.cfg.model_name == "ResNetHeatmap":
            keypoints = process_heatmap_keypoints(
                cfg=self.cfg,
                keypoints=keypoints,
                threshold=-2,
                pixel_distance=10,
                image_width=original_w,
                image_height=original_h
            )
        else: # change to elif afer resnet pipeline
            # TODO: make post-processing pipeline for resnet, probably drop for rcnn
            keypoints = keypoints.squeeze().cpu().numpy()
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
        
        # Warm up GPU with proper dtype
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        print("Warming up GPU...")
        for _ in range(10):  # More warmup iterations
            detector.predict_frame(dummy_frame)

        if self.cfg.device == "cuda":
            torch.cuda.synchronize()  # Ensure GPU warmup is complete
        
        frame_count = 0
        total_inference_time = 0
        prev_time = time.time()

        print(f"Starting video processing with {'FP16' if self.use_fp16 else 'FP32'}...")
        
        # Process frames in batches for better benchmarking
        inference_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Time only the inference
            if self.cfg.device == "cuda": torch.cuda.synchronize()
            inference_start = time.time()
            keypoints = detector.predict_frame(frame_rgb)
            if self.cfg.device == "cuda": torch.cuda.synchronize()  # Wait for GPU to finish
            inference_end = time.time()
            
            inference_time = inference_end - inference_start
            inference_times.append(inference_time)
            total_inference_time += inference_time
            frame_count += 1

            frame_with_kps = detector.draw_keypoints_on_frame(frame, keypoints)
            writer.write(frame_with_kps)

            # Calculate overall FPS
            curr_time = time.time()
            overall_fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Calculate inference FPS
            inference_fps = 1 / inference_time if inference_time > 0 else 0

            fps_text = f"Overall FPS: {overall_fps:.2f} | Inference FPS: {inference_fps:.2f}"
            cv2.putText(frame_with_kps, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Keypoint detection", frame_with_kps)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
        # create as function instead? only need frame_count, inference_times and if use_fp16?
        # Print detailed statistics
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            median_inference_time = np.median(inference_times)
            
            print(f"\n{'='*50}")
            print(f"Performance Statistics ({'FP16' if self.use_fp16 else 'FP32'})")
            print(f"{'='*50}")
            print(f"Processed frames: {frame_count}")
            print(f"Average inference time: {avg_inference_time*1000:.2f}ms ± {std_inference_time*1000:.2f}ms")
            print(f"Median inference time: {median_inference_time*1000:.2f}ms")
            print(f"Average inference FPS: {1/avg_inference_time:.2f}")
            print(f"Min inference time: {min(inference_times)*1000:.2f}ms")
            print(f"Max inference time: {max(inference_times)*1000:.2f}ms")
            print(f"{'='*50}")