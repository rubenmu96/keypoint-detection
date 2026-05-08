import time

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import onnxruntime as ort

from src.utils import keypoint_unscaler, scale_keypoints_to_original

from src.inference.processing  import (
    process_heatmap_keypoints,
    load_fp16_model
)

class KeypointPredictor:
    def __init__(self, cfg, model=None, load_model=None, use_fp16=True, onnx_path=None):
        self.cfg = cfg
        self.use_onnx = onnx_path is not None
        
        if self.use_onnx:
            try:
                self._init_onnx(onnx_path)
            except Exception as e:
                print(f"ONNX model failed to load ({e}). Falling back to PyTorch.")
                self.use_onnx = False
                self._init_pytorch(model, load_model, use_fp16)
        else:
            self._init_pytorch(model, load_model, use_fp16)
        
        self.width = cfg.width
        self.height = cfg.height
        self.threshold = getattr(cfg, 'threshold', -2.0)
        self.pixel_distance = getattr(cfg, 'pixel_distance', 10)

        if cfg.model_name == "KeypointRCNN":
            self.transform = A.Compose([
                A.Resize(width=cfg.width, height=cfg.height),
                A.ToFloat(max_value=255),
                ToTensorV2(p=1.0),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(width=cfg.width, height=cfg.height),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0),
            ])
    
    # Initialization
    
    def _init_onnx(self, onnx_path):
        """Initialize ONNX."""
        
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                }),
                "CPUExecutionProvider"
            ]
            self.device = "cuda"
        else:
            providers = ["CPUExecutionProvider"]
            self.device = "cpu"
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        input_type = self.session.get_inputs()[0].type
        if 'float16' in input_type or 'Half' in input_type:
            self.onnx_dtype = np.float16
            self.use_fp16 = True
            print(f"ONNX model expects FP16 input")
        else:
            self.onnx_dtype = np.float32
            self.use_fp16 = False
            print(f"ONNX model expects FP32 input")
        
        self.input_array = np.zeros(
            (1, 3, self.cfg.height, self.cfg.width), 
            dtype=self.onnx_dtype
        )
        
        print(f"ONNX model loaded: {onnx_path}")
        print(f"  Providers: {self.session.get_providers()}")
    
    def _init_pytorch(self, model, load_model, use_fp16):
        """Initialize PyTorch model."""
        self.use_fp16 = use_fp16 and self.cfg.device != 'cpu'
        self.device = self.cfg.device
        
        self.model = load_fp16_model(model, load_model, self.cfg.device)
        self.model = self.model.to(self.cfg.device)
        
        if self.use_fp16:
            self.model = self.model.half()
            print(f"Model dtype: {next(self.model.parameters()).dtype}")
        
        self.model.eval()
        
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.input_tensor = torch.zeros(
            (1, 3, self.cfg.height, self.cfg.width),
            device=self.cfg.device, 
            dtype=dtype
        )
        
        if self.cfg.device != 'cpu':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            if self.use_fp16:
                torch.set_float32_matmul_precision('high')
    
    # Main infernece
    
    def predict(self, image):
        """Predict keypoints for a single image."""
        if isinstance(image, str):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        
        original_h, original_w = image.shape[:2]
        transformed = self.transform(image=image)["image"].unsqueeze(0)
        
        if self.use_onnx:
            heatmaps = self._inference_onnx(transformed)
        else:
            heatmaps = self._inference_pytorch(transformed)
        
        return self._postprocess(heatmaps, original_w, original_h)
    
    def _inference_onnx(self, transformed):
        """Run ONNX inference."""
        transformed_np = transformed.numpy().astype(self.onnx_dtype)
        np.copyto(self.input_array, transformed_np)
        return self.session.run(None, {self.input_name: self.input_array})[0]
    
    @torch.no_grad()
    def _inference_pytorch(self, transformed):
        """Run PyTorch inference."""
        if self.use_fp16:
            transformed = transformed.half()
        self.input_tensor.copy_(transformed)
        return self.model(self.input_tensor)
    
    def _postprocess(self, heatmaps, original_w, original_h):
        """Convert model output to keypoints."""
        if self.cfg.model_name == "ResNetHeatmap":
            if isinstance(heatmaps, np.ndarray):
                heatmaps = torch.from_numpy(heatmaps)

            return process_heatmap_keypoints(
                cfg=self.cfg,
                keypoints=heatmaps,
                threshold=self.threshold,
                pixel_distance=self.pixel_distance,
                image_width=original_w,
                image_height=original_h
            )
        elif self.cfg.model_name == "KeypointRCNN":
            # heatmaps is a list of prediction dicts (one per image in the batch)
            pred = heatmaps[0] if isinstance(heatmaps, list) else heatmaps
            scores = pred["scores"]
            if scores.numel() == 0:
                # No detections — return zeros so callers don't crash
                return np.zeros((self.cfg.num_kps, 2), dtype=np.float32)
            kps = pred["keypoints"]  # (N, K, 3) — pixel coords at model input resolution
            best = scores.argmax()
            kps_best = kps[best, :, :2].cpu().numpy()  # (K, 2)
            return scale_keypoints_to_original(
                kps_best, self.cfg.width, self.cfg.height, original_w, original_h
            )
        else:
            if torch.is_tensor(heatmaps):
                heatmaps = heatmaps.squeeze().cpu().numpy()
            else:
                heatmaps = heatmaps.squeeze()

            return keypoint_unscaler(self.cfg, heatmaps, original_w, original_h)
        
    # Batch inference

    def predict_batch(self, images):
        """Predict keypoints for a batch of images."""
        if not images:
            return []
        
        originals = []
        transformed_list = []
        
        for image in images:
            if isinstance(image, str):
                image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            
            originals.append((image.shape[1], image.shape[0]))
            transformed = self.transform(image=image)["image"]
            transformed_list.append(transformed)
        
        batch = torch.stack(transformed_list)
        
        if self.use_onnx:
            heatmaps = self._inference_onnx_batch(batch)
        else:
            heatmaps = self._inference_pytorch_batch(batch)
        
        keypoints_list = []
        for i, (orig_w, orig_h) in enumerate(originals):
            kps = self._postprocess(heatmaps[i:i+1], orig_w, orig_h)
            keypoints_list.append(kps)
        
        return keypoints_list

    def _inference_onnx_batch(self, batch):
        """Run ONNX inference on a batch."""
        batch_np = batch.numpy().astype(self.onnx_dtype)
        return self.session.run(None, {self.input_name: batch_np})[0]

    @torch.no_grad()
    def _inference_pytorch_batch(self, batch):
        """Run PyTorch inference on a batch."""
        batch = batch.to(self.device)
        if self.use_fp16:
            batch = batch.half()
        return self.model(batch)
    
    # Visualization
    
    def draw_keypoints(self, image, keypoints, bgr=False, save_path=None, show=True):
        """Draw keypoints on image. Set bgr=True for BGR frames (e.g. video)."""
        if keypoints.ndim == 2:
            keypoints = keypoints.flatten()

        if isinstance(image, str):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        image = image.copy()

        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i+1])
            if x > 0 and y > 0:
                x = max(0, min(x, image.shape[1] - 1))
                y = max(0, min(y, image.shape[0] - 1))
                cv2.putText(image, str(i//2), (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 2)
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

        if not bgr and (save_path or show):
            plt.imshow(image)

            if save_path:
                plt.savefig(save_path)

            if show:
                plt.show()

        return image
    
    # Video processing
    
    def _warmup(self, height, width, iterations=10):
        """Warmup for consistent timing."""
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        print(f"Warming up {'ONNX' if self.use_onnx else 'PyTorch'}...")
        
        for _ in range(iterations):
            self.predict(dummy_frame)
        
        if not self.use_onnx and self.device == "cuda":
            torch.cuda.synchronize()

    def predict_video(self, video_path, output_path, limit_fps=False, show=True):
        """Process video and save with keypoint overlays."""
        cap = cv2.VideoCapture(video_path)
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))

        target_frame_time = 1.0 / original_fps if original_fps > 0 else 0
        
        if self.device == "cuda":
            self._warmup(height, width)
        
        frame_count = 0
        prev_time = time.time()
        inference_times = []
        
        backend = "ONNX" if self.use_onnx else f"PyTorch {'FP16' if self.use_fp16 else 'FP32'}"
        print(f"Processing video with {backend}...")
        print(f"Target FPS: {original_fps:.2f} (Frame time: {target_frame_time*1000:.2f}ms)")
        
        while True:
            frame_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Sync for accurate timing (PyTorch CUDA only)
            if not self.use_onnx and self.device == "cuda":
                torch.cuda.synchronize()
            
            inference_start = time.time()
            keypoints = self.predict(frame_rgb)
            
            if not self.use_onnx and self.device == "cuda":
                torch.cuda.synchronize()
            
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            frame_count += 1

            frame_with_kps = self.draw_keypoints(frame, keypoints, bgr=True)
            writer.write(frame_with_kps)

            if show:
                curr_time = time.time()
                overall_fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                inference_fps = 1 / inference_time if inference_time > 0 else 0

                fps_text = f"Overall FPS: {overall_fps:.2f} | Inference FPS: {inference_fps:.2f}"
                cv2.putText(frame_with_kps, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                cv2.namedWindow('Keypoint detection', cv2.WINDOW_KEEPRATIO)
                cv2.imshow("Keypoint detection", frame_with_kps)
                cv2.resizeWindow('Keypoint detection', 600, 800)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if limit_fps: # reduce fps to video fps
                elapsed = time.time() - frame_start_time
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()
        
        if inference_times:
            summary_statistics(
                inference_times, frame_count, 
                self.use_fp16 if not self.use_onnx else False
            )

def summary_statistics(inference_times, frame_count, use_fp16):
    """Calculate FPS statistics"""
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    median_inference_time = np.median(inference_times)
    
    print(f"\n{'='*50}")
    print(f"Performance Statistics ({'FP16' if use_fp16 else 'FP32'})")
    print(f"{'='*50}")
    print(f"Processed frames: {frame_count}")
    print(f"Average inference time: {avg_inference_time*1000:.2f}ms ± {std_inference_time*1000:.2f}ms")
    print(f"Median inference time: {median_inference_time*1000:.2f}ms")
    print(f"Average inference FPS: {1/avg_inference_time:.2f}")
    print(f"Min inference time: {min(inference_times)*1000:.2f}ms")
    print(f"Max inference time: {max(inference_times)*1000:.2f}ms")
    print(f"{'='*50}")