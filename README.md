# Keypoint detection
Court keypoint detection for tennis match footage. The model detects **14 keypoints** — 7 per half of the court. Three model architectures are supported with a shared training and inference pipeline:

| Model | Architecture | Output |
|---|---|---|
| `resnet` | ResNet-34 backbone + regression head | `[B, 28]` flat coordinates |
| `heatmap` | ResNet-34 backbone + dilated conv + heatmap head | `[B, 14, H, W]` heatmaps |
| `rcnn` | Keypoint R-CNN (ResNet-50 FPN) | list of detection dicts |

The **heatmap model** (`resnet34` backbone) is the primary model. It produces one Gaussian heatmap per keypoint, extracts the argmax coordinate, and passes the result through a post-processing pipeline (confidence filtering, overlap removal, minimum keypoint count check).

---
### Dataset
The dataset is taken from https://github.com/yastrebksv/TennisCourtDetector. In total there are 8841 images.

--- 
## Training

```bash
# Train the heatmap model with FP16 (default)
python main.py --name heatmap --num_workers 4

# Train with FP32
python main.py --name heatmap --num_workers 4 --fp32

# Train the regression model
python main.py --name resnet --num_workers 4

# Train Keypoint R-CNN
python main.py --name rcnn --num_workers 4
```

---
## Inference

The `inference.py` entry point accepts an image, a video file, or a folder of images. The input type is detected automatically from the file extension.

```bash
# Single image (PyTorch FP16)
python inference.py \
    --media "dataset/sample_images/clay.jpg" \
    --model_folder "models/resnet34-hm/"

# Single image (ONNX)
python inference.py \
    --media "dataset/sample_images/clay.jpg" \
    --model_folder "models/resnet34-hm/" \
    --use_onnx

# Video (ONNX FP16, with FPS limiting to match source video)
python inference.py \
    --media "examples/tennis_match_shortened.mp4" \
    --model_folder "models/resnet34-hm/" \
    --use_onnx

# Video (ONNX FP32)
python inference.py \
    --media "examples/tennis_match_shortened.mp4" \
    --model_folder "models/resnet34-hm/" \
    --use_onnx --fp32

# Folder of images (batch size 4)
python inference.py \
    --media "dataset/sample_images/" \
    --model_folder "models/resnet34-hm/" \
    --use_onnx \
    --batch_size 4 \
    --output_dir "predictions/"
```


## Future updates
- Checkpoint training
- Multiscale training
- Batch inference for video
- Optimize Keypoint R-CNN more
- Create better data augmentation pipeline
- Homography for post-processing