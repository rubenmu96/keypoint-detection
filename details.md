# Keypoint Detection - more detailed description

Court keypoint detection for tennis match footage. The model detects **14 keypoints** — 7 per half of the court — that define the court geometry from a broadcast camera angle.

Three model architectures are supported with a shared training and inference pipeline:

| Model | Architecture | Output |
|---|---|---|
| `resnet` | ResNet backbone + regression head | `[B, 28]` flat coordinates |
| `heatmap` | ResNet backbone + dilated conv + heatmap head | `[B, 14, H, W]` heatmaps |
| `rcnn` | Keypoint R-CNN (ResNet-50 FPN) | list of detection dicts |

The **ResNet heatmap model** is the primary model. It produces one Gaussian heatmap per keypoint, extracts the argmax coordinate, and passes the result through a post-processing pipeline (confidence filtering, overlap removal, minimum keypoint count check).


The ResNet heatmap model and Keypoint R-CNN both uses heatmaps to predict keypoints. At the keypoint location $(x_k^\star, y_k^\star)$ we have the value

$$\mathbf{H}_k(x, y) = \exp \left(-\frac{(x - x_{k}^\star)^2 + (y - y_{k}^\star)^2}{2\sigma^2}\right),$$

where $\sigma$ controls the spread of the Gaussian. A small $\sigma$ will give us higher localization accuracy but will make optimziation harder. A larger $\sigma$ is more forgiving but less precise. 

At inference time, the predicted keypoint location is recovered as the position of the maximum activation in the heatmap:

$$(\hat{x}_k, \hat{y}_k) = \arg \max_{(x,y)} \mathbf{H}_k(x, y).$$

---

## Project structure
The main structure of the project (excluding the testing of functions under `tests/`)
```
keypoint-detection/
├── main.py                    # Training entry point
├── inference.py               # Inference entry point
├── test.py                    # For testing main.py on a small sample
├── config/
│   ├── config.py              # BaseConfig — shared dataset and training parameters
│   ├── model_configs.py       # ResNetConfig, HeatmapConfig, RCNNConfig
│   └── utils.py               # Config serialisation (dict ↔ class/namespace)
└── src/
    ├── core/
    │   ├── dataset.py         # KeypointPyTorch dataset, KeypointData loader, CollateFunction
    │   └── metrics.py         # compute_loss, compute_accuracy, PCK, MPJPE
    ├── models/
    │   ├── resnet.py          # ResNetKeypoint
    │   ├── heatmap.py         # ResNetHeatmap
    │   └── keypoint_rcnn.py   # KeypointRCNN wrapper
    ├── trainer/
    │   ├── trainer.py         # Trainer class (train loop, early stopping, FP16/FP32 saving)
    │   └── convert_onnx.py    # ONNX export (FP32 and FP16)
    ├── inference/
    │   ├── inference.py       # KeypointPredictor (PyTorch and ONNX backends)
    │   ├── processing.py      # Heatmap post-processing pipeline
    │   └── video_processor.py # VideoProcessor (mp4 trimming, YouTube download)
    └── utils/
        ├── utils.py           # Model factory, file-type detection, folder prediction
        ├── heatmap_funcs.py   # create_heatmap, extract_keypoints
        └── processing.py      # keypoint_scaler, keypoint_unscaler, keypoints_region
```

---

## Installation

Requires **Python ≥ 3.12** and **Poetry**.

```bash
poetry install --no-root
```

On **Windows**, PyTorch is installed from the CUDA 12.8 index automatically via the source declared in `pyproject.toml`. On Linux/macOS the standard PyPI wheel is used instead.

To find the virtual environment path (useful for selecting the interpreter in VS Code):

```bash
poetry show -v
```

Then in VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → paste the path.

---

## Dataset

The dataset is a JSON-annotated collection of tennis broadcast frames. Each record contains:

- `id`: image filename stem (image file is `<img_dir>/<id>.png`)
- `kps`: list of `[x, y]` pixel coordinates, one pair per keypoint (14 keypoints = 28 values)

Expected layout:

```
dataset/tennis_court_det_dataset/data/
├── data_train.json
├── data_val.json
└── images/
    ├── <id>.png
    └── ...
```

The paths are configured in `config/config.py` (`BaseConfig.path`).

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

> **Windows note:** `num_workers > 0` can cause issues with PyTorch's multiprocessing on Windows. If you encounter errors, use `--num_workers 0`.

### What happens during training

1. The model and its config are created by `get_model_and_config(name)`.
2. The dataset is loaded and optionally cleaned (images with out-of-bounds keypoints are removed).
3. A cosine learning rate schedule with linear warmup (15 % of total steps) is applied.
4. The `Trainer` runs the training loop with optional AMP (Automatic Mixed Precision).
5. Early stopping saves the best checkpoint based on validation loss.
6. Each epoch's metrics (loss, PCK@0.05, MPJPE, timing breakdown) are written to `<model_folder>/tracking.csv`.
7. After training, if `cfg.onnx = True`, the model is exported to ONNX (FP32 and, if AMP was used, FP16).

### Key hyperparameters (`config/config.py`, `config/model_configs.py`)

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 35 | Maximum training epochs |
| `batch_size` | 16 | Batch size |
| `learn_rate` | 5e-4 | Peak learning rate (AdamW) |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `warmup_ratio` | 0.15 | Fraction of steps used for LR warmup |
| `patience` | `None` (= `epochs`) | Early stopping patience |
| `width` / `height` | 672 / 448 | Model input resolution |
| `scale` | `(0, 1)` | Coordinate normalisation range |
| `sigma` | 1.4 | Gaussian sigma for heatmap targets (heatmap model only) |

### Output files

Saved under the folder defined by `cfg.folder` (e.g. `models/resnet34-hm/`):

```
models/resnet34-hm/
├── heatmap_config.json          # Full config snapshot for reproducibility
├── heatmap_resnet34_672x448_fp32.pth
├── heatmap_resnet34_672x448_fp16.pth   # only when AMP is used
├── heatmap_resnet34_672x448_fp32.onnx  # only when cfg.onnx = True
├── heatmap_resnet34_672x448_fp16.onnx  # only when cfg.onnx = True and AMP is used
└── tracking.csv
```

---

## Inference

The `inference.py` entry point accepts an image, a video file, or a folder of images. The input type is detected automatically from the file extension.

```bash
# Single image (PyTorch FP16)
python inference.py \
    --media "examples/test-images/clay.jpg" \
    --model_folder "models/resnet34-hm/"

# Single image (ONNX)
python inference.py \
    --media "examples/test-images/clay.jpg" \
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
    --media "examples/test-images/" \
    --model_folder "models/resnet34-hm/" \
    --use_onnx \
    --batch_size 4 \
    --output_dir "predictions/"
```

### Inference arguments

| Argument | Default | Description |
|---|---|---|
| `--name` | `heatmap` | Model type: `resnet`, `heatmap`, or `rcnn` |
| `--media` | — | Path to image, video, or folder |
| `--model_folder` | — | Path to the folder produced by training |
| `--use_onnx` | off | Use the ONNX runtime instead of PyTorch |
| `--fp32` | off | Use FP32 weights (default is FP16) |
| `--output_dir` | `predictions/` | Output directory for saved results |
| `--batch_size` | 4 | Batch size for folder prediction |

### Backend selection

`KeypointPredictor` tries ONNX first (if `--use_onnx` is set) and falls back to PyTorch if the session fails to initialise. When a CUDA-capable GPU is present, ONNX uses the `CUDAExecutionProvider`; otherwise it falls back to `CPUExecutionProvider`.

### Heatmap post-processing pipeline

For the heatmap model, raw heatmap outputs go through three sequential steps before the final keypoints are returned:

1. **Confidence filtering** — keypoints whose heatmap peak value is below a threshold are set to `(-1, -1)`.
2. **Overlap removal** — if two keypoints are within `pixel_distance` pixels of each other, the one with the lower confidence is removed.
3. **Minimum count check** — if fewer than `num_kps` valid keypoints remain, all keypoints are invalidated.

Invalid keypoints are represented as `(-1, -1)` throughout.

---

## Metrics

Two metrics are computed on the validation set after each epoch:

- **PCK@0.05** (Percentage of Correct Keypoints) — a keypoint is correct if its Euclidean distance to the ground truth is within 5 % of the image diagonal.
- **MPJPE** (Mean Per-Joint Position Error) — mean Euclidean distance in pixels between predicted and ground-truth keypoints after unscaling to original image size.

---

## Models

### ResNetHeatmap (primary)

ResNet-34 backbone with the final average-pool and fully-connected layers replaced. The last residual block uses dilated convolutions (dilation=2) to preserve spatial resolution. A lightweight heatmap head — two conv layers followed by two transposed convolutions — upsamples the feature map back to a resolution proportional to the input, producing one heatmap per keypoint.

### ResNetKeypoint

ResNet-18 backbone with the classifier head replaced by a two-layer MLP regression head (`Linear(512, 256) → ReLU → Dropout(0.25) → Linear(256, 28)`). Outputs 28 values representing the flat `[x1, y1, ..., x14, y14]` coordinates normalised to `[0, 1]`.

### KeypointRCNN

Wraps `torchvision.models.detection.keypointrcnn_resnet50_fpn` (pretrained on COCO). The keypoint predictor head is replaced to match the project's keypoint count. In training mode the model receives image tensors and target dicts and returns a loss dict. In eval mode it returns a list of detection dicts per image.

---

## Adding a new model

1. Implement an `nn.Module` in `src/models/` and export it from `src/models/__init__.py`.
2. Add a config class in `config/model_configs.py` inheriting from `BaseConfig`. Set at minimum: `model_name`, `criterion`, `scale`, `save_path`, `folder`.
3. Register the model in `src/utils/utils.py` inside `get_model_and_config()` and `load_model_inference()`.
4. If the model uses a non-standard loss or a different forward signature, add a branch in `compute_loss()` in `src/core/metrics.py`.