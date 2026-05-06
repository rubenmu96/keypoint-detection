# Keypoint detection
Court keypoint detection for tennis match footage. The model detects **14 keypoints** — 7 per half of the court. Three model architectures are supported with a shared training and inference pipeline:

| Model | Architecture | Output |
|---|---|---|
| `resnet` | ResNet backbone + regression head | `[B, 28]` flat coordinates |
| `heatmap` | ResNet backbone + dilated conv + heatmap head | `[B, 14, H, W]` heatmaps |
| `rcnn` | Keypoint R-CNN (ResNet-50 FPN) | list of detection dicts |

The **heatmap model** (with `resnet` as backbone) is the primary model. It produces one Gaussian heatmap per keypoint, extracts the argmax coordinate, and passes the result through a post-processing pipeline (confidence filtering, overlap removal, minimum keypoint count check).

The ResNet heatmap model and Keypoint R-CNN both uses heatmaps to predict keypoints. At the keypoint location $(x_k^*, y_k^*)$ we have the value

$$\mathbf{H}_k(x, y) = \exp\!\left(-\frac{(x - x_k^*)^2 + (y - y_k^*)^2}{2\sigma^2}\right),$$

where $\sigma$ controls the spread of the Gaussian. A small $\sigma$ will give us higher localization accuracy but will make optimziation harder. A larger $\sigma$ is more forgiving but less precise. 

At inference time, the predicted keypoint location is recovered as the position of the maximum activation in the heatmap:

$$(\hat{x}_k, \hat{y}_k) = \arg \max_{(x,y)}\, \mathbf{H}_k(x, y).$$

Read this: https://medium.com/@kosolapov.aetp/tennis-analysis-using-deep-learning-and-machine-learning-a5a74db7e2ee

### Metrics
Accuracy metrics used to evalulate the models are PCK@0.10, PCK@0.05 and MPJPE. PCK stands for **percentage of correct keypoints** and MPJPE stands for **mean per joint position error**. MPJPE is mainly used for 3D human pose estimation, but can be used in 2D also by calculating the pixel-wise distance. 
$$
    \text{MPJPE}_{2\text{D}} = \frac{1}{K} \sum_{k=1}^K \sqrt{(x_k - \hat{x}_k)(y_k - \hat{y}_k^*)}
$$
PCK also used the Euclidean distance, but is normalized by the image diagonal. Then given a threshold, we count how many normalized distances are within the threshold. 


---
### Dataset
The dataset is taken from https://github.com/yastrebksv/TennisCourtDetector. In total there are 8841 images, where 75% are training images and 25% are validation images.

The dataset is a JSON-annotated collection of tennis broadcast frames. Each record contains:

- `id` — image filename stem (image file is `<img_dir>/<id>.png`)
- `kps` — list of `[x, y]` pixel coordinates, one pair per keypoint (14 keypoints = 28 values)

Expected layout:

```
dataset/tennis_court_det_dataset/data/
├── data_train.json
├── data_val.json
└── images/
    ├── <id>.png
    └── ...
```

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


TODO: put the sample images under examples

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
    --media "examples/tennis_match.mp4" \
    --model_folder "models/resnet34-hm/" \
    --use_onnx

# Video (ONNX FP32)
python inference.py \
    --media "examples/tennis_match.mp4" \
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

Below are two videos displaying the results of ResNet34-heatmap. Second video shows that the post-processing is able to handle the camera moving. 

![Tennis match predictions](examples/example1.gif)

![Tennis match predictions](examples/example2.gif)


## Future updates
- Checkpoint training
- Multiscale training
- Video inference with batch size
- Optimize Keypoint R-CNN more
- Create better data augmentation pipeline and try more augmentation techniques
- Homography for post-processing