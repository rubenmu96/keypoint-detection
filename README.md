# Keypoint detection
Keypoint detection for tennis field.

At each tennis half, we want to detect 7 keypoints. 

### Dataset
The dataset consists of...

### How to run
Create virtual environment and install packages by using ```poetry install --no-root```

Help: ```poetry show -v``` to get virtualenv path, then change to this path if necessary (in VSCode: ```ctrl + shifft + p```, choose ```Python: Select Interpreter```, copy the path). 

To train the model, run...

To run the model on a tennis match, use inference_image.py for image and inference_video.py for video.

TEST MODEL ON YOUTUBE TENNIS MATCH! Look into inference time of the 3 different models

## Models
- ResNet
- ResNet with heatmap classification head
- keypoint-rcnn

The inference.py supports folder prediction (for images), image prediction and video prediction. The format (folder, image or video) is automatically detected.

Main focus on ResNet with heatmap classification head.

## Performance



### Improvements (show performance on the 4 test images?)


## TODO
- converting models to onnx and inference pipeline with support for onnx
- get a better understanding of keypoint-rcnn
- improve loss calculations + add more losses?
- Test model_size change in heatmap
- Write a readme
- poetry for newest cuda
- Optimize dataset.py
- Decide if dropping or keeping Keypoint-RCNN


## TODO (later)
- Checkpoint training
- Multiscale training
- Batch inference for video
- Keypoint-rcnn?
- Create better data augmentation pipeline