from config import BaseConfig

class ResNetConfig(BaseConfig):
    task = "resnet"
    model = "resnet18"
    pretrained = True
    criterion = "smoothl1loss" # loss function
    scale = (0, 1)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    width = 672
    height = 448
    # Saving and converting to onnx
    save_path = f"{model}_{width}x{height}.pth"
    folder = f"models/{model}"
    onnx = False
    onnx_save_path = f"{model}_{width}x{height}"

class HeatmapConfig(BaseConfig):
    task = "heatmap"
    model = "resnet34"
    pretrained = True
    criterion = "bcelogitloss" # loss function
    sigma = 1.4
    scale = (0, 1)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    width = 672
    height = 448
    # Saving and converting to onnx
    save_path = f"heatmap_{model}_{width}x{height}.pth"
    folder = f"models/{model}-hm"
    onnx = True # will save onnx
    onnx_save_path = f"heatmap_{model}_{width}x{height}"

class RCNNConfig(BaseConfig):
    task = "rcnn"
    base_config = BaseConfig
    # Is being skipped for Keypoint R-CNN
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    scale = None
    criterion = "mseloss" # loss function
    width = 672
    height = 448
    # Saving and converting to onnx
    folder = f"models/keypoint-rcnn"
    save_path = f"keypoint_rcnn_{width}x{height}.pth"
    onnx = True