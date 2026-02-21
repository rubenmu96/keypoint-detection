from config import BaseConfig

# TODO: need to add folder for saving model

class ResNetConfig(BaseConfig):
    model = "resnet18"
    pretrained = True
    criterion = "smoothl1loss"
    scale = (0, 1)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # num_kps = 14
    # num_coords = 28
    width = 672
    height = 448
    # Saving and converting to onnx
    save_path = f"{model}_{width}x{height}.pth"
    folder = f"models/{model}"
    onnx = False # need to add support
    onnx_save_path = f"{model}_{width}x{height}"

class HeatmapConfig(BaseConfig):
    model = "resnet34"
    pretrained = True
    criterion = "bcelogitloss"
    sigma = 1.4
    scale = (0, 1)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # num_kps = 14
    # num_coords = 28
    width = 672
    height = 448
    # Saving and converting to onnx
    save_path = f"heatmap_{model}_{width}x{height}.pth"
    folder = f"models/{model}-hm"
    onnx = True # will save onnx
    onnx_save_path = f"heatmap_{model}_{width}x{height}"

class RCNNConfig(BaseConfig):
    base_config = BaseConfig
    mean = (0, 0, 0)
    std = (1, 1, 1)
    criterion = "mseloss"
    scale = None
    # num_kps = 14
    # num_coords = 28
    width = 672
    height = 448
    # Saving and converting to onnx
    folder = f"models/keypoint-rcnn"
    save_path = f"keypoint_rcnn_{width}x{height}.pth"
    onnx = False # need to add support