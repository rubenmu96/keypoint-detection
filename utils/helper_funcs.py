import torch
from models import (
    ResNetKeypoint,
    ResNetHeatmap,
    KeypointRCNN
)
from config import (
    ResNetConfig,
    HeatmapConfig,
    RCNNConfig,
)

def keypoint_scaler(kps, width, height, scale=None):
    if scale == (-1, 1):
        kps[::2] = 2.0 * kps[::2] / width - 1 
        kps[1::2] = 2.0 * kps[1::2] / height - 1
    elif scale == (0, 1):
        kps[::2] = kps[::2] / width
        kps[1::2] = kps[1::2] / height
    return kps

def keypoint_unscaler(cfg, kps, orig_width, orig_height):
    if cfg.scale == (-1, 1):
        kps[::2] = (kps[::2] + 1) / 2
        kps[1::2] = (kps[1::2] + 1) / 2
    elif cfg.scale is None:
        kps[::2] = kps[::2] / cfg.width
        kps[1::2] = kps[1::2] / cfg.height

    kps[::2] = kps[::2] * orig_width
    kps[1::2] = kps[1::2] * orig_height
    return kps

def are_keypoints_valid(keypoints_list, max_width=1280, max_height=720, offset=10):
    """Check if all keypoints in a list are within bounds."""
    # do something with visibility here?
    for x, y in keypoints_list:
        if not (offset <= x <= max_width - offset and offset <= y <= max_height - offset):
            return False
    return True

# change name, as it creates a boundary box around all the boundary boxes
def keypoints_to_bbox(keypoints, offset=10, width=None, height=None):
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.numpy()
    if keypoints.ndim == 2 and keypoints.shape[1] == 3:
        keypoints = keypoints[:, :2]
    
    x_min = max(0, keypoints[:, 0].min() - offset)
    y_min = max(0, keypoints[:, 1].min() - offset)
    x_max = keypoints[:, 0].max() + offset
    y_max = keypoints[:, 1].max() + offset
    
    # Clip to image bounds
    if width is not None:
        x_max = min(x_max, width - 1)
    if height is not None:
        y_max = min(y_max, height - 1)
    
    return torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)

def keypoints_with_visibility(kps, visibility=None):
    if visibility is None:
        visibility = [1] * len(kps)
    return torch.tensor([[x, y, v] for (x, y), v in zip(kps, visibility)], dtype=torch.float32)

def get_model_and_config(name="resnet", classes=None):
    name = name.lower()
    
    if name == "resnet":
        cfg = ResNetConfig
        # check if possible to load something like **kwargs (cfg), then if they are different from default, use them instead?
        model = ResNetKeypoint(
            model=cfg.model,
            weights=cfg.weights,
            num_kps=cfg.num_coords,
            pretrained=cfg.pretrained
        )
    elif name == "heatmap":
        cfg = HeatmapConfig
        model = ResNetHeatmap(
            model=cfg.model,
            weights=cfg.weights,
            num_kps=cfg.num_kps,
            model_size=cfg.model_size,
            input_size=cfg.input_size,
            pretrained=cfg.pretrained
        )
    elif name == "rcnn":
        cfg = RCNNConfig
        model = KeypointRCNN(cfg.num_kps)
    else: # TODO: create assert error if not model name is segformer or smp
        pass 
    cfg.model_name = model.__class__.__name__
    return model, cfg