import torch

def keypoint_scaler(kps, width, height, scale=None):
    """
    Scale keypoints given scale.
    
    Supports (-1, 1), (0, 1) and no scaling.
    """
    if scale == (-1, 1):
        kps[::2] = 2.0 * kps[::2] / width - 1 
        kps[1::2] = 2.0 * kps[1::2] / height - 1
    elif scale == (0, 1):
        kps[::2] = kps[::2] / width
        kps[1::2] = kps[1::2] / height
    return kps

# def keypoint_unscaler(cfg, kps, orig_width, orig_height):
#     kps = kps.clone() if torch.is_tensor(kps) else kps.copy()
#     if cfg.scale == (-1, 1):
#         kps[::2] = (kps[::2] + 1) / 2
#         kps[1::2] = (kps[1::2] + 1) / 2
#     elif cfg.scale is None:
#         kps[::2] = kps[::2] / cfg.width
#         kps[1::2] = kps[1::2] / cfg.height
#     kps[::2] = kps[::2] * orig_width
#     kps[1::2] = kps[1::2] * orig_height
    
#     return kps

def keypoint_unscaler(cfg, kps, orig_width, orig_height):
    """Unscale keypoints given scale."""
    kps = kps.clone() if torch.is_tensor(kps) else kps.copy()
    # kps = kps.float() # TODO: hmm
    
    # [B, K, 2] or [K, 2] shape
    if kps.ndim >= 2 and kps.shape[-1] == 2:
        if cfg.scale == (-1, 1):
            kps[..., 0] = (kps[..., 0] + 1) / 2
            kps[..., 1] = (kps[..., 1] + 1) / 2
        elif cfg.scale is None:
            kps[..., 0] = kps[..., 0] / cfg.width
            kps[..., 1] = kps[..., 1] / cfg.height
        kps[..., 0] = kps[..., 0] * orig_width
        kps[..., 1] = kps[..., 1] * orig_height
    
    # [x1, y1, x2, y2, ...]
    else:
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
    for x, y in keypoints_list:
        if not (offset <= x <= max_width - offset and offset <= y <= max_height - offset):
            return False
    return True

def keypoints_region(keypoints, offset=10, width=None, height=None):
    """Calculates a region all the keypoints can be found in."""
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.numpy()
    if keypoints.ndim == 2 and keypoints.shape[1] == 3:
        keypoints = keypoints[:, :2]
    
    x_min = max(0, keypoints[:, 0].min() - offset)
    y_min = max(0, keypoints[:, 1].min() - offset)
    x_max = keypoints[:, 0].max() + offset
    y_max = keypoints[:, 1].max() + offset
    
    if width is not None:
        x_max = min(x_max, width - 1)
    if height is not None:
        y_max = min(y_max, height - 1)
    
    return torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)

def keypoints_with_visibility(kps, visibility=None):
    """Get if keypoints are visible or not."""
    if visibility is None:
        visibility = [1] * len(kps)
    return torch.tensor([[x, y, v] for (x, y), v in zip(kps, visibility)], dtype=torch.float32)