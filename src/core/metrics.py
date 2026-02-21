import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import create_heatmap, keypoint_unscaler


loss_dictionary = {
    "bcelogitloss": nn.BCEWithLogitsLoss(),
    "mseloss": nn.MSELoss(),
    "smoothl1loss": nn.SmoothL1Loss(),
    "bcelogitloss_unreduced": nn.BCEWithLogitsLoss(reduction='none'),
    "mseloss_unreduced": nn.MSELoss(reduction='none'),
}


def _calculate_keypoint_loss(
    pred_kps,
    target_kps,
    criterion,
    num_kps,
    use_visibility_mask=True,
    visibility_threshold=0,  # 0=not labeled, 1=occluded, 2=visible
):
    """
    Calculate keypoint loss with proper visibility masking and multi-object support.
    
    Args:
        pred_kps: List of prediction dicts with 'keypoints' tensor of shape [N, K, 3]
        target_kps: List of target dicts with 'keypoints' tensor of shape [M, K, 3]
        criterion: Loss function to use
        num_kps: Number of keypoints
        use_visibility_mask: Whether to mask loss by visibility flags
        visibility_threshold: Minimum visibility value to include in loss
    
    Returns:
        Computed loss value
    """
    device = pred_kps[0]["keypoints"].device if pred_kps[0]["keypoints"].numel() > 0 else "cpu"
    
    total_loss = torch.tensor(0.0, device=device)
    valid_count = 0
    
    for pred_dict, target_dict in zip(pred_kps, target_kps):
        pred = pred_dict["keypoints"]  # [N_pred, K, 3]
        target = target_dict["keypoints"]  # [N_target, K, 3]
        
        # Skip if either is empty
        if pred.shape[0] == 0 or target.shape[0] == 0:
            continue
        
        # Match predictions to targets (simple: use min of both counts)
        # For better matching, consider using Hungarian algorithm with IoU
        num_objects = min(pred.shape[0], target.shape[0])
        
        pred_coords = pred[:num_objects, :, :2]  # [N, K, 2]
        target_coords = target[:num_objects, :, :2]  # [N, K, 2]
        target_visibility = target[:num_objects, :, 2]  # [N, K]
        
        if use_visibility_mask:
            # Create mask for valid keypoints (visibility > threshold)
            visibility_mask = (target_visibility > visibility_threshold).float()  # [N, K]
            
            # Compute per-keypoint loss
            per_kp_loss = F.mse_loss(pred_coords, target_coords, reduction='none')  # [N, K, 2]
            per_kp_loss = per_kp_loss.mean(dim=-1)  # [N, K]
            
            # Apply visibility mask
            masked_loss = per_kp_loss * visibility_mask
            
            # Normalize by number of valid keypoints
            num_valid = visibility_mask.sum()
            if num_valid > 0:
                total_loss += masked_loss.sum() / num_valid
                valid_count += 1
        else:
            total_loss += criterion(pred_coords, target_coords)
            valid_count += 1
    
    return total_loss / max(valid_count, 1)

def compute_loss(cfg, criterion, preds, targets):
    if cfg.model_name == "ResNetHeatmap":
        criterion_fn = loss_dictionary[criterion]
        targets = create_heatmap(
            targets, output_shape=(preds.shape[2], preds.shape[3]), sigma=cfg.sigma
        )
        return criterion_fn(preds, targets)
    
    elif cfg.model_name == "KeypointRCNN":        
        num_kps = getattr(cfg, 'num_kps', 14)
        
        criterion_fn = loss_dictionary.get(criterion, nn.MSELoss())
        return _calculate_keypoint_loss(
            preds, targets, 
            criterion=criterion_fn,
            num_kps=num_kps,
            use_visibility_mask=True
        )
    
    elif cfg.model_name == "ResNetKeypoint":
        criterion_fn = loss_dictionary[criterion]
        return criterion_fn(preds, targets)
    
    else:
        raise ValueError(f"Unknown model: {cfg.model_name}")
    

from src.utils import extract_keypoints
import numpy as np

"""
Simple accuracy metrics for keypoint detection
- pck - 
- mpjpe - 
"""

def compute_pck(
    preds,
    targets,
    image_size,
    threshold=0.05,
):
    # Reshape to [B, K, 2] if flattened
    if preds.ndim == 2:
        preds = preds.view(preds.shape[0], -1, 2)
    if targets.ndim == 2:
        targets = targets.view(targets.shape[0], -1, 2)
    
    # Cast to float to avoid norm() error with integer tensors
    preds = preds.float()
    targets = targets.float()
    
    batch_size, num_keypoints, _ = preds.shape
    
    # Compute Euclidean distances
    distances = torch.norm(preds - targets, dim=-1)  # [B, K]

    h, w = image_size
    normalizer = np.sqrt(h**2 + w**2)  # Image diagonal
    
    # Normalize distances and check against threshold
    normalized_distances = distances / normalizer
    correct = (normalized_distances <= threshold).float()
    
    # Compute metrics
    pck_per_keypoint = correct.mean(dim=0)  # [K]
    pck_overall = correct.mean().item()
    
    return {
        "pck": pck_overall,
        "pck_per_keypoint": pck_per_keypoint.cpu().numpy(),
        "num_correct": correct.sum().item(),
        "num_total": batch_size * num_keypoints,
    }

def compute_mpjpe(
    preds,
    targets,
):
    if preds.ndim == 2:
        preds = preds.view(preds.shape[0], -1, 2)
    if targets.ndim == 2:
        targets = targets.view(targets.shape[0], -1, 2)
    
    # Cast to float
    preds = preds.float()
    targets = targets.float()
    
    # Euclidean distance per keypoint
    distances = torch.norm(preds - targets, dim=-1)  # [B, K]
    
    mpjpe = distances.mean()
    mpjpe_per_kp = distances.mean(dim=0)
    
    return {
        "mpjpe": mpjpe.item(),
        "mpjpe_per_keypoint": mpjpe_per_kp.cpu().numpy(),
        "std": distances.std().item(),
    }

def compute_accuracy(cfg, preds, targets, image_size=None):
    if cfg.model_name == "ResNetHeatmap":
        preds = extract_keypoints(preds)

    elif cfg.model_name == "KeypointRCNN":
        pred_coords = []
        target_coords = []
        
        for pred_dict, target_dict in zip(preds, targets):
            if pred_dict["keypoints"].shape[0] > 0:
                pred_coords.append(pred_dict["keypoints"][0, :, :2])
                target_coords.append(target_dict["keypoints"][0, :, :2])
        
        if not pred_coords:
            return {"pck": 0.0, "mpjpe": float('inf')}
        
        preds = torch.stack(pred_coords)
        targets = torch.stack(target_coords)

    if image_size is None:
        image_size = (cfg.height, cfg.width)
    
    h, w = image_size

    preds = keypoint_unscaler(
        cfg, preds, orig_width=w, orig_height=h
    )
    targets = keypoint_unscaler(
        cfg, targets, orig_width=w, orig_height=h
    )

    # Compute metrics
    pck_results = compute_pck(preds, targets, threshold=0.05, image_size=image_size)
    mpjpe_results = compute_mpjpe(preds, targets)
    
    return {
        "pck@0.05": pck_results["pck"],
        "pck@0.1": compute_pck(preds, targets, threshold=0.1, image_size=image_size)["pck"],
        "mpjpe": mpjpe_results["mpjpe"],
    }