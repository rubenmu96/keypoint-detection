import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import (
    extract_keypoints,
    create_heatmap,
    keypoint_unscaler
)

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
    visibility_threshold=0,  # 0=not labeled, 1=occluded, 2=visible
):
    """
    Calculate keypoint loss with visibility masking and multi-object support.

    Args:
        pred_kps: List of prediction dicts with 'keypoints' tensor of shape [N_pred, K, 3]
        target_kps: List of target dicts with 'keypoints' tensor of shape [N_target, K, 3]
        visibility_threshold: Minimum visibility value to include in loss

    Returns:
        Mean loss over all images that had at least one valid detection.
        Returns 0.0 with a warning when every image in the batch had zero detections.
    """
    device = pred_kps[0]["keypoints"].device if pred_kps[0]["keypoints"].numel() > 0 else "cpu"

    total_loss = torch.tensor(0.0, device=device)
    valid_count = 0

    for pred_dict, target_dict in zip(pred_kps, target_kps):
        pred = pred_dict["keypoints"]    # [N_pred, K, 3]
        target = target_dict["keypoints"]  # [N_target, K, 3]

        if pred.shape[0] == 0 or target.shape[0] == 0:
            continue

        """
        Pair each ground-truth instance with the best-matching prediction.
        torchvision returns predictions sorted by descending score, so index 0
        is the highest-confidence detection, the right choice for the
        single-instance case. For multi-instance datasets a proper assignment
        algorithm (e.g. Hungarian matching on IoU) would be needed here.
        """
        num_objects = min(pred.shape[0], target.shape[0])

        pred_coords = pred[:num_objects, :, :2].float()    # [N, K, 2]
        target_coords = target[:num_objects, :, :2].float()  # [N, K, 2]
        target_visibility = target[:num_objects, :, 2]     # [N, K]

        visibility_mask = (target_visibility > visibility_threshold).float()  # [N, K]
        per_kp_loss = F.mse_loss(pred_coords, target_coords, reduction='none')  # [N, K, 2]
        per_kp_loss = per_kp_loss.mean(dim=-1)  # [N, K]

        masked_loss = per_kp_loss * visibility_mask

        num_valid = visibility_mask.sum()
        if num_valid > 0:
            total_loss += masked_loss.sum() / num_valid
            valid_count += 1

    if valid_count == 0:
        warnings.warn(
            "KeypointRCNN: no valid detections in this batch — loss is set to 0.0",
            RuntimeWarning,
            stacklevel=2,
        )

    return total_loss / max(valid_count, 1)


def compute_loss(cfg, preds, targets):
    """Compute the loss"""
    criterion_fn = loss_dictionary[cfg.criterion]

    if cfg.model_name == "ResNetHeatmap":
        targets = create_heatmap(
            targets, output_shape=(preds.shape[2], preds.shape[3]), sigma=cfg.sigma
        )
        return criterion_fn(preds, targets)
    
    elif cfg.model_name == "KeypointRCNN":
        return _calculate_keypoint_loss(preds, targets)
    
    elif cfg.model_name == "ResNetKeypoint":
        return criterion_fn(preds, targets)
    
    else:
        raise ValueError(f"Unknown model: {cfg.model_name}")

def compute_pck(preds, targets, image_size, threshold=0.05):
    """
    PCK (Percentage of Correct Keypoints).

    A keypoint is "correct" if its Euclidean distance to the ground-truth is
    within `threshold * diag` pixels, where diag = sqrt(H^2 + W^2) is the
    image diagonal used as the normalizer.  Normalizing by the diagonal makes
    the threshold scale-invariant across different image resolutions.

    PCK@0.05 means threshold = 5 % of the image diagonal.

    A bigger value is better, between 0-1.

    Args:
        preds: Predicted keypoint coordinates, shape [B, K, 2] or [B, K*2].
        targets: Ground-truth keypoint coordinates, same shape as preds.
        image_size: (height, width) of the image used to compute the diagonal
            normalizer.
        threshold: Distance threshold as a fraction of the image diagonal.

    Returns:
        dict with keys:
            'pck' (float): Mean fraction of correct keypoints across the batch.
            'pck_per_keypoint' (np.ndarray, shape [K]): Per-keypoint PCK.
            'num_correct' (float): Total correct keypoint predictions.
            'num_total' (int): Total keypoint predictions (B * K).
    """
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

def compute_mpjpe(preds, targets):
    """
    MPJPE (Mean Per-Joint Position Error).

    For each keypoint the Euclidean distance between prediction and ground-truth
    is computed, then averaged across all keypoints and all samples in the batch.
    Reported in pixels (at the unscaled / original image resolution).

    A lower value is better.

    Args:
        preds: Predicted keypoint coordinates, shape [B, K, 2] or [B, K*2].
        targets: Ground-truth keypoint coordinates, same shape as preds.

    Returns:
        dict with keys:
            'mpjpe' (float): Mean error in pixels across all keypoints and samples.
            'mpjpe_per_keypoint' (np.ndarray, shape [K]): Per-keypoint mean error.
            'std' (float): Standard deviation of per-keypoint distances.
    """
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

def compute_accuracy(cfg, preds, targets, image_size=None, thresholds=[0.05, 0.10]):
    """
    Compute accuracy metrics for keypoint detection models.

    Args:
        cfg: Config object with model_name, height, width, and scaler settings.
        preds: Model predictions. Shape depends on model — heatmap tensor for
            ResNetHeatmap, flat/shaped keypoint tensor for ResNetKeypoint, or a
            list of dicts with 'keypoints' of shape [N, K, 3] for KeypointRCNN.
        targets: Ground-truth keypoints in the same format as preds.
        image_size: (height, width) of the original image used as the PCK
            normalizer. Falls back to (cfg.height, cfg.width) when None.
        thresholds: PCK distance thresholds as fractions of the image diagonal.
            Defaults to [0.05, 0.10].

    Returns:
        dict with keys "mpjpe" (float, pixels) and "pck@{thr}" (float, 0–1)
        for each threshold. Returns mpjpe=inf and pck=0.0 for all thresholds
        when KeypointRCNN produces no detections in the batch.
    """
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
            metrics = {}
            metrics["mpjpe"] = float("inf")
            for thr in thresholds:
                metrics[f"pck@{thr}"] = 0.0
            return metrics
        
        preds = torch.stack(pred_coords)
        targets = torch.stack(target_coords)

    if image_size is None:
        image_size = (cfg.height, cfg.width)
    
    h, w = image_size

    # Skip scaling of Keypoint R-CNN
    if cfg.model_name != "KeypointRCNN":
        preds = keypoint_unscaler(cfg, preds, orig_width=w, orig_height=h)
        targets = keypoint_unscaler(cfg, targets, orig_width=w, orig_height=h)

    metrics = {}
    metrics["mpjpe"] = compute_mpjpe(preds, targets)["mpjpe"]
    
    for thr in thresholds:
        metrics[f"pck@{thr}"] = compute_pck(
            preds, targets, image_size=(h, w), threshold=thr
        )["pck"]

    return metrics