from src.utils import create_heatmap
import torch
import torch.functional as F

loss_dictionary = {
    "bcelogitloss": torch.nn.BCEWithLogitsLoss(),
    "mseloss": torch.nn.MSELoss(),
    "smoothl1loss": torch.nn.SmoothL1Loss()
}

def _calculate_keypoint_mse(pred_kps, target_kps, criterion=None):
    # Also use num_kps as argument instead of 14? or -1 if it works?
    def select_first_object(keypoints_list):
        return [kps[0:1] if kps.shape[0] > 0 else torch.zeros((1, 14, 3), device=kps.device) for kps in keypoints_list]
    
    # Usage:
    target_kps = select_first_object([t["keypoints"] for t in target_kps])
    pred_kps = select_first_object([p["keypoints"] for p in pred_kps])
    
    # Now stack safely (all [1, 14, 3])
    stacked_targets = torch.stack(target_kps)
    stacked_preds = torch.stack(pred_kps)
    
    # Compute MSE (swap with criterion instead?)
    mse = F.mse_loss(stacked_preds[..., :2], stacked_targets[..., :2])
    return mse

# def compute_loss(cfg, criterion, preds, targets):
#     if cfg.model_name == "ResNetHeatmap":
#         target_heatmap = create_heatmap(
#             targets, output_shape=(preds.shape[2], preds.shape[3]), sigma=cfg.sigma
#         )
#         return criterion(preds, target_heatmap)
#     elif cfg.model_name == "KeypointRCNN": # only during evaluation
#         return _calculate_keypoint_mse(preds, targets, criterion)
#     else:
#         return criterion(preds, targets)

def compute_loss(cfg, criterion, preds, targets):
    criterion = loss_dictionary[criterion]
    if cfg.model_name == "ResNetHeatmap":
        targets = create_heatmap(
            targets, output_shape=(preds.shape[2], preds.shape[3]), sigma=cfg.sigma
        )
    elif cfg.model_name == "KeypointRCNN": # only during evaluation
        loss = _calculate_keypoint_mse(preds, targets, criterion)
        return loss
    
    loss = criterion(preds, targets)
    return loss