import numpy as np
import torch

from src.utils import extract_keypoints, keypoint_unscaler

def overlapping_kps(keypoints, max_values, pixel_distance):
    """Remove overlapping keypoints."""
    def remove_kps(pairs, max_values):
        remove = []
        for row, column in pairs:
            if max_values[row] > max_values[column]:
                remove.append(column)
            else:
                remove.append(row)
        return remove

    if keypoints.ndim == 1 or keypoints.shape[1] != 2:
        keypoints = keypoints.reshape(-1, 2)
    
    keypoints_copy = keypoints.copy()
    
    valid_mask = ~np.all(keypoints_copy == -1, axis=1)
    if not np.any(valid_mask):
        return keypoints_copy.flatten()
    
    valid_keypoints = keypoints_copy[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    n_valid = len(valid_keypoints)
    if n_valid < 2:
        return keypoints_copy.flatten()
    
    diff = valid_keypoints[:, np.newaxis, :] - valid_keypoints[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    
    close_pairs = []
    for i in range(n_valid):
        for j in range(i+1, n_valid):
            if dist_matrix[i, j] < pixel_distance:
                # Map back to original indices
                orig_i = valid_indices[i]
                orig_j = valid_indices[j]
                close_pairs.append((orig_i, orig_j))

    if close_pairs:
        remove_indices = remove_kps(close_pairs, max_values)
        keypoints_copy[remove_indices] = -1
    
    return keypoints_copy.flatten()

def filter_low_probabilities(keypoints, max_values, threshold):
    """Remove keypoints (set to (-1, -1)) if confidence is below threshold."""
    if keypoints.ndim == 1 or keypoints.shape[1] != 2:
        keypoints = keypoints.reshape(-1, 2)
    
    keypoints_copy = keypoints.copy()
    
    remove_kps = []
    for i, value in enumerate(max_values):
        if value < threshold:
            remove_kps.append(i)

    if remove_kps:
        keypoints_copy[remove_kps] = -1
    
    return keypoints_copy.flatten()


def num_kps_req(keypoints, num_kps=7, required_indices=None):
    """
    Invalidate all keypoints if quality thresholds are not met.

    Two independent checks can be configured:
    - ``num_kps``: minimum number of valid keypoints required in total.
    - ``required_indices``: list of keypoint indices that *must all* be valid
      (e.g. ``[-2, -1]`` for the two center-court keypoints).

    If either check fails, every keypoint is set to (-1, -1).
    """
    if keypoints.ndim == 1 or keypoints.shape[1] != 2:
        keypoints = keypoints.reshape(-1, 2)

    valid_mask = np.all(keypoints != -1, axis=1)
    failed = np.sum(valid_mask) < num_kps

    if not failed and required_indices is not None:
        n = len(keypoints)
        for idx in required_indices:
            if not valid_mask[idx % n]:
                failed = True
                break

    if failed:
        keypoints_copy = keypoints.copy()
        keypoints_copy[:] = -1
        return keypoints_copy.flatten()

    return keypoints.flatten()


def kps_postprocessor(
        keypoints, max_values, threshold, pixel_distance, num_kps=7, required_indices=None
    ):
    """Apply all keypoint post-processing steps"""
    keypoints = filter_low_probabilities(keypoints, max_values, threshold)
    keypoints = overlapping_kps(keypoints, max_values, pixel_distance)
    keypoints = num_kps_req(keypoints, num_kps, required_indices)
    return keypoints

def process_heatmap_keypoints(
        cfg, keypoints, threshold=-2, pixel_distance=10,
        num_kps=7, image_width=None, image_height=None
    ):
    """
    Pipeline for processing a batch of heatmaps into filtered keypoints during inference.
    """
    # Turn heatmap into keypoints
    keypoints, max_values = extract_keypoints(keypoints, return_max_values=True)
    
    processed_keypoints = []
    for b in range(len(keypoints)):
        batch_kps = keypoints[b]
        batch_max_vals = max_values[b]
        
        # Unscale keypoints
        scaled_kps = keypoint_unscaler(
            cfg, batch_kps, image_width, image_height
        )
        
        kps_np = scaled_kps.cpu().numpy() if torch.is_tensor(scaled_kps) else scaled_kps
        
        # Post-processing to remove low-threshold and overlapping keypoints
        processed_kps = kps_postprocessor(
            kps_np, batch_max_vals, threshold, pixel_distance, num_kps
        )
        
        processed_keypoints.append(processed_kps)
    
    return np.array(processed_keypoints)


def load_fp16_model(model, checkpoint_path, device):
    """Load model from FP16 checkpoint else FP32"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        dtype_info = checkpoint.get('dtype', 'fp32')
        model.load_state_dict(state_dict)
        
        if dtype_info == 'fp16' and device != 'cpu':
            model = model.half()
        print(f"Loaded model with dtype: {dtype_info}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model from legacy format (FP32)")
    
    return model