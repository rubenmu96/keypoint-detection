import numpy as np
import torch

from src.utils import extract_keypoints, keypoint_unscaler

def overlapping_kps(keypoints, max_values, pixel_distance):
    """
    Remove overlapping keypoints
    
    Args:
        keypoints: numpy array of shape (n_keypoints, 2) or (n_keypoints*2,)
        max_values: list or array of confidence values for each keypoint
        pixel_distance: minimum distance in pixels between keypoints
        image_width, image_height: dimensions for converting normalized coords to pixels
    """
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
    """Remove keypoints (set to (-1, -1)) if confidence is below threshold"""
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


def num_kps_req(keypoints, num_kps=7):
    """Remove all keypoints if the number of valid keypoints is less than required"""
    # TODO: need some rework, maybe require that both center keypoints and certain amount of keypoints should be present
    if keypoints.ndim == 1 or keypoints.shape[1] != 2:
        keypoints = keypoints.reshape(-1, 2)
    
    valid_kps = np.sum(np.all(keypoints != -1, axis=1))
    
    if valid_kps < num_kps:
        keypoints_copy = keypoints.copy()
        keypoints_copy[:] = -1
        return keypoints_copy.flatten()
    
    return keypoints.flatten()


def kps_postprocessor(keypoints, max_values, threshold, pixel_distance, num_kps=7):
    """
    Apply all keypoint post-processing steps
    
    Args:
        keypoints: numpy array of keypoints
        max_values: confidence values for each keypoint
        threshold: minimum confidence threshold
        pixel_distance: minimum pixel distance between keypoints
        num_kps: minimum number of required keypoints
    
    Returns:
        processed keypoints as flattened array
    """
    # Step 1: Filter low probability keypoints
    keypoints = filter_low_probabilities(keypoints, max_values, threshold)
    
    # Step 2: Remove overlapping keypoints
    keypoints = overlapping_kps(
        keypoints, max_values, pixel_distance
    )
    
    # Step 3: Check if we have enough keypoints
    keypoints = num_kps_req(keypoints, num_kps)
    
    return keypoints

def process_heatmap_keypoints(
        cfg, keypoints, threshold=-2, pixel_distance=10, num_kps=7, image_width=None, image_height=None
    ):
    """
    Complete pipeline for processing a batch of heatmaps into filtered keypoints.
    """
    keypoints, max_values = extract_keypoints(keypoints, return_max_values=True)
    
    processed_keypoints = []
    
    for b in range(len(keypoints)):
        batch_kps = keypoints[b]
        batch_max_vals = max_values[b]
        
        scaled_kps = keypoint_unscaler(
            cfg, batch_kps, image_width, image_height
        )
        
        kps_np = scaled_kps.cpu().numpy() if torch.is_tensor(scaled_kps) else scaled_kps
        
        processed_kps = kps_postprocessor(
            kps_np, batch_max_vals, threshold, pixel_distance, num_kps
        )
        
        processed_keypoints.append(processed_kps)
    
    return np.array(processed_keypoints)


def load_fp16_model(model, checkpoint_path, device):
    """Load model from FP16 checkpoint"""
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