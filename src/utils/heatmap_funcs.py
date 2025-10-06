import torch

def create_heatmap(keypoints, output_shape, sigma=1):
    batch_size = keypoints.shape[0]
    num_keys = keypoints.shape[1] // 2
    H, W = output_shape
    
    heatmaps = torch.zeros(batch_size, num_keys, H, W, device=keypoints.device)
    
    for b in range(batch_size):
        for k in range(num_keys):
            x, y = keypoints[b, 2*k], keypoints[b, 2*k+1]
            
            if x > 0 or y > 0:
                x_coord = torch.arange(0, W, device=keypoints.device).float()
                y_coord = torch.arange(0, H, device=keypoints.device).float()
                yy, xx = torch.meshgrid(y_coord, x_coord, indexing="ij")
                
                x_px = x * (W - 1)
                y_px = y * (H - 1)
                
                heatmaps[b,k] = torch.exp(-((xx - x_px)**2 + (yy - y_px)**2) / (2 * sigma**2))
    return heatmaps

# def extract_keypoints(heatmaps, return_max_values=False):
#     """
#     Extract keypoints from heatmaps
    
#     Coordinate calculation
#     - y = idx // W: gets the row (y-coordinate)
#     - x = idx % W: gets the column (x-coordinate)
#     This follows standard row-major indexing where idx = y*W + x
#     """
#     batch_size, num_keys, H, W = heatmaps.shape
#     device = heatmaps.device
    
#     keypoints = torch.zeros(batch_size, num_keys * 2, device=device)
#     max_values = []
#     for b in range(batch_size):
#         for k in range(num_keys):
#             heatmap = heatmaps[b, k]
#             max_val, max_idx = torch.max(heatmap.view(-1), dim=0)
            
#             ## why y uses // and x uses %?
#             y_px = max_idx // W
#             x_px = max_idx % W # why not H?
            
#             x = x_px.float() / (W - 1)
#             y = y_px.float() / (H - 1)
            
#             keypoints[b, 2*k] = x
#             keypoints[b, 2*k+1] = y
#             max_values.append(max_val)

#     if return_max_values:
#         return keypoints, max_values
#     return keypoints

def extract_keypoints(heatmaps, return_max_values=False):
    """
    Extract keypoints from heatmaps
    
    Coordinate calculation
    - y = idx // W: gets the row (y-coordinate)
    - x = idx % W: gets the column (x-coordinate)
    This follows standard row-major indexing where idx = y*W + x
    """
    batch_size, num_keys, H, W = heatmaps.shape
    device = heatmaps.device
    
    keypoints = torch.zeros(batch_size, num_keys * 2, device=device)
    max_values = []
    
    for b in range(batch_size):
        batch_max_values = []
        for k in range(num_keys):
            heatmap = heatmaps[b, k]
            max_val, max_idx = torch.max(heatmap.view(-1), dim=0)
            
            # Convert flattened index back to 2D coordinates
            y_px = max_idx // W  # Row index
            x_px = max_idx % W   # Column index (not % H because we're indexing columns)
            
            # Normalize coordinates to [0, 1]
            x = x_px.float() / (W - 1) if W > 1 else 0.0
            y = y_px.float() / (H - 1) if H > 1 else 0.0
            
            keypoints[b, 2*k] = x
            keypoints[b, 2*k+1] = y
            batch_max_values.append(max_val.cpu().item())
        
        max_values.append(batch_max_values)

    if return_max_values:
        return keypoints, max_values
    return keypoints