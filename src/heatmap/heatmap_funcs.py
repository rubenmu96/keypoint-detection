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

def extract_keypoints(heatmaps):
    batch_size, num_keys, H, W = heatmaps.shape
    device = heatmaps.device
    
    keypoints = torch.zeros(batch_size, num_keys * 2, device=device)
    
    for b in range(batch_size):
        for k in range(num_keys):
            heatmap = heatmaps[b, k]
            _, max_idx = torch.max(heatmap.view(-1), dim=0)
            
            ## why y uses // and x uses %?
            y_px = max_idx // W
            x_px = max_idx % W # why not H?
            
            x = x_px.float() / (W - 1)
            y = y_px.float() / (H - 1)
            
            keypoints[b, 2*k] = x
            keypoints[b, 2*k+1] = y

    return keypoints