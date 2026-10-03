"""Convert between keypoint coordinates and Gaussian heatmaps."""
import torch

def create_heatmap(keypoints, output_shape, sigma=1) -> torch.Tensor:
    """Render one Gaussian heatmap per keypoint.

    keypoints is [B, K * 2] with x and y normalized to [0, 1]. Returns a
    [B, K, H, W] tensor for output_shape (H, W), where each channel holds a
    Gaussian with standard deviation sigma pixels centered on its keypoint.
    A keypoint with a negative coordinate is treated as missing and gives an
    all-zero channel.
    """
    batch_size = keypoints.shape[0]
    num_keys = keypoints.shape[1] // 2
    height, width = output_shape

    heatmaps = torch.zeros(
        batch_size, num_keys, height, width,
        device=keypoints.device
    )

    for b in range(batch_size):
        for k in range(num_keys):
            x, y = keypoints[b, 2*k], keypoints[b, 2*k+1]

            if x >= 0 and y >= 0:
                x_coord = torch.arange(0, width, device=keypoints.device).float()
                y_coord = torch.arange(0, height, device=keypoints.device).float()
                yy, xx = torch.meshgrid(y_coord, x_coord, indexing="ij")

                x_px = x * (width - 1)
                y_px = y * (height - 1)

                heatmaps[b,k] = torch.exp(-((xx - x_px)**2 + (yy - y_px)**2) / (2 * sigma**2))
    return heatmaps

def extract_keypoints(heatmaps, return_max_values=False):
    """
    Extract keypoints from heatmaps
    
    Coordinate calculation
    - y = idx // W: gets the row (y-coordinate)
    - x = idx % W: gets the column (x-coordinate)
    This follows standard row-major indexing where idx = y*W + x
    """
    batch_size, num_keys, height, width = heatmaps.shape
    device = heatmaps.device

    keypoints = torch.zeros(batch_size, num_keys * 2, device=device)
    max_values = []

    for b in range(batch_size):
        batch_max_values = []
        for k in range(num_keys):
            heatmap = heatmaps[b, k]
            max_val, max_idx = torch.max(heatmap.view(-1), dim=0)

            y_px = max_idx // width
            x_px = max_idx % width

            # Normalize coordinates to [0, 1]
            x = x_px.float() / (width - 1) if width > 1 else 0.0
            y = y_px.float() / (height - 1) if height > 1 else 0.0

            keypoints[b, 2*k] = x
            keypoints[b, 2*k+1] = y
            batch_max_values.append(max_val.cpu().item())

        max_values.append(batch_max_values)

    if return_max_values:
        return keypoints, max_values

    return keypoints
