"""
Tests for heatmap creation and keypoint extraction (round-trip).

Core invariant: create_heatmap(kps) -> extract_keypoints() should recover
the original keypoints within one pixel of tolerance.
"""
import pytest
import torch

from src.utils.heatmap_funcs import create_heatmap, extract_keypoints


# ---------------------------------------------------------------------------
# create_heatmap
# ---------------------------------------------------------------------------

class TestCreateHeatmap:
    def test_output_shape(self):
        """Returns correct [B, K, H, W] tensor."""
        B, K, H, W = 2, 7, 56, 84
        kps = torch.rand(B, K * 2)
        out = create_heatmap(kps, (H, W), sigma=1.0)
        assert out.shape == (B, K, H, W)

    def test_values_in_zero_one(self):
        """Gaussian heatmap values must lie in [0, 1]."""
        kps = torch.rand(2, 14)
        out = create_heatmap(kps, (56, 84), sigma=1.0)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0 + 1e-6

    def test_peak_at_correct_pixel(self):
        """The argmax of each channel must fall at the specified keypoint pixel."""
        # Use H=W=65 (odd) so 0.5 * (65-1) = 32.0 is an exact integer pixel.
        H, W = 65, 65
        kps = torch.tensor([[0.5, 0.5]])
        hm = create_heatmap(kps, (H, W), sigma=1.0)
        flat_idx = hm[0, 0].argmax()
        assert flat_idx // W == 32  # y
        assert flat_idx % W == 32   # x

    def test_corner_top_left(self):
        """Keypoint (0, 0) -> peak at pixel (0, 0)."""
        H, W = 64, 64
        kps = torch.tensor([[0.0, 0.0]])
        hm = create_heatmap(kps, (H, W), sigma=1.0)
        flat_idx = hm[0, 0].argmax()
        assert flat_idx // W == 0
        assert flat_idx % W == 0

    def test_corner_bottom_right(self):
        """Keypoint (1, 1) -> peak at pixel (W-1, H-1)."""
        H, W = 64, 64
        kps = torch.tensor([[1.0, 1.0]])
        hm = create_heatmap(kps, (H, W), sigma=1.0)
        flat_idx = hm[0, 0].argmax()
        assert flat_idx // W == H - 1
        assert flat_idx % W == W - 1

    def test_negative_keypoint_produces_zero_heatmap(self):
        """Keypoints with x < 0 or y < 0 are invalid and yield all-zero channels."""
        kps = torch.tensor([[-0.1, -0.1]])
        hm = create_heatmap(kps, (32, 32), sigma=1.0)
        assert hm.sum().item() == 0.0

    def test_multiple_keypoints_independent_channels(self):
        """Each keypoint maps to its own channel without cross-channel bleeding."""
        H, W = 64, 64
        # Two keypoints placed far apart
        kps = torch.tensor([[0.1, 0.1, 0.9, 0.9]])
        hm = create_heatmap(kps, (H, W), sigma=0.5)
        assert hm.shape[1] == 2
        # Peak of channel 0 should be near (0.1*63, 0.1*63) = (6, 6)
        # Peak of channel 1 should be near (0.9*63, 0.9*63) = (57, 57)
        ch0_peak = hm[0, 0].argmax()
        ch1_peak = hm[0, 1].argmax()
        assert ch0_peak // W < 20   # y ≈ 6
        assert ch1_peak // W > 44   # y ≈ 57

    def test_batch_dimension_independent(self):
        """Each batch element is processed independently."""
        H, W = 32, 32
        kps = torch.zeros(3, 2)
        kps[0] = torch.tensor([0.0, 0.0])
        kps[1] = torch.tensor([0.5, 0.5])
        kps[2] = torch.tensor([1.0, 1.0])
        hm = create_heatmap(kps, (H, W), sigma=0.5)
        peaks = [hm[b, 0].argmax() for b in range(3)]
        # Each batch element should have a different peak
        assert peaks[0] != peaks[1]
        assert peaks[1] != peaks[2]


# ---------------------------------------------------------------------------
# extract_keypoints
# ---------------------------------------------------------------------------

class TestExtractKeypoints:
    def test_output_shape(self):
        """Returns [B, K*2] tensor."""
        B, K, H, W = 3, 7, 56, 84
        hm = torch.rand(B, K, H, W)
        out = extract_keypoints(hm)
        assert out.shape == (B, K * 2)

    def test_values_in_zero_one(self):
        """Normalised coordinates must lie in [0, 1]."""
        hm = torch.rand(2, 7, 56, 84)
        out = extract_keypoints(hm)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0 + 1e-6

    def test_return_max_values_shape(self):
        """return_max_values=True gives (keypoints, list[list[float]])."""
        B, K = 2, 4
        hm = torch.rand(B, K, 32, 32)
        kps, max_vals = extract_keypoints(hm, return_max_values=True)
        assert kps.shape == (B, K * 2)
        assert len(max_vals) == B
        assert len(max_vals[0]) == K

    def test_max_value_matches_heatmap_peak(self):
        """Reported max value should equal the true channel maximum."""
        H, W = 16, 16
        hm = torch.zeros(1, 1, H, W)
        hm[0, 0, 3, 7] = 0.9
        _, max_vals = extract_keypoints(hm, return_max_values=True)
        assert max_vals[0][0] == pytest.approx(0.9, abs=1e-6)

    def test_single_hot_pixel(self):
        """With a single hot pixel, extracted coordinate matches that pixel exactly."""
        H, W = 16, 16
        hm = torch.zeros(1, 1, H, W)
        # Place peak at pixel (x=4, y=10) -> normalised (4/15, 10/15)
        hm[0, 0, 10, 4] = 1.0
        kps = extract_keypoints(hm)
        assert kps[0, 0].item() == pytest.approx(4.0 / 15, abs=1e-5)   # x
        assert kps[0, 1].item() == pytest.approx(10.0 / 15, abs=1e-5)  # y


# ---------------------------------------------------------------------------
# Round-trip: create_heatmap -> extract_keypoints
# ---------------------------------------------------------------------------

class TestHeatmapRoundTrip:
    """
    Tight sigma + sufficiently large heatmap -> argmax falls on the correct pixel,
    so the recovered coordinate should be within one pixel of the original.
    """

    def _pixel_tol(self, H, W):
        return 1.5 / (W - 1), 1.5 / (H - 1)

    def test_single_center_keypoint(self):
        """Round-trip for a keypoint at the image centre."""
        H, W = 64, 64
        kps_in = torch.tensor([[0.5, 0.5]])
        hm = create_heatmap(kps_in, (H, W), sigma=0.5)
        kps_out = extract_keypoints(hm)
        tol_x, tol_y = self._pixel_tol(H, W)
        assert abs(kps_out[0, 0].item() - 0.5) <= tol_x
        assert abs(kps_out[0, 1].item() - 0.5) <= tol_y

    def test_two_keypoints_per_image(self):
        """Round-trip for two keypoints in a single image."""
        H, W = 64, 64
        kps_in = torch.tensor([[0.2, 0.8, 0.7, 0.3]])  # (x1,y1,x2,y2)
        hm = create_heatmap(kps_in, (H, W), sigma=0.5)
        kps_out = extract_keypoints(hm)
        tol_x, tol_y = self._pixel_tol(H, W)
        assert abs(kps_out[0, 0].item() - 0.2) <= tol_x
        assert abs(kps_out[0, 1].item() - 0.8) <= tol_y
        assert abs(kps_out[0, 2].item() - 0.7) <= tol_x
        assert abs(kps_out[0, 3].item() - 0.3) <= tol_y

    def test_batch_round_trip(self):
        """Round-trip is consistent across all images in a batch."""
        H, W = 64, 64
        B, K = 4, 3
        # Place keypoints on an evenly spaced grid so they fall on clean pixels
        kps_in = torch.zeros(B, K * 2)
        for b in range(B):
            for k in range(K):
                kps_in[b, 2 * k]     = round((b + 1) / (B + 1) * (W - 1)) / (W - 1)
                kps_in[b, 2 * k + 1] = round((k + 1) / (K + 1) * (H - 1)) / (H - 1)
        hm = create_heatmap(kps_in, (H, W), sigma=0.5)
        kps_out = extract_keypoints(hm)
        tol = 1.5 / min(H - 1, W - 1)
        assert torch.allclose(kps_in, kps_out, atol=tol)

    def test_corner_keypoints_round_trip(self):
        """Keypoints at all four corners survive the round-trip."""
        H, W = 32, 32
        corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        flat = [c for xy in corners for c in xy]
        kps_in = torch.tensor([flat])
        hm = create_heatmap(kps_in, (H, W), sigma=0.5)
        kps_out = extract_keypoints(hm)
        tol_x, tol_y = self._pixel_tol(H, W)
        for k, (x_ref, y_ref) in enumerate(corners):
            assert abs(kps_out[0, 2 * k].item()     - x_ref) <= tol_x
            assert abs(kps_out[0, 2 * k + 1].item() - y_ref) <= tol_y
