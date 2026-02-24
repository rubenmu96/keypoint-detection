"""
Tests for keypoint metrics and loss computation.

Covers:
  - compute_pck      – PCK correctness and regression gate
  - compute_mpjpe    – MPJPE correctness
  - compute_loss     – per-model loss routing
  - _calculate_keypoint_loss – visibility masking, empty-detection warning
"""
import warnings

import numpy as np
import pytest
import torch

from src.core.metrics import (
    _calculate_keypoint_loss,
    compute_loss,
    compute_mpjpe,
    compute_pck,
)
from src.utils.heatmap_funcs import create_heatmap


# ---------------------------------------------------------------------------
# Shared image size used across metric tests
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 448, 672
IMG_DIAG = np.sqrt(IMG_H**2 + IMG_W**2)   # ≈ 807.6 px


# ---------------------------------------------------------------------------
# Minimal config stub (avoids loading the real dataset in BaseConfig.__init__)
# ---------------------------------------------------------------------------
class _Cfg:
    sigma = 1.0
    num_kps = 7


# ---------------------------------------------------------------------------
# compute_pck
# ---------------------------------------------------------------------------

class TestComputePCK:
    def test_perfect_predictions_give_one(self):
        """Identical preds and targets → PCK = 1.0."""
        B, K = 4, 7
        preds = torch.rand(B, K, 2) * torch.tensor([IMG_W, IMG_H])
        result = compute_pck(preds, preds.clone(), image_size=(IMG_H, IMG_W))
        assert result["pck"] == pytest.approx(1.0)

    def test_far_predictions_give_zero(self):
        """Predictions far beyond the threshold → PCK = 0.0."""
        B, K = 4, 7
        targets = torch.zeros(B, K, 2)
        # distance = sqrt(2)*IMG_DIAG >> 0.05*IMG_DIAG
        preds = torch.full((B, K, 2), float(IMG_DIAG))
        result = compute_pck(preds, targets, image_size=(IMG_H, IMG_W), threshold=0.05)
        assert result["pck"] == pytest.approx(0.0)

    def test_half_correct_gives_half_pck(self):
        """Exactly half the keypoints within threshold → PCK ≈ 0.5."""
        K = 4
        B = 1
        targets = torch.zeros(B, K, 2)
        preds = targets.clone()
        # Push the first K//2 keypoints far beyond the threshold
        preds[0, : K // 2, 0] = IMG_DIAG * 10
        result = compute_pck(preds, targets, image_size=(IMG_H, IMG_W), threshold=0.05)
        assert result["pck"] == pytest.approx(0.5)

    def test_per_keypoint_shape(self):
        """pck_per_keypoint must have shape (K,)."""
        B, K = 3, 7
        preds = torch.rand(B, K, 2)
        targets = torch.rand(B, K, 2)
        result = compute_pck(preds, targets, image_size=(IMG_H, IMG_W))
        assert result["pck_per_keypoint"].shape == (K,)

    def test_num_total(self):
        """num_total must equal B * K."""
        B, K = 3, 5
        preds = torch.rand(B, K, 2)
        result = compute_pck(preds, preds.clone(), image_size=(IMG_H, IMG_W))
        assert result["num_total"] == B * K

    def test_flat_input_accepted(self):
        """Flat [B, K*2] tensors are reshaped internally."""
        B, K = 2, 7
        preds = torch.rand(B, K * 2) * IMG_W
        result = compute_pck(preds, preds.clone(), image_size=(IMG_H, IMG_W))
        assert result["pck"] == pytest.approx(1.0)

    def test_threshold_sensitivity(self):
        """Tighter threshold should yield equal or lower PCK."""
        B, K = 4, 7
        targets = torch.zeros(B, K, 2)
        preds = targets + 20.0   # 20-pixel offset
        pck_loose = compute_pck(preds, targets, image_size=(IMG_H, IMG_W), threshold=0.1)["pck"]
        pck_tight = compute_pck(preds, targets, image_size=(IMG_H, IMG_W), threshold=0.01)["pck"]
        assert pck_tight <= pck_loose

    # --- Regression gate ---------------------------------------------------

    def test_pck_regression_gate_near_perfect_predictions(self):
        """
        PCK@0.05 regression gate.

        With ±2 px Gaussian noise on pixel-space predictions, the normalised
        distance is ≈ 2-3 px / 807.6 px ≈ 0.003, well below the 0.05 threshold.
        PCK must stay above 0.95. A failure here indicates a regression in the
        metric computation itself (not in model quality).
        """
        torch.manual_seed(42)
        B, K = 16, 7
        targets = torch.rand(B, K, 2) * torch.tensor([IMG_W, IMG_H])
        preds = targets + torch.randn_like(targets) * 2.0   # σ = 2 px noise
        result = compute_pck(preds, targets, image_size=(IMG_H, IMG_W), threshold=0.05)
        assert result["pck"] >= 0.95, (
            f"PCK@0.05 = {result['pck']:.3f} with 2 px noise – "
            "a value below 0.95 signals a regression in compute_pck."
        )

    def test_pck_regression_gate_medium_noise(self):
        """
        PCK@0.1 gate with ±10 px noise.

        10 px / 807.6 px ≈ 0.012, still below the 0.1 threshold.
        PCK@0.1 must exceed 0.9.
        """
        torch.manual_seed(0)
        B, K = 16, 7
        targets = torch.rand(B, K, 2) * torch.tensor([IMG_W, IMG_H])
        preds = targets + torch.randn_like(targets) * 10.0
        result = compute_pck(preds, targets, image_size=(IMG_H, IMG_W), threshold=0.1)
        assert result["pck"] >= 0.9, (
            f"PCK@0.1 = {result['pck']:.3f} with 10 px noise – regression detected."
        )


# ---------------------------------------------------------------------------
# compute_mpjpe
# ---------------------------------------------------------------------------

class TestComputeMPJPE:
    def test_zero_error(self):
        """Identical predictions → MPJPE = 0."""
        preds = torch.rand(4, 7, 2)
        result = compute_mpjpe(preds, preds.clone())
        assert result["mpjpe"] == pytest.approx(0.0, abs=1e-5)

    def test_known_euclidean_distance(self):
        """MPJPE of two identical 3-4-5 triangles is exactly 5.0."""
        # Use B=1, K=2 so std() has 2 elements and doesn't warn.
        preds = torch.zeros(1, 2, 2)
        targets = torch.tensor([[[3.0, 4.0], [3.0, 4.0]]])
        result = compute_mpjpe(preds, targets)
        assert result["mpjpe"] == pytest.approx(5.0, abs=1e-5)

    def test_per_keypoint_shape(self):
        """mpjpe_per_keypoint has shape (K,)."""
        B, K = 3, 7
        preds = torch.rand(B, K, 2)
        targets = torch.rand(B, K, 2)
        result = compute_mpjpe(preds, targets)
        assert result["mpjpe_per_keypoint"].shape == (K,)

    def test_std_is_non_negative(self):
        """Standard deviation of per-keypoint errors must be ≥ 0."""
        preds = torch.rand(4, 7, 2)
        targets = torch.rand(4, 7, 2)
        result = compute_mpjpe(preds, targets)
        assert result["std"] >= 0.0

    def test_flat_input_accepted(self):
        """Flat [B, K*2] tensors are reshaped internally."""
        B, K = 2, 4
        preds = torch.rand(B, K * 2)
        result = compute_mpjpe(preds, preds.clone())
        assert result["mpjpe"] == pytest.approx(0.0, abs=1e-5)

    def test_symmetric(self):
        """MPJPE(A, B) == MPJPE(B, A) since it uses L2 distance."""
        preds = torch.rand(3, 5, 2)
        targets = torch.rand(3, 5, 2)
        r1 = compute_mpjpe(preds, targets)["mpjpe"]
        r2 = compute_mpjpe(targets, preds)["mpjpe"]
        assert r1 == pytest.approx(r2, abs=1e-5)


# ---------------------------------------------------------------------------
# compute_loss – heatmap branch
# ---------------------------------------------------------------------------

class TestComputeLossHeatmap:
    def _cfg(self):
        cfg = _Cfg()
        cfg.model_name = "ResNetHeatmap"
        return cfg

    def test_perfect_preds_lower_loss_than_random(self):
        """
        Predictions generated from the same heatmap function as the targets
        should give lower loss than fully random predictions.
        """
        cfg = self._cfg()
        B, K, H, W = 2, 7, 56, 84
        targets_kps = torch.rand(B, K * 2)
        preds_good = create_heatmap(targets_kps, (H, W), sigma=cfg.sigma)
        preds_bad  = torch.rand_like(preds_good)

        loss_good = compute_loss(cfg, "bcelogitloss", preds_good, targets_kps)
        loss_bad  = compute_loss(cfg, "bcelogitloss", preds_bad,  targets_kps)
        assert loss_good.item() < loss_bad.item()

    def test_returns_scalar_tensor(self):
        """Loss must be a scalar tensor (shape = ())."""
        cfg = self._cfg()
        B, K, H, W = 2, 7, 56, 84
        targets_kps = torch.rand(B, K * 2)
        preds = torch.rand(B, K, H, W)
        loss = compute_loss(cfg, "bcelogitloss", preds, targets_kps)
        assert loss.shape == torch.Size([])

    def test_loss_is_non_negative(self):
        """BCEWithLogitsLoss is always non-negative."""
        cfg = self._cfg()
        B, K, H, W = 2, 7, 56, 84
        preds = torch.randn(B, K, H, W)
        targets_kps = torch.rand(B, K * 2)
        loss = compute_loss(cfg, "bcelogitloss", preds, targets_kps)
        assert loss.item() >= 0.0

    def test_unknown_model_raises_value_error(self):
        cfg = self._cfg()
        cfg.model_name = "ThisModelDoesNotExist"
        with pytest.raises(ValueError, match="Unknown model"):
            compute_loss(cfg, "mseloss", torch.rand(2, 14), torch.rand(2, 14))


# ---------------------------------------------------------------------------
# _calculate_keypoint_loss (RCNN validation loss helper)
# ---------------------------------------------------------------------------

def _make_kps_dict(coords_val: float, visibility: int = 2, n_det: int = 1, K: int = 7):
    """Build a minimal prediction/target dict like KeypointRCNN outputs."""
    kps = torch.full((n_det, K, 3), coords_val)
    kps[:, :, 2] = visibility
    return {"keypoints": kps}


class TestCalculateKeypointLoss:
    def test_zero_loss_when_preds_match_targets(self):
        """Predictions equal to targets → loss = 0."""
        K = 7
        pred   = _make_kps_dict(100.0, K=K)
        target = _make_kps_dict(100.0, K=K)
        loss = _calculate_keypoint_loss([pred], [target], None, K)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_non_zero_loss_when_preds_differ(self):
        """Different predictions produce positive loss."""
        K = 7
        pred   = _make_kps_dict(0.0,   K=K)
        target = _make_kps_dict(100.0, K=K)
        loss = _calculate_keypoint_loss([pred], [target], None, K)
        assert loss.item() > 0.0

    def test_occluded_keypoints_excluded_from_loss(self):
        """
        Keypoints with visibility = 0 (not labeled) are masked out.
        A wrong prediction on an invisible keypoint should not affect loss.
        """
        K = 2
        # pred: kp0 correct, kp1 very wrong
        pred_kps = torch.zeros(1, K, 3)
        pred_kps[0, 1, :2] = 9999.0
        # target: kp0 visible, kp1 invisible
        tgt_kps = torch.zeros(1, K, 3)
        tgt_kps[0, 0, 2] = 2   # visible
        tgt_kps[0, 1, 2] = 0   # not labeled

        loss = _calculate_keypoint_loss(
            [{"keypoints": pred_kps}],
            [{"keypoints": tgt_kps}],
            criterion=None,
            num_kps=K,
            use_visibility_mask=True,
            visibility_threshold=0,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_empty_detection_emits_runtime_warning(self):
        """A batch with zero detections should trigger RuntimeWarning."""
        K = 7
        pred   = {"keypoints": torch.zeros(0, K, 3)}   # empty
        target = {"keypoints": torch.zeros(1, K, 3)}
        with pytest.warns(RuntimeWarning, match="no valid detections"):
            _calculate_keypoint_loss([pred], [target], None, K)

    def test_empty_detection_returns_zero(self):
        """Even with zero detections the function returns a finite 0.0."""
        K = 7
        pred   = {"keypoints": torch.zeros(0, K, 3)}
        target = {"keypoints": torch.zeros(1, K, 3)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loss = _calculate_keypoint_loss([pred], [target], None, K)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_batch_averages_over_valid_images(self):
        """Loss is averaged only over images that have detections."""
        K = 3
        # Image 0: valid detection
        pred0   = _make_kps_dict(0.0,   K=K)
        target0 = _make_kps_dict(10.0,  K=K)
        # Image 1: no detection (empty)
        pred1   = {"keypoints": torch.zeros(0, K, 3)}
        target1 = {"keypoints": torch.zeros(1, K, 3)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loss_mixed = _calculate_keypoint_loss(
                [pred0, pred1], [target0, target1], None, K
            )
        loss_single = _calculate_keypoint_loss([pred0], [target0], None, K)
        assert loss_mixed.item() == pytest.approx(loss_single.item(), abs=1e-5)
