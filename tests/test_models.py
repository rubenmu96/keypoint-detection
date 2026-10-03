"""
Model input/output verification tests.

All tests run on CPU with pretrained=False to avoid network downloads in CI.
Tests cover:
  - ResNetKeypoint  – regression head model
  - ResNetHeatmap   – heatmap head model
  - KeypointRCNN    – detection model with a resized keypoint head
"""
import pytest
import torch
import torch.hub
import torchvision.models._api

from src.models.heatmap import ResNetHeatmap
from src.models.keypoint_rcnn import KeypointRCNN
from src.models.resnet import ResNetKeypoint
from src.utils.processing import keypoints_region, keypoints_with_visibility

# ---------------------------------------------------------------------------
# Shared model fixtures (module-scoped: built once per test session)
# ---------------------------------------------------------------------------

NUM_KPS = 7


@pytest.fixture(scope="module")
def resnet_kp():
    """ResNetKeypoint (resnet34 backbone) – CPU, random weights."""
    model = ResNetKeypoint(pretrained=False, num_kps=NUM_KPS)
    return model.eval()


@pytest.fixture(scope="module")
def resnet_hm():
    """ResNetHeatmap (resnet34 backbone) – CPU, random weights, 224-px calibration."""
    model = ResNetHeatmap(pretrained=False, num_kps=NUM_KPS, input_size=224)
    return model.eval()


# KeypointRCNN resizes every image so its shorter side is min_size (default 800).
# Pinning the resize to the test image size makes CPU forward passes ~6x faster.
RCNN_IMG_H, RCNN_IMG_W = 224, 336


@pytest.fixture(scope="module")
def keypoint_rcnn():
    """KeypointRCNN – CPU, random weights, resize pinned to the test image size."""
    model = KeypointRCNN(
        num_kps=NUM_KPS, pretrained=False, min_size=RCNN_IMG_H, max_size=RCNN_IMG_W
    )
    return model.eval()


# ---------------------------------------------------------------------------
# ResNetKeypoint
# ---------------------------------------------------------------------------

class TestResNetKeypoint:
    IMG_H, IMG_W = 448, 672

    def _img(self, B=1):
        return torch.rand(B, 3, self.IMG_H, self.IMG_W)

    def test_output_shape_batch_2(self, resnet_kp):
        """Output is [B, num_kps * 2] for a batch of 2."""
        with torch.no_grad():
            out = resnet_kp(self._img(B=2))
        assert out.shape == (2, NUM_KPS * 2)

    def test_output_shape_batch_1(self, resnet_kp):
        """Single-image batch also works."""
        with torch.no_grad():
            out = resnet_kp(self._img(B=1))
        assert out.shape == (1, NUM_KPS * 2)

    def test_output_is_finite(self, resnet_kp):
        """No NaN or Inf in output."""
        with torch.no_grad():
            out = resnet_kp(self._img(B=2))
        assert torch.isfinite(out).all()

    def test_output_dtype_float32(self, resnet_kp):
        """Output must be float32 by default."""
        with torch.no_grad():
            out = resnet_kp(self._img())
        assert out.dtype == torch.float32

    def test_eval_mode_consistent_across_batch_sizes(self, resnet_kp):
        """
        In eval mode (dropout disabled, BN uses running stats) the same image
        repeated in a batch should yield identical per-image outputs.
        """
        x = self._img(B=1)
        x_batch = x.expand(3, -1, -1, -1)
        with torch.no_grad():
            out_single = resnet_kp(x)
            out_batch  = resnet_kp(x_batch)
        assert torch.allclose(out_single.expand(3, -1), out_batch, atol=1e-5)

    def test_gradient_flows(self, resnet_kp):
        """Gradient should flow from output back through the model."""
        model = ResNetKeypoint(pretrained=False, num_kps=NUM_KPS).train()
        x = self._img()
        out = model(x)
        loss = out.sum()
        loss.backward()
        # At least one parameter must have a non-None gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_different_spatial_inputs_work(self, resnet_kp):
        """
        ResNetKeypoint uses AdaptiveAvgPool2d so it accepts any spatial size.
        """
        for H, W in [(224, 224), (320, 480)]:
            x = torch.rand(1, 3, H, W)
            with torch.no_grad():
                out = resnet_kp(x)
            assert out.shape == (1, NUM_KPS * 2), f"Failed for {H}x{W}"


# ---------------------------------------------------------------------------
# ResNetHeatmap
# ---------------------------------------------------------------------------

class TestResNetHeatmap:
    # Use 224x224 to match input_size=224 set in the fixture
    IMG_SIZE = 224

    def _img(self, B=1):
        return torch.rand(B, 3, self.IMG_SIZE, self.IMG_SIZE)

    def test_output_channels_equals_num_kps(self, resnet_hm):
        """Output must have exactly num_kps channels."""
        with torch.no_grad():
            out = resnet_hm(self._img())
        assert out.shape[1] == NUM_KPS

    def test_output_spatial_larger_than_backbone_stride(self, resnet_hm):
        """
        The heatmap head upsamples beyond the 1/32 backbone stride.
        For 224-px input the backbone produces 7x7 feature maps;
        the head must output strictly larger spatial dimensions.
        """
        with torch.no_grad():
            out = resnet_hm(self._img())
        backbone_spatial = self.IMG_SIZE // 32   # = 7
        assert out.shape[2] > backbone_spatial
        assert out.shape[3] > backbone_spatial

    def test_output_batch_dimension(self, resnet_hm):
        """Batch dimension is preserved."""
        B = 3
        with torch.no_grad():
            out = resnet_hm(self._img(B=B))
        assert out.shape[0] == B

    def test_output_is_finite(self, resnet_hm):
        """No NaN or Inf in output."""
        with torch.no_grad():
            out = resnet_hm(self._img(B=2))
        assert torch.isfinite(out).all()

    def test_output_dtype_float32(self, resnet_hm):
        """Output must be float32 by default."""
        with torch.no_grad():
            out = resnet_hm(self._img())
        assert out.dtype == torch.float32

    def test_eval_mode_consistent_across_batch_sizes(self, resnet_hm):
        """Same image repeated in a batch yields identical per-image outputs."""
        x = self._img(B=1)
        x_batch = x.expand(3, -1, -1, -1)
        with torch.no_grad():
            out_single = resnet_hm(x)
            out_batch  = resnet_hm(x_batch)
        # Expand single-image output to match batch then compare
        assert torch.allclose(
            out_single.expand(3, -1, -1, -1), out_batch, atol=1e-5
        )

    def test_gradient_flows(self):
        """Gradient must flow through the full forward pass."""
        model = ResNetHeatmap(pretrained=False, num_kps=NUM_KPS, input_size=224).train()
        x = torch.rand(1, 3, self.IMG_SIZE, self.IMG_SIZE)
        out = model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_num_kps_respected(self):
        """Model instantiated with a different num_kps outputs that many channels."""
        for k in [4, 7, 14]:
            model = ResNetHeatmap(pretrained=False, num_kps=k, input_size=224).eval()
            with torch.no_grad():
                out = model(self._img())
            assert out.shape[1] == k, f"Expected {k} channels, got {out.shape[1]}"


# ---------------------------------------------------------------------------
# KeypointRCNN
# ---------------------------------------------------------------------------

LOSS_KEYS = {
    "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "loss_keypoint"
}
PREDICTION_KEYS = {"boxes", "labels", "scores", "keypoints", "keypoints_scores"}


class _DownloadBlocked(Exception):
    """Raised in place of a weight download, so no test can reach the network."""


@pytest.fixture
def blocked_downloads(monkeypatch):
    """Intercept torchvision weight downloads and record the URLs requested."""
    requested = []

    def fake_download(url, *args, **kwargs):
        requested.append(url)
        raise _DownloadBlocked(url)

    # torchvision imports the function into its own namespace, so patch both names.
    monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", fake_download)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fake_download)
    return requested


class TestKeypointRCNN:
    IMG_H, IMG_W = RCNN_IMG_H, RCNN_IMG_W

    def _images(self, B=1):
        return [torch.rand(3, self.IMG_H, self.IMG_W) for _ in range(B)]

    def _fresh_model(self, **kwargs):
        """Separate model for train-mode tests: training updates BatchNorm running stats."""
        return KeypointRCNN(
            num_kps=NUM_KPS, pretrained=False,
            min_size=self.IMG_H, max_size=self.IMG_W, **kwargs
        )

    def _target(self, seed=0):
        """One training target, built with the same helpers as KeypointPyTorch."""
        gen = torch.Generator().manual_seed(seed)
        xs = torch.randint(20, self.IMG_W - 20, (NUM_KPS,), generator=gen)
        ys = torch.randint(20, self.IMG_H - 20, (NUM_KPS,), generator=gen)
        kps = torch.stack([xs, ys], dim=1).float().numpy()
        return {
            "boxes": keypoints_region(kps, offset=10, width=self.IMG_W, height=self.IMG_H),
            "labels": torch.ones((1,), dtype=torch.int64),
            "keypoints": keypoints_with_visibility(kps).unsqueeze(0),
        }

    # Construction

    def test_pretrained_false_downloads_nothing(self, blocked_downloads):
        """pretrained=False builds without fetching any weights, backbone included."""
        KeypointRCNN(num_kps=NUM_KPS, pretrained=False)
        assert not blocked_downloads

    def test_pretrained_true_requests_coco_keypoint_weights(self, blocked_downloads):
        """pretrained=True asks for the full COCO Keypoint R-CNN checkpoint."""
        with pytest.raises(_DownloadBlocked):
            KeypointRCNN(num_kps=NUM_KPS, pretrained=True)
        assert len(blocked_downloads) == 1
        assert "keypointrcnn_resnet50_fpn_coco" in blocked_downloads[0]

    def test_keypoint_head_resized_to_num_kps(self, keypoint_rcnn):
        """The COCO 17-keypoint predictor is replaced by one with num_kps outputs."""
        predictor = keypoint_rcnn.model.roi_heads.keypoint_predictor
        assert predictor.kps_score_lowres.out_channels == NUM_KPS

    def test_constructor_options_applied(self):
        """num_classes, score_thresh and pass-through kwargs reach the torchvision model."""
        model = self._fresh_model(num_classes=3, score_thresh=0.3)
        assert model.model.roi_heads.box_predictor.cls_score.out_features == 3
        assert model.model.roi_heads.score_thresh == 0.3
        assert model.model.transform.min_size == (self.IMG_H,)
        assert model.model.transform.max_size == self.IMG_W

    # Inference

    def test_eval_returns_one_prediction_per_image(self, keypoint_rcnn):
        """Eval mode returns a list with one prediction dict per input image."""
        with torch.no_grad():
            out = keypoint_rcnn.eval()(self._images(B=2))
        assert isinstance(out, list)
        assert len(out) == 2
        for pred in out:
            assert set(pred) == PREDICTION_KEYS

    def test_eval_output_shapes_consistent(self, keypoint_rcnn):
        """All per-detection tensors share the same N; keypoints are [N, num_kps, 3]."""
        with torch.no_grad():
            pred = keypoint_rcnn.eval()(self._images())[0]
        n = pred["boxes"].shape[0]
        assert pred["boxes"].shape == (n, 4)
        assert pred["labels"].shape == (n,)
        assert pred["scores"].shape == (n,)
        assert pred["keypoints"].shape == (n, NUM_KPS, 3)
        assert pred["keypoints_scores"].shape == (n, NUM_KPS)

    def test_eval_outputs_finite(self, keypoint_rcnn):
        """No NaN or Inf in any floating-point prediction."""
        with torch.no_grad():
            pred = keypoint_rcnn.eval()(self._images())[0]
        for key in ("boxes", "scores", "keypoints", "keypoints_scores"):
            assert torch.isfinite(pred[key]).all(), key

    # Training

    def test_train_returns_all_losses(self):
        """Train mode with dataset-format targets returns every loss as a finite scalar."""
        model = self._fresh_model().train()
        losses = model(self._images(B=2), [self._target(seed=0), self._target(seed=1)])
        assert set(losses) == LOSS_KEYS
        for name, value in losses.items():
            assert value.ndim == 0, name
            assert torch.isfinite(value), name

    def test_gradient_reaches_keypoint_head(self):
        """loss_keypoint backpropagates into the replaced keypoint predictor."""
        model = self._fresh_model().train()
        losses = model(self._images(), [self._target()])
        losses["loss_keypoint"].backward()
        grad = model.model.roi_heads.keypoint_predictor.kps_score_lowres.weight.grad
        assert grad is not None
        assert grad.abs().sum() > 0

    def test_get_loss_switches_to_train_mode(self):
        """get_loss puts the model in train mode and returns the loss dict."""
        model = self._fresh_model().eval()
        losses = model.get_loss(self._images(), [self._target()])
        assert model.training
        assert set(losses) == LOSS_KEYS

    def test_predict_switches_to_eval_mode_without_grad(self):
        """predict puts the model in eval mode and returns detached predictions."""
        model = self._fresh_model().train()
        preds = model.predict(self._images())
        assert not model.training
        assert isinstance(preds, list)
        assert not preds[0]["keypoints"].requires_grad
