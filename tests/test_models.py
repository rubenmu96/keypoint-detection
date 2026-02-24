"""
Model input/output verification tests.

All tests run on CPU with pretrained=False to avoid network downloads in CI.
Tests cover:
  - ResNetKeypoint  – regression head model
  - ResNetHeatmap   – heatmap head model

KeypointRCNN is excluded: it hardcodes weights='DEFAULT', requiring a
network download that is unsuitable for lightweight CI.
"""
import pytest
import torch

from src.models.heatmap import ResNetHeatmap
from src.models.resnet import ResNetKeypoint

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
