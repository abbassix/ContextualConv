"""Unit tests for ContextualConv1d and ContextualConv2d."""

import pytest
import torch
from contextual_conv import ContextProcessor, ContextualConv1d, ContextualConv2d
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def make_random_input(dim: int = 2):
    """Generate (x, c) for testing: x is input, c is context."""
    if dim == 1:
        x = torch.randn(4, 3, 64)
    else:
        x = torch.randn(4, 3, 32, 32)
    c = torch.randn(4, 8)
    return x, c

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def test_requires_scale_or_bias():
    with pytest.raises(ValueError, match="At least one of `use_scale` or `use_bias` must be True."):
        _ = ContextualConv1d(3, 6, 3, use_scale=False, use_bias=False)

# -----------------------------------------------------------------------------
# ContextualConv1d
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("use_scale, use_bias, scale_mode", [
    (True, False, "film"),
    (True, False, "scale"),
    (False, True, "film"),
    (True, True, "scale"),
])
def test_conv1d_output_shape_and_modes(use_scale, use_bias, scale_mode):
    x, c = make_random_input(dim=1)
    layer = ContextualConv1d(
        in_channels=3,
        out_channels=6,
        kernel_size=3,
        padding=1,
        context_dim=8,
        use_scale=use_scale,
        use_bias=use_bias,
        scale_mode=scale_mode,
    )
    y = layer(x, c)
    assert y.shape == (4, 6, 64)


def test_conv1d_behaves_like_conv1d_without_context():
    x, _ = make_random_input(dim=1)
    conv = torch.nn.Conv1d(3, 6, 3, padding=1)
    ctx_layer = ContextualConv1d(3, 6, 3, padding=1)
    ctx_layer.conv.load_state_dict(conv.state_dict())

    out_ref = conv(x)
    out_ctx = ctx_layer(x)
    assert torch.allclose(out_ctx, out_ref, atol=1e-6)

# -----------------------------------------------------------------------------
# ContextualConv2d
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("use_scale, use_bias, h_dim, scale_mode", [
    (True, False, None, "film"),
    (True, True, None, "scale"),
    (True, True, 16, "film"),
])
def test_conv2d_output_shape_and_modes(use_scale, use_bias, h_dim, scale_mode):
    x, c = make_random_input(dim=2)
    layer = ContextualConv2d(
        in_channels=3,
        out_channels=6,
        kernel_size=3,
        padding=1,
        context_dim=8,
        h_dim=h_dim,
        use_scale=use_scale,
        use_bias=use_bias,
        scale_mode=scale_mode,
    )
    y = layer(x, c)
    assert y.shape == (4, 6, 32, 32)

def test_conv2d_context_dim_mismatch_raises():
    x, _ = make_random_input(dim=2)
    c_bad = torch.randn(4, 5)
    layer = ContextualConv2d(
        in_channels=3,
        out_channels=6,
        kernel_size=3,
        padding=1,
        context_dim=8,
        use_scale=True,
    )
    with pytest.raises(RuntimeError):
        _ = layer(x, c_bad)

# -----------------------------------------------------------------------------
# ContextProcessor
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("h_dim", [None, 16, [16, 8]])
def test_context_processor_output_shape(h_dim):
    c = torch.randn(4, 10)
    processor = ContextProcessor(context_dim=10, out_dim=6, h_dim=h_dim)
    out = processor(c)
    assert out.shape == (4, 6)

# -----------------------------------------------------------------------------
# infer_context
# -----------------------------------------------------------------------------

def test_infer_context_shape_and_validity():
    x = torch.randn(4, 3, 64)
    layer = ContextualConv1d(
        3, 3, 3, padding=1,
        context_dim=6,
        use_scale=True,
        use_bias=False,
        linear_bias=False,
    )
    context = layer.infer_context(x)
    assert context.shape == (4, 6)

@pytest.mark.parametrize("setting", [
    {"context_dim": None},
    {"use_bias": True},
    {"h_dim": 16},
    {"linear_bias": True},
])
def test_infer_context_raises(setting):
    x = torch.randn(4, 3, 64)
    kwargs = dict(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        padding=1,
        use_scale=True,
        use_bias=False,
        context_dim=5,
        h_dim=None,
        linear_bias=False,
    )
    kwargs.update(setting)
    layer = ContextualConv1d(**kwargs)
    if setting.get("linear_bias", False):
        if isinstance(layer.context_processor.processor, torch.nn.Linear):
            layer.context_processor.processor.bias.data.fill_(1.0)
    with pytest.raises(RuntimeError):
        _ = layer.infer_context(x)

@pytest.mark.parametrize("scale_mode", ["film", "scale"])
def test_scale_mode_identity_initialization(scale_mode):
    x = torch.randn(4, 3, 64)

    if scale_mode == "scale":
        # one-hot context: each row has a single 1
        indices = torch.randint(0, 6, (4,))
        c = F.one_hot(indices, num_classes=6).float()
    else:
        c = torch.randn(4, 6)

    layer = ContextualConv1d(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        padding=1,
        context_dim=6,
        use_scale=True,
        use_bias=False,
        linear_bias=False,
        scale_mode=scale_mode,
    )

    out_plain = layer(x, None)
    out_modulated = layer(x, c)

    assert torch.allclose(out_plain, out_modulated, atol=1e-3)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for timing test")
def test_contextualconv_cuda_forward_timing():
    x = torch.randn(64, 32, 128).cuda()
    c = torch.randn(64, 16).cuda()
    model = ContextualConv1d(
        32, 64, 3, padding=1,
        context_dim=16,
        use_scale=True,
        use_bias=True,
        h_dim=32,
    ).cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    _ = model(x, c)
    ender.record()
    torch.cuda.synchronize()
    elapsed_ms = starter.elapsed_time(ender)
    assert elapsed_ms < 100, f"Forward pass too slow: {elapsed_ms:.2f} ms"
