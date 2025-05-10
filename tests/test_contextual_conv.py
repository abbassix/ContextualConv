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
        x = torch.randn(4, 3, 64)       # (B, C_in, L)
    else:
        x = torch.randn(4, 3, 32, 32)   # (B, C_in, H, W)
    c = torch.randn(4, 8)               # (B, context_dim)
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

@pytest.mark.parametrize("use_scale, use_bias", [
    (False, True), (True, False), (True, True)
])
def test_conv1d_output_shape(use_scale, use_bias):
    x, c = make_random_input(dim=1)
    layer = ContextualConv1d(
        in_channels=3,
        out_channels=6,
        kernel_size=3,
        padding=1,
        context_dim=8,
        use_scale=use_scale,
        use_bias=use_bias,
    )
    y = layer(x, c)
    assert y.shape == (4, 6, 64)


def test_conv1d_behaves_like_conv1d_without_context():
    """Should reduce to regular Conv1d if no context is passed."""
    x, _ = make_random_input(dim=1)
    conv = torch.nn.Conv1d(3, 6, 3, padding=1)
    ctx_layer = ContextualConv1d(3, 6, 3, padding=1)
    ctx_layer.conv.load_state_dict(conv.state_dict())  # sync weights

    out_ref = conv(x)
    out_ctx = ctx_layer(x)  # no context provided

    assert torch.allclose(out_ctx, out_ref, atol=1e-6)


# -----------------------------------------------------------------------------
# ContextualConv2d
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("use_scale, use_bias, h_dim", [
    (False, True, None),   # bias only, linear
    (True, False, None),   # scale only, linear
    (True, True, None),    # FiLM, linear
    (True, True, 16),      # FiLM, MLP
])
def test_conv2d_output_shape(use_scale, use_bias, h_dim):
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
# ContextProcessor Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("h_dim", [None, 16, [16, 8]])
def test_context_processor_output_shape(h_dim):
    c = torch.randn(4, 10)  # (B, context_dim)
    processor = ContextProcessor(context_dim=10, out_dim=6, h_dim=h_dim)
    out = processor(c)
    assert out.shape == (4, 6)


# -----------------------------------------------------------------------------
# infer_context() — 1D only (same logic applies to 2D)
# -----------------------------------------------------------------------------

def test_infer_context_matches_known_projection():
    """Check that infer_context() gives expected shape and is callable."""
    x, _ = torch.randn(4, 3, 64), None
    layer = ContextualConv1d(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        padding=1,
        context_dim=6,
        use_scale=True,
        use_bias=False,
        linear_bias=False,
    )
    context = layer.infer_context(x)
    assert context.shape == (4, 6)


@pytest.mark.parametrize("setting", [
    {"context_dim": None},                          # no context
    {"use_bias": True},                             # bias enabled
    {"h_dim": 16},                                  # MLP, not linear
    {"linear_bias": True},                          # bias=True in Linear
])
def test_infer_context_raises_invalid(setting):
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

    # If bias=True, set nonzero bias explicitly
    if setting.get("linear_bias", False):
        if isinstance(layer.context_processor.processor, torch.nn.Linear):
            layer.context_processor.processor.bias.data.fill_(1.0)

    with pytest.raises(RuntimeError):
        _ = layer.infer_context(x)


def test_infer_context_modulation_effect():
    """Ensure that inferred context alters output as expected."""
    x = torch.randn(4, 3, 64)
    layer = ContextualConv1d(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        padding=1,
        context_dim=6,
        use_scale=True,
        use_bias=False,
        linear_bias=False,
    )

    # Perturb context processor weights to break identity initialization
    with torch.no_grad():
        layer.context_processor.processor.weight.uniform_(-1.0, 1.0)

    out_base = layer(x, None)
    c_infer = layer.infer_context(x)
    out_modulated = layer(x, c_infer)

    assert not torch.allclose(out_base, out_modulated, atol=1e-4)


def test_identity_init_approximates_no_modulation():
    """If initialized with gamma ≈ 0, FiLM should not alter output."""
    x, c = torch.randn(4, 3, 64), torch.randn(4, 6)
    layer = ContextualConv1d(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        padding=1,
        context_dim=6,
        use_scale=True,
        use_bias=False,
        linear_bias=False,
    )

    # gamma ≈ 0 ⇒ scale ≈ 1.0
    out_modulated = layer(x, c)
    out_plain = layer(x, None)

    # Should be close (not necessarily identical due to bias/init)
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
