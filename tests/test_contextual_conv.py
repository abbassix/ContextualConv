import pytest
import torch

from contextual_conv import ContextualConv1d, ContextualConv2d


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def random_tensors(dim: int = 2):
    """Return a pair (x, c) of random tensors suitable for 1‑D or 2‑D layers."""
    if dim == 1:
        x = torch.randn(4, 3, 64)           # (B, C_in, L)
    else:
        x = torch.randn(4, 3, 32, 32)       # (B, C_in, H, W)
    c = torch.randn(4, 8)                   # (B, context_dim)
    return x, c


# -----------------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------------

def test_requires_scale_or_bias():
    with pytest.raises(ValueError):
        _ = ContextualConv1d(3, 6, 3, use_scale=False, use_bias=False)


# -----------------------------------------------------------------------------
# 1‑D tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("use_scale, use_bias", [(False, True), (True, False), (True, True)])
def test_conv1d_shapes(use_scale: bool, use_bias: bool):
    x, c = random_tensors(dim=1)

    model = ContextualConv1d(
        3,
        6,
        kernel_size=3,
        padding=1,
        context_dim=8,
        use_scale=use_scale,
        use_bias=use_bias,
    )

    out = model(x, c)
    assert out.shape == (4, 6, 64)


def test_conv1d_no_context_fallback():
    """Layer should act like plain Conv1d when no context is given."""
    x, _ = random_tensors(dim=1)
    model_ctx = ContextualConv1d(3, 6, 3, padding=1)
    model_ref = torch.nn.Conv1d(3, 6, 3, padding=1)
    # copy weights so both give identical outputs
    model_ref.weight.data.copy_(model_ctx.conv.weight)
    model_ref.bias.data.copy_(model_ctx.conv.bias)

    out_ctx = model_ctx(x)  # no c passed
    out_ref = model_ref(x)
    assert torch.allclose(out_ctx, out_ref, atol=1e-6)


# -----------------------------------------------------------------------------
# 2‑D tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("use_scale, use_bias, h_dim", [
    (False, True, None),  # bias only (linear)
    (True, False, None),  # scale only (linear)
    (True, True, None),   # FiLM (linear)
    (True, True, 16),     # FiLM (MLP)
])
def test_conv2d_shapes(use_scale: bool, use_bias: bool, h_dim):
    x, c = random_tensors(dim=2)

    model = ContextualConv2d(
        3,
        6,
        kernel_size=3,
        padding=1,
        context_dim=8,
        h_dim=h_dim,
        use_scale=use_scale,
        use_bias=use_bias,
    )

    out = model(x, c)
    assert out.shape == (4, 6, 32, 32)


def test_conv2d_context_dim_mismatch():
    x, _ = random_tensors(dim=2)
    c_wrong = torch.randn(4, 5)
    model = ContextualConv2d(3, 6, 3, padding=1, context_dim=8, use_scale=True)
    with pytest.raises(RuntimeError):  # Linear will receive mismatched in‑features
        _ = model(x, c_wrong)
