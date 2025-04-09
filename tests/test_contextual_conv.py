import torch
import torch.nn as nn
import pytest
from contextual_conv import ContextualConv1d, ContextualConv2d


def copy_conv_weights(ref_layer: nn.Module, custom_layer: nn.Module) -> None:
    assert ref_layer.weight.shape == custom_layer.weight.shape, "Weight shape mismatch"
    custom_layer.weight.data.copy_(ref_layer.weight.data)

    if ref_layer.bias is not None and custom_layer.bias is not None:
        assert ref_layer.bias.shape == custom_layer.bias.shape, "Bias shape mismatch"
        custom_layer.bias.data.copy_(ref_layer.bias.data)


@pytest.mark.parametrize("groups", [1, 2])
def test_contextual_conv2d_no_context(groups):
    conv_ref = nn.Conv2d(6, 12, kernel_size=3, padding=1, groups=groups)
    conv_custom = ContextualConv2d(6, 12, kernel_size=3, padding=1, groups=groups)

    copy_conv_weights(conv_ref, conv_custom)

    x = torch.randn(2, 6, 16, 16)
    out_ref = conv_ref(x)
    out_custom = conv_custom(x)

    assert out_ref.shape == out_custom.shape
    assert torch.allclose(out_ref, out_custom, atol=1e-6)


@pytest.mark.parametrize("groups", [1, 2])
def test_contextual_conv1d_no_context(groups):
    conv_ref = nn.Conv1d(4, 8, kernel_size=5, padding=2, groups=groups)
    conv_custom = ContextualConv1d(4, 8, kernel_size=5, padding=2, groups=groups)

    copy_conv_weights(conv_ref, conv_custom)

    x = torch.randn(2, 4, 32)
    out_ref = conv_ref(x)
    out_custom = conv_custom(x)

    assert out_ref.shape == out_custom.shape
    assert torch.allclose(out_ref, out_custom, atol=1e-6)


@pytest.mark.parametrize("conv_cls, input_shape, kernel_size", [
    (ContextualConv1d, (2, 4, 32), 3),
    (ContextualConv2d, (2, 4, 16, 16), 3),
])
def test_contextual_conv_with_context(conv_cls, input_shape, kernel_size):
    c_dim = 10
    in_channels = 4
    out_channels = 8

    conv = conv_cls(in_channels, out_channels, kernel_size=kernel_size, padding=1, c_dim=c_dim)
    x = torch.randn(*input_shape)
    c = torch.randn(input_shape[0], c_dim)  # context shape: (N, c_dim)

    out = conv(x, c)

    # Basic shape check
    if len(input_shape) == 3:
        assert out.shape == (input_shape[0], out_channels, input_shape[2])
    elif len(input_shape) == 4:
        assert out.shape == (input_shape[0], out_channels, input_shape[2], input_shape[3])


def test_invalid_context_dim():
    conv = ContextualConv2d(3, 6, kernel_size=3, padding=1, c_dim=5)
    x = torch.randn(1, 3, 16, 16)
    c = torch.randn(1, 6)  # wrong c_dim

    with pytest.raises(ValueError):
        _ = conv(x, c)
