import torch
import torch.nn as nn
from contextual_conv import ContextualConv1d, ContextualConv2d


def copy_conv_weights(ref_layer: nn.Module, custom_layer: nn.Module) -> None:
    """
    Safely copies weights and bias from a standard nn.Conv1d/2d layer
    to a custom ContextualConv1d/2d layer (without context).
    """
    custom_layer.weight.data.copy_(ref_layer.weight.data)
    if ref_layer.bias is not None and custom_layer.bias is not None:
        custom_layer.bias.data.copy_(ref_layer.bias.data)


def test_contextual_conv2d():
    conv_ref = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    conv_custom = ContextualConv2d(3, 8, kernel_size=3, padding=1)

    copy_conv_weights(conv_ref, conv_custom)

    x = torch.randn(1, 3, 16, 16)

    out_ref = conv_ref(x)
    out_custom = conv_custom(x)

    assert torch.allclose(out_ref, out_custom, atol=1e-6), "ContextualConv2d output mismatch"


def test_contextual_conv1d():
    conv_ref = nn.Conv1d(3, 8, kernel_size=5, padding=2)
    conv_custom = ContextualConv1d(3, 8, kernel_size=5, padding=2)

    copy_conv_weights(conv_ref, conv_custom)

    x = torch.randn(1, 3, 16)

    out_ref = conv_ref(x)
    out_custom = conv_custom(x)

    assert torch.allclose(out_ref, out_custom, atol=1e-6), "ContextualConv1d output mismatch"
