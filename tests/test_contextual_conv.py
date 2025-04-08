import torch
import torch.nn as nn
from contextual_conv import ContextualConv1d, ContextualConv2d

def test_contextual_conv2d():
    # Initialize reference (standard) and custom convolution layers
    conv_ref = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    conv_custom = ContextualConv2d(3, 8, kernel_size=3, padding=1)
    
    # Manually copy weights from reference conv to custom conv
    conv_custom.weight.data = conv_ref.weight.data.clone()
    conv_custom.bias.data = conv_ref.bias.data.clone()

    x = torch.randn(1, 3, 16, 16)  # Dummy input

    # Run both layers
    out_ref = conv_ref(x)
    out_custom = conv_custom(x)

    # Validate outputs match
    assert torch.allclose(out_ref, out_custom, atol=1e-6), "Test failed for Conv2D"

def test_contextual_conv1d():
    # Initialize reference (standard) and custom convolution layers
    conv_ref = nn.Conv1d(3, 8, kernel_size=5, padding=2)
    conv_custom = ContextualConv1d(3, 8, kernel_size=5, padding=2)
    
    # Manually copy weights from reference conv to custom conv
    conv_custom.weight.data = conv_ref.weight.data.clone()
    conv_custom.bias.data = conv_ref.bias.data.clone()

    x = torch.randn(1, 3, 16)  # Dummy input

    # Run both layers
    out_ref = conv_ref(x)
    out_custom = conv_custom(x)

    # Validate outputs match
    assert torch.allclose(out_ref, out_custom, atol=1e-6), "Test failed for Conv1D"
