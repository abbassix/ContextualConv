import torch
import torch.nn as nn
import pytest
from contextual_conv import ContextualConv1d, ContextualConv2d


# ----- Helpers -----

def has_same_output_shape(out1, out2):
    return out1.shape == out2.shape


# ----- 1D Tests -----

def test_conv1d_without_context():
    model = ContextualConv1d(3, 6, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 64)
    out = model(x)
    assert out.shape == (2, 6, 64)


def test_conv1d_with_context_linear():
    model = ContextualConv1d(3, 6, kernel_size=3, padding=1, context_dim=10)
    x = torch.randn(2, 3, 64)
    c = torch.randn(2, 10)
    out = model(x, c)
    assert out.shape == (2, 6, 64)


def test_conv1d_with_context_mlp():
    model = ContextualConv1d(3, 6, kernel_size=3, padding=1, context_dim=10, h_dim=16)
    x = torch.randn(2, 3, 64)
    c = torch.randn(2, 10)
    out = model(x, c)
    assert out.shape == (2, 6, 64)


# ----- 2D Tests -----

def test_conv2d_without_context():
    model = ContextualConv2d(3, 6, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 6, 32, 32)


def test_conv2d_with_context_linear():
    model = ContextualConv2d(3, 6, kernel_size=3, padding=1, context_dim=10)
    x = torch.randn(2, 3, 32, 32)
    c = torch.randn(2, 10)
    out = model(x, c)
    assert out.shape == (2, 6, 32, 32)


def test_conv2d_with_context_mlp():
    model = ContextualConv2d(3, 6, kernel_size=3, padding=1, context_dim=10, h_dim=16)
    x = torch.randn(2, 3, 32, 32)
    c = torch.randn(2, 10)
    out = model(x, c)
    assert out.shape == (2, 6, 32, 32)


# ----- Error case -----

def test_context_shape_mismatch():
    model = ContextualConv2d(3, 6, kernel_size=3, padding=1, context_dim=10)
    x = torch.randn(2, 3, 32, 32)
    c = torch.randn(2, 5)  # wrong context dim
    with pytest.raises(RuntimeError):
        _ = model(x, c)


# ----- Multi-layer MLP Tests -----

def test_contextual_conv2d_multi_layer_mlp():
    model = ContextualConv2d(
        in_channels=3,
        out_channels=10,
        kernel_size=3,
        padding=1,
        context_dim=8,
        h_dim=[32, 64, 16]
    )

    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    c = torch.randn(2, 8, requires_grad=True)

    out = model(x, c)

    assert out.shape == (2, 10, 16, 16)

    # Test backward
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert c.grad is not None
