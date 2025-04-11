# Usage Guide

This guide shows how to use `ContextualConv1d` and `ContextualConv2d` in your PyTorch models.

## 🔧 1D Example (no context)

```python
from contextual_conv import ContextualConv1d
import torch

layer = ContextualConv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
x = torch.randn(2, 4, 32)
out = layer(x)  # shape: (2, 8, 32)
```

## 🧠 1D Example with context (bias only)

```python
layer = ContextualConv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1, context_dim=10)
c = torch.randn(2, 10)
out = layer(x, c)  # shape: (2, 8, 32)
```

## 🧠 1D with MLP for context

```python
layer = ContextualConv1d(
    in_channels=4,
    out_channels=8,
    kernel_size=3,
    padding=1,
    context_dim=10,
    h_dim=16
)
out = layer(x, c)
```

## 🖼️ 2D Example with context

```python
from contextual_conv import ContextualConv2d

conv = ContextualConv2d(
    in_channels=3,
    out_channels=6,
    kernel_size=3,
    padding=1,
    context_dim=10,
    h_dim=32
)

x = torch.randn(2, 3, 16, 16)
c = torch.randn(2, 10)
out = conv(x, c)  # shape: (2, 6, 16, 16)
```

## ✅ Notes

- If `context_dim` is set, the context vector `c` is passed through a linear layer or MLP.
- The result is used as a **per-output-channel bias**, added uniformly across all locations.
- You can disable the MLP by omitting `h_dim`.
