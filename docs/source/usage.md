# Usage Guide

This guide shows how to use `ContextualConv1d` and `ContextualConv2d` in your PyTorch models.

## 🔧 1D Example (no context)

```python
from contextual_conv import ContextualConv1d
import torch

layer = ContextualConv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
x = torch.randn(2, 4, 32)
out = layer(x)
```

## 🔧 2D Example with context

```python
from contextual_conv import ContextualConv2d
import torch

layer = ContextualConv2d(
    in_channels=3,
    out_channels=6,
    kernel_size=3,
    padding=1,
    c_dim=10
)

x = torch.randn(4, 3, 16, 16)
c = torch.randn(4, 10)
out = layer(x, c)
```

## ✅ Notes

- Context vector `c` is broadcasted to all spatial/temporal locations.
- Works like grouped `nn.Conv1d` and `nn.Conv2d`.
- Use `groups > 1` for grouped convolution behavior.
