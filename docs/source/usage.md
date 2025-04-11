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

## 🧠 Context with Hidden Layer (`h_dim`)

You can specify `h_dim` to use a learnable MLP to process the context vector before it's added to the output.

```python
from contextual_conv import ContextualConv2d
import torch

conv = ContextualConv2d(
    in_channels=3,
    out_channels=12,
    kernel_size=3,
    padding=1,
    c_dim=8,    # context dimension
    h_dim=32    # hidden dimension for context MLP
)

x = torch.randn(4, 3, 16, 16)
c = torch.randn(4, 8)
out = conv(x, c)  # shape: (4, 12, 16, 16)
```

The context vector `c` is passed through an MLP:
`c → Linear(c_dim → h_dim) → ReLU → Linear(h_dim → out_channels)`,
and the result is added to the convolution output.


## ✅ Notes

- Context vector `c` is broadcasted to all spatial/temporal locations.
- Works like grouped `nn.Conv1d` and `nn.Conv2d`.
- Use `groups > 1` for grouped convolution behavior.
