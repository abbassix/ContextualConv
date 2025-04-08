# ContextualConv

**ContextualConv** is a family of custom PyTorch convolutional layers (`ContextualConv1d`, `ContextualConv2d`, etc.) that extend standard convolutions with support for **global context conditioning**.

Instead of using built-in convolution operators, these layers apply convolution via **im2col + matrix multiplication**, while enabling **location-invariant modulation** via an optional context vector `c`.

---

## 🔧 Features

- ⚙️ Drop-in replacement for `nn.Conv1d` and `nn.Conv2d` (with grouped conv support)
- 🧠 Context-aware: injects global information into every spatial or temporal location
- 🧱 Uses `unfold` (im2col) to compute convolution explicitly via matrix multiplication
- 📦 Fully differentiable and autograd-compatible

---

## 📦 Installation

Copy `contextual_conv.py` into your project directory and import the required layer:

```python
from contextual_conv import ContextualConv1d, ContextualConv2d
```

No external dependencies beyond PyTorch.

---

## 🚀 Usage
2D Example
```
import torch
from contextual_conv import ContextualConv2d

conv = ContextualConv2d(
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    padding=1,
    c_dim=10  # context dimensionality
)

x = torch.randn(8, 16, 32, 32)       # input image tensor
c = torch.randn(8, 10)               # global context vector

out = conv(x, c)  # shape: (8, 32, 32, 32)

If you don’t need context, just omit it:

conv = ContextualConv2d(16, 32, kernel_size=3, padding=1)
out = conv(x)  # works without `c`
```

1D Example
```
from contextual_conv import ContextualConv1d

conv1d = ContextualConv1d(
    in_channels=16,
    out_channels=32,
    kernel_size=5,
    padding=2,
    c_dim=6
)

x = torch.randn(4, 16, 100)  # input time series
c = torch.randn(4, 6)        # context vector

out = conv1d(x, c)           # shape: (4, 32, 100)
```

You can omit the context vector `c` entirely if you don't need it:
```
conv = ContextualConv1d(16, 32, kernel_size=3, padding=1)
out = conv(x)  # still works without context
```

---

## 📐 Context Vector

- The context tensor `c` can be of shape `(N, c_dim)` or `(N, 1, c_dim)`.

- It is *broadcasted* over all spatial positions and concatenated to the local input patch at each location.

- Learnable weights `c_weight` are applied to this context and used in the convolution computation.

---

## 🔍 When to Use

Use `ContextualConv` when you need:

- Feature maps influenced by external signals (e.g., class embeddings, latent codes).

- Global conditioning of convolutional outputs without spatial variation.

- Interpretable, customizable convolutional logic using explicit matrix ops.

---

## 📄 License

GNU GPLv3 License

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Add support for ContextualConv3d

- Improve speed via einsum or GPU-specific tricks

- Add performance benchmarks or gradient checks

Please open an issue or pull request.
---

## 🧪 Tests

Unit tests are included in the tests/ directory. To verify the correctness of ContextualConv1d and ContextualConv2d, we compare their output against PyTorch’s built-in nn.Conv1d and nn.Conv2d using identical weights and no context.

### ✅ Running the tests

Make sure you have pytest installed:
```
pip install pytest
```
Then run the tests:
```
pytest tests/
```
This will execute the test cases defined in tests/test_contextual_conv.py and confirm that the custom layers behave as expected.

---

## 📫 Contact

For questions or suggestions, open an issue or reach out via GitHub.
