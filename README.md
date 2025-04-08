# ContextualConv2d

**ContextualConv2d** is a custom PyTorch module that extends standard 2D convolution with support for **global context conditioning**. It mimics the behavior of `nn.Conv2d` using an **im2col + matrix multiplication** approach, while enabling **location-invariant conditioning** on an optional global context vector `c`.

---

## 🔧 Features

- ⚙️ Drop-in replacement for `nn.Conv2d` (with grouped conv support)
- 🧠 Context-aware: injects global information into every spatial location
- 🧱 Uses `unfold` (im2col) to explicitly compute convolution as matrix multiplication
- 📦 Fully differentiable and compatible with autograd

---

## 📦 Installation

Just copy `contextual_conv.py` into your project directory and import:

```python
from contextual_conv import ContextualConv2d
