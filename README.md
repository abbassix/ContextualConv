[![PyPI version](https://img.shields.io/pypi/v/contextual-conv)](https://pypi.org/project/contextual-conv/)
[![CI](https://github.com/abbassix/ContextualConv/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/abbassix/ContextualConv/actions/workflows/test.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://readthedocs.org/projects/contextualconv/badge/?version=latest)](https://contextualconv.readthedocs.io/en/latest/)

# ContextualConv

**ContextualConv** is a family of custom PyTorch convolutional layers (`ContextualConv1d`, `ContextualConv2d`) that support **global context conditioning**.

These layers behave like standard PyTorch `nn.Conv1d` and `nn.Conv2d`, but allow a global vector `c` to inject **per-channel bias** into the output, modulating it with contextual information (e.g., class embeddings, latent vectors, etc.).

---

## 🔧 Features

- ⚙️ Drop-in replacement for `nn.Conv1d` and `nn.Conv2d`
- 🧠 Context-aware: injects global vector as output bias
- 🧱 Based on standard PyTorch convolution
- 🧠 Optional hidden layer (`h_dim`) for MLP processing of `c`
- 📦 Fully differentiable and unit-tested

---

## 📦 Installation

Clone the repo or copy `contextual_conv.py` into your project, then:

```bash
pip install -r requirements.txt
```

To install PyTorch, follow the official guide:
https://pytorch.org/get-started/locally/

Example (CPU only):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 🚀 Usage

### 2D Example (with context and MLP)

```python
from contextual_conv import ContextualConv2d
import torch

conv = ContextualConv2d(
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    padding=1,
    context_dim=10,
    h_dim=64
)

x = torch.randn(8, 16, 32, 32)
c = torch.randn(8, 10)

out = conv(x, c)  # shape: (8, 32, 32, 32)
```

### 1D Example (linear context projection)

```python
from contextual_conv import ContextualConv1d

conv = ContextualConv1d(
    in_channels=16,
    out_channels=32,
    kernel_size=5,
    padding=2,
    context_dim=6
)

x = torch.randn(4, 16, 100)
c = torch.randn(4, 6)

out = conv(x, c)  # shape: (4, 32, 100)
```

### Without context

```python
conv = ContextualConv2d(16, 32, kernel_size=3, padding=1)
out = conv(x)  # standard conv2d
```

---

## 📐 Context Vector

- Shape: `(B, context_dim)`
- Passed through a `ContextProcessor` (either `Linear` or `MLP`)
- Output shape: `(B, out_channels)` → added as a bias to the output

---

## 🧪 Tests

All tests live in `tests/test_contextual_conv.py`.

Run them with:

```bash
pytest tests/
```

---

## 📘 Documentation

Full documentation is available at:

👉 https://contextualconv.readthedocs.io

Includes API reference, architecture explanation, and usage tips.

---

## 📄 License

Licensed under GNU GPLv3.

---

## 🤝 Contributing

You're welcome to:
- Add `ContextualConv3d`
- Suggest other context conditioning strategies
- Add notebook examples
- Improve performance

Open an issue or PR to contribute!

---

## 📫 Contact

Questions? Issues? Reach out on GitHub or open a discussion.
