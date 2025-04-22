# ContextualConv  

[![PyPI version](https://img.shields.io/pypi/v/contextual-conv)](https://pypi.org/project/contextual-conv/)
[![CI](https://github.com/abbassix/ContextualConv/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/abbassix/ContextualConv/actions/workflows/test.yml)
[![Docs](https://readthedocs.org/projects/contextualconv/badge/?version=latest)](https://contextualconv.readthedocs.io/en/latest/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **ContextualConv** – PyTorch convolutional layers with **global context conditioning**: per‑channel **bias**, **scale**, or **FiLM‑style** *scale + bias*.

---

## 🚀 Quick start

```python
from contextual_conv import ContextualConv2d
import torch

# FiLM‑style (scale + bias)
conv = ContextualConv2d(
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    padding=1,
    context_dim=10,   # size of global vector c
    h_dim=64,         # optional MLP hidden dim
    use_scale=True,   # γ(c)
    use_bias=True     # β(c)
)

x = torch.randn(8, 16, 32, 32)  # feature map
c = torch.randn(8, 10)          # context vector

out = conv(x, c)  # shape: (8, 32, 32, 32)
```

### Modes at a glance
| `use_scale` | `use_bias` | Behaviour |
|-------------|-----------|-----------|
| `False`     | `True`    | **Contextual bias** (original behaviour) |
| `True`      | `False`   | **Per‑channel scale** only |
| `True`      | `True`    | **FiLM** – scale **and** bias |

If *both* flags are `False`, the constructor raises `ValueError`.

---

## 🔧 Key features

* ⚙️ **Drop‑in replacement** for `nn.Conv1d` / `nn.Conv2d`  
  → Same arguments + optional context options.
* 🧠 **Global vector conditioning** via learnable γ(c) and/or β(c).
* 🪶 **Lightweight** – one small MLP (or single `Linear`) per layer.
* 🧑‍🔬 **FiLM ready** – reproduce Feature‑wise Linear Modulation with two lines.
* 🧩 **Modular** – combine with any architecture, works on CPU / GPU.
* ✅ **Unit‑tested** and documented.

---

## 📦 Installation

```bash
pip install contextual-conv  # version 0.3.0 on PyPI
```

Or install from source:

```bash
git clone https://github.com/abbassix/ContextualConv.git
cd ContextualConv
pip install -e .[dev]
```

---

## 📐 Context vector details

* Shape: **`(B, context_dim)`**  
  (one global descriptor per sample – class label embedding, latent code, etc.)
* Processed by a **`ContextProcessor`**:
  * `Linear(context_dim, out_dim)` *(bias‑only / scale‑only)*
  * Small **MLP** if `h_dim` is set.
* Output dims:
  * `out_channels` → bias **or** scale
  * `2 × out_channels` → FiLM (scale + bias)

---

## 🧪 Running tests

```bash
pytest tests/
```

---

## 📘 Documentation

Full API reference & tutorials: **<https://contextualconv.readthedocs.io>**

---

## 🤝 Contributing

Bug reports, feature requests, and PRs are welcome! See `CONTRIBUTING.md`.

---

## 📄 License

GNU GPLv3 – see `LICENSE` file for details.

