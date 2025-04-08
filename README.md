# ContextualConv

**ContextualConv** is a family of custom PyTorch convolutional layers (`ContextualConv1d`, `ContextualConv2d`) that support **global context conditioning**.

These layers mimic standard PyTorch convolutions using **im2col + matrix multiplication**, while allowing a global context vector `c` to modulate the output at all spatial or temporal positions.

---

## 🔧 Features

- ⚙️ Drop-in replacement for `nn.Conv1d` and `nn.Conv2d`
- 🧠 Context-aware: injects global information into every location
- 🧱 Uses `unfold` (im2col) to compute convolution explicitly
- 📦 Fully differentiable, supports grouped convolutions

---

## 📦 Installation

Clone the repo or copy `contextual_conv.py` into your project, then install dependencies:

```bash
pip install -r requirements.txt
```

Then install the correct PyTorch version **for your system (CPU or CUDA)** by following the official instructions:

🔗 https://pytorch.org/get-started/locally/

Examples:

- CPU-only:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

- CUDA 11.8:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

---

## 🚀 Usage

### 2D Example

```python
import torch
from contextual_conv import ContextualConv2d

conv2d = ContextualConv2d(
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    padding=1,
    c_dim=10  # context dimensionality
)

x = torch.randn(8, 16, 32, 32)
c = torch.randn(8, 10)

out = conv2d(x, c)  # shape: (8, 32, 32, 32)
```

### 1D Example

```python
from contextual_conv import ContextualConv1d

conv1d = ContextualConv1d(
    in_channels=16,
    out_channels=32,
    kernel_size=5,
    padding=2,
    c_dim=6
)

x = torch.randn(4, 16, 100)
c = torch.randn(4, 6)

out = conv1d(x, c)  # shape: (4, 32, 100)
```

### Without context

```python
conv = ContextualConv2d(16, 32, kernel_size=3, padding=1)
out = conv(x)  # works even without `c`
```

---

## 📐 Context Vector

- Shape: `(N, c_dim)` or `(N, 1, c_dim)`
- Broadcasted to all positions (spatial or temporal)
- Concatenated to each unfolded input patch
- Modulated by learnable `c_weight` before being added to the convolution output

---

## 🔍 When to Use

Use `ContextualConv` layers when:

- You want to inject external or global information into feature maps
- You need interpretable and customizable convolution logic
- You want context-aware dynamic filtering with no extra spatial modeling

---

## 🧪 Tests

Unit tests are included in `tests/test_contextual_conv.py`.

### ✅ To run the tests:

```bash
pip install -r requirements.txt
pip install torch  # see installation instructions above
pytest tests/
```

The tests compare `ContextualConv1d` and `ContextualConv2d` against standard PyTorch layers with context disabled, ensuring correctness.

---

## 🤖 GitHub Actions (CI)

A GitHub Actions workflow in `.github/workflows/test.yml` automatically runs tests on push and pull requests using the CPU version of PyTorch.

---

## 📄 License

GNU GPLv3 License

---

## 🤝 Contributing

Contributions welcome! You can:

- Add `ContextualConv3d` support
- Improve performance (e.g., using `einsum`)
- Write more advanced tests or benchmarks
- Create useful networks that use these layers

Open a pull request or issue to get started.

---

## 📫 Contact

Questions or suggestions? Open an issue or reach out via GitHub.

