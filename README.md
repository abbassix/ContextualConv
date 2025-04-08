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
```

No external dependencies beyond PyTorch.

---

## 🚀 Usage
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

---

## 📐 Context Vector

- The context tensor `c` can be of shape `(N, c_dim)` or `(N, 1, c_dim)`.

- It is broadcasted over all spatial positions and concatenated to the local input patch at each location.

- Learnable weights `c_weight` are applied to this context and used in the convolution computation.

---

## 🔍 When to Use

Use `ContextualConv2d` when you need:

- Feature maps that are modulated by external signals (e.g., class labels, embeddings, latent vectors).

- Dynamic filtering or conditioning without spatial variation.

- A custom convolutional layer you can fully inspect and modify.

---

## 📄 License

GNU GPLv3 License

---

## 🤝 Contributing

Feel free to open issues or pull requests! Contributions are welcome if you want to extend this to support depthwise convolutions, different padding strategies, or performance optimizations.

---

## 🧪 Tests

Unit tests are under development. You can validate basic functionality by comparing it with `nn.Conv2d` on toy data with context disabled.

---

## 📫 Contact

For questions or suggestions, open an issue or reach out via GitHub.


---

Let me know if you'd like to add a diagram, benchmarks, or example notebooks to go with it!

