# How It Works

## 🧠 ContextualConv Layers

`ContextualConv1d` and `ContextualConv2d` extend standard convolutions by allowing the output to be **modulated by a global context vector**.

## 🔄 Forward Pass

1. Input tensor `x` is passed through a standard `nn.Conv1d` or `nn.Conv2d` layer.
2. If a context vector `c` is provided:
   - It is passed through a shared `ContextProcessor` module:
     - If `h_dim` is not set: just a `Linear(context_dim → out_channels)`
     - If `h_dim` is set: an MLP `Linear → ReLU → Linear`
     - If both scale and bias are used: output is `2 * out_channels` and split into γ and β
   - The result is a scale (`γ`) and/or bias (`β`) of shape `(B, out_channels)`
   - These parameters are broadcast and applied to the conv output:

```python
y = γ(c) * conv(x) + β(c)
```

## 🧩 Architecture (1D/2D)

```python
x → Conv1d/2d → y
c → ContextProcessor → γ, β
y → γ * y + β → final output
```

## ✅ Notes

- The context vector is **global** — it does not vary across spatial/temporal locations.
- The `ContextProcessor` is **shared** across all locations and all groups (if used).
- If `context_dim` is not set, the layer behaves like a standard convolution.
