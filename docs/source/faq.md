# FAQ

### ❓ What is the context vector used for?

The context vector is a global input (e.g., class label, latent code) that is transformed into a **per-channel bias**. This bias is added uniformly across all spatial (Conv2d) or temporal (Conv1d) positions in the output.

---

### ❓ How is the context processed?

The context is passed through a `ContextProcessor`:
- If `h_dim` is not set → a single linear layer:  
  ```python
  Linear(context_dim, out_channels)
  ```
- If `h_dim` is set → a two-layer MLP with ReLU:
  ```python
  Linear(context_dim, h_dim) → ReLU → Linear(h_dim, out_channels)
  ```

---

### ❓ Does the context affect each spatial location differently?

No — the same per-channel bias is **broadcast** across all positions. The context acts globally.

---

### ❓ What happens if I don’t provide `context_dim`?

The layer behaves exactly like a standard `nn.Conv1d` or `nn.Conv2d`.

---

### ❓ Can this replace SE blocks or FiLM?

In spirit, yes. While it’s not spatially aware like attention, the design enables **global conditioning** and is similar to SE blocks (in additive form).

---

### ❓ Is it compatible with grouped/depthwise convolutions?

Yes. The same context bias is shared across groups, respecting the weight-sharing behavior of grouped convolutions.
