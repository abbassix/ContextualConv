# FAQ

### ❓ What is the context vector used for?

The context vector is a global signal (e.g., class label, latent state) broadcasted to all local patches to condition the output.

---

### ❓ What happens if I don’t provide `c`?

The layer falls back to standard convolution behavior (no context conditioning).

---

### ❓ Does it support grouped convolutions?

Yes! Just set `groups > 1`, and the layer will split the input/output channels per group, like PyTorch’s built-in `Conv1d`/`Conv2d`.

---

### ❓ Can it be used for dynamic filtering?

Yes. You can use the context vector to modulate the output dynamically, like attention or class conditioning.
