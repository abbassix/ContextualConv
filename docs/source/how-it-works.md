# How It Works

`ContextualConv1d` and `ContextualConv2d` layers:

- Use `unfold()` to extract local patches (like `im2col`)
- Flatten patches to matrices
- Apply group-wise matrix multiplication with learned kernels
- Optionally concatenate a global context vector to each patch
- Add learned bias and reshape the output

These layers are fully differentiable and compatible with PyTorch autograd.
