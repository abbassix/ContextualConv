# Architecture Overview

This diagram shows how `ContextualConv2d` works internally.

```{graphviz}
:caption: ContextualConv2d Architecture

digraph ContextualConv2d {
    rankdir=TB;
    Input -> Unfold [label="im2col"];
    Unfold -> SliceGroups [label="split by group"];
    SliceGroups -> MatMul [label="grouped weights"];
    MatMul -> AddContext [label="(optional) + scale/bias"];
    AddContext -> BiasAdd [label="FiLM: γ * conv(x) + β"];
    BiasAdd -> Reshape [label="(N, C_out, H, W)"];
}
```

Each convolution group handles a slice of the input and combines it with optional global context parameters (scale and/or bias).
