# Architecture Overview

This diagram shows how `ContextualConv2d` works internally.

```{graphviz}
digraph ContextualConv2d {
    rankdir=LR;
    Input -> Unfold [label="im2col"];
    Unfold -> SliceGroups [label="split by group"];
    SliceGroups -> MatMul [label="grouped weights"];
    MatMul -> AddContext [label="(optional) + context"];
    AddContext -> BiasAdd [label="+ bias"];
    BiasAdd -> Reshape [label="reshape to (N, C_out, H, W)"];
}
```

Each convolution group handles a slice of the input and combines it with optional global context.