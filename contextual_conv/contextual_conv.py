import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Literal

__all__ = ["ContextProcessor", "ContextualConv1d", "ContextualConv2d"]


class ContextProcessor(nn.Module):
    """Maps a global context vector `c` to per-channel parameters (gamma, beta)."""

    def __init__(
        self,
        context_dim: int,
        out_dim: int,
        h_dim: Optional[int] = None,
        linear_bias: bool = False,
    ) -> None:
        super().__init__()
        if h_dim is None or (isinstance(h_dim, int) and h_dim <= 0):
            self.processor = nn.Linear(context_dim, out_dim, bias=linear_bias)
        else:
            layers = []
            input_dim = context_dim
            hidden_dims = h_dim if isinstance(h_dim, list) else [h_dim]
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=linear_bias))
                layers.append(nn.ReLU(inplace=True))
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, out_dim, bias=linear_bias))
            self.processor = nn.Sequential(*layers)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.processor(c)


class _ContextualConvBase(nn.Module):
    """Base class for contextual convolution layers with FiLM or scale+shift modulation."""

    _NDIMS: int  # to be defined in subclasses

    def __init__(
        self,
        conv: nn.Module,
        *,
        context_dim: Optional[int] = None,
        h_dim: Optional[int] = None,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_scale: bool = False,
        use_bias: bool = True,
        linear_bias: bool = False,
        g: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        scale_mode: Literal["film", "scale"] = "film",
    ) -> None:
        super().__init__()

        if not use_scale and not use_bias:
            raise ValueError("At least one of `use_scale` or `use_bias` must be True.")

        self.conv = conv
        self.activation = activation
        self.g_fn = g if g is not None else self._default_g_fn
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.use_context = context_dim is not None and context_dim > 0
        self.out_channels = conv.out_channels
        self.scale_mode = scale_mode

        if self.use_context:
            n_parts = (self.use_scale + self.use_bias) * self.out_channels
            self.context_processor = ContextProcessor(context_dim, n_parts, h_dim, linear_bias=linear_bias)

            if self.use_scale:
                self._init_scale()

    def _default_g_fn(self, out: torch.Tensor) -> torch.Tensor:
        squared = out.pow(2)
        dims = list(range(2, 2 + self._NDIMS))
        return squared.mean(dim=dims)

    def _split_ctx(self, ctx: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        idx = 0
        gamma = beta = None
        if self.use_scale:
            gamma = ctx[:, idx:idx + self.out_channels]
            idx += self.out_channels
        if self.use_bias:
            beta = ctx[:, idx:idx + self.out_channels]
        return gamma, beta

    def _apply_modulation(self, out: torch.Tensor, gamma: Optional[torch.Tensor], beta: Optional[torch.Tensor]):
        if gamma is not None:
            for _ in range(self._NDIMS):
                gamma = gamma.unsqueeze(-1)
            if self.scale_mode == "film":
                out = out * (1.0 + gamma)
            elif self.scale_mode == "scale":
                out = out * gamma
        if beta is not None:
            for _ in range(self._NDIMS):
                beta = beta.unsqueeze(-1)
            out = out + beta
        return out

    def _init_scale(self) -> None:
        """Initialize scale parameters to identity depending on scale mode."""
        if not self.use_context:
            return

        processor = self.context_processor.processor
        last_linear = None
        if isinstance(processor, nn.Linear):
            last_linear = processor
        elif isinstance(processor, nn.Sequential):
            last_linear = processor[-1]

        if last_linear is not None:
            if self.use_scale:
                if self.scale_mode == "film":
                    nn.init.zeros_(last_linear.weight)
                    if last_linear.bias is not None:
                        nn.init.zeros_(last_linear.bias)
                elif self.scale_mode == "scale":
                    nn.init.ones_(last_linear.weight)
                    if last_linear.bias is not None:
                        nn.init.zeros_(last_linear.bias)

    def _forward_impl(self, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)

        if self.use_context and c is not None:
            ctx = self.context_processor(c)
            gamma, beta = self._split_ctx(ctx)
            out = self._apply_modulation(out, gamma, beta)

        return out

    @torch.no_grad()
    def infer_context(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_context:
            raise RuntimeError("Context is not enabled in this layer.")
        processor = self.context_processor.processor
        if not isinstance(processor, nn.Linear):
            raise RuntimeError("ContextProcessor must be a single Linear layer.")
        if processor.bias is not None and not torch.allclose(processor.bias, torch.zeros_like(processor.bias)):
            raise RuntimeError("Linear bias must be disabled (bias=False).")
        if self.use_bias:
            raise RuntimeError("use_bias must be False for reversibility.")

        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)

        V = self.g_fn(out)
        W_plus_1 = processor.weight + 1
        return V @ W_plus_1


class ContextualConv1d(_ContextualConvBase):
    _NDIMS = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        activation: Optional[Callable] = None,
        context_dim: Optional[int] = None,
        h_dim: Optional[int] = None,
        use_scale: bool = False,
        use_bias: bool = True,
        linear_bias: bool = False,
        g: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        scale_mode: Literal["film", "scale"] = "film",
        **conv_kwargs,
    ) -> None:
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, **conv_kwargs)
        super().__init__(
            conv,
            activation=activation,
            context_dim=context_dim,
            h_dim=h_dim,
            use_scale=use_scale,
            use_bias=use_bias,
            linear_bias=linear_bias,
            g=g,
            scale_mode=scale_mode,
        )

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._forward_impl(x, c)


class ContextualConv2d(_ContextualConvBase):
    _NDIMS = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        activation: Optional[Callable] = None,
        context_dim: Optional[int] = None,
        h_dim: Optional[int] = None,
        use_scale: bool = False,
        use_bias: bool = True,
        linear_bias: bool = False,
        g: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        scale_mode: Literal["film", "scale"] = "film",
        **conv_kwargs,
    ) -> None:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)
        super().__init__(
            conv,
            activation=activation,
            context_dim=context_dim,
            h_dim=h_dim,
            use_scale=use_scale,
            use_bias=use_bias,
            linear_bias=linear_bias,
            g=g,
            scale_mode=scale_mode,
        )

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._forward_impl(x, c)
