import torch
import torch.nn as nn
from typing import Optional, Tuple

__all__ = ["ContextProcessor", "ContextualConv1d", "ContextualConv2d"]


class ContextProcessor(nn.Module):
    """Maps a *global* context vector ``c`` to per‑channel parameters.

    The processor is deliberately lightweight – a single ``Linear`` layer by
    default or an MLP with one hidden layer if ``h_dim`` is provided.  The
    output dimension is chosen by the calling layer and can represent scale
    (``\gamma``), bias (``\beta``) or both.
    """

    def __init__(
        self,
        context_dim: int,
        out_dim: int,
        h_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if h_dim is None or (isinstance(h_dim, int) and h_dim <= 0):
            # Simple linear projection
            self.processor = nn.Linear(context_dim, out_dim)
        else:
            # Build MLP
            layers = []
            input_dim = context_dim
            hidden_dims = h_dim if isinstance(h_dim, list) else [h_dim]

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                input_dim = hidden_dim

            layers.append(nn.Linear(input_dim, out_dim))
            self.processor = nn.Sequential(*layers)

    def forward(self, c: torch.Tensor) -> torch.Tensor:  # noqa: D401  (keep short doc)
        """Return context‑dependent parameters.

        Args:
            c: Tensor of shape ``(B, context_dim)``.
        Returns:
            Tensor of shape ``(B, out_dim)``.
        """
        return self.processor(c)


class _ContextualConvBase(nn.Module):
    """Shared implementation details for 1‑D and 2‑D contextual conv layers."""

    _NDIMS: int  # to be set by subclasses

    def __init__(
        self,
        conv: nn.Module,
        *,
        context_dim: Optional[int] = None,
        h_dim: Optional[int] = None,
        use_scale: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if not use_scale and not use_bias:
            raise ValueError("At least one of `use_scale` or `use_bias` must be True.")

        self.conv = conv
        self.use_scale = bool(use_scale)
        self.use_bias = bool(use_bias)
        self.use_context = context_dim is not None and context_dim > 0
        self.out_channels = conv.out_channels

        if self.use_context:
            n_parts = (self.use_scale + self.use_bias) * self.out_channels
            self.context_processor = ContextProcessor(context_dim, n_parts, h_dim)

        # Optional: init context processor so the layer starts as identity.
        if self.use_context and self.use_scale:
            self._init_scale_to_one()

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _split_ctx(self, ctx: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Split *ctx* into ``gamma`` and/or ``beta`` parts depending on flags."""
        idx = 0
        gamma = beta = None
        if self.use_scale:
            gamma = ctx[:, idx : idx + self.out_channels]
            idx += self.out_channels
        if self.use_bias:
            beta = ctx[:, idx : idx + self.out_channels]
        return gamma, beta

    def _apply_film(self, out: torch.Tensor, gamma: Optional[torch.Tensor], beta: Optional[torch.Tensor]):
        """Broadcast and apply FiLM parameters."""
        if gamma is not None:
            # Broadcast: (B, C, *1) – add trailing singleton dims to match ND.
            for _ in range(self._NDIMS):
                gamma = gamma.unsqueeze(-1)
            out = out * (1.0 + gamma)  # centre scale at 1 for stability
        if beta is not None:
            for _ in range(self._NDIMS):
                beta = beta.unsqueeze(-1)
            out = out + beta
        return out

    def _init_scale_to_one(self) -> None:
        """Initialise scale parameters such that *gamma*≈0 ⇒ effective scale ≈1."""
        last_linear: nn.Linear = None  # type: ignore
        if isinstance(self.context_processor.processor, nn.Linear):
            last_linear = self.context_processor.processor
        elif isinstance(self.context_processor.processor, nn.Sequential):
            last_linear = self.context_processor.processor[-1]  # type: ignore

        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    # ---------------------------------------------------------------------
    # Forward – subclasses will wrap and add dimensionality logic.
    # ---------------------------------------------------------------------
    def _forward_impl(self, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        out = self.conv(x)
        if self.use_context and c is not None:
            ctx = self.context_processor(c)  # (B, n_parts)
            gamma, beta = self._split_ctx(ctx)
            out = self._apply_film(out, gamma, beta)
        return out


class ContextualConv1d(_ContextualConvBase):
    """1‑D convolution with optional FiLM‑style global conditioning.

    Works as a drop‑in replacement for :class:`torch.nn.Conv1d`.  When a global
    context vector ``c`` is provided, the layer can predict a per‑channel
    *scale* (``\gamma``), *bias* (``\beta``) or both and apply them to the
    convolution output, following the FiLM formulation.
    """

    _NDIMS = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        context_dim: Optional[int] = None,
        h_dim: Optional[int] = None,
        use_scale: bool = False,
        use_bias: bool = True,
        **conv_kwargs,
    ) -> None:
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, **conv_kwargs)
        super().__init__(
            conv,
            context_dim=context_dim,
            h_dim=h_dim,
            use_scale=use_scale,
            use_bias=use_bias,
        )

    # pyre-ignore[3]: We intentionally match ``nn.Conv1d`` signature (+c).
    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401
        """Apply convolution followed by optional FiLM modulation."""
        return self._forward_impl(x, c)


class ContextualConv2d(_ContextualConvBase):
    """2‑D convolution with optional FiLM‑style global conditioning.

    Usage is identical to :class:`torch.nn.Conv2d` except for the extra
    *context* arguments that control scaling and biasing.
    """

    _NDIMS = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        context_dim: Optional[int] = None,
        h_dim: Optional[int] = None,
        use_scale: bool = False,
        use_bias: bool = True,
        **conv_kwargs,
    ) -> None:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)
        super().__init__(
            conv,
            context_dim=context_dim,
            h_dim=h_dim,
            use_scale=use_scale,
            use_bias=use_bias,
        )

    # pyre-ignore[3]
    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401
        """Apply convolution followed by optional FiLM modulation."""
        return self._forward_impl(x, c)
