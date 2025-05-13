# contextual_conv.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Literal, Union, List

__all__ = ["ContextProcessor", "ContextualConv1d", "ContextualConv2d"]


# -----------------------------------------------------------------------------#
#                               helper modules                                 #
# -----------------------------------------------------------------------------#
class ContextProcessor(nn.Module):
    """
    Maps a global context vector c ∈ ℝ^{context_dim} to per-channel parameters.

    By default it is a single Linear layer.  If ``h_dim`` is an int or list of
    ints, an MLP (Linear → ReLU → … → Linear) is built automatically.
    """

    def __init__(
        self,
        context_dim: int,
        out_dim: int,
        h_dim: Optional[Union[int, List[int]]] = None,
        linear_bias: bool = False,
    ) -> None:
        super().__init__()

        if h_dim is None or (isinstance(h_dim, int) and h_dim <= 0):
            self.processor = nn.Linear(context_dim, out_dim, bias=linear_bias)
        else:
            layers: List[nn.Module] = []
            hidden_dims = h_dim if isinstance(h_dim, list) else [h_dim]
            in_dim = context_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(in_dim, h, bias=linear_bias), nn.ReLU(inplace=True)])
                in_dim = h
            layers.append(nn.Linear(in_dim, out_dim, bias=linear_bias))
            self.processor = nn.Sequential(*layers)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.processor(c)

    # Convenience for weight initialisation
    @property
    def last_linear(self) -> nn.Linear:
        return self.processor if isinstance(self.processor, nn.Linear) else self.processor[-1]


# -----------------------------------------------------------------------------#
#                             contextual conv base                             #
# -----------------------------------------------------------------------------#
class _ContextualConvBase(nn.Module):
    """
    Base class for contextual convolution layers (1-D / 2-D) with FiLM-style
    or pure scaling modulation.

    Two independent heads are created:
        gamma-head → per-channel scale
        beta-head → per-channel shift
    """

    _NDIMS: int  # spatial rank (1 for Conv1d, 2 for Conv2d); set in subclasses

    def __init__(
        self,
        conv: nn.Module,
        *,
        context_dim: Optional[int] = None,
        h_dim: Optional[Union[int, List[int]]] = None,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_scale: bool = True,
        use_bias: bool = False,
        linear_bias: bool = False,
        g: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        scale_mode: Literal["film", "scale"] = "scale",
    ) -> None:
        super().__init__()

        # -------------------- sanity checks ---------------------------------- #
        if context_dim is not None and context_dim > 0:
            if not use_scale and not use_bias:
                raise ValueError(
                    "If `context_dim` is set, at least one of `use_scale` or "
                    "`use_bias` must be True."
                )

        # -------------------- core attributes -------------------------------- #
        self.conv = conv
        self.activation = activation
        self.g_fn = g if g is not None else self._default_g_fn
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.use_context = context_dim is not None and context_dim > 0
        self.out_channels = conv.out_channels
        self.scale_mode = scale_mode

        # -------------------- context processors ----------------------------- #
        if self.use_context:
            if self.use_scale:
                self.gamma_proc = ContextProcessor(
                    context_dim, self.out_channels, h_dim, linear_bias=linear_bias
                )
                self._init_gamma_weights()
            else:
                self.gamma_proc = None

            if self.use_bias:
                self.beta_proc = ContextProcessor(
                    context_dim, self.out_channels, h_dim, linear_bias=linear_bias
                )
            else:
                self.beta_proc = None

    # -------------------------------- utils ----------------------------------#
    def _default_g_fn(self, feats: torch.Tensor) -> torch.Tensor:
        """Non-negative per-channel weights   v_b = mean_{spatial}(out²)."""
        squared = feats.pow(2)
        dims = list(range(2, 2 + self._NDIMS))
        return squared.mean(dim=dims)

    def _unsqueeze_to_match(self, x: torch.Tensor) -> torch.Tensor:
        """Add NDIMS trailing singleton dims so x can broadcast across feature maps."""
        for _ in range(self._NDIMS):
            x = x.unsqueeze(-1)
        return x

    def _apply_modulation(
        self,
        feats: torch.Tensor,
        gamma: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply (1+gamma)·x or gamma·x, then add beta, with correct broadcasting."""
        if gamma is not None:
            gamma_exp = self._unsqueeze_to_match(gamma)
            feats = feats * (1.0 + gamma_exp) if self.scale_mode == "film" else feats * gamma_exp
        if beta is not None:
            feats = feats + self._unsqueeze_to_match(beta)
        return feats

    def _init_gamma_weights(self) -> None:
        """Identity init so un-trained gamma leaves the network unchanged."""
        last = self.gamma_proc.last_linear
        if self.scale_mode == "film":
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)
        else:  # "scale"
            nn.init.ones_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    # ------------------------------ forward ----------------------------------#
    def _forward_impl(self, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        feats = self.conv(x)
        if self.activation is not None:
            feats = self.activation(feats)

        if self.use_context and c is not None:
            gamma = self.gamma_proc(c) if self.use_scale else None
            beta = self.beta_proc(c) if self.use_bias else None
            feats = self._apply_modulation(feats, gamma, beta)

        return feats

    # ---------------------------- wieghted goodness ---------------------------#
    def weighted_goodness(self, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        feats = self.conv(x)
        if self.activation is not None:
            feats = self.activation(feats)

        g = self.g_fn(feats)  # shape: (B, C)

        if self.use_context and c is not None:
            gamma = self.gamma_proc(c) if self.use_scale else None  # shape: (B, C)
            beta = self.beta_proc(c) if self.use_bias else None     # shape: (B, C)

            if self.scale_mode == "film":
                if gamma is not None:
                    g = g * (1.0 + gamma)
            else:
                if gamma is not None:
                    g = g * gamma

            if beta is not None:
                g = g + beta

        return g

    # --------------------------- analytic inference --------------------------#
    @torch.no_grad()
    def infer_context(
        self,
        inputs: torch.Tensor,
        return_raw_output: bool = False,
    ):
        """
        Compute the weighted-average scores for *all* possible one-hot context
        vectors without scanning them one by one.

        Returns
        -------
        ctx_scores : Tensor,  shape (B, context_dim)
        raw_feats  : Optional[Tensor]  (if requested)
        """

        if not self.use_context:
            raise RuntimeError("Context is not enabled in this layer.")

        # 1) Forward conv → activation
        feats = self.conv(inputs)
        if self.activation is not None:
            feats = self.activation(feats)

        v = self.g_fn(feats)  # (B, C)  non-negative sample weights

        # 2) Accumulate contributions from gamma and beta heads
        # Infer context_dim from whichever head is available
        if self.use_scale and hasattr(self.gamma_proc, "last_linear"):
            context_dim = self.gamma_proc.last_linear.in_features
        elif self.use_bias and hasattr(self.beta_proc, "last_linear"):
            context_dim = self.beta_proc.last_linear.in_features
        else:
            raise RuntimeError("At least one of use_scale or use_bias must be True, and head must end in nn.Linear.")

        ctx_scores = torch.zeros(v.size(0), context_dim, device=v.device)

        # ---- gamma-head --------------------------------------------------------- #
        if self.use_scale:
            g_last = self.gamma_proc.last_linear
            W_gamma = g_last.weight  # (out_channels, context_dim)
            if self.scale_mode == "film":
                W_gamma = W_gamma + 1.0
            ctx_scores += v @ W_gamma
            if g_last.bias is not None:
                ctx_scores += (v @ g_last.bias).unsqueeze(-1)

        # ---- beta-head --------------------------------------------------------- #
        if self.use_bias:
            b_last = self.beta_proc.last_linear
            W_beta = b_last.weight  # (out_channels, context_dim)
            ctx_scores += v @ W_beta
            if b_last.bias is not None:
                ctx_scores += (v @ b_last.bias).unsqueeze(-1)

        # 3) Normalise → weighted average
        denom = v.sum(dim=1, keepdim=True).clamp_min(1e-8)
        ctx_scores = ctx_scores / denom

        return (ctx_scores, feats) if return_raw_output else ctx_scores


# -----------------------------------------------------------------------------#
#                           concrete 1-D / 2-D classes                          #
# -----------------------------------------------------------------------------#
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
        h_dim: Optional[Union[int, List[int]]] = None,
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

    def weighted_goodness(self, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        return super().weighted_goodness(x, c)


class ContextualConv2d(_ContextualConvBase):
    _NDIMS = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        *,
        activation: Optional[Callable] = None,
        context_dim: Optional[int] = None,
        h_dim: Optional[Union[int, List[int]]] = None,
        use_scale: bool = False,
        use_bias: bool = True,
        linear_bias: bool = False,
        g: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        scale_mode: Literal["film", "scale"] = "scale",
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
    
    def weighted_goodness(self, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        return super().weighted_goodness(x, c)