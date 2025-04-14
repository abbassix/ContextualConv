import torch
import torch.nn as nn
from typing import Optional


class ContextProcessor(nn.Module):
    """
    Processes the context vector into a per-channel bias.
    Supports single Linear layer, shallow MLP (int), or multi-layer MLP (list of ints).
    """

    def __init__(self, context_dim: int, out_channels: int, h_dim: Optional[Union[int, List[int]]] = None):
        super().__init__()

        layers = []
        in_dim = context_dim

        if h_dim is None or h_dim == 0:
            # Simple projection
            layers.append(nn.Linear(in_dim, out_channels))
        else:
            # Multi-layer MLP: normalize to list
            if isinstance(h_dim, int):
                h_dim = [h_dim]
            for hidden_dim in h_dim:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_channels))

        self.processor = nn.Sequential(*layers)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c (Tensor): Context vector of shape (B, context_dim)
        Returns:
            Tensor: Bias of shape (B, out_channels)
        """
        return self.processor(c)


class ContextualConv1d(nn.Module):
    """
    1D Convolution layer with optional context-aware output modulation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 context_dim: Optional[int] = None,
                 h_dim: Optional[int] = None,
                 **kwargs):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            context_dim: Size of the context vector (set to None/0 to disable).
            h_dim: Hidden layer size for MLP if used (None or 0 disables MLP).
            kwargs: All other arguments passed to nn.Conv1d.
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)

        if context_dim and context_dim > 0:
            self.use_context = True
            self.context_processor = ContextProcessor(context_dim, out_channels, h_dim)
        else:
            self.use_context = False

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C_in, L)
            c: Optional context tensor of shape (B, context_dim)
        Returns:
            Output tensor of shape (B, C_out, L_out)
        """
        out = self.conv(x)

        if self.use_context and c is not None:
            bias = self.context_processor(c).unsqueeze(-1)  # (B, C_out, 1)
            out = out + bias  # Broadcast bias over temporal dimension

        return out


class ContextualConv2d(nn.Module):
    """
    2D Convolution layer with optional context-aware output modulation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 context_dim: Optional[int] = None,
                 h_dim: Optional[int] = None,
                 **kwargs):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            context_dim: Size of the context vector (set to None/0 to disable).
            h_dim: Hidden layer size for MLP if used (None or 0 disables MLP).
            kwargs: All other arguments passed to nn.Conv2d.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

        if context_dim and context_dim > 0:
            self.use_context = True
            self.context_processor = ContextProcessor(context_dim, out_channels, h_dim)
        else:
            self.use_context = False

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            c: Optional context tensor of shape (B, context_dim)
        Returns:
            Output tensor of shape (B, C_out, H_out, W_out)
        """
        out = self.conv(x)

        if self.use_context and c is not None:
            bias = self.context_processor(c).unsqueeze(-1).unsqueeze(-1)  # (B, C_out, 1, 1)
            out = out + bias  # Broadcast bias over spatial dimensions

        return out
