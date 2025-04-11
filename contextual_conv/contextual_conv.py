import torch
import torch.nn as nn
from typing import Optional


class ContextProcessor(nn.Module):
    """
    Processes the context vector into a per-channel bias.
    Uses a single linear layer or a small MLP depending on `h_dim`.
    """

    def __init__(self, context_dim: int, out_channels: int, h_dim: Optional[int] = None):
        super().__init__()
        if h_dim is None or h_dim == 0:
            self.processor = nn.Linear(context_dim, out_channels)
        else:
            self.processor = nn.Sequential(
                nn.Linear(context_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Linear(h_dim, out_channels)
            )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: Context vector of shape (B, context_dim)
        Returns:
            Output tensor of shape (B, out_channels)
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
