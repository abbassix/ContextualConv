import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class ContextualConv1d(nn.Module):
    """
    Custom 1D convolutional layer using unfold + matrix multiplication,
    with optional global context vector injection at every temporal step.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the 1D convolution kernel.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int, optional): Zero-padding. Default: 0.
        dilation (int, optional): Dilation rate. Default: 1.
        groups (int, optional): Number of groups. Default: 1.
        bias (bool, optional): Whether to include a learnable bias. Default: True.
        c_dim (int, optional): Dimensionality of the optional global context vector. Default: None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        c_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.c_dim = c_dim

        self.group_in_channels = in_channels // groups
        self.group_out_channels = out_channels // groups

        self.weight = nn.Parameter(torch.randn(
            out_channels, self.group_in_channels, kernel_size
        ))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

        self.c_weight = (
            nn.Parameter(torch.randn(out_channels, c_dim)) if c_dim is not None else None
        )

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (N, C_in, L).
            c (Tensor, optional): Context tensor of shape (N, c_dim).

        Returns:
            Tensor: Output tensor of shape (N, C_out, L_out).
        """
        N, _, L = x.shape
        k, s, p, d = self.kernel_size, self.stride, self.padding, self.dilation

        # Apply padding
        x_padded = F.pad(x, (p, p))

        # Unfold: extract 1D patches
        patches = x_padded.unfold(dimension=2, size=k, step=s)  # shape: (N, C_in, L_out, K)
        L_out = patches.shape[2]
        input_matrix = patches.permute(0, 2, 1, 3).reshape(N, L_out, -1)  # (N, L_out, C_in * K)

        # If context is used, expand and concatenate
        if self.c_dim is not None and c is not None:
            if c.shape[-1] != self.c_dim:
                raise ValueError(f"Expected c.shape[-1] = {self.c_dim}, got {c.shape[-1]}")
            c_expanded = c.view(N, 1, self.c_dim).expand(N, L_out, self.c_dim)
            input_matrix = torch.cat([input_matrix, c_expanded], dim=-1)

        outputs = []

        for g in range(self.groups):
            # Group weight: (out_ch_per_group, in_ch_per_group, K)
            weight_g = self.weight[
                g * self.group_out_channels : (g + 1) * self.group_out_channels
            ].reshape(self.group_out_channels, -1)  # (out_ch_per_group, in_ch_per_group * K)

            if self.c_dim is not None and self.c_weight is not None:
                c_weight_g = self.c_weight[
                    g * self.group_out_channels : (g + 1) * self.group_out_channels
                ]  # (out_ch_per_group, c_dim)
                weight_g = torch.cat([weight_g, c_weight_g], dim=1)

            # (N, L_out, out_ch_per_group)
            output_g = torch.matmul(input_matrix, weight_g.T)
            outputs.append(output_g)

        # Concatenate group outputs: (N, L_out, out_channels)
        out = torch.cat(outputs, dim=-1)

        if self.bias is not None:
            out += self.bias.view(1, 1, -1)

        return out.permute(0, 2, 1)  # (N, out_channels, L_out)


class ContextualConv2d(nn.Module):
    """
    A custom 2D convolutional layer using im2col with optional global context conditioning.

    This layer mimics a standard grouped 2D convolution (like nn.Conv2d) but:
    - Uses im2col (unfold) + matrix multiplication instead of conv.
    - Supports conditioning on a global context vector `c` shared across all spatial locations.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or Tuple[int, int]): Convolution kernel size.
        stride (int or Tuple[int, int], optional): Convolution stride. Default: 1.
        padding (int or Tuple[int, int], optional): Zero-padding added to both sides. Default: 0.
        dilation (int or Tuple[int, int], optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of groups for grouped convolution. Default: 1.
        padding_mode (str, optional): Padding mode ('zeros', 'reflect', etc.). Default: 'zeros'.
        bias (bool, optional): If True, adds a learnable bias. Default: True.
        c_dim (int, optional): Dimensionality of the context vector. If None, no context is used.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        padding_mode: str = 'zeros',
        bias: bool = True,
        c_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Normalize inputs to (h, w) tuples
        self.kernel_size = self._to_pair(kernel_size)
        self.stride = self._to_pair(stride)
        self.padding = self._to_pair(padding)
        self.dilation = self._to_pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.padding_mode = padding_mode
        self.c_dim = c_dim

        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")

        self.group_in_channels = in_channels // groups
        self.group_out_channels = out_channels // groups

        # Main convolutional weights (like nn.Conv2d)
        self.weight = nn.Parameter(torch.randn(
            out_channels, self.group_in_channels, *self.kernel_size
        ))

        # Optional bias
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

        # Contextual weights if context vector is used
        self.c_weight = (
            nn.Parameter(torch.randn(out_channels, c_dim)) if c_dim is not None else None
        )

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (N, C_in, H, W).
            c (Tensor, optional): Context tensor of shape (N, c_dim) or (N, 1, c_dim).

        Returns:
            Tensor: Output tensor of shape (N, C_out, H_out, W_out).
        """
        N, _, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        # Pad input
        if self.padding_mode == 'zeros':
            x_padded = F.pad(x, (pw, pw, ph, ph), mode='constant', value=0)
        else:
            x_padded = F.pad(x, (pw, pw, ph, ph), mode=self.padding_mode)

        # Extract sliding windows (patches) using unfold
        patches = x_padded.unfold(2, kh, sh).unfold(3, kw, sw)
        # Apply dilation (by slicing every dh/dw-th row/column)
        patches = patches[..., ::dh, ::dw]

        out_h, out_w = patches.shape[2], patches.shape[3]
        # Shape: (N, out_h * out_w, C_in * kh * kw)
        input_matrix = patches.permute(0, 2, 3, 1, 4, 5).reshape(N, out_h * out_w, -1)

        # Context integration (if provided)
        if self.c_dim is not None and c is not None:
            if c.shape[-1] != self.c_dim:
                raise ValueError(f"Expected c.shape[-1] = {self.c_dim}, got {c.shape[-1]}")
            c_expanded = c.view(N, 1, self.c_dim).expand(N, out_h * out_w, self.c_dim)
            input_matrix = torch.cat([input_matrix, c_expanded], dim=-1)

        outputs = []

        for g in range(self.groups):
            # Extract group-specific weights and flatten
            weight_g = self.weight[
                g * self.group_out_channels : (g + 1) * self.group_out_channels
            ].reshape(self.group_out_channels, -1)

            # Add context weights if needed
            if self.c_dim is not None and self.c_weight is not None:
                c_weight_g = self.c_weight[
                    g * self.group_out_channels : (g + 1) * self.group_out_channels
                ]
                weight_g = torch.cat([weight_g, c_weight_g], dim=1)

            # Group-wise matrix multiplication
            output_g = torch.matmul(input_matrix, weight_g.T)  # (N, out_h*out_w, out_channels_per_group)
            outputs.append(output_g)

        # Concatenate across groups and reshape
        out = torch.cat(outputs, dim=-1)  # (N, out_h*out_w, out_channels)
        if self.bias is not None:
            out += self.bias.view(1, 1, -1)

        # Reshape to image-like output: (N, C_out, out_h, out_w)
        return out.permute(0, 2, 1).reshape(N, self.out_channels, out_h, out_w)

    @staticmethod
    def _to_pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        return (value, value) if isinstance(value, int) else value
