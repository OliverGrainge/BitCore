"""Binary convolutional layer implementations with bitops quantization utilities."""

import torch
from typing import Union, Tuple
from torch import nn
import torch.nn.functional as F
from .quantizers import get_quantizers


class BitConv2d(nn.Module):
    """
    A binary neural network convolutional layer that quantizes weights to {-1, 0, 1}.

    This layer applies quantized 2D convolution while preserving gradient flow during training.
    Currently, both training and deployment modes use the same execution path, as the
    optimized bitops kernel for convolutions is not yet implemented.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to both sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input channels to output channels.
        bias: Whether to include a bias term.
        padding_mode: Padding mode ('zeros', 'reflect', 'replicate', 'circular').
        eps: Small epsilon for numerical stability in quantization.
        quant_type: Quantization type identifier (e.g., "bitnet", "binary").
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
        bias: bool = True,
        padding_mode: str = 'zeros',
        eps: float = 1e-6,
        quant_type: str = "bitnet",
    ):
        """
        Initialize a binary convolutional layer with learnable parameters.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel (int or tuple).
            stride: Stride of the convolution (default: 1).
            padding: Zero-padding added to both sides (default: 0).
            dilation: Spacing between kernel elements (default: 1).
            groups: Number of blocked connections (default: 1).
            bias: Whether to include a learnable bias term (default: True).
            padding_mode: Padding mode (default: 'zeros').
            eps: Small constant for numerical stability (default: 1e-6).
            quant_type: Quantization type identifier (e.g., "bitnet", "binary").
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.eps = eps
        self.quant_type = quant_type
        
        # Get quantizer class from registry and instantiate
        # For conv layers, treat flattened weight dimensions
        weight_out_features = out_channels
        weight_in_features = (in_channels // groups) * self.kernel_size[0] * self.kernel_size[1]
        quantizer_cls = get_quantizers(quant_type)
        self.quantizer = quantizer_cls(weight_out_features, weight_in_features)

        # Initialize parameters
        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Initialize weights
        self._init_weights()
        
        # Flag to track deployment state (not used yet, but kept for future compatibility)
        self._is_deployed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantization-aware convolutional transformation.
        
        Args:
            x: Input tensor of shape [batch, in_channels, height, width]
            
        Returns:
            Output tensor of shape [batch, out_channels, out_height, out_width]
        """
        # Use quantizer to get dequantized activations and weights (with STE)
        dqx, dqw = self.quantizer(x, self.weight)
        return F.conv2d(
            dqx, dqw, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, eps: float = 1e-6, quant_type: str = "bitnet") -> 'BitConv2d':
        """
        Convert a floating-point Conv2d layer into a `BitConv2d` instance.

        The resulting layer copies over trained weights/bias values and enables
        quantization-aware execution.

        Args:
            conv: Source `nn.Conv2d` module whose parameters should be cloned.
            eps: Small epsilon safeguarding activation quantization operations.
            quant_type: Quantization type identifier (e.g., "bitnet", "binary").

        Returns:
            `BitConv2d` instance initialized to mirror the supplied `conv`.
        """
        layer = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            eps,
            quant_type
        )
        layer.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            layer.bias.data.copy_(conv.bias.data)
        return layer

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming uniform initialization (suitable for Conv2d)."""
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _deploy(self) -> None:
        """
        Deploy the layer for efficient inference.
        
        Note: This is a placeholder for future implementation. Currently, the optimized
        bitops kernel for convolutions is not available, so deployment mode uses the
        same execution path as training/eval mode.
        """
        if self._is_deployed:
            return
        
        # Mark as deployed but keep using the same forward method
        self._is_deployed = True
        # No parameter swapping yet - optimized kernel not implemented
        # self.forward remains unchanged



