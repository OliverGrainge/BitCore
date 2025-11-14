"""Binary convolutional layer implementations built on BitLab quantization."""

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..bitquantizer import BitQuantizer
from ..functional import bitconv2d
from .base import BaseBitLayer


class BitConv2d(BaseBitLayer):
    """
    Binary `Conv2d` that aligns training and deployment quantization workflows.

    The layer mirrors the interface of `nn.Conv2d` while introducing the ability
    to quantize activations/weights during training and to swap parameters for
    packed buffers at deployment time.
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
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
    ):
        """
        Construct a binary convolution layer with learnable parameters and quantizer.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of convolutional filters to produce.
            kernel_size: Spatial size of the convolution kernel.
            stride: Convolution stride for height and width.
            padding: Implicit zero padding applied to the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections (as in grouped convolution).
            bias: Whether to allocate a learnable bias term.
            eps: Small constant injected for quantization stability.
            quant_type: Identifier describing the activation/weight quantization pairing.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation)
        )
        self.groups = groups
        self.eps = eps
        self.quant_type = quant_type

        self.weight = nn.Parameter(
            torch.zeros(
                out_channels,
                in_channels // groups,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self._init_weights()
        self.quantizer = BitQuantizer(eps=eps, quant_type=quant_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization-aware convolution suitable for training loops."""
        dqx, dqw = self.quantizer(x, self.weight)
        return F.conv2d(
            dqx, dqw, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

    @classmethod
    def from_conv2d(cls, conv2d: nn.Conv2d, quant_type: str, eps: float = 1e-6) -> None:
        """
        Instantiate a `BitConv2d` module using an existing floating-point layer.

        Args:
            conv2d: Source `nn.Conv2d` module whose parameters are copied.
            quant_type: Identifier describing the activation/weight quantization
                configuration to adopt.
            eps: Stability constant used by the quantizer.

        Returns:
            `BitConv2d` instance initialized with the source layer's parameters.
        """
        layer = cls(conv2d.in_channels, conv2d.out_channels, conv2d.kernel_size, conv2d.stride, conv2d.padding, conv2d.dilation, conv2d.groups, conv2d.bias is not None, eps, quant_type)
        layer.weight.data.copy_(conv2d.weight.data)
        if conv2d.bias is not None:
            layer.bias.data.copy_(conv2d.bias.data)
        return layer

    def _init_weights(self) -> None:
        """Use Kaiming init for weights and zeros for bias."""
        nn.init.kaiming_uniform_(self.weight, a=0, mode="fan_in", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _deploy(self) -> None:
        """Freeze parameters into quantized buffers and switch to deploy forward."""
        # prepare_weights now handles both weights and bias
        buffer_data = bitconv2d.prepare_weights(
            self.weight,
            self.bias,
            self.eps,
            self.quant_type
        )
        
        # Delete original parameters
        del self.bias, self.weight

        # Store the keys for later unpacking
        self._buffer_keys = list(buffer_data.keys())
        
        # Register all tensors as buffers
        for key, tensor in buffer_data.items():
            self.register_buffer(key, tensor)

        self.forward = self._deploy_forward

    def _deploy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the packed-weight quantized convolution used during deployment."""
        # Reconstruct all buffers
        buffer_kwargs = {key: getattr(self, key) for key in self._buffer_keys}
        
        return bitconv2d(
            x,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            eps=self.eps,
            quant_type=self.quant_type,
            **buffer_kwargs,
        )
