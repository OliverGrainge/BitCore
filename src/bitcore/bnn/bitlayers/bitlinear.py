"""Binary linear layer implementations with shared quantization utilities."""

import torch
from torch import nn
import torch.nn.functional as F
from ..bitquantizer import BitQuantizer
from ..functional import bitlinear
from .base import BaseBitLayer


class BitLinear(BaseBitLayer):
    """
    A binary neural network linear layer that quantizes weights to {-1, 0, 1}.

    This layer supports two execution modes:
    - Training mode: Applies quantized weights while preserving gradient flow.
    - Deployment mode: Swaps parameters for packed buffers and a lightweight
      inference-only computation path.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include a bias term.
        eps: Small epsilon for numerical stability in quantization.
        quant_type: Identifier for activation/weight quantization pair to use.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
    ):
        """
        Initialize a binary linear layer with learnable parameters and a quantizer.

        Args:
            in_features: Number of input activations per sample.
            out_features: Number of output activations per sample.
            bias: Whether to include a learnable bias term.
            eps: Small constant added during quantization to avoid division by zero.
            quant_type: String identifier that selects the activation/weight quantization pair.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.quant_type = quant_type

        # Initialize parameters
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Initialize weights and quantizer
        self._init_weights()
        self.quantizer = BitQuantizer(eps=eps, quant_type=quant_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization-aware linear transformation suitable for training."""
        dqx, dqw = self.quantizer(x, self.weight)
        return F.linear(dqx, dqw, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, quant_type: str, eps: float = 1e-6) -> None:
        """
        Convert a floating-point linear layer into a `BitLinear` instance.

        The resulting layer copies over trained weights/bias values and enables
        quantization-aware execution using the provided quantizer configuration.

        Args:
            linear: Source `nn.Linear` module whose parameters should be cloned.
            quant_type: Identifier describing the activation/weight quantization
                flavors to adopt.
            eps: Small epsilon safeguarding activation quantization operations.

        Returns:
            `BitLinear` instance initialized to mirror the supplied `linear`.
        """
        qt = getattr(linear, "quant_type", quant_type)
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None, eps, qt)
        layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _deploy(self) -> None:
        """
        Deploy the layer for efficient inference by:
        1. Quantizing and packing weights
        2. Removing original parameters
        3. Switching to optimized forward pass
        """
        # Quantize and pack weights for deployment
        qs, qw = bitlinear.prepare_weights(self.weight, self.eps, self.quant_type)
        bias_data = self.bias.detach().clone() if self.bias is not None else None
        del self.bias, self.weight

        # Replace parameters with quantized buffers
        self.register_buffer("qws", qs)
        self.register_buffer("qw", qw)
        self.register_buffer("bias", bias_data)

        # Switch to optimized forward pass
        self.forward = self._deploy_forward

    def _deploy(self) -> None:
        """
        Deploy the layer for efficient inference by:
        1. Quantizing and packing weights (and bias)
        2. Removing original parameters
        3. Switching to optimized forward pass
        """
        # prepare_weights now handles both weights and bias
        buffer_data = bitlinear.prepare_weights(
            self.weight, 
            self.bias,  # Pass bias to prepare_weights
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

        # Switch to optimized forward pass
        self.forward = self._deploy_forward

    def _deploy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the quantized inference pathway after `deploy` has packed the weights."""
        # Reconstruct all buffers
        buffer_kwargs = {key: getattr(self, key) for key in self._buffer_keys}
        return bitlinear(x, **buffer_kwargs, eps=self.eps, quant_type=self.quant_type)

