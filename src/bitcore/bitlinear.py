"""Binary linear layer implementations with bitops quantization utilities."""

import torch
from typing import Tuple
from torch import nn
import torch.nn.functional as F
from .quantizers import get_quantizers


class BitLinear(nn.Module):
    """
    A binary neural network linear layer that quantizes weights to {-1, 0, 1}.

    This layer supports two execution modes:
    - Training mode: Applies quantized weights while preserving gradient flow.
    - Deployment mode: Swaps parameters for packed buffers and a lightweight
      inference-only computation path using bitops.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include a bias term.
        eps: Small epsilon for numerical stability in quantization.
        quant_type: Quantization type identifier (e.g., "bitnet", "binary").
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-6,
        quant_type: str = "bitnet",
    ):
        """
        Initialize a binary linear layer with learnable parameters and a quantizer.

        Args:
            in_features: Number of input activations per sample.
            out_features: Number of output activations per sample.
            bias: Whether to include a learnable bias term.
            eps: Small constant added during quantization to avoid division by zero.
            quant_type: Quantization type identifier (e.g., "bitnet", "binary").
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.quant_type = quant_type
        
        # Get quantizer class from registry and instantiate
        quantizer_cls = get_quantizers(quant_type)
        self.quantizer = quantizer_cls(out_features, in_features)

        # Initialize parameters
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Initialize weights
        self._init_weights()
        
        # Flag to track deployment state
        self._is_deployed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization-aware linear transformation suitable for training."""
        # Use quantizer to get dequantized activations and weights (with STE)
        dqx, dqw = self.quantizer(x, self.weight)
        return F.linear(dqx, dqw, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, quant_type: str = "bitnet", eps: float = 1e-6) -> 'BitLinear':
        """
        Convert a floating-point linear layer into a `BitLinear` instance.

        The resulting layer copies over trained weights/bias values and enables
        quantization-aware execution using the provided quantizer configuration.

        Args:
            linear: Source `nn.Linear` module whose parameters should be cloned.
            quant_type: Quantization type identifier (e.g., "bitnet", "binary").
            eps: Small epsilon safeguarding activation quantization operations.

        Returns:
            `BitLinear` instance initialized to mirror the supplied `linear`.
        """
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None, eps, quant_type)
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
        Deploy the layer for efficient inference using the quantizer's deployment API.
        
        This method:
        1. Uses quantizer.get_deployment_weights() to prepare weights
        2. Gets the appropriate inference function from quantizer.get_inference_fn()
        3. Removes original parameters
        4. Switches to optimized forward pass
        """
        if self._is_deployed:
            return
        
        # Get device for tensor creation
        device = self.weight.device
        
        # Get deployment weights from quantizer (handles packing if bitops available)
        w_scale, w_packed = self.quantizer.get_deployment_weights(self.weight)
        
        # Store bias if it exists, otherwise create zeros
        if self.bias is not None:
            bias_data = self.bias.data.clone()
        else:
            bias_data = torch.zeros(self.out_features, dtype=torch.float32, device=device)
        
        # Delete original parameters
        del self.weight
        if self.bias is not None:
            del self.bias
        
        # Register buffers for deployment
        self.register_buffer('w_scale', w_scale)
        self.register_buffer('w_packed', w_packed)
        self.register_buffer('bias_buffer', bias_data)
        
        # Get inference function from quantizer
        self.inference_fn = self.quantizer.get_inference_fn()
        
        # Mark as deployed and switch forward method
        self._is_deployed = True
        self.forward = self._deploy_forward

    def _deploy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the quantized inference pathway using the quantizer's inference function.
        
        The inference function handles:
        - Quantizing input activations appropriately
        - Performing matrix multiplication with packed/quantized weights
        - Adding bias
        - Supporting both 2D and 3D inputs
        
        Args:
            x: Input tensor of shape [batch, in_features] or [batch, seq_len, in_features]
        
        Returns:
            Output tensor of shape [batch, out_features] or [batch, seq_len, out_features]
        """
        return self.inference_fn(
            x,
            self.w_scale,
            self.w_packed,
            self.bias_buffer
        )