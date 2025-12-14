"""Binary linear layer implementations with bitops quantization utilities."""

import torch
from typing import Tuple
from torch import nn
import torch.nn.functional as F
from .quantizers import get_quantizers

try:
    import bitops
except ImportError:
    raise ImportError("bitops is required for deployment. Install with: pip install -e .")


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
        
        # Get quantization functions from registry
        self.activation_quant, self.weight_quant = get_quantizers(quant_type)

        # Initialize parameters
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Initialize weights and quantizer
        self._init_weights()
        
        # Flag to track deployment state
        self._is_deployed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization-aware linear transformation suitable for training."""
        dqx = x - (x - self.activation_quant(x)).detach()
        dqw = self.weight - (self.weight - self.weight_quant(self.weight)).detach()
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
        Deploy the layer for efficient inference using bitops by:
        1. Quantizing weights to 2-bit {-1, 0, 1} using bitops.quant_t2spg
        2. Packing entire weight tensor using bitops.pack_t2s for efficient storage
        3. Removing original parameters
        4. Switching to optimized forward pass
        """
        if self._is_deployed:
            return
        
        # Get device for tensor creation
        device = self.weight.device
        
        # Quantize weights to 2-bit using bitops (per-tensor quantization)
        # group_size = total elements means single scale factor (per-tensor)
        group_size = self.out_features * self.in_features
        w_scale, w_quant = bitops.quant_t2spg(self.weight, group_size)
        
        # Pack the entire quantized weight tensor to 2-bit format
        # Flatten first, pack, then reshape to maintain structure
        w_quant_flat = w_quant.flatten()
        w_packed_flat = bitops.pack_t2s(w_quant_flat)
        # Reshape packed weights: each packed element contains 4 values (2-bit each in int8)
        packed_per_row = (self.in_features + 3) // 4  # Ceiling division
        w_packed = w_packed_flat.reshape(self.out_features, packed_per_row)
        
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
        
        # Mark as deployed and switch forward method
        self._is_deployed = True
        self.forward = self._deploy_forward

    def _deploy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the quantized inference pathway using bitops matmul.
        
        Uses bitops.matmul_f32_i8spr_t2spt which:
        - Quantizes input activations to int8 per-row
        - Performs matrix multiplication with packed 2-bit weights (per-tensor scale)
        - Adds bias
        - Supports both 2D and 3D inputs natively
        
        Args:
            x: Input tensor of shape [batch, in_features] or [batch, seq_len, in_features]
        
        Returns:
            Output tensor of shape [batch, out_features] or [batch, seq_len, out_features]
        """
        return bitops.matmul_f32_i8spr_t2spt(
            x,
            self.w_scale,
            self.w_packed,
            self.bias_buffer
        )