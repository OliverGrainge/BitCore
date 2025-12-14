"""Quantization function registry for BitCore layers."""

import torch
from typing import Callable, Tuple

# ============================================================================
# BitNet Quantization (default)
# ============================================================================

def bitnet_activation_quant(x: torch.Tensor) -> torch.Tensor:
    """
    8-bit per-token activation quantization from BitNet paper.
    
    Args:
        x: Activation tensor with shape [n, d] for linear or [n, c, h, w] for conv
        
    Returns:
        Quantized activation tensor with same shape as input
    """
    dim = 1 if x.dim() == 4 else -1  # channels for conv, features for linear
    scale = 127.0 / x.abs().max(dim=dim, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def bitnet_weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    1.58-bit ternary weight quantization from BitNet paper.
    
    Quantizes weights to {-1, 0, 1} using per-tensor scaling.
    
    Args:
        w: Weight tensor of any shape
        
    Returns:
        Quantized weight tensor with same shape as input
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


# ============================================================================
# Binary Quantization
# ============================================================================

def binary_activation_quant(x: torch.Tensor) -> torch.Tensor:
    """
    Binary activation quantization.
    
    Args:
        x: Activation tensor with shape [n, d] for linear or [n, c, h, w] for conv
        
    Returns:
        Quantized activation tensor with same shape as input
    """
    dim = 1 if x.dim() == 4 else -1
    scale = x.abs().max(dim=dim, keepdim=True).values.clamp_(min=1e-5)
    y = torch.sign(x) * scale
    return y


def binary_weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    Binary weight quantization.
    
    Quantizes weights to {-1, 1} using sign function.
    
    Args:
        w: Weight tensor of any shape
        
    Returns:
        Quantized weight tensor with same shape as input
    """
    scale = w.abs().mean().clamp_(min=1e-5)
    u = torch.sign(w) * scale
    return u


# ============================================================================
# Registry: Maps quant_type -> (activation_quant_fn, weight_quant_fn)
# ============================================================================

QUANTIZERS = {
    "bitnet": (bitnet_activation_quant, bitnet_weight_quant),
    "binary": (binary_activation_quant, binary_weight_quant),
}


def get_quantizers(quant_type: str) -> Tuple[Callable, Callable]:
    """
    Get activation and weight quantization functions for a given type.
    
    Args:
        quant_type: Quantization type identifier (e.g., "bitnet", "binary")
        
    Returns:
        Tuple of (activation_quant_fn, weight_quant_fn)
        
    Raises:
        ValueError: If quant_type is not recognized
        
    Example:
        >>> act_quant, weight_quant = get_quantizers("bitnet")
        >>> quantized_acts = act_quant(activations)
        >>> quantized_weights = weight_quant(weights)
    """
    if quant_type not in QUANTIZERS:
        raise ValueError(
            f"Unknown quant_type: '{quant_type}'. "
            f"Available options: {list(QUANTIZERS.keys())}"
        )
    return QUANTIZERS[quant_type]
