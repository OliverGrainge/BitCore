"""Quantization configurations for BitLinear layers - CORRECTED VERSION

CRITICAL FIX FOR BITNET QUANTIZER:
To match CustomAutoBitLinear (online_quant=False) behavior:
- get_deployment_weights() returns ORIGINAL weights (not quantized)
- _fallback_inference_fn() uses these original weights
- This gives: F.linear(quantized_activations, original_weights, bias)
"""

from calendar import c
import torch
from torch import nn
from typing import Tuple, Callable
import torch.nn.functional as F 
from abc import ABC, abstractmethod
import torch
try:
    import bitops
    HAS_BITOPS = True
except ImportError:
    HAS_BITOPS = False

HAS_BITOPS = False 



# ============================================================================
# Base Quantizer Class
# ============================================================================

class Quantizer(nn.Module, ABC):
    """
    Base class for quantizers that define both training and inference behavior.
    
    A quantizer specifies:
    - How to quantize weights and activations during training
    - How to prepare weights for deployment
    - Which inference function to use (bitops or fallback)
    """
    
    
    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.out_features = w.shape[0]
        self.in_features = w.shape[1]

    @abstractmethod
    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize/dequantize activations and weights during training (with STE).
        
        Args:
            x: Input activations
            w: Input weights

        Returns:
            Tuple of (quantized activations, quantized weights)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_deployment_weights(
        self, 
        weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get deployment-ready scales and quantized weights.
        
        Returns:
            Tuple of (scales, quantized_weights) for packing
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_inference_fn(self) -> Callable:
        """
        Get the inference function to use for this quantization scheme.
        
        Returns either a bitops kernel or a fallback PyTorch implementation.
        
        Returns:
            Function with signature: fn(x, w_scale, w_packed, bias) -> output
        """
        raise NotImplementedError


#======================== BITNET QUANTIZER ================================
#==========================================================================

class _STEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quant_fn):
        # quant_fn is a Python callable that returns the dequantized tensor
        # Forward must return quantized value directly (no x + (.. - x) trick)
        return quant_fn(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradient through unchanged
        return grad_output, None



class BitNetQuantizer(Quantizer):  # Replace with: class BitNetQuantizer(Quantizer):
    def __init__(self, w: torch.Tensor, use_fallback: bool = False):
        super().__init__(w)
        self.use_fallback = use_fallback

    @torch.compile
    def _activation_quant_dequant(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize activations to 8-bit range [-128, 127]."""
        dtype = x.dtype
        x = x.float()
        max_vals = x.abs().max(dim=-1, keepdim=True)[0].clamp_(min=1e-5)
        inv_scale = 127.0 / max_vals
        x = (x * inv_scale).round().clamp_(-128, 127) / inv_scale
        return x.to(dtype)

    @torch.compile
    def _weight_quant_dequant(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize weights to ternary values {-1, 0, 1}."""
        dtype = w.dtype
        w = w.float()
        inv_scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        w = (w * inv_scale).round().clamp_(-1, 1) / inv_scale
        return w.to(dtype)

    @torch.compile
    def _activation_quant(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize activations to 8-bit integers for bitops.
        Returns: (scale, quantized_tensor)
        """
        x = x.float()
        max_vals = x.abs().max(dim=-1, keepdim=False)[0].clamp_(min=1e-5)
        scale = max_vals / 127.0  # [batch] or [batch * seq_len]
        x_quant = (x / scale.unsqueeze(-1)).round().clamp_(-128, 127)
        return scale.contiguous().to(torch.float32), x_quant.contiguous().to(torch.int8)

    @torch.compile
    def _weight_quant(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights to ternary values for bitops.
        Returns: (scale, quantized_tensor)
        """
        w = w.float()
        scale = w.abs().mean().clamp_(min=1e-5)
        w_quant = (w / scale).round().clamp_(-1, 1)
        return scale.contiguous().to(torch.float32), w_quant.contiguous().to(torch.int8)
    
    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training/eval path: Apply quantization with straight-through estimator.
        Returns: (quantized_dequantized_activations, quantized_dequantized_weights)
        """
        x_ste = _STEQuant.apply(x, self._activation_quant_dequant)
        w_ste = _STEQuant.apply(w, self._weight_quant_dequant)
        return x_ste, w_ste
        
    def get_deployment_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare weights for deployment.
        
        For fallback mode:
            - Pre-applies weight quantization once
            - Returns dummy scale and quantized-dequantized weights
        
        For bitops mode:
            - Quantizes weights to int8
            - Packs them using bitops.bitmatmulpack
            - Returns scale vector and packed weights
        """
        # Import here to avoid circular dependency issues
        # Replace these lines with your actual imports:
        
        if self.use_fallback or not HAS_BITOPS:
            # Fallback mode: pre-apply weight quantization
            w_scale = torch.ones((1,), dtype=weight.dtype, device=weight.device)
            weight_dequant = self._weight_quant_dequant(weight)
            return w_scale, weight_dequant
        else:
            # Bitops mode: quantize and pack weights
            w_scale, w_quant = self._weight_quant(weight)
            w_scale_vec = w_scale.expand(self.out_features).contiguous().to(torch.float32)
            w_packed = bitops.bitmatmulpack(w_quant)
            return w_scale_vec, w_packed

    def get_inference_fn(self) -> Callable:
        """
        Return the appropriate inference function based on whether bitops is available.
        """
        if self.use_fallback or not HAS_BITOPS:
            return self._fallback_inference_fn
        else:
            return self._bitops_inference_fn
    
    def _bitops_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Bitops inference with dynamic activation quantization.
        
        This matches the training-time quantization behavior using per-token
        quantization (max per row).
        
        Args:
            x: Input activations [batch, seq_len, in_features] or [batch, in_features]
            w_scale: Weight scale vector [out_features]
            w_packed: Packed quantized weights
            bias: Bias term [out_features]
        
        Returns:
            Output tensor with same shape prefix as input
        """
        # Import here to avoid circular dependency issues
        dtype = x.dtype
        x_scale, x_quant = self._activation_quant(x)
        y = bitops.bitmatmul(x_quant, x_scale, w_packed, w_scale, bias)
        return y.type(dtype)
    
    def _fallback_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Fallback inference when bitops is not available.
        
        Note: w_packed already contains the quantized-dequantized weights from
        get_deployment_weights, so we only need to quantize-dequantize activations.
        
        Args:
            x: Input activations [batch, seq_len, in_features] or [batch, in_features]
            w_scale: Dummy scale (not used in fallback)
            w_packed: Pre-quantized-dequantized weights [out_features, in_features]
            bias: Bias term [out_features]
        
        Returns:
            Output tensor with same shape prefix as input
        """
        
        dtype = x.dtype
        # w_packed already contains quantized-dequantized weights from get_deployment_weights
        x_dequant = self._activation_quant_dequant(x)
        output = F.linear(x_dequant, w_packed, bias)
        return output.type(dtype)
    


class BitNetChannelQuantizer(Quantizer):  # Replace with: class BitNetQuantizer(Quantizer):
    def __init__(self, w: torch.Tensor, use_fallback: bool = False):
        super().__init__(w) 
        self.use_fallback = use_fallback

    @torch.compile
    def _activation_quant_dequant(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize activations to 8-bit range [-128, 127]."""
        dtype = x.dtype
        x = x.float()
        max_vals = x.abs().max(dim=-1, keepdim=True)[0].clamp_(min=1e-5)
        inv_scale = 127.0 / max_vals
        x = (x * inv_scale).round().clamp_(-128, 127) / inv_scale
        return x.to(dtype)

    @torch.compile
    def _weight_quant_dequant(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize weights to ternary values {-1, 0, 1}."""
        dtype = w.dtype
        w = w.float()
        inv_scale = 1.0 / w.abs().mean(dim=1, keepdim=True).clamp_(min=1e-5)
        w = (w * inv_scale).round().clamp_(-1, 1) / inv_scale
        return w.to(dtype)

    @torch.compile
    def _activation_quant(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize activations to 8-bit integers for bitops.
        Returns: (scale, quantized_tensor)
        """
        x = x.float()
        max_vals = x.abs().max(dim=-1, keepdim=False)[0].clamp_(min=1e-5)
        scale = max_vals / 127.0  # [batch] or [batch * seq_len]
        x_quant = (x / scale.unsqueeze(-1)).round().clamp_(-128, 127)
        return scale.contiguous().to(torch.float32), x_quant.contiguous().to(torch.int8)

    @torch.compile
    def _weight_quant(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights to ternary values for bitops.
        Returns: (scale, quantized_tensor)
        """
        w = w.float()
        scale = w.abs().mean(dim=1, keepdim=True).clamp_(min=1e-5)
        w_quant = (w / scale).round().clamp_(-1, 1)
        return scale.contiguous().to(torch.float32), w_quant.contiguous().to(torch.int8)
    
    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training/eval path: Apply quantization with straight-through estimator.
        Returns: (quantized_dequantized_activations, quantized_dequantized_weights)
        """
        x_ste = _STEQuant.apply(x, self._activation_quant_dequant)
        w_ste = _STEQuant.apply(w, self._weight_quant_dequant)
        return x_ste, w_ste
        
    def get_deployment_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare weights for deployment.
        
        For fallback mode:
            - Pre-applies weight quantization once
            - Returns dummy scale and quantized-dequantized weights
        
        For bitops mode:
            - Quantizes weights to int8
            - Packs them using bitops.bitmatmulpack
            - Returns scale vector and packed weights
        """
        # Import here to avoid circular dependency issues
        # Replace these lines with your actual imports:
        
        if self.use_fallback or not HAS_BITOPS:
            # Fallback mode: pre-apply weight quantization
            w_scale = torch.ones((self.out_features,), dtype=weight.dtype, device=weight.device)
            weight_dequant = self._weight_quant_dequant(weight)
            return w_scale, weight_dequant
        else:
            # Bitops mode: quantize and pack weights
            w_scale, w_quant = self._weight_quant(weight)
            w_scale = w_scale.contiguous().to(torch.float32)
            w_packed = bitops.bitmatmulpack(w_quant)
            return w_scale, w_packed

    def get_inference_fn(self) -> Callable:
        """
        Return the appropriate inference function based on whether bitops is available.
        """
        if self.use_fallback or not HAS_BITOPS:
            return self._fallback_inference_fn
        else:
            return self._bitops_inference_fn
    
    def _bitops_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Bitops inference with dynamic activation quantization.
        
        This matches the training-time quantization behavior using per-token
        quantization (max per row).
        
        Args:
            x: Input activations [batch, seq_len, in_features] or [batch, in_features]
            w_scale: Weight scale vector [out_features]
            w_packed: Packed quantized weights
            bias: Bias term [out_features]
        
        Returns:
            Output tensor with same shape prefix as input
        """
        # Import here to avoid circular dependency issues
        dtype = x.dtype
        x_scale, x_quant = self._activation_quant(x)
        y = bitops.bitmatmul(x_quant, x_scale, w_packed, w_scale, bias)
        return y.type(dtype)
    
    def _fallback_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Fallback inference when bitops is not available.
        
        Note: w_packed already contains the quantized-dequantized weights from
        get_deployment_weights, so we only need to quantize-dequantize activations.
        
        Args:
            x: Input activations [batch, seq_len, in_features] or [batch, in_features]
            w_scale: Dummy scale (not used in fallback)
            w_packed: Pre-quantized-dequantized weights [out_features, in_features]
            bias: Bias term [out_features]
        
        Returns:
            Output tensor with same shape prefix as input
        """
        
        dtype = x.dtype
        # w_packed already contains quantized-dequantized weights from get_deployment_weights
        x_dequant = self._activation_quant_dequant(x)
        output = F.linear(x_dequant, w_packed, bias)
        return output.type(dtype)
    





class LSQWeightTernary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, w_scale):
        dtype = w.dtype
        w_f = w.float()
        w_scale_f = w_scale.float()
        
        # Ensure w_scale has shape [output_channel, 1] for broadcasting
        if w_scale_f.dim() == 1:
            w_scale_f = w_scale_f.unsqueeze(1)  # [out_ch] -> [out_ch, 1]

        qmin, qmax = -1, 1

        z = w_f / w_scale_f                    # w_scaled [out_ch, in_ch]
        q = z.round().clamp(qmin, qmax)        # w_clipped [out_ch, in_ch]
        w_hat = q * w_scale_f                  # dequant [out_ch, in_ch]

        ctx.save_for_backward(z, q)
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.N = w_f.shape[1]  # Number of elements per output channel (input_channel)
        ctx.original_scale_shape = w_scale.shape  # Save original shape

        return w_hat.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        z, q = ctx.saved_tensors
        qmax = ctx.qmax
        N = ctx.N
        original_scale_shape = ctx.original_scale_shape

        go = grad_output.float()

        # d w_hat / d w (STE + clipping mask)
        mask = (z >= ctx.qmin) & (z <= ctx.qmax)
        gw = go * mask

        # d w_hat / d s  ≈ q - z   (LSQ)
        # Compute per-channel gradient
        g_scale = 1.0 / ((N * qmax) ** 0.5)
        gs = (go * (q - z)).sum(dim=1, keepdim=True) * g_scale  # [out_ch, 1]
        
        # Reshape to match original w_scale shape
        if len(original_scale_shape) == 1:
            gs = gs.squeeze(1)  # [out_ch, 1] -> [out_ch]

        # Cast back
        return gw.to(grad_output.dtype), gs.to(grad_output.dtype)

class BitNetLSQQuantizer(Quantizer):
    """
    BitNet quantizer with Learned Step Size Quantization (LSQ) for weights.
    
    Uses learnable scale parameter for weight quantization with LSQ gradients,
    while activations use standard STE quantization.
    """
    
    def __init__(self, w: torch.Tensor, use_fallback: bool = False):
        super().__init__(w)
        self.use_fallback = use_fallback
        
        # Learnable weight scale parameter
        init_scale = w.abs().mean().clamp_(min=1e-5)
        self.w_scale = nn.Parameter(torch.tensor(init_scale), requires_grad=True)

    @torch.compile
    def _activation_quant_dequant(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize activations to 8-bit range [-128, 127]."""
        dtype = x.dtype
        x = x.float()
        max_vals = x.abs().max(dim=-1, keepdim=True)[0].clamp_(min=1e-5)
        inv_scale = 127.0 / max_vals
        x = (x * inv_scale).round().clamp_(-128, 127) / inv_scale
        return x.to(dtype)

    @torch.compile
    def _weight_quant_dequant(self, w: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize weights to ternary values {-1, 0, 1}."""
        dtype = w.dtype
        w = w.float()
        w_scale_f = w_scale.float()
        inv_scale = 1.0 / w_scale_f
        w = (w * inv_scale).round().clamp_(-1, 1) / inv_scale
        return w.to(dtype)

    @torch.compile
    def _activation_quant(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize activations to 8-bit integers for bitops.
        Returns: (scale, quantized_tensor)
        """
        x = x.float()
        max_vals = x.abs().max(dim=-1, keepdim=False)[0].clamp_(min=1e-5)
        scale = max_vals / 127.0
        x_quant = (x / scale.unsqueeze(-1)).round().clamp_(-128, 127)
        return scale.contiguous().to(torch.float32), x_quant.contiguous().to(torch.int8)

    @torch.compile
    def _weight_quant(self, w: torch.Tensor, w_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights to ternary values for bitops.
        Returns: (scale, quantized_tensor)
        """
        w = w.float()
        w_scale_f = w_scale.float()
        w_quant = (w / w_scale_f).round().clamp_(-1, 1)
        return w_scale_f.contiguous().to(torch.float32), w_quant.contiguous().to(torch.int8)
    
    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training/eval path: Apply quantization with straight-through estimator.
        Returns: (quantized_dequantized_activations, quantized_dequantized_weights)
        """
        x_ste = _STEQuant.apply(x, self._activation_quant_dequant)
        w_ste = LSQWeightTernary.apply(w, self.w_scale)
        return x_ste, w_ste

    def get_deployment_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare weights for deployment.
        
        For fallback mode:
            - Pre-applies weight quantization once
            - Returns dummy scale and quantized-dequantized weights
        
        For bitops mode:
            - Quantizes weights to int8
            - Packs them using bitops.bitmatmulpack
            - Returns scale vector and packed weights
        """
        if self.use_fallback or not HAS_BITOPS:
            # Fallback mode: pre-apply weight quantization
            w_scale = torch.ones((1,), dtype=weight.dtype, device=weight.device)
            weight_dequant = self._weight_quant_dequant(weight, self.w_scale.data)
            return w_scale, weight_dequant
        else:
            # Bitops mode: quantize and pack weights
            w_scale, w_quant = self._weight_quant(weight, self.w_scale.data)
            w_scale_vec = w_scale.expand(self.out_features).contiguous().to(torch.float32)
            w_packed = bitops.bitmatmulpack(w_quant)
            return w_scale_vec, w_packed

    def get_inference_fn(self) -> Callable:
        """
        Return the appropriate inference function based on whether bitops is available.
        """
        if self.use_fallback or not HAS_BITOPS:
            return self._fallback_inference_fn
        else:
            return self._bitops_inference_fn
    
    def _bitops_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Bitops inference with dynamic activation quantization.
        
        This matches the training-time quantization behavior using per-token
        quantization (max per row).
        
        Args:
            x: Input activations [batch, seq_len, in_features] or [batch, in_features]
            w_scale: Weight scale vector [out_features]
            w_packed: Packed quantized weights
            bias: Bias term [out_features]
        
        Returns:
            Output tensor with same shape prefix as input
        """
        dtype = x.dtype
        x_scale, x_quant = self._activation_quant(x)
        y = bitops.bitmatmul(x_quant, x_scale, w_packed, w_scale, bias)
        return y.type(dtype)
    
    def _fallback_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Fallback inference when bitops is not available.
        
        Note: w_packed already contains the quantized-dequantized weights from
        get_deployment_weights, so we only need to quantize-dequantize activations.
        
        Args:
            x: Input activations [batch, seq_len, in_features] or [batch, in_features]
            w_scale: Dummy scale (not used in fallback)
            w_packed: Pre-quantized-dequantized weights [out_features, in_features]
            bias: Bias term [out_features]
        
        Returns:
            Output tensor with same shape prefix as input
        """
        dtype = x.dtype
        x_dequant = self._activation_quant_dequant(x)
        output = F.linear(x_dequant, w_packed, bias)
        return output.type(dtype)








class BitNetLSQChannelQuantizer(Quantizer):
    """
    BitNet quantizer with Learned Step Size Quantization (LSQ) for weights.
    
    Uses learnable scale parameter for weight quantization with LSQ gradients,
    while activations use standard STE quantization.
    """
    
    def __init__(self, w: torch.Tensor, use_fallback: bool = False):
        super().__init__(w)
        self.use_fallback = use_fallback
        
        # Learnable weight scale parameter
        init_scale = w.abs().mean(dim=1, keepdim=False).clamp_(min=1e-5)
        self.w_scale = nn.Parameter(torch.tensor(init_scale), requires_grad=True)

    @torch.compile
    def _activation_quant_dequant(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize activations to 8-bit range [-128, 127]."""
        dtype = x.dtype
        x = x.float()
        max_vals = x.abs().max(dim=-1, keepdim=True)[0].clamp_(min=1e-5)
        inv_scale = 127.0 / max_vals
        x = (x * inv_scale).round().clamp_(-128, 127) / inv_scale
        return x.to(dtype)

    @torch.compile
    def _weight_quant_dequant(self, w: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize weights to ternary values {-1, 0, 1}."""
        dtype = w.dtype
        w = w.float()
        w_scale_f = w_scale.float().unsqueeze(1)
        inv_scale = 1.0 / w_scale_f
        w = (w * inv_scale).round().clamp_(-1, 1) / inv_scale
        return w.to(dtype)

    @torch.compile
    def _activation_quant(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize activations to 8-bit integers for bitops.
        Returns: (scale, quantized_tensor)
        """
        x = x.float()
        max_vals = x.abs().max(dim=-1, keepdim=False)[0].clamp_(min=1e-5)
        scale = max_vals / 127.0
        x_quant = (x / scale.unsqueeze(-1)).round().clamp_(-128, 127)
        return scale.contiguous().to(torch.float32), x_quant.contiguous().to(torch.int8)

    @torch.compile
    def _weight_quant(self, w: torch.Tensor, w_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights to ternary values for bitops.
        Returns: (scale, quantized_tensor)
        """
        w = w.float()
        w_scale_f = w_scale.float().unsqueeze(1)
        w_quant = (w / w_scale_f).round().clamp_(-1, 1)
        return w_scale_f.contiguous().to(torch.float32), w_quant.contiguous().to(torch.int8)
    
    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training/eval path: Apply quantization with straight-through estimator.
        Returns: (quantized_dequantized_activations, quantized_dequantized_weights)
        """
        x_ste = _STEQuant.apply(x, self._activation_quant_dequant)
        w_ste = LSQWeightTernary.apply(w, self.w_scale)
        return x_ste, w_ste

    def get_deployment_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare weights for deployment.
        
        For fallback mode:
            - Pre-applies weight quantization once
            - Returns dummy scale and quantized-dequantized weights
        
        For bitops mode:
            - Quantizes weights to int8
            - Packs them using bitops.bitmatmulpack
            - Returns scale vector and packed weights
        """
        if self.use_fallback or not HAS_BITOPS:
            # Fallback mode: pre-apply weight quantization
            w_scale = torch.ones((self.out_features,), dtype=weight.dtype, device=weight.device)
            weight_dequant = self._weight_quant_dequant(weight, self.w_scale.data)
            return w_scale, weight_dequant
        else:
            # Bitops mode: quantize and pack weights
            w_scale, w_quant = self._weight_quant(weight, self.w_scale.data)
            w_scale = w_scale.contiguous().to(torch.float32)
            w_packed = bitops.bitmatmulpack(w_quant)
            return w_scale, w_packed

    def get_inference_fn(self) -> Callable:
        """
        Return the appropriate inference function based on whether bitops is available.
        """
        if self.use_fallback or not HAS_BITOPS:
            return self._fallback_inference_fn
        else:
            return self._bitops_inference_fn
    
    def _bitops_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Bitops inference with dynamic activation quantization.
        
        This matches the training-time quantization behavior using per-token
        quantization (max per row).
        
        Args:
            x: Input activations [batch, seq_len, in_features] or [batch, in_features]
            w_scale: Weight scale vector [out_features]
            w_packed: Packed quantized weights
            bias: Bias term [out_features]
        
        Returns:
            Output tensor with same shape prefix as input
        """
        dtype = x.dtype
        x_scale, x_quant = self._activation_quant(x)
        y = bitops.bitmatmul(x_quant, x_scale, w_packed, w_scale, bias)
        return y.type(dtype)
    
    def _fallback_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Fallback inference when bitops is not available.
        
        Note: w_packed already contains the quantized-dequantized weights from
        get_deployment_weights, so we only need to quantize-dequantize activations.
        
        Args:
            x: Input activations [batch, seq_len, in_features] or [batch, in_features]
            w_scale: Dummy scale (not used in fallback)
            w_packed: Pre-quantized-dequantized weights [out_features, in_features]
            bias: Bias term [out_features]
        
        Returns:
            Output tensor with same shape prefix as input
        """
        dtype = x.dtype
        x_dequant = self._activation_quant_dequant(x)
        output = F.linear(x_dequant, w_packed, bias)
        return output.type(dtype)







# ============================================================================
# Registry: Maps quant_type
# ============================================================================

QUANTIZERS = {
    "bitnet": BitNetQuantizer,
    "bitnet_channel": BitNetChannelQuantizer,
    "bitnet_lsq": BitNetLSQQuantizer,
    "bitnet_lsq_channel": BitNetLSQChannelQuantizer
}


def get_quantizers(quant_type: str) -> Callable:
    """
    Get quantizer class for a given type.
    
    Args:
        quant_type: Quantization type identifier (e.g., "bitnet", "twn", "paretoq")
    
    Returns:
        Quantizer class
    """
    if quant_type not in QUANTIZERS:
        raise ValueError(
            f"Unknown quant_type: '{quant_type}'. "
            f"Available options: {list(QUANTIZERS.keys())}"
        )
    return QUANTIZERS[quant_type]