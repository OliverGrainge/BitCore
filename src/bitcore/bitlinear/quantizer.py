"""Quantization configurations for BitLinear layers."""

import torch
from torch import nn
from typing import Tuple, Callable
import torch.nn.functional as F 
try:
    import bitops
    HAS_BITOPS = True
except ImportError:
    HAS_BITOPS = False


# ============================================================================
# Base Quantizer Class
# ============================================================================

class Quantizer(nn.Module):
    """
    Base class for quantizers that define both training and inference behavior.
    
    A quantizer specifies:
    - How to quantize weights and activations during training
    - How to prepare weights for deployment
    - Which inference function to use (bitops or fallback)
    """
    
    
    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize/dequantize activations and weights during training (with STE).
        
        Args:
            x: Input activations
            w: Input weights

        Returns:
            Tuple of (quantized activations, quantized weights)
        """
        raise NotImplementedError
    
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

class BitNetQuantizer(Quantizer): 
    def __init__(self, out_features: int, in_features: int, use_fallback: bool = False):
        super().__init__(out_features, in_features)
        self.use_fallback = use_fallback

    def _quantize_act(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize activations to int8.
        
        Matches reference implementation exactly:
        Qn = -2^(num_bits-1) = -128
        Qp = 2^(num_bits-1) - 1 = 127
        s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (x * s).round().clamp(Qn, Qp) / s
        
        Returns:
            Tuple of (scale, quantized_activations)
            where scale is per-row and quantized is int8 in [-128, 127]
        """
        # Match reference exactly: Qn = -128, Qp = 127
        Qn = -128
        Qp = 127
        # Match reference: s = Qp / max_val (not max_val / Qp)
        # Use clamp (not clamp_) to match reference behavior
        max_val = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        s = Qp / max_val
        # Match reference: (x * s).round().clamp(Qn, Qp)
        # Use clamp (not clamp_) to match reference behavior
        quantized = (x * s).round().clamp(Qn, Qp)
        # For dequantization, we need scale = 1/s = max_val / Qp
        scale = 1.0 / s  # This is max_val / Qp
        return scale, quantized

    def _quantize_weight(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        Quantize weights to ternary {-1, 0, 1}.
        
        Matches reference implementation exactly:
        s = 1 / weight.abs().mean().clamp(min=1e-5)
        result = (weight * s).round().clamp(-1, 1) / s
        
        Returns:
            Tuple of (scale, quantized_weights) - scale is per-tensor (scalar)
        """
        # Match reference exactly: s = 1 / mean
        # Use clamp (not clamp_) to match reference behavior
        s = 1.0 / w.abs().mean().clamp(min=1e-5)
        # Match reference: (weight * s).round().clamp(-1, 1)
        # Use clamp (not clamp_) to match reference behavior
        quantized = (w * s).round().clamp(-1, 1)
        # For dequantization, we need scale = 1/s = mean
        scale = 1.0 / s
        return scale, quantized
    
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Match reference implementation exactly
        # Reference: quant_input = input + (activation_quant(input, self.input_bits) - input).detach()
        # Reference: quant_weight = self.weight + (weight_quant(self.weight, self.weight_bits) - self.weight).detach()
        
        # Use helper methods to get dequantized values
        x_scale, x_quant = self._quantize_act(x)
        x_dequant = x_quant * x_scale  # This equals (x * s).round().clamp(Qn, Qp) / s
        x_dequant_ste = x + (x_dequant - x).detach()
        
        w_scale, w_quant = self._quantize_weight(w)
        w_dequant = w_quant * w_scale  # This equals (w * s).round().clamp(-1, 1) / s
        w_dequant_ste = w + (w_dequant - w).detach()
        
        return x_dequant_ste, w_dequant_ste
        
    def get_deployment_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use fallback format if explicitly requested or if bitops is not available
        if self.use_fallback or not HAS_BITOPS:
            w_scale, w_quant = self._quantize_weight(weight)
            w_scale = w_scale.reshape(1).to(torch.float32)
            w_quant = w_quant.to(torch.float32)
            return w_scale, w_quant
        else:
            w_scale, w_quant = self._quantize_weight(weight)
            w_quant = w_quant.to(torch.int8)
            # bitops expects w_scale as [N] (per output channel)
            # For per-tensor quantization, broadcast scalar to all channels
            w_scale_vec = w_scale.expand(self.out_features).contiguous().to(torch.float32)
            # Use new bitops packing function
            w_packed = bitops.bitmatmulpack(w_quant)
            return w_scale_vec, w_packed

    def get_inference_fn(self) -> Callable:
        if self.use_fallback or not HAS_BITOPS:
            return self._fallback_inference_fn
        else:
            return self._bitops_inference_fn
    
    def _bitops_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Bitops inference with dynamic activation quantization.
        
        The bitops API requires:
        - x: int8 quantized activations [M, K]
        - x_scale: float32 per-row scales [M]
        - w_packed: packed ternary weights [N, K/4]
        - w_scale: float32 per-channel scales [N]
        - bias: float32 bias [N]
        """
        original_shape = x.shape
        # Handle 3D input (batch, seq_len, features)
        if x.dim() == 3:
            batch, seq_len, features = x.shape
            x = x.reshape(batch * seq_len, features)
        
        M = x.shape[0]
        
        # Quantize activations to int8 with per-row scales
        x_scale, x_quant = self._quantize_act(x)
        x_quant = x_quant.to(torch.int8)
        # x_scale is [M, 1], bitops expects [M]
        x_scale = x_scale.squeeze(-1).to(torch.float32)
        
        # Call bitops with pre-quantized activations
        # Signature: bitmatmul(x, x_scale, w_packed, w_scale, bias)
        y = bitops.bitmatmul(x_quant, x_scale, w_packed, w_scale, bias)
        
        # Restore original shape if needed
        if len(original_shape) == 3:
            y = y.reshape(batch, seq_len, -1)
        
        return y
    
    def _fallback_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_quant: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # Quantize activations using the same method as forward
        x_scale, x_quant = self._quantize_act(x)
        x_dequant = x_quant * x_scale
        # w_scale is [1] for fallback, broadcast works naturally
        w_dequant = w_quant * w_scale
        return F.linear(x_dequant, w_dequant, bias)
    

#======================== TWN QUANTIZER ===================================
#==========================================================================

class TWNQuantizer(Quantizer): 
    def __init__(self, out_features: int, in_features: int, use_fallback: bool = False):
        super().__init__(out_features, in_features)
        self.use_fallback = use_fallback

    def _quantize_act(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize activations to int8.
        
        Returns:
            Tuple of (scale, quantized_activations)
            where scale is per-row and quantized is int8 in [-128, 127]
        """
        # For linear layers, quantize over the last dimension (features)
        max_val = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        scale = max_val / 127.0
        y = (x / scale).round().clamp_(-128, 127)
        return scale, y

    def _quantize_weight(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TWN (Ternary Weight Networks) quantization.
        threshold Δ ≈ 0.75 E[|W|], scale α = mean(|W_i|) over non-zero entries
        
        Returns:
            Tuple of (scale, quantized_weights) - scale is per-tensor (scalar)
        """
        # threshold Δ ≈ 0.75 E[|W|]
        delta = 0.75 * w.abs().mean().clamp(min=1e-5)

        # ternary mask
        mask = w.abs() > delta

        # scaling factor α = mean(|W_i|) over non-zero entries
        if mask.any():
            scale = w[mask].abs().mean()
        if not mask.any():
            scale = w.new_tensor(0.0)
            w_ternary = torch.zeros_like(w)
            return scale, w_ternary

        # ternary weights
        w_ternary = torch.zeros_like(w)
        w_ternary[w > delta] = 1.0
        w_ternary[w < -delta] = -1.0

        return scale, w_ternary

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_scale, x_quant = self._quantize_act(x)
        w_scale, w_quant = self._quantize_weight(w)
        x_dequant = x_quant * x_scale
        w_dequant = w_quant * w_scale
        x_dequant_ste = x + (x_dequant - x).detach()
        w_dequant_ste = w + (w_dequant - w).detach()
        return x_dequant_ste, w_dequant_ste
        
    def get_deployment_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use fallback format if explicitly requested or if bitops is not available
        if self.use_fallback or not HAS_BITOPS:
            w_scale, w_quant = self._quantize_weight(weight)
            w_scale = w_scale.reshape(1).to(torch.float32)
            w_quant = w_quant.to(torch.float32)
            return w_scale, w_quant
        else:
            w_scale, w_quant = self._quantize_weight(weight)
            w_quant = w_quant.to(torch.int8)
            # bitops expects w_scale as [N] (per output channel)
            # For per-tensor quantization, broadcast scalar to all channels
            w_scale_vec = w_scale.expand(self.out_features).contiguous().to(torch.float32)
            # Use new bitops packing function
            w_packed = bitops.bitmatmulpack(w_quant)
            return w_scale_vec, w_packed

    def get_inference_fn(self) -> Callable:
        if self.use_fallback or not HAS_BITOPS:
            return self._fallback_inference_fn
        else:
            return self._bitops_inference_fn
    
    def _bitops_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Bitops inference with dynamic activation quantization.
        
        The bitops API requires:
        - x: int8 quantized activations [M, K]
        - x_scale: float32 per-row scales [M]
        - w_packed: packed ternary weights [N, K/4]
        - w_scale: float32 per-channel scales [N]
        - bias: float32 bias [N]
        """
        original_shape = x.shape
        # Handle 3D input (batch, seq_len, features)
        if x.dim() == 3:
            batch, seq_len, features = x.shape
            x = x.reshape(batch * seq_len, features)
        
        M = x.shape[0]
        
        # Quantize activations to int8 with per-row scales
        x_scale, x_quant = self._quantize_act(x)
        x_quant = x_quant.to(torch.int8)
        # x_scale is [M, 1], bitops expects [M]
        x_scale = x_scale.squeeze(-1).to(torch.float32)
        
        # Call bitops with pre-quantized activations
        y = bitops.bitmatmul(x_quant, x_scale, w_packed, w_scale, bias)
        
        # Restore original shape if needed
        if len(original_shape) == 3:
            y = y.reshape(batch, seq_len, -1)
        
        return y
    
    def _fallback_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_quant: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # Quantize activations using the same method as forward
        x_scale, x_quant = self._quantize_act(x)
        x_dequant = x_quant * x_scale
        # w_scale is [1] for fallback, broadcast works naturally
        w_dequant = w_quant * w_scale
        return F.linear(x_dequant, w_dequant, bias)


#======================== PARETOQ 1.58-BIT (TERNARY) QUANTIZER ==============
#==========================================================================


class ParetoQQuantizer(Quantizer):
    """
    ParetoQ SEQ-style 1.58-bit (ternary) weight quantization:
      W_Q = α * SEQ( Clip(W/α, -1, 1), k=3 )

    Key: α is TRAINABLE (learnable range).

    scale_granularity: Hard-coded to "channel" - one α per output channel/row (more accurate)
    
    Note: This quantizer uses per-channel weight scales, which is natively supported
    by bitops (w_scale shape [N]).
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        eps: float = 1e-5,
        use_fallback: bool = False,
    ):
        super().__init__(out_features, in_features)
        self.eps = eps
        self.use_fallback = use_fallback
        # We store an unconstrained parameter and map -> positive scale via softplus.
        # Shape: [out_features, 1] for per-channel scale (broadcasts over in_features)
        self._logit_alpha = nn.Parameter(torch.zeros(out_features, 1))
        self._alpha_inited = False

    def _alpha(self) -> torch.Tensor:
        # positive, stable
        return F.softplus(self._logit_alpha) + self.eps

    @torch.no_grad()
    def _maybe_init_alpha_from_weight(self, w: torch.Tensor) -> None:
        # Paper mentions initializing clip range (α) for SEQ; common choice is max(|W|).
        if self._alpha_inited:
            return

        # per-channel initialization: max(|W_channel|) for each output channel
        # Linear layer: weight shape is [out_features, in_features]
        # Max over in_features dimension
        init_alpha = w.abs().max(dim=1, keepdim=True).values.clamp(min=self.eps)

        # set softplus(logit)=init_alpha  => logit = softplus^{-1}(a) = log(exp(a)-1)
        self._logit_alpha.copy_(torch.log(torch.expm1(init_alpha)))
        self._alpha_inited = True

    def _quantize_act(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize activations to int8.
        
        Returns:
            Tuple of (scale, quantized_activations)
            where scale is per-row and quantized is int8 in [-128, 127]
        """
        # For linear layers, quantize over the last dimension (features)
        max_val = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=self.eps)
        scale = max_val / 127.0
        y = (x / scale).round().clamp_(-128, 127)
        return scale, y

    def _seq_ternary(self, w_hat: torch.Tensor) -> torch.Tensor:
        # SEQ for k=3 levels over [-1,1] gives symmetric balanced levels.
        # Equivalent simple partition: [-1,-1/3)->-1, [-1/3,1/3]->0, (1/3,1]->+1
        q = torch.zeros_like(w_hat)
        q[w_hat >  (1.0 / 3.0)] =  1.0
        q[w_hat < -(1.0 / 3.0)] = -1.0
        return q

    def _quantize_weight(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          alpha: per-channel scales [out_features, 1]
          q: ternary weights in {-1,0,1}
          ste_mask: mask for STE (|w/alpha| < 1)
        """
        self._maybe_init_alpha_from_weight(w)

        alpha = self._alpha()  # [out_features, 1]
        
        w_hat = (w / alpha).clamp_(-1.0, 1.0)
        q = self._seq_ternary(w_hat)

        # Paper's STE mask idea for these low-bit settings uses the clip region.
        ste_mask = (w_hat.abs() < 1.0).to(w.dtype)
        return alpha, q, ste_mask

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # activations: int8 fake-quant
        x_scale, x_q = self._quantize_act(x)
        x_deq = x_q * x_scale
        x_ste = x + (x_deq - x).detach()

        # weights: SEQ ternary with trainable alpha
        alpha, w_q, ste_mask = self._quantize_weight(w)
        w_deq = w_q * alpha

        # STE (masked): pass grads through within clip range
        w_ste = w + ((w_deq - w) * ste_mask).detach()

        return x_ste, w_ste

    def get_deployment_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha, w_q, _ = self._quantize_weight(weight)

        if self.use_fallback or not HAS_BITOPS:
            # Fallback: keep [out_features, 1] shape for broadcasting in F.linear
            w_scale = alpha.to(torch.float32)
            return w_scale, w_q.to(torch.float32)
        else:
            # bitops: w_scale should be [N] (per output channel)
            # alpha is [out_features, 1], squeeze to [out_features]
            w_scale_vec = alpha.squeeze(-1).contiguous().to(torch.float32)
            w_q_int8 = w_q.to(torch.int8)
            w_packed = bitops.bitmatmulpack(w_q_int8)
            return w_scale_vec, w_packed

    def get_inference_fn(self) -> Callable:
        if self.use_fallback or not HAS_BITOPS:
            return self._fallback_inference_fn
        else:
            return self._bitops_inference_fn

    def _bitops_inference_fn(
        self,
        x: torch.Tensor,
        w_scale: torch.Tensor,
        w_packed: torch.Tensor,
        bias: torch.Tensor
    ) -> torch.Tensor:
        """
        Bitops inference with dynamic activation quantization.
        
        ParetoQ uses per-channel weight scales which bitops natively supports.
        """
        original_shape = x.shape
        # Handle 3D input (batch, seq_len, features)
        if x.dim() == 3:
            batch, seq_len, features = x.shape
            x = x.reshape(batch * seq_len, features)
        
        # Quantize activations to int8 with per-row scales
        x_scale, x_quant = self._quantize_act(x)
        x_quant = x_quant.to(torch.int8)
        # x_scale is [M, 1], bitops expects [M]
        x_scale = x_scale.squeeze(-1).to(torch.float32)
        
        # Call bitops - w_scale is already [N] from get_deployment_weights
        y = bitops.bitmatmul(x_quant, x_scale, w_packed, w_scale, bias)
        
        # Restore original shape if needed
        if len(original_shape) == 3:
            y = y.reshape(batch, seq_len, -1)
        
        return y

    def _fallback_inference_fn(
        self,
        x: torch.Tensor,
        w_scale: torch.Tensor,
        w_quant: torch.Tensor,
        bias: torch.Tensor
    ) -> torch.Tensor:
        x_scale, x_q = self._quantize_act(x)
        x_deq = x_q * x_scale

        # w_scale is [out_features, 1] for fallback, broadcasts correctly
        w_deq = w_quant * w_scale
        return F.linear(x_deq, w_deq, bias)


# ============================================================================
# Registry: Maps quant_type
# ============================================================================

QUANTIZERS = {
    "bitnet": BitNetQuantizer,
    "twn": TWNQuantizer,
    "paretoq": ParetoQQuantizer,
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