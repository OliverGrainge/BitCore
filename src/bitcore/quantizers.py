"""Quantization configurations for BitCore layers."""

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
    def __init__(self, out_features: int, in_features: int):
        super().__init__(out_features, in_features)

    def _quantize_act(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if x.dim() == 4 else -1  # channels for conv, features for linear
        inv_scale = 127.0 / x.abs().max(dim=dim, keepdim=True).values.clamp_(min=1e-5)
        y = (x * inv_scale).round().clamp_(-128, 127)
        return inv_scale, y

    def _quantize_weight(self, w: torch.Tensor) -> torch.Tensor: 
        inv_scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * inv_scale).round().clamp_(-1, 1) 
        return inv_scale, u
    
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_inv_scale, x_quant = self._quantize_act(x)
        w_inv_scale, w_quant = self._quantize_weight(w)
        x_dequant = x_quant / x_inv_scale
        w_dequant = w_quant / w_inv_scale
        x_dequant_ste = x + (x_dequant - x).detach()
        w_dequant_ste = w + (w_dequant - w).detach()
        return x_dequant_ste, w_dequant_ste
        
    def get_deployment_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if HAS_BITOPS: 
            w_inv_scale, w_quant = self._quantize_weight(weight)
            # Convert inv_scale to actual scale for deployment (bitops expects scale, not inv_scale)
            w_scale = (1.0 / w_inv_scale).reshape(1).to(torch.float32)
            w_quant = w_quant.to(torch.int8)
            w_quant_flat = w_quant.flatten() 
            w_packed_flat = bitops.pack_t2s(w_quant_flat)
            packed_per_row = (self.in_features + 3) // 4  # Ceiling division
            w_packed = w_packed_flat.reshape(self.out_features, packed_per_row)
            return w_scale, w_packed
        else:
            w_inv_scale, w_quant = self._quantize_weight(weight)
            # Convert inv_scale to actual scale for fallback
            w_scale = (1.0 / w_inv_scale).reshape(1).to(torch.float32)
            w_quant = w_quant.to(torch.float32)
            return w_scale, w_quant

    def get_inference_fn(self) -> Callable:
        if HAS_BITOPS:
            return bitops.matmul_f32_i8spr_t2spt
        else:
            return self._fallback_inference_fn

    def _fallback_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_quant: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # w_scale here is the actual scale (from get_deployment_weights)
        # Quantize activations
        x_inv_scale, x_quant = self._quantize_act(x)
        x_dequant = x_quant / x_inv_scale
        w_dequant = w_quant * w_scale
        return F.linear(x_dequant, w_dequant, bias)
    

#======================== TWN QUANTIZER ===================================
#==========================================================================

class TWNQuantizer(Quantizer): 
    def __init__(self, out_features: int, in_features: int):
        super().__init__(out_features, in_features)

    def _quantize_act(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if x.dim() == 4 else -1  # channels for conv, features for linear
        max_val = x.abs().max(dim=dim, keepdim=True).values.clamp_(min=1e-5)
        scale = max_val / 127.0
        y = (x / scale).round().clamp_(-128, 127)
        return scale, y

    def _quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
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

        # scaled ternary weights
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
        if HAS_BITOPS: 
            w_scale, w_quant = self._quantize_weight(weight)
            w_quant = w_quant.to(torch.int8)
            # Ensure w_scale is at least 1D (bitops expects [1] shape for per-tensor scale)
            w_scale = w_scale.reshape(1).to(torch.float32)
            w_quant_flat = w_quant.flatten() 
            w_packed_flat = bitops.pack_t2s(w_quant_flat)
            packed_per_row = (self.in_features + 3) // 4  # Ceiling division
            w_packed = w_packed_flat.reshape(self.out_features, packed_per_row)
            return w_scale, w_packed
        else:
            w_scale, w_quant = self._quantize_weight(weight)
            # Ensure w_scale is at least 1D for consistency
            w_scale = w_scale.reshape(1).to(torch.float32)
            w_quant = w_quant.to(torch.float32)
            return w_scale, w_quant

    def get_inference_fn(self) -> Callable:
        if HAS_BITOPS:
            return bitops.matmul_f32_i8spr_t2spt
        else:
            return self._fallback_inference_fn

    def _fallback_inference_fn(self, x: torch.Tensor, w_scale: torch.Tensor, w_quant: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # w_scale here is the actual scale (from get_deployment_weights)
        # TWNQuantizer returns actual scale, so multiply to dequantize
        x_scale, x_quant = self._quantize_act(x)
        x_dequant = x_quant * x_scale
        w_dequant = w_quant * w_scale
        return F.linear(x_dequant, w_dequant, bias)


#======================== PARETOQ 1.58-BIT (TERNARY) QUANTIZER ==============
#==========================================================================


class ParetoQQuantizer(Quantizer):
    """
    ParetoQ SEQ-style 1.58-bit (ternary) weight quantization:
      W_Q = α * SEQ( Clip(W/α, -1, 1), k=3 )    [oai_citation:3‡pareto1.pdf](sediment://file_00000000190471f8a645681155922abe)

    Key: α is TRAINABLE (learnable range).  [oai_citation:4‡pareto1.pdf](sediment://file_00000000190471f8a645681155922abe)

    scale_granularity: Hard-coded to "channel" - one α per output channel/row (more accurate)
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        eps: float = 1e-5,
    ):
        super().__init__(out_features, in_features)
        self.eps = eps
        # We store an unconstrained parameter and map -> positive scale via softplus.
        # Shape: [out_features, 1] for per-channel scale (broadcasts over in_features)
        self._logit_alpha = nn.Parameter(torch.zeros(out_features, 1))
        self._alpha_inited = False

    def _alpha(self) -> torch.Tensor:
        # positive, stable
        return F.softplus(self._logit_alpha) + self.eps

    @torch.no_grad()
    def _maybe_init_alpha_from_weight(self, w: torch.Tensor) -> None:
        # Paper mentions initializing clip range (α) for SEQ; common choice is max(|W|).  [oai_citation:5‡pareto1.pdf](sediment://file_00000000190471f8a645681155922abe)
        if self._alpha_inited:
            return

        # per-channel initialization: max(|W_channel|) for each output channel
        if w.dim() == 4:
            # Conv layer: weight shape is [out_channels, in_channels, H, W]
            # Max over [in_channels, H, W] dimensions
            init_alpha = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=self.eps).view(-1, 1)
        else:
            # Linear layer: weight shape is [out_features, in_features]
            # Max over in_features dimension
            init_alpha = w.abs().max(dim=1, keepdim=True).values.clamp(min=self.eps)

        # set softplus(logit)=init_alpha  => logit = softplus^{-1}(a) = log(exp(a)-1)
        self._logit_alpha.copy_(torch.log(torch.expm1(init_alpha)))
        self._alpha_inited = True

    def _quantize_act(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = 1 if x.dim() == 4 else -1
        inv_scale = 127.0 / x.abs().max(dim=dim, keepdim=True).values.clamp_(min=self.eps)
        y = (x * inv_scale).round().clamp_(-128, 127)
        return inv_scale, y

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
          alpha (broadcastable),
          q in {-1,0,1},
          mask for STE (|w/alpha| < 1)
        """
        self._maybe_init_alpha_from_weight(w)

        alpha = self._alpha()  # [out_features, 1]
        
        # For conv layers (4D), reshape alpha to [out_features, 1, 1, 1] for proper broadcasting
        # For linear layers (2D), keep as [out_features, 1]
        if w.dim() == 4:
            alpha = alpha.view(-1, 1, 1, 1)
        
        w_hat = (w / alpha).clamp_(-1.0, 1.0)
        q = self._seq_ternary(w_hat)

        # Paper's STE mask idea for these low-bit settings uses the clip region.  [oai_citation:6‡pareto1.pdf](sediment://file_00000000190471f8a645681155922abe)
        ste_mask = (w_hat.abs() < 1.0).to(w.dtype)
        return alpha, q, ste_mask

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # activations: keep your int8 fake-quant
        x_inv_scale, x_q = self._quantize_act(x)
        x_deq = x_q / x_inv_scale
        x_ste = x + (x_deq - x).detach()

        # weights: SEQ ternary with trainable alpha
        alpha, w_q, ste_mask = self._quantize_weight(w)
        w_deq = w_q * alpha

        # STE (masked): pass grads through within clip range
        w_ste = w + ((w_deq - w) * ste_mask).detach()

        return x_ste, w_ste

    def get_deployment_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha, w_q, _ = self._quantize_weight(weight)

        # Per-channel quantization - fallback only (bitops doesn't support per-channel scales)
        # w_scale shape: [out_features, 1] for per-channel scaling
        w_scale = alpha.to(torch.float32)
        return w_scale, w_q.to(torch.float32)

    def get_inference_fn(self) -> Callable:
        # Per-channel quantization requires fallback (bitops doesn't support per-channel scales)
        return self._fallback_inference_fn

    def _fallback_inference_fn(
        self,
        x: torch.Tensor,
        w_scale: torch.Tensor,
        w_quant: torch.Tensor,
        bias: torch.Tensor
    ) -> torch.Tensor:
        x_inv_scale, x_q = self._quantize_act(x)
        x_deq = x_q / x_inv_scale

        # w_scale is [out_features, 1] for per-channel scaling
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


def get_quantizers(quant_type: str) -> Tuple[Callable, Callable]:
    """
    Get activation and weight quantization functions for a given type.
    """
    if quant_type not in QUANTIZERS:
        raise ValueError(
            f"Unknown quant_type: '{quant_type}'. "
            f"Available options: {list(QUANTIZERS.keys())}"
        )
    return QUANTIZERS[quant_type]
