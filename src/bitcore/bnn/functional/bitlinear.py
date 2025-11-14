"""Functional helpers that back the deployment path for binary linear layers."""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..bitquantizer import (
    _parse_quant_type,
    dequantize,
    quantize_act,
    quantize_weight,
)


class _BitLinearFunctional:
    """Deployment helper for `BitLinear` that handles quantized state."""

    def prepare_weights(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
    ) -> Dict[str, torch.Tensor]:
        """
        Produce packed weight buffers suitable for deployment.

        Args:
            weight: Floating-point weight tensor copied from the training module.
            bias: Optional bias tensor from the training module.
            eps: Numerical stabilizer forwarded to the quantizer.
            quant_type: Identifier describing the desired quantization pairing.

        Returns:
            Dictionary containing all buffers needed for deployment:
            - 'qws': Weight scale tensor
            - 'qw': Packed quantized weight tensor
            - 'bias': Bias tensor (may be None or processed)
        """
        _, weight_quant_type = _parse_quant_type(quant_type)
        scale, qtensor = quantize_weight(weight, eps, weight_quant_type)
        
        return {
            'qws': scale,
            'qw': qtensor,
            'bias': bias,  # Could also quantize/process bias here if needed
        }

    def __call__(
        self,
        x: torch.Tensor,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
        **buffers,
    ) -> torch.Tensor:
        """
        Execute the deployment-time linear operator using packed quantized weights.

        Args:
            x: Input activation tensor expected in floating point.
            eps: Numerical stabilizer for activation quantization.
            quant_type: Identifier that selects activation/weight quantization pair.
            **buffers: Dictionary of buffers produced by `prepare_weights`, including:
                - qws: Weight scale tensor
                - qw: Packed quantized weight tensor
                - bias: Optional bias term

        Returns:
            Dequantized output tensor resulting from `F.linear`.
        """
        # Extract required buffers
        qws = buffers['qws']
        qw = buffers['qw']
        bias = buffers.get('bias', None)
        
        # Dequantize weights
        dqweight = dequantize(qws, qw)
        
        # Quantize and dequantize activations
        act_quant_type, _ = _parse_quant_type(quant_type)
        qxs, qx = quantize_act(x, eps, act_quant_type)
        dqx = dequantize(qxs, qx)
        
        return F.linear(dqx, dqweight, bias)


bitlinear = _BitLinearFunctional()