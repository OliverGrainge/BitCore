"""Functional counterpart used by deployment-ready binary convolution layers."""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ..bitquantizer import (
    _parse_quant_type,
    dequantize,
    quantize_act,
    quantize_weight,
)


class _BitConv2dFunctional:
    """Namespace + callable that mirrors the deployment API used by layers."""

    def prepare_weights(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
    ) -> Dict[str, torch.Tensor]:
        """
        Convert floating-point convolution weights into deployable buffers.

        Args:
            weight: Weight tensor cloned from the training-time convolution.
            bias: Optional bias tensor from the training module.
            eps: Small numerical constant passed to the weight quantizer.
            quant_type: Identifier describing the activation/weight quantization
                flavors expected during inference.

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
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
        **buffers,
    ) -> torch.Tensor:
        """
        Run deployment-time binary convolution leveraging quantized buffers.

        Args:
            x: Floating-point activation tensor to be quantized.
            stride: Convolution stride applied after quantization.
            padding: Padding to apply to the input tensor.
            dilation: Kernel dilation factor for convolution.
            groups: Number of feature groups processed independently.
            eps: Numerical stabilizer for activation quantization.
            quant_type: Selector describing activation/weight quantization pairing.
            **buffers: Dictionary of buffers produced by `prepare_weights`, including:
                - qws: Weight scale tensor
                - qw: Packed quantized weight tensor
                - bias: Optional bias term

        Returns:
            Tensor produced by `F.conv2d` after dequantization.
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
        
        return F.conv2d(dqx, dqweight, bias, stride, padding, dilation, groups)


bitconv2d = _BitConv2dFunctional()