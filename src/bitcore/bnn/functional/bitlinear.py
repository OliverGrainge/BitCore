"""Functional helpers for binary linear layers with kernel backend support."""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..bitquantizer import (
    _parse_quant_type,
    dequantize,
    quantize_act,
    quantize_weight,
)

# Try to import C++ extension
try:
    from bitcore import _C
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False
    _C = None


class _BitLinearReference:
    """Reference PyTorch implementation."""

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
            'bias': bias,
        }

    def forward(
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


class _BitLinearKernel:
    """C++/CUDA kernel implementation."""
    
    def __init__(self):
        self.available = HAS_CPP_EXTENSION
        if self.available:
            self.ops = _C
    
    def prepare_weights(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
    ) -> Dict[str, torch.Tensor]:
        """Prepare weights using C++/CUDA kernels."""
        if not self.available:
            raise RuntimeError("C++ extension not available")
        
        # C++ function returns (qws, qw, bias)
        qws, qw, bias_out = self.ops.bitlinear_prepare_weights(weight, bias, eps, quant_type)
        
        return {
            'qws': qws,
            'qw': qw,
            'bias': bias_out if (bias_out is not None and bias_out.numel() > 0) else None,
        }

    def forward(
        self,
        x: torch.Tensor,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
        **buffers,
    ) -> torch.Tensor:
        """Execute forward pass using C++/CUDA kernels."""
        if not self.available:
            raise RuntimeError("C++ extension not available")
        
        qws = buffers['qws']
        qw = buffers['qw']
        bias = buffers.get('bias', None)
        
        return self.ops.bitlinear_forward(x, qws, qw, bias, eps, quant_type)


class _BitLinearDispatcher:
    """Dispatcher that selects between reference and kernel implementations."""
    
    def __init__(self):
        self.reference = _BitLinearReference()
        self.kernel = _BitLinearKernel()
        # Auto-select kernel if available, otherwise use reference
        self.backend = 'kernel' if self.kernel.available else 'reference'
    
    def set_backend(self, backend: str):
        """
        Switch between 'reference' and 'kernel' backends.
        
        Args:
            backend: Either 'reference' or 'kernel'
            
        Raises:
            ValueError: If backend is not recognized
            RuntimeError: If kernel backend requested but not available
        """
        if backend not in ('reference', 'kernel'):
            raise ValueError(
                f"Unknown backend: {backend}. Must be 'reference' or 'kernel'"
            )
        if backend == 'kernel' and not self.kernel.available:
            raise RuntimeError(
                "Kernel backend not available. "
                "Rebuild with C++ extensions: pip install -e ."
            )
        self.backend = backend
    
    def get_backend(self) -> str:
        """Get the name of the currently active backend."""
        return self.backend
    
    def is_kernel_available(self) -> bool:
        """Check if the kernel backend is available."""
        return self.kernel.available
    
    def prepare_weights(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare weights using selected backend.
        
        Args:
            weight: Floating-point weight tensor [out_features, in_features]
            bias: Optional bias tensor [out_features]
            eps: Epsilon for numerical stability
            quant_type: Quantization type identifier
            
        Returns:
            Dictionary with keys: 'qws', 'qw', 'bias'
        """
        impl = self.kernel if self.backend == 'kernel' else self.reference
        return impl.prepare_weights(weight, bias, eps, quant_type)
    
    def __call__(
        self,
        x: torch.Tensor,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
        **buffers,
    ) -> torch.Tensor:
        """
        Execute forward pass using selected backend.
        
        Args:
            x: Input tensor
            eps: Epsilon for numerical stability
            quant_type: Quantization type identifier
            **buffers: Dictionary containing 'qws', 'qw', and optional 'bias'
            
        Returns:
            Output tensor
        """
        impl = self.kernel if self.backend == 'kernel' else self.reference
        return impl.forward(x, eps, quant_type, **buffers)


# Global instance - this is what users import and use
bitlinear = _BitLinearDispatcher()