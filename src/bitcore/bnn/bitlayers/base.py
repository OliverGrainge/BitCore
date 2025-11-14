"""Abstract base definition for BitLab binary neural network layers."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class BaseBitLayer(nn.Module, ABC):
    """Common interface shared by BitLab layers operating in quantized domains."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass used during training.

        Subclasses should implement a quantization-aware forward pass that
        maintains gradient flow. The default deployment-time version is provided
        by `_deploy_forward`.

        Args:
            x: Input activation tensor in floating point precision.

        Returns:
            Tensor with the same batch dimension representing the layer output.
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_linear(cls, linear: nn.Linear, quant_type: str) -> None:
        """
        Construct a bit-layer from a standard floating-point linear layer.

        Implementations are expected to copy learned parameters and allocate the
        appropriate quantization utilities so the resulting binary layer can be
        used interchangeably with the original `nn.Linear`.

        Args:
            linear: Source PyTorch linear layer to be converted.
            quant_type: Identifier selecting the activation/weight quantization
                pairing that should be used after conversion.

        Returns:
            Instance of `BaseBitLayer` matching the source layer configuration.
        """
        pass

    def _init_weights(self) -> None:
        """
        Initialize the weights of the layer.
        """
        pass

    @abstractmethod
    def _deploy(self) -> None:
        """
        Deploy the layer for efficient inference by:
        1. Quantizing and packing weights
        2. Removing original parameters
        3. Switching to optimized _deploy_forward forward pass
        """
        pass

    def _deploy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the quantized inference pathway after `deploy` has packed the weights.
        """
        pass

    def __repr__(self) -> str:
        """Return a concise string representation summarizing the layer config."""
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, eps={self.eps}, quant_type={self.quant_type})"
    