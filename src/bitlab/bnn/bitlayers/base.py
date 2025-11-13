from abc import ABC, abstractmethod
import torch 
import torch.nn as nn 

class BaseBitLayer(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @classmethod
    @abstractmethod
    def from_linear(cls, linear: nn.Linear, quant_type: str) -> None:
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
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, eps={self.eps}, quant_type={self.quant_type})"
    