"""
Basic usage examples for BitLinear layer.

This example demonstrates:
- Creating BitLinear layers
- Basic forward pass
- Layer parameters and configuration
"""

import torch
from bitcore import BitLinear


def main():
    print("=" * 60)
    print("Basic Usage of BitLinear")
    print("=" * 60)
    print()
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Example 1: Create a BitLinear layer with default settings
    print("Example 1: Basic BitLinear layer")
    print("-" * 60)
    layer = BitLinear(in_features=128, out_features=64, bias=True).to(device)
    
    # Create some input data
    batch_size = 8
    x = torch.randn(batch_size, 128, device=device)
    
    # Forward pass
    y = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Layer parameters: {sum(p.numel() for p in layer.parameters())}")
    print(f"Quantizer type: {layer.quant_type}")
    print()
    
    # Example 2: BitLinear without bias
    print("Example 2: BitLinear without bias")
    print("-" * 60)
    layer_no_bias = BitLinear(in_features=64, out_features=32, bias=False).to(device)
    x2 = torch.randn(4, 64, device=device)
    y2 = layer_no_bias(x2)
    
    print(f"Layer has bias: {layer_no_bias.bias is not None}")
    print(f"Input shape: {x2.shape}")
    print(f"Output shape: {y2.shape}")
    print()
    
    # Example 3: Inspecting layer properties
    print("Example 3: Layer properties")
    print("-" * 60)
    layer = BitLinear(in_features=256, out_features=128, bias=True).to(device)
    print(f"In features: {layer.in_features}")
    print(f"Out features: {layer.out_features}")
    print(f"Has bias: {layer.bias is not None}")
    print(f"Epsilon: {layer.eps}")
    print(f"Quantizer type: {layer.quant_type}")
    print(f"Deployed: {layer._is_deployed}")
    print()


if __name__ == "__main__":
    main()

