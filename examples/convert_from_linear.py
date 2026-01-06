"""
Examples for converting standard PyTorch Linear layers to BitLinear.

This example demonstrates:
- Converting nn.Linear to BitLinear
- Preserving weights and biases
- Using different quantizer types during conversion
"""

import torch
import torch.nn as nn
from bitcore import BitLinear


def example_basic_conversion():
    """Example: Basic conversion from Linear to BitLinear."""
    print("=" * 60)
    print("Basic Conversion")
    print("=" * 60)
    
    # Create a standard PyTorch Linear layer
    linear = nn.Linear(in_features=128, out_features=64, bias=True)
    
    # Initialize with some weights (for demonstration)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)
    
    print("Original Linear layer:")
    print(f"  Weight shape: {linear.weight.shape}")
    print(f"  Bias shape: {linear.bias.shape}")
    print(f"  Weight mean: {linear.weight.mean().item():.4f}")
    print(f"  Weight std: {linear.weight.std().item():.4f}")
    
    # Convert to BitLinear
    bit_linear = BitLinear.from_linear(linear, quant_type="bitnet")
    
    print("\nConverted BitLinear layer:")
    print(f"  Weight shape: {bit_linear.weight.shape}")
    print(f"  Bias shape: {bit_linear.bias.shape}")
    print(f"  Quantizer type: {bit_linear.quant_type}")
    
    # Verify weights and bias are preserved
    weights_match = torch.allclose(bit_linear.weight, linear.weight, atol=1e-6)
    bias_match = torch.allclose(bit_linear.bias, linear.bias, atol=1e-6)
    
    print(f"\n  Weights match: {weights_match}")
    print(f"  Bias matches: {bias_match}")
    print()


def example_conversion_with_different_quantizers():
    """Example: Convert with different quantizer types."""
    print("=" * 60)
    print("Conversion with Different Quantizers")
    print("=" * 60)
    
    # Create a Linear layer
    linear = nn.Linear(in_features=64, out_features=32, bias=True)
    nn.init.xavier_uniform_(linear.weight)
    
    # Convert with different quantizers
    quantizers = ["bitnet", "twn", "paretoq"]
    
    for quant_type in quantizers:
        bit_linear = BitLinear.from_linear(linear, quant_type=quant_type)
        print(f"Quantizer '{quant_type}':")
        print(f"  Weight shape: {bit_linear.weight.shape}")
        print(f"  Quantizer type: {bit_linear.quant_type}")
        print(f"  Weights match: {torch.allclose(bit_linear.weight, linear.weight)}")
        print()


def example_conversion_no_bias():
    """Example: Convert Linear layer without bias."""
    print("=" * 60)
    print("Conversion Without Bias")
    print("=" * 60)
    
    # Create Linear layer without bias
    linear = nn.Linear(in_features=64, out_features=32, bias=False)
    nn.init.xavier_uniform_(linear.weight)
    
    print("Original Linear layer:")
    print(f"  Has bias: {linear.bias is not None}")
    print(f"  Weight shape: {linear.weight.shape}")
    
    # Convert to BitLinear
    bit_linear = BitLinear.from_linear(linear, quant_type="bitnet")
    
    print("\nConverted BitLinear layer:")
    print(f"  Has bias: {bit_linear.bias is not None}")
    print(f"  Weight shape: {bit_linear.weight.shape}")
    print(f"  Weights match: {torch.allclose(bit_linear.weight, linear.weight)}")
    print()


def example_convert_model():
    """Example: Convert an entire model's Linear layers to BitLinear."""
    print("=" * 60)
    print("Converting Model Layers")
    print("=" * 60)
    
    # Create a model with standard Linear layers
    class StandardNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Create standard model
    standard_model = StandardNet()
    x = torch.randn(8, 128)
    
    # Get output from standard model
    with torch.no_grad():
        y_standard = standard_model(x)
    
    print("Standard model:")
    print(f"  Output shape: {y_standard.shape}")
    print(f"  Linear layers: 3")
    
    # Convert to BitLinear model
    class BitNet(nn.Module):
        def __init__(self, standard_model):
            super().__init__()
            self.fc1 = BitLinear.from_linear(standard_model.fc1, quant_type="bitnet")
            self.fc2 = BitLinear.from_linear(standard_model.fc2, quant_type="bitnet")
            self.fc3 = BitLinear.from_linear(standard_model.fc3, quant_type="bitnet")
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    bit_model = BitNet(standard_model)
    
    with torch.no_grad():
        y_bit = bit_model(x)
    
    print("\nBitLinear model:")
    print(f"  Output shape: {y_bit.shape}")
    print(f"  BitLinear layers: 3")
    
    # Note: outputs will differ due to quantization
    max_diff = (y_standard - y_bit).abs().max().item()
    print(f"\nMax output difference: {max_diff:.4f}")
    print("(Expected difference due to quantization)")
    print()


def example_conversion_preserves_gradients():
    """Example: Verify conversion preserves gradient computation."""
    print("=" * 60)
    print("Conversion Preserves Gradients")
    print("=" * 60)
    
    # Create and train a Linear layer (simulated)
    linear = nn.Linear(64, 32, bias=True)
    x = torch.randn(4, 64, requires_grad=True)
    
    # Forward pass with Linear
    y_linear = linear(x)
    loss_linear = y_linear.sum()
    loss_linear.backward()
    
    print("Original Linear layer:")
    print(f"  Input grad computed: {x.grad is not None}")
    print(f"  Weight grad computed: {linear.weight.grad is not None}")
    
    # Convert to BitLinear
    bit_linear = BitLinear.from_linear(linear, quant_type="bitnet")
    
    # Reset gradients
    x.grad = None
    bit_linear.zero_grad()
    
    # Forward pass with BitLinear
    y_bit = bit_linear(x)
    loss_bit = y_bit.sum()
    loss_bit.backward()
    
    print("\nConverted BitLinear layer:")
    print(f"  Input grad computed: {x.grad is not None}")
    print(f"  Weight grad computed: {bit_linear.weight.grad is not None}")
    print(f"  Bias grad computed: {bit_linear.bias.grad is not None}")
    print()


if __name__ == "__main__":
    example_basic_conversion()
    example_conversion_with_different_quantizers()
    example_conversion_no_bias()
    example_convert_model()
    example_conversion_preserves_gradients()

