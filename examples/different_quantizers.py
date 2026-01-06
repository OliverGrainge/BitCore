"""
Examples demonstrating different quantizer types available in BitLinear.

This example demonstrates:
- Available quantizer types (bitnet, twn, paretoq)
- Differences between quantizers
- Using quantizers in models
"""

import torch
from bitcore import BitLinear, QUANTIZERS


def example_list_quantizers():
    """Example: List all available quantizers."""
    print("=" * 60)
    print("Available Quantizers")
    print("=" * 60)
    
    print(f"Available quantizer types: {list(QUANTIZERS.keys())}")
    print(f"Total quantizers: {len(QUANTIZERS)}")
    print()
    
    for quant_type in QUANTIZERS.keys():
        print(f"  - {quant_type}")
    print()


def example_compare_quantizers():
    """Example: Compare different quantizers on the same input."""
    print("=" * 60)
    print("Comparing Quantizers")
    print("=" * 60)
    
    # Create same input for all quantizers
    x = torch.randn(8, 64)
    
    print("Input shape:", x.shape)
    print()
    
    # Test each quantizer
    for quant_type in QUANTIZERS.keys():
        layer = BitLinear(in_features=64, out_features=32, quant_type=quant_type)
        y = layer(x)
        
        print(f"Quantizer '{quant_type}':")
        print(f"  Output shape: {y.shape}")
        print(f"  Output mean: {y.mean().item():.4f}")
        print(f"  Output std: {y.std().item():.4f}")
        print()


def example_quantizer_in_model():
    """Example: Use different quantizers in different layers."""
    print("=" * 60)
    print("Mixed Quantizers in Model")
    print("=" * 60)
    
    import torch.nn as nn
    
    class MixedQuantizerNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Use different quantizers for different layers
            self.layer1 = BitLinear(128, 64, quant_type="bitnet")
            self.layer2 = BitLinear(64, 32, quant_type="twn")
            self.layer3 = BitLinear(32, 10, quant_type="paretoq")
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.layer3(x)
            return x
    
    model = MixedQuantizerNet()
    x = torch.randn(8, 128)
    y = model(x)
    
    print("Model with mixed quantizers:")
    print(f"  Layer 1 quantizer: {model.layer1.quant_type}")
    print(f"  Layer 2 quantizer: {model.layer2.quant_type}")
    print(f"  Layer 3 quantizer: {model.layer3.quant_type}")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print()


def example_quantizer_default():
    """Example: Default quantizer type."""
    print("=" * 60)
    print("Default Quantizer")
    print("=" * 60)
    
    # Create layer without specifying quant_type
    layer = BitLinear(in_features=64, out_features=32)
    
    print(f"Default quantizer type: {layer.quant_type}")
    print("(If not specified, defaults to 'bitnet')")
    print()


def example_quantizer_training():
    """Example: Training with different quantizers."""
    print("=" * 60)
    print("Training with Different Quantizers")
    print("=" * 60)
    
    import torch.nn as nn
    
    x = torch.randn(8, 64, requires_grad=True)
    target = torch.randn(8, 32)
    
    for quant_type in QUANTIZERS.keys():
        layer = BitLinear(in_features=64, out_features=32, quant_type=quant_type)
        layer.train()
        
        y = layer(x)
        loss = nn.MSELoss()(y, target)
        loss.backward()
        
        print(f"Quantizer '{quant_type}':")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradients computed: {layer.weight.grad is not None}")
        print()


def example_quantizer_deployment():
    """Example: Deployment with different quantizers."""
    print("=" * 60)
    print("Deployment with Different Quantizers")
    print("=" * 60)
    
    x = torch.randn(8, 64)
    
    for quant_type in QUANTIZERS.keys():
        layer = BitLinear(in_features=64, out_features=32, quant_type=quant_type)
        
        # Eval mode (before deployment)
        layer.eval()
        with torch.no_grad():
            y_eval = layer(x)
        
        # Deploy mode (after deployment)
        layer._deploy()
        with torch.no_grad():
            y_deploy = layer(x)
        
        # Calculate max difference between eval and deploy outputs
        max_diff = (y_eval - y_deploy).abs().max().item()
        mean_diff = (y_eval - y_deploy).abs().mean().item()
        
        print(f"Quantizer '{quant_type}':")
        print(f"  Eval output shape: {y_eval.shape}")
        print(f"  Deploy output shape: {y_deploy.shape}")
        print(f"  Max difference (eval vs deploy): {max_diff:.6f}")
        print(f"  Mean difference (eval vs deploy): {mean_diff:.6f}")
        print()


if __name__ == "__main__":
    example_list_quantizers()
    example_compare_quantizers()
    example_quantizer_in_model()
    example_quantizer_default()
    example_quantizer_training()
    example_quantizer_deployment()

