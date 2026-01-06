"""
Examples demonstrating BitLinear with different input shapes.

This example demonstrates:
- 2D inputs (batch, features) - standard case
- 3D inputs (batch, sequence, features) - for NLP/transformer models
- 4D inputs (batch, ..., features) - for various architectures
"""

import torch
from bitcore import BitLinear


def example_2d_input():
    """Example: Standard 2D input (batch, features)."""
    print("=" * 60)
    print("2D Input (batch, features)")
    print("=" * 60)
    
    layer = BitLinear(in_features=64, out_features=32, bias=True)
    
    # 2D input: (batch, features)
    x = torch.randn(8, 64)
    y = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("This is the standard use case for feedforward networks.")
    print()


def example_3d_input():
    """Example: 3D input (batch, sequence, features) for NLP models."""
    print("=" * 60)
    print("3D Input (batch, sequence, features)")
    print("=" * 60)
    
    layer = BitLinear(in_features=64, out_features=32, bias=True)
    
    # 3D input: (batch, sequence_length, features)
    # Common for transformer models, RNNs, etc.
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 64)
    y = layer(x)
    
    print(f"Input shape: {x.shape}  # (batch, seq_len, features)")
    print(f"Output shape: {y.shape}  # (batch, seq_len, out_features)")
    print("This is useful for sequence models like transformers.")
    print()


def example_4d_input():
    """Example: 4D input for various architectures."""
    print("=" * 60)
    print("4D Input (batch, ..., features)")
    print("=" * 60)
    
    layer = BitLinear(in_features=64, out_features=32, bias=True)
    
    # 4D input: (batch, channels, height, features)
    x = torch.randn(2, 3, 5, 64)
    y = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("BitLinear handles arbitrary input dimensions (last dim must match in_features).")
    print()


def example_various_batch_sizes():
    """Example: Different batch sizes with 2D input."""
    print("=" * 60)
    print("Various Batch Sizes")
    print("=" * 60)
    
    layer = BitLinear(in_features=64, out_features=32, bias=True)
    
    batch_sizes = [1, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 64)
        y = layer(x)
        print(f"Batch size {batch_size:2d}: {x.shape} -> {y.shape}")
    print()


def example_transformer_like():
    """Example: Transformer-like usage with 3D input."""
    print("=" * 60)
    print("Transformer-like Usage")
    print("=" * 60)
    
    import torch.nn as nn
    
    # Simulate a transformer feedforward layer
    class TransformerFFN(nn.Module):
        def __init__(self, d_model=128, d_ff=256):
            super().__init__()
            self.w1 = BitLinear(d_model, d_ff, quant_type="bitnet")
            self.w2 = BitLinear(d_ff, d_model, quant_type="bitnet")
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # x shape: (batch, seq_len, d_model)
            x = self.relu(self.w1(x))
            x = self.w2(x)
            return x
    
    model = TransformerFFN(d_model=128, d_ff=256)
    
    # Input: (batch, seq_len, d_model)
    batch_size = 4
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 128)
    
    y = model(x)
    
    print(f"Input shape: {x.shape}  # (batch, seq_len, d_model)")
    print(f"Output shape: {y.shape}  # (batch, seq_len, d_model)")
    print("This demonstrates BitLinear in transformer architectures.")
    print()


def example_sequence_various_lengths():
    """Example: Different sequence lengths with 3D input."""
    print("=" * 60)
    print("Various Sequence Lengths")
    print("=" * 60)
    
    layer = BitLinear(in_features=64, out_features=32, bias=True)
    batch_size = 4
    
    seq_lengths = [5, 10, 20, 50, 100]
    
    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, 64)
        y = layer(x)
        print(f"Seq length {seq_len:3d}: {x.shape} -> {y.shape}")
    print()


def example_mixed_shapes():
    """Example: Processing inputs of different shapes with the same layer."""
    print("=" * 60)
    print("Mixed Input Shapes")
    print("=" * 60)
    
    layer = BitLinear(in_features=64, out_features=32, bias=True)
    
    # Process different input shapes
    inputs = [
        torch.randn(8, 64),           # 2D
        torch.randn(4, 10, 64),       # 3D
        torch.randn(2, 3, 5, 64),     # 4D
    ]
    
    for i, x in enumerate(inputs, 1):
        y = layer(x)
        print(f"Input {i} shape: {x.shape} -> Output shape: {y.shape}")
    print()


if __name__ == "__main__":
    example_2d_input()
    example_3d_input()
    example_4d_input()
    example_various_batch_sizes()
    example_transformer_like()
    example_sequence_various_lengths()
    example_mixed_shapes()

