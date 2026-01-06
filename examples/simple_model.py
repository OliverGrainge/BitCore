"""
Examples of building complete models with BitLinear layers.

This example demonstrates:
- Building neural networks with BitLinear
- Different architectures (MLP, transformer-like, etc.)
- Combining BitLinear with other PyTorch layers
"""

import torch
import torch.nn as nn
from bitcore import BitLinear


def example_simple_mlp():
    """Example: Simple Multi-Layer Perceptron with BitLinear."""
    print("=" * 60)
    print("Simple MLP")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=128, hidden_dims=[64, 32], output_dim=10):
            super().__init__()
            layers = []
            
            # Input layer
            layers.append(BitLinear(input_dim, hidden_dims[0], quant_type="bitnet"))
            layers.append(nn.LayerNorm(hidden_dims[0]))
            layers.append(nn.ReLU())
            
            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                layers.append(BitLinear(hidden_dims[i], hidden_dims[i+1], quant_type="bitnet"))
                layers.append(nn.LayerNorm(hidden_dims[i+1]))
                layers.append(nn.ReLU())
            
            # Output layer
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    model = SimpleMLP(input_dim=128, hidden_dims=[64, 32], output_dim=10).to(device)
    x = torch.randn(16, 128, device=device)
    y = model(x)
    
    print(f"Model architecture:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    print()


def example_classifier():
    """Example: Image classifier-like model."""
    print("=" * 60)
    print("Classifier Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class BitClassifier(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            # Simulate flattened image features (e.g., from CNN)
            self.fc1 = BitLinear(784, 256, quant_type="bitnet")  # 28x28 = 784
            self.fc2 = BitLinear(256, 128, quant_type="bitnet")
            self.fc3 = BitLinear(128, num_classes, quant_type="bitnet")
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    model = BitClassifier(num_classes=10).to(device)
    x = torch.randn(32, 784, device=device)  # Batch of flattened images
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Number of classes: 10")
    print()


def example_transformer_ffn():
    """Example: Transformer Feed-Forward Network."""
    print("=" * 60)
    print("Transformer Feed-Forward Network")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class TransformerFFN(nn.Module):
        def __init__(self, d_model=128, d_ff=256):
            super().__init__()
            self.w1 = BitLinear(d_model, d_ff, quant_type="bitnet")
            self.w2 = BitLinear(d_ff, d_model, quant_type="bitnet")
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # x: (batch, seq_len, d_model)
            return self.w2(self.relu(self.w1(x)))
    
    model = TransformerFFN(d_model=128, d_ff=256).to(device)
    x = torch.randn(4, 20, 128, device=device)  # (batch, seq_len, d_model)
    y = model(x)
    
    print(f"Input shape: {x.shape}  # (batch, seq_len, d_model)")
    print(f"Output shape: {y.shape}  # (batch, seq_len, d_model)")
    print("This is the feedforward component of transformer blocks.")
    print()


def example_residual_block():
    """Example: Residual block with BitLinear."""
    print("=" * 60)
    print("Residual Block")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class ResidualBlock(nn.Module):
        def __init__(self, dim=128):
            super().__init__()
            self.fc1 = BitLinear(dim, dim, quant_type="bitnet")
            self.fc2 = BitLinear(dim, dim, quant_type="bitnet")
            self.relu = nn.ReLU()
        
        def forward(self, x):
            residual = x
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return self.relu(x + residual)  # Residual connection
    
    model = ResidualBlock(dim=128).to(device)
    x = torch.randn(8, 128, device=device)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Residual connections help with gradient flow in deep networks.")
    print()


def example_sequential_model():
    """Example: Using nn.Sequential with BitLinear."""
    print("=" * 60)
    print("Sequential Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = nn.Sequential(
        BitLinear(128, 64, quant_type="bitnet"),
        nn.ReLU(),
        nn.Dropout(0.1),
        BitLinear(64, 32, quant_type="bitnet"),
        nn.ReLU(),
        nn.Dropout(0.1),
        BitLinear(32, 10, quant_type="bitnet"),
    ).to(device)
    
    x = torch.randn(16, 128, device=device)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Number of layers: {len(model)}")
    print()


def example_mixed_precision_model():
    """Example: Model mixing BitLinear and standard Linear layers."""
    print("=" * 60)
    print("Mixed Precision Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Use standard Linear for first layer
            self.fc1 = nn.Linear(128, 64)
            # Use BitLinear for subsequent layers
            self.fc2 = BitLinear(64, 32, quant_type="bitnet")
            self.fc3 = BitLinear(32, 10, quant_type="bitnet")
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = MixedModel().to(device)
    x = torch.randn(16, 128, device=device)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("You can mix BitLinear with standard PyTorch layers.")
    print()


if __name__ == "__main__":
    example_simple_mlp()
    example_classifier()
    example_transformer_ffn()
    example_residual_block()
    example_sequential_model()
    example_mixed_precision_model()

