"""
Training examples with BitLinear layers.

This example demonstrates:
- Training mode forward pass
- Gradient computation
- Using BitLinear in training loops
- Loss computation and backpropagation
- Training on MNIST dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from bitcore import BitLinear


def example_single_layer_training():
    """Example: Training a single BitLinear layer."""
    print("=" * 60)
    print("Single Layer Training")
    print("=" * 60)
    
    # Create layer and input
    layer = BitLinear(in_features=128, out_features=64, bias=True)
    x = torch.randn(8, 128, requires_grad=True)
    
    # Forward pass in training mode
    layer.train()
    y = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output requires_grad: {y.requires_grad}")
    
    # Compute loss and backward
    target = torch.randn(8, 64)
    loss = nn.MSELoss()(y, target)
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Input gradient computed: {x.grad is not None}")
    print(f"Weight gradient computed: {layer.weight.grad is not None}")
    print(f"Bias gradient computed: {layer.bias.grad is not None}")
    print()


def example_model_training():
    """Example: Training a model with multiple BitLinear layers."""
    print("=" * 60)
    print("Model Training")
    print("=" * 60)
    
    class SimpleBitNet(nn.Module):
        def __init__(self, input_dim=128, hidden_dim=64, output_dim=10):
            super().__init__()
            self.layer1 = BitLinear(input_dim, hidden_dim, quant_type="bitnet")
            self.layer2 = BitLinear(hidden_dim, hidden_dim, quant_type="bitnet")
            self.layer3 = BitLinear(hidden_dim, output_dim, quant_type="bitnet")
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.layer3(x)
            return x
    
    # Create model
    model = SimpleBitNet(input_dim=128, hidden_dim=64, output_dim=10)
    model.train()
    
    # Create dummy data
    batch_size = 16
    x = torch.randn(batch_size, 128)
    target = torch.randint(0, 10, (batch_size,))
    
    # Forward pass
    output = model(x)
    
    print(f"Model input shape: {x.shape}")
    print(f"Model output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print("Gradients computed for all layers!")
    print()


def example_mnist_training():
    """Example: Training an MLP on MNIST with BitLinear."""
    print("=" * 60)
    print("MNIST Training with BitLinear")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5
    input_dim = 28 * 28  # MNIST images are 28x28
    hidden_dims = [256, 128]
    output_dim = 10
    
    # Define model (same architecture as simple_model.py)
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10, quant_type="bitnet"):
            super().__init__()
            layers = []
            
            # Input layer
            layers.append(BitLinear(input_dim, hidden_dims[0], quant_type=quant_type))
            layers.append(nn.LayerNorm(hidden_dims[0]))
            layers.append(nn.ReLU())
            
            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                layers.append(BitLinear(hidden_dims[i], hidden_dims[i+1], quant_type=quant_type))
                layers.append(nn.LayerNorm(hidden_dims[i+1]))
                layers.append(nn.ReLU())
            
            # Output layer
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            # Flatten input
            x = x.view(x.size(0), -1)
            return self.network(x)
    
    # Create model
    model = SimpleMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Data loading
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate training accuracy
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        # Calculate test accuracy
        test_acc = 100. * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{num_epochs}] Summary: \n')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print()
    
    print("Training completed!")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print()


def example_gradient_flow():
    """Example: Verify gradient flow through quantization."""
    print("=" * 60)
    print("Gradient Flow Verification")
    print("=" * 60)
    
    layer = BitLinear(in_features=64, out_features=32, quant_type="bitnet")
    x = torch.randn(4, 64, requires_grad=True)
    
    # Forward pass
    y = layer(x)
    loss = y.sum()
    
    # Backward pass
    loss.backward()
    
    print(f"Input gradient shape: {x.grad.shape if x.grad is not None else None}")
    print(f"Weight gradient shape: {layer.weight.grad.shape if layer.weight.grad is not None else None}")
    print(f"Bias gradient shape: {layer.bias.grad.shape if layer.bias.grad is not None else None}")
    print("Gradients flow successfully through quantization!")
    print()


if __name__ == "__main__":
    # Run basic examples
    example_single_layer_training()
    example_model_training()
    example_gradient_flow()
    
    # Run MNIST training (this will download the dataset on first run)
    print("\n" + "=" * 60)
    print("Running MNIST Training Example")
    print("=" * 60 + "\n")
    example_mnist_training()

