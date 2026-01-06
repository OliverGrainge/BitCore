# BitCore

BitCore provides quantization-aware binary linear layers (`BitLinear`) that can seamlessly swap into deployment mode using the accompanying `bitops` extension for efficient inference.

## Features

- **Drop-in Replacement**: `BitLinear` is a drop-in replacement for `nn.Linear` with binary quantization
- **Training & Deployment Modes**: Seamlessly switch between training (gradient-aware) and deployment (optimized) modes
- **Multiple Quantizers**: Support for BitNet, TWN (Ternary Weight Networks), and ParetoQ quantization schemes
- **BitOps Integration**: Optional integration with `bitops` for accelerated inference in deployment mode
- **PyTorch Native**: Built on PyTorch with full compatibility with existing models and training pipelines

## Installation

Install BitCore in editable mode during development:

```bash
git clone https://github.com/olivergrainge/BitCore.git
cd BitCore
pip install -e .
```

For production use:

```bash
pip install git+https://github.com/olivergrainge/BitCore.git
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- (Optional) `bitops` for deployment mode acceleration

## Quick Start

```python
import torch
from bitcore import BitLinear

# Create a binary linear layer
layer = BitLinear(in_features=128, out_features=64, quant_type="bitnet")

# Use in training mode (default)
x = torch.randn(32, 128)
y = layer(x)  # Forward pass with quantization-aware gradients

# Switch to deployment mode for optimized inference
layer.deploy()
y_deployed = layer(x)  # Uses optimized inference path
```

## Available Quantizers

BitCore supports multiple quantization schemes:

- **`bitnet`**: BitNet quantization (default)
- **`twn`**: Ternary Weight Networks (weights: -1, 0, 1)
- **`paretoq`**: ParetoQ quantization scheme

```python
from bitcore import BitLinear, QUANTIZERS

# List available quantizers
print(QUANTIZERS.keys())  # dict_keys(['bitnet', 'twn', 'paretoq'])

# Use different quantizers
layer1 = BitLinear(128, 64, quant_type="bitnet")
layer2 = BitLinear(128, 64, quant_type="twn")
layer3 = BitLinear(128, 64, quant_type="paretoq")
```

## Usage

### Basic Usage

```python
import torch
import torch.nn as nn
from bitcore import BitLinear

# Create a simple model with BitLinear
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BitLinear(784, 256, quant_type="bitnet")
        self.fc2 = BitLinear(256, 128, quant_type="bitnet")
        self.fc3 = BitLinear(128, 10, quant_type="bitnet")
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleModel()
```

### Converting from Standard Linear Layers

```python
from bitcore import BitLinear
import torch.nn as nn

# Convert a standard Linear layer to BitLinear
linear = nn.Linear(128, 64)
bitlinear = BitLinear.from_linear(linear, quant_type="bitnet")

# Convert an entire model
def convert_to_bitlinear(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, BitLinear.from_linear(module))
        else:
            convert_to_bitlinear(module)
    return model
```

### Training

BitLinear layers work seamlessly with standard PyTorch training loops:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from bitcore import BitLinear

model = nn.Sequential(
    BitLinear(784, 256, quant_type="bitnet"),
    nn.ReLU(),
    BitLinear(256, 10, quant_type="bitnet"),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
model.train()
for batch in dataloader:
    x, y = batch
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
```

### Deployment

Switch to deployment mode for optimized inference:

```python
# After training, switch to deployment mode
model.eval()
for module in model.modules():
    if isinstance(module, BitLinear):
        module.deploy()

# Now inference uses optimized path (bitops if available)
with torch.no_grad():
    output = model(x)
```

## Examples

The `examples/` directory contains comprehensive examples:

- **`basic_usage.py`**: Basic usage and layer configuration
- **`training.py`**: Training examples with gradients and backpropagation
- **`deployment.py`**: Deployment mode and performance comparisons
- **`convert_from_linear.py`**: Converting standard Linear layers to BitLinear
- **`different_quantizers.py`**: Comparing different quantization schemes
- **`input_shapes.py`**: Handling various input shapes (2D, 3D, 4D)
- **`simple_model.py`**: Building complete models with BitLinear

Run examples:

```bash
python examples/basic_usage.py
python examples/training.py
python examples/deployment.py
```

See [examples/README.md](examples/README.md) for detailed documentation.

## API Reference

### BitLinear

```python
BitLinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    eps: float = 1e-6,
    quant_type: str = "bitnet"
)
```

**Parameters:**
- `in_features`: Number of input features
- `out_features`: Number of output features
- `bias`: Whether to include a bias term (default: `True`)
- `eps`: Small epsilon for numerical stability (default: `1e-6`)
- `quant_type`: Quantization type - `"bitnet"`, `"twn"`, or `"paretoq"` (default: `"bitnet"`)

**Methods:**
- `forward(x)`: Forward pass with quantization-aware gradients
- `deploy()`: Switch to deployment mode for optimized inference
- `from_linear(linear_layer, quant_type="bitnet")`: Class method to convert `nn.Linear` to `BitLinear`

## Related Projects

- **[BitOps](https://github.com/OliverGrainge/BitOps)**: Low-level bitwise operations for accelerated inference

## License

MIT License - see LICENSE file for details

## Author

Oliver Grainge

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

