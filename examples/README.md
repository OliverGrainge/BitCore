# BitLinear Examples

This directory contains example scripts demonstrating how to use the `BitLinear` layer from BitCore. Each example focuses on a specific aspect of using BitLinear.

## Example Files

### `basic_usage.py`
Basic usage examples for BitLinear:
- Creating BitLinear layers
- Basic forward pass
- Layer parameters and configuration
- Layers with and without bias

**Run:** `python examples/basic_usage.py`

### `training.py`
Training examples with BitLinear:
- Training mode forward pass
- Gradient computation
- Using BitLinear in training loops
- Loss computation and backpropagation
- Complete training loop example

**Run:** `python examples/training.py`

### `deployment.py`
Deployment examples for BitLinear:
- Switching to deployment mode
- Inference in deployment mode
- Comparison between eval and deploy modes
- Optimized inference path
- Deploying entire models

**Run:** `python examples/deployment.py`

### `convert_from_linear.py`
Examples for converting standard PyTorch Linear layers to BitLinear:
- Converting `nn.Linear` to `BitLinear`
- Preserving weights and biases
- Using different quantizer types during conversion
- Converting entire models

**Run:** `python examples/convert_from_linear.py`

### `different_quantizers.py`
Examples demonstrating different quantizer types:
- Available quantizer types (bitnet, twn, paretoq)
- Differences between quantizers
- Using quantizers in models
- Training and deployment with different quantizers

**Run:** `python examples/different_quantizers.py`

### `input_shapes.py`
Examples demonstrating BitLinear with different input shapes:
- 2D inputs (batch, features) - standard case
- 3D inputs (batch, sequence, features) - for NLP/transformer models
- 4D inputs (batch, ..., features) - for various architectures
- Various batch and sequence lengths

**Run:** `python examples/input_shapes.py`

### `simple_model.py`
Examples of building complete models with BitLinear:
- Building neural networks with BitLinear
- Different architectures (MLP, transformer-like, etc.)
- Combining BitLinear with other PyTorch layers
- Residual blocks, sequential models, mixed precision

**Run:** `python examples/simple_model.py`

## Quick Start

To run all examples, you can execute each file individually:

```bash
# Basic usage
python examples/basic_usage.py

# Training
python examples/training.py

# Deployment
python examples/deployment.py

# And so on...
```

## Requirements

Make sure you have BitCore installed and the required dependencies:

```bash
pip install -e .
```

## Notes

- All examples use dummy/random data for demonstration purposes
- Examples are designed to be self-contained and runnable
- Each example includes detailed comments explaining the concepts
- Examples demonstrate both CPU and GPU-compatible code (GPU examples will work if CUDA is available)

