import pytest
import torch
from torch import nn

from bitcore import BitLinear, QUANTIZERS

# Get all available quantizer types
ALL_QUANTIZERS = list(QUANTIZERS.keys())


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_forward_cpu(quant_type):
    torch.manual_seed(0)
    layer = BitLinear(in_features=16, out_features=32, quant_type=quant_type)
    x = torch.randn(4, 16, requires_grad=True)

    y = layer(x)

    assert y.shape == (4, 32)
    y.sum().backward()
    assert layer.weight.grad is not None
    if layer.bias is not None:
        assert layer.bias.grad is not None


def test_bitlinear_from_linear_clones_weights():
    torch.manual_seed(1)
    linear = nn.Linear(16, 16, bias=True)
    bit_layer = BitLinear.from_linear(linear)

    assert torch.allclose(bit_layer.weight, linear.weight)
    assert bit_layer.bias is not None
    assert torch.allclose(bit_layer.bias, linear.bias)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_deploy_forward_cpu(quant_type):
    torch.manual_seed(2)
    layer = BitLinear(in_features=16, out_features=16, bias=False, quant_type=quant_type)
    x = torch.randn(8, 16)

    layer.deploy()
    assert layer._is_deployed is True

    y = layer(x)
    assert y.shape == (8, 16)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_eval_vs_deploy_cpu_equivalence(quant_type):
    torch.manual_seed(4)
    layer = BitLinear(in_features=16, out_features=16, bias=True, quant_type=quant_type)
    layer.eval()
    x = torch.randn(4, 16)

    with torch.no_grad():
        y_eval = layer(x)

    layer.deploy()

    with torch.no_grad():
        y_deploy = layer(x)

    max_diff = (y_eval - y_deploy).abs().max().item()
    assert y_eval.shape == y_deploy.shape == (4, 16)
    assert max_diff < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitlinear_forward_gpu():
    torch.manual_seed(3)
    device = torch.device("cuda")
    layer = BitLinear(in_features=16, out_features=32, quant_type=quant_type).to(device)
    x = torch.randn(4, 16, device=device, requires_grad=True)

    y = layer(x)

    assert y.device.type == "cuda"
    assert y.shape == (4, 32)
    y.sum().backward()
    assert layer.weight.grad is not None
    if layer.bias is not None:
        assert layer.bias.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitlinear_eval_vs_deploy_gpu_equivalence():
    torch.manual_seed(5)
    device = torch.device("cuda")
    layer = BitLinear(in_features=16, out_features=16, bias=True, quant_type=quant_type).to(device)
    layer.eval()
    x = torch.randn(4, 16, device=device)

    with torch.no_grad():
        y_eval = layer(x)

    layer.deploy()

    with torch.no_grad():
        y_deploy = layer(x)

    max_diff = (y_eval - y_deploy).abs().max().item()
    assert y_eval.shape == y_deploy.shape == (4, 16)
    assert max_diff < 0.5


# Additional comprehensive tests with dimensions divisible by 4


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_various_dimensions_cpu(quant_type):
    """Test various dimension combinations (all divisible by 4)."""
    test_configs = [
        (4, 8),
        (8, 4),
        (12, 16),
        (32, 64),
        (64, 32),
        (128, 256),
    ]
    
    for in_features, out_features in test_configs:
        torch.manual_seed(42)
        layer = BitLinear(in_features=in_features, out_features=out_features, quant_type=quant_type)
        x = torch.randn(2, in_features)
        
        y = layer(x)
        assert y.shape == (2, out_features), f"Failed for {in_features}x{out_features}"


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_batch_dimensions_cpu(quant_type):
    """Test different batch dimensions."""
    torch.manual_seed(10)
    layer = BitLinear(in_features=16, out_features=32, quant_type=quant_type)
    
    # Test various batch sizes
    for batch_size in [1, 4, 8, 16, 32]:
        x = torch.randn(batch_size, 16)
        y = layer(x)
        assert y.shape == (batch_size, 32)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_3d_input_cpu(quant_type):
    """Test 3D input (batch, sequence, features)."""
    torch.manual_seed(11)
    layer = BitLinear(in_features=16, out_features=32, quant_type=quant_type)
    
    x = torch.randn(4, 10, 16)  # (batch, seq_len, features)
    y = layer(x)
    
    assert y.shape == (4, 10, 32)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_4d_input_cpu(quant_type):
    """Test 4D input."""
    torch.manual_seed(12)
    layer = BitLinear(in_features=16, out_features=32, quant_type=quant_type)
    
    x = torch.randn(2, 3, 4, 16)  # Multiple dimensions
    y = layer(x)
    
    assert y.shape == (2, 3, 4, 32)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_no_bias_cpu(quant_type):
    """Test BitLinear without bias."""
    torch.manual_seed(13)
    layer = BitLinear(in_features=16, out_features=32, bias=False, quant_type=quant_type)
    
    assert layer.bias is None
    
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 32)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_gradient_flow_cpu(quant_type):
    """Test that gradients flow properly through quantization."""
    torch.manual_seed(14)
    layer = BitLinear(in_features=16, out_features=32, quant_type=quant_type)
    x = torch.randn(4, 16, requires_grad=True)
    
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_training_vs_eval_mode_cpu(quant_type):
    """Test that training and eval modes produce different behaviors."""
    torch.manual_seed(15)
    layer = BitLinear(in_features=16, out_features=16, bias=True, quant_type=quant_type)
    x = torch.randn(4, 16)
    
    # Training mode
    layer.train()
    y_train = layer(x)
    
    # Eval mode
    layer.eval()
    with torch.no_grad():
        y_eval = layer(x)
    
    # Shapes should match
    assert y_train.shape == y_eval.shape == (4, 16)
    # Outputs should be similar but may differ slightly
    max_diff = (y_train - y_eval).abs().max().item()
    assert max_diff < 1.0  # Allow reasonable difference


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_multiple_forward_passes_cpu(quant_type):
    """Test multiple forward passes with same layer."""
    torch.manual_seed(16)
    layer = BitLinear(in_features=16, out_features=32, quant_type=quant_type)
    
    for i in range(5):
        x = torch.randn(4, 16)
        y = layer(x)
        assert y.shape == (4, 32)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_eval_vs_deploy_various_dims_cpu(quant_type):
    """Test eval vs deploy equivalence for various dimensions."""
    test_configs = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
    ]
    
    for in_features, out_features in test_configs:
        torch.manual_seed(20)
        layer = BitLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True
        , quant_type=quant_type)
        layer.eval()
        x = torch.randn(4, in_features)
        
        with torch.no_grad():
            y_eval = layer(x)
        
        layer.deploy()
        
        with torch.no_grad():
            y_deploy = layer(x)
        
        max_diff = (y_eval - y_deploy).abs().max().item()
        assert y_eval.shape == y_deploy.shape == (4, out_features)
        assert max_diff < 0.5, f"Failed for {in_features}x{out_features}: diff={max_diff}"


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_eval_vs_deploy_no_bias_cpu(quant_type):
    """Test eval vs deploy equivalence without bias."""
    torch.manual_seed(21)
    layer = BitLinear(in_features=16, out_features=32, bias=False, quant_type=quant_type)
    layer.eval()
    x = torch.randn(8, 16)
    
    with torch.no_grad():
        y_eval = layer(x)
    
    layer.deploy()
    
    with torch.no_grad():
        y_deploy = layer(x)
    
    max_diff = (y_eval - y_deploy).abs().max().item()
    assert y_eval.shape == y_deploy.shape == (8, 32)
    assert max_diff < 0.5


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_eval_vs_deploy_3d_input_cpu(quant_type):
    """Test eval vs deploy equivalence with 3D input."""
    torch.manual_seed(22)
    layer = BitLinear(in_features=16, out_features=32, bias=True, quant_type=quant_type)
    layer.eval()
    x = torch.randn(4, 5, 16)  # (batch, seq_len, features)
    
    with torch.no_grad():
        y_eval = layer(x)
    
    layer.deploy()
    
    with torch.no_grad():
        y_deploy = layer(x)
    
    max_diff = (y_eval - y_deploy).abs().max().item()
    assert y_eval.shape == y_deploy.shape == (4, 5, 32)
    assert max_diff < 0.5


def test_bitlinear_from_linear_no_bias():
    """Test from_linear with no bias."""
    torch.manual_seed(23)
    linear = nn.Linear(16, 32, bias=False)
    bit_layer = BitLinear.from_linear(linear)
    
    assert torch.allclose(bit_layer.weight, linear.weight)
    assert bit_layer.bias is None


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitlinear_deploy_idempotent_cpu(quant_type):
    """Test that calling _deploy multiple times is safe."""
    torch.manual_seed(24)
    layer = BitLinear(in_features=16, out_features=32, bias=True, quant_type=quant_type)
    x = torch.randn(4, 16)
    
    layer.deploy()
    with torch.no_grad():
        y1 = layer(x)
    
    # Deploy again - should be idempotent
    layer.deploy()
    with torch.no_grad():
        y2 = layer(x)
    
    assert torch.allclose(y1, y2)
    assert layer._is_deployed is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitlinear_various_dimensions_gpu():
    """Test various dimension combinations on GPU (all divisible by 4)."""
    device = torch.device("cuda")
    test_configs = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
    ]
    
    for in_features, out_features in test_configs:
        torch.manual_seed(30)
        layer = BitLinear(in_features=in_features, out_features=out_features, quant_type=quant_type).to(device)
        x = torch.randn(4, in_features, device=device)
        
        y = layer(x)
        assert y.shape == (4, out_features)
        assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitlinear_batch_dimensions_gpu():
    """Test different batch dimensions on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(31)
    layer = BitLinear(in_features=16, out_features=32, quant_type=quant_type).to(device)
    
    for batch_size in [1, 4, 8, 16, 32]:
        x = torch.randn(batch_size, 16, device=device)
        y = layer(x)
        assert y.shape == (batch_size, 32)
        assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitlinear_3d_input_gpu():
    """Test 3D input on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(32)
    layer = BitLinear(in_features=16, out_features=32, quant_type=quant_type).to(device)
    
    x = torch.randn(4, 10, 16, device=device)
    y = layer(x)
    
    assert y.shape == (4, 10, 32)
    assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitlinear_no_bias_gpu():
    """Test BitLinear without bias on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(33)
    layer = BitLinear(in_features=16, out_features=32, bias=False, quant_type=quant_type).to(device)
    
    assert layer.bias is None
    
    x = torch.randn(4, 16, device=device)
    y = layer(x)
    assert y.shape == (4, 32)
    assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitlinear_deploy_forward_gpu():
    """Test deploy mode on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(34)
    layer = BitLinear(in_features=16, out_features=32, bias=True, quant_type=quant_type).to(device)
    x = torch.randn(8, 16, device=device)
    
    layer.deploy()
    assert layer._is_deployed is True
    
    y = layer(x)
    assert y.shape == (8, 32)
    assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitlinear_eval_vs_deploy_various_dims_gpu():
    """Test eval vs deploy equivalence for various dimensions on GPU."""
    device = torch.device("cuda")
    test_configs = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
    ]
    
    for in_features, out_features in test_configs:
        torch.manual_seed(40)
        layer = BitLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True
        , quant_type=quant_type).to(device)
        layer.eval()
        x = torch.randn(4, in_features, device=device)
        
        with torch.no_grad():
            y_eval = layer(x)
        
        layer.deploy()
        
        with torch.no_grad():
            y_deploy = layer(x)
        
        max_diff = (y_eval - y_deploy).abs().max().item()
        assert y_eval.shape == y_deploy.shape == (4, out_features)
        assert max_diff < 0.5, f"GPU: Failed for {in_features}x{out_features}: diff={max_diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitlinear_eval_vs_deploy_no_bias_gpu():
    """Test eval vs deploy equivalence without bias on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(41)
    layer = BitLinear(in_features=16, out_features=32, bias=False, quant_type=quant_type).to(device)
    layer.eval()
    x = torch.randn(8, 16, device=device)
    
    with torch.no_grad():
        y_eval = layer(x)
    
    layer.deploy()
    
    with torch.no_grad():
        y_deploy = layer(x)
    
    max_diff = (y_eval - y_deploy).abs().max().item()
    assert y_eval.shape == y_deploy.shape == (8, 32)
    assert max_diff < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitlinear_eval_vs_deploy_3d_input_gpu():
    """Test eval vs deploy equivalence with 3D input on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    layer = BitLinear(in_features=16, out_features=32, bias=True, quant_type=quant_type).to(device)
    layer.eval()
    x = torch.randn(4, 5, 16, device=device)
    
    with torch.no_grad():
        y_eval = layer(x)
    
    layer.deploy()
    
    with torch.no_grad():
        y_deploy = layer(x)
    
    max_diff = (y_eval - y_deploy).abs().max().item()
    assert y_eval.shape == y_deploy.shape == (4, 5, 32)
    assert max_diff < 0.5

