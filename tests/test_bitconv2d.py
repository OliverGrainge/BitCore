import pytest
import torch
from torch import nn

from bitcore import BitConv2d


def test_bitconv2d_forward_cpu():
    """Test basic forward pass on CPU."""
    torch.manual_seed(0)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    x = torch.randn(4, 8, 32, 32, requires_grad=True)

    y = layer(x)

    assert y.shape == (4, 16, 32, 32)
    y.sum().backward()
    assert layer.weight.grad is not None
    if layer.bias is not None:
        assert layer.bias.grad is not None


def test_bitconv2d_from_conv2d_clones_weights():
    """Test from_conv2d preserves weights and bias."""
    torch.manual_seed(1)
    conv = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=True)
    bit_layer = BitConv2d.from_conv2d(conv)

    assert torch.allclose(bit_layer.weight, conv.weight)
    assert bit_layer.bias is not None
    assert torch.allclose(bit_layer.bias, conv.bias)


def test_bitconv2d_no_bias_cpu():
    """Test BitConv2d without bias."""
    torch.manual_seed(2)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, bias=False)
    
    assert layer.bias is None
    
    x = torch.randn(4, 8, 32, 32)
    y = layer(x)
    assert y.shape == (4, 16, 32, 32)


def test_bitconv2d_various_kernel_sizes_cpu():
    """Test various kernel sizes (odd sizes for symmetric padding)."""
    test_configs = [
        (1, 0),  # 1x1 conv
        (3, 1),  # 3x3 conv with padding
        (5, 2),  # 5x5 conv with padding
    ]
    
    for kernel_size, padding in test_configs:
        torch.manual_seed(10)
        layer = BitConv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=kernel_size,
            padding=padding
        )
        x = torch.randn(2, 8, 32, 32)
        
        y = layer(x)
        assert y.shape == (2, 16, 32, 32), f"Failed for kernel_size={kernel_size}"


def test_bitconv2d_stride_cpu():
    """Test various stride configurations."""
    test_configs = [
        (1, 32),  # stride 1, same size
        (2, 16),  # stride 2, half size
        (4, 8),   # stride 4, quarter size
    ]
    
    for stride, expected_size in test_configs:
        torch.manual_seed(11)
        layer = BitConv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        x = torch.randn(2, 8, 32, 32)
        
        y = layer(x)
        assert y.shape == (2, 16, expected_size, expected_size), f"Failed for stride={stride}"


def test_bitconv2d_various_channel_dimensions_cpu():
    """Test various channel dimensions (all divisible by 4)."""
    test_configs = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
    ]
    
    for in_channels, out_channels in test_configs:
        torch.manual_seed(12)
        layer = BitConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        x = torch.randn(2, in_channels, 16, 16)
        
        y = layer(x)
        assert y.shape == (2, out_channels, 16, 16), \
            f"Failed for {in_channels}x{out_channels}"


def test_bitconv2d_batch_dimensions_cpu():
    """Test different batch dimensions."""
    torch.manual_seed(13)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    
    for batch_size in [1, 4, 8, 16]:
        x = torch.randn(batch_size, 8, 32, 32)
        y = layer(x)
        assert y.shape == (batch_size, 16, 32, 32)


def test_bitconv2d_spatial_dimensions_cpu():
    """Test various spatial dimensions (all divisible by 4)."""
    spatial_sizes = [4, 8, 16, 32, 64]
    
    torch.manual_seed(14)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    
    for size in spatial_sizes:
        x = torch.randn(2, 8, size, size)
        y = layer(x)
        assert y.shape == (2, 16, size, size), f"Failed for spatial size {size}x{size}"


def test_bitconv2d_rectangular_input_cpu():
    """Test non-square input dimensions."""
    torch.manual_seed(15)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    
    x = torch.randn(2, 8, 16, 32)  # height=16, width=32
    y = layer(x)
    
    assert y.shape == (2, 16, 16, 32)


def test_bitconv2d_gradient_flow_cpu():
    """Test that gradients flow properly through quantization."""
    torch.manual_seed(16)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    x = torch.randn(4, 8, 16, 16, requires_grad=True)
    
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_bitconv2d_training_vs_eval_mode_cpu():
    """Test that training and eval modes produce consistent behavior."""
    torch.manual_seed(17)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    x = torch.randn(4, 8, 32, 32)
    
    # Training mode
    layer.train()
    y_train = layer(x)
    
    # Eval mode
    layer.eval()
    with torch.no_grad():
        y_eval = layer(x)
    
    # Shapes should match
    assert y_train.shape == y_eval.shape == (4, 16, 32, 32)
    # Outputs should be similar but may differ slightly
    max_diff = (y_train - y_eval).abs().max().item()
    assert max_diff < 1.0


def test_bitconv2d_multiple_forward_passes_cpu():
    """Test multiple forward passes with same layer."""
    torch.manual_seed(18)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    
    for i in range(5):
        x = torch.randn(4, 8, 32, 32)
        y = layer(x)
        assert y.shape == (4, 16, 32, 32)


def test_bitconv2d_deploy_mode_cpu():
    """Test deploy mode (currently same as eval)."""
    torch.manual_seed(19)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    x = torch.randn(4, 8, 32, 32)
    
    layer.eval()
    with torch.no_grad():
        y_eval = layer(x)
    
    layer._deploy()
    assert layer._is_deployed is True
    
    with torch.no_grad():
        y_deploy = layer(x)
    
    # Should be identical since no optimized kernel is implemented yet
    assert torch.allclose(y_eval, y_deploy)


def test_bitconv2d_from_conv2d_no_bias():
    """Test from_conv2d with no bias."""
    torch.manual_seed(20)
    conv = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
    bit_layer = BitConv2d.from_conv2d(conv)
    
    assert torch.allclose(bit_layer.weight, conv.weight)
    assert bit_layer.bias is None


def test_bitconv2d_from_conv2d_stride_padding():
    """Test from_conv2d preserves stride and padding."""
    torch.manual_seed(21)
    conv = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=True)
    bit_layer = BitConv2d.from_conv2d(conv)
    
    assert bit_layer.stride == conv.stride
    assert bit_layer.padding == conv.padding
    
    x = torch.randn(2, 8, 32, 32)
    y_conv = conv(x)
    y_bit = bit_layer(x)
    
    assert y_bit.shape == y_conv.shape


def test_bitconv2d_groups_cpu():
    """Test grouped convolution."""
    torch.manual_seed(22)
    # 8 input channels, 16 output channels, groups=4
    # Each group: 2 input channels -> 4 output channels
    layer = BitConv2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=1,
        groups=4
    )
    x = torch.randn(2, 8, 16, 16)
    
    y = layer(x)
    assert y.shape == (2, 16, 16, 16)


def test_bitconv2d_dilation_cpu():
    """Test dilated convolution."""
    torch.manual_seed(23)
    layer = BitConv2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=2,
        dilation=2
    )
    x = torch.randn(2, 8, 32, 32)
    
    y = layer(x)
    assert y.shape == (2, 16, 32, 32)


def test_bitconv2d_deploy_idempotent_cpu():
    """Test that calling _deploy multiple times is safe."""
    torch.manual_seed(24)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    x = torch.randn(4, 8, 32, 32)
    
    layer._deploy()
    with torch.no_grad():
        y1 = layer(x)
    
    # Deploy again - should be idempotent
    layer._deploy()
    with torch.no_grad():
        y2 = layer(x)
    
    assert torch.allclose(y1, y2)
    assert layer._is_deployed is True


# GPU Tests

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconv2d_forward_gpu():
    """Test basic forward pass on GPU."""
    torch.manual_seed(30)
    device = torch.device("cuda")
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1).to(device)
    x = torch.randn(4, 8, 32, 32, device=device, requires_grad=True)

    y = layer(x)

    assert y.device.type == "cuda"
    assert y.shape == (4, 16, 32, 32)
    y.sum().backward()
    assert layer.weight.grad is not None
    if layer.bias is not None:
        assert layer.bias.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconv2d_various_kernel_sizes_gpu():
    """Test various kernel sizes on GPU."""
    device = torch.device("cuda")
    test_configs = [
        (1, 0),
        (3, 1),
        (5, 2),
    ]
    
    for kernel_size, padding in test_configs:
        torch.manual_seed(31)
        layer = BitConv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=kernel_size,
            padding=padding
        ).to(device)
        x = torch.randn(2, 8, 32, 32, device=device)
        
        y = layer(x)
        assert y.shape == (2, 16, 32, 32)
        assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconv2d_stride_gpu():
    """Test various stride configurations on GPU."""
    device = torch.device("cuda")
    test_configs = [
        (1, 32),
        (2, 16),
        (4, 8),
    ]
    
    for stride, expected_size in test_configs:
        torch.manual_seed(32)
        layer = BitConv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=stride,
            padding=1
        ).to(device)
        x = torch.randn(2, 8, 32, 32, device=device)
        
        y = layer(x)
        assert y.shape == (2, 16, expected_size, expected_size)
        assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconv2d_batch_dimensions_gpu():
    """Test different batch dimensions on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(33)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1).to(device)
    
    for batch_size in [1, 4, 8, 16]:
        x = torch.randn(batch_size, 8, 32, 32, device=device)
        y = layer(x)
        assert y.shape == (batch_size, 16, 32, 32)
        assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconv2d_no_bias_gpu():
    """Test BitConv2d without bias on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(34)
    layer = BitConv2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=1,
        bias=False
    ).to(device)
    
    assert layer.bias is None
    
    x = torch.randn(4, 8, 32, 32, device=device)
    y = layer(x)
    assert y.shape == (4, 16, 32, 32)
    assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconv2d_deploy_mode_gpu():
    """Test deploy mode on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(35)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1).to(device)
    x = torch.randn(4, 8, 32, 32, device=device)
    
    layer._deploy()
    assert layer._is_deployed is True
    
    y = layer(x)
    assert y.shape == (4, 16, 32, 32)
    assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconv2d_groups_gpu():
    """Test grouped convolution on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(36)
    layer = BitConv2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=1,
        groups=4
    ).to(device)
    x = torch.randn(2, 8, 16, 16, device=device)
    
    y = layer(x)
    assert y.shape == (2, 16, 16, 16)
    assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconv2d_various_channel_dimensions_gpu():
    """Test various channel dimensions on GPU."""
    device = torch.device("cuda")
    test_configs = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
    ]
    
    for in_channels, out_channels in test_configs:
        torch.manual_seed(37)
        layer = BitConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        ).to(device)
        x = torch.randn(2, in_channels, 16, 16, device=device)
        
        y = layer(x)
        assert y.shape == (2, out_channels, 16, 16)
        assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconv2d_spatial_dimensions_gpu():
    """Test various spatial dimensions on GPU."""
    device = torch.device("cuda")
    spatial_sizes = [4, 8, 16, 32, 64]
    
    torch.manual_seed(38)
    layer = BitConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1).to(device)
    
    for size in spatial_sizes:
        x = torch.randn(2, 8, size, size, device=device)
        y = layer(x)
        assert y.shape == (2, 16, size, size)
        assert y.device.type == "cuda"



