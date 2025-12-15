import pytest
import torch
from torch import nn

from bitcore import BitConvTranspose2d, QUANTIZERS

# Get all available quantizer types
ALL_QUANTIZERS = list(QUANTIZERS.keys())


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_forward_cpu(quant_type):
    """Test basic forward pass on CPU."""
    torch.manual_seed(0)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type)
    x = torch.randn(4, 8, 32, 32, requires_grad=True)

    y = layer(x)

    assert y.shape == (4, 16, 32, 32)
    y.sum().backward()
    assert layer.weight.grad is not None
    if layer.bias is not None:
        assert layer.bias.grad is not None


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_upsampling_cpu(quant_type):
    """Test upsampling with stride > 1."""
    torch.manual_seed(1)
    # stride=2 should double spatial dimensions
    layer = BitConvTranspose2d(
        in_channels=8,
        out_channels=16,
        kernel_size=4,
        stride=2,
        padding=1
    , quant_type=quant_type)
    x = torch.randn(2, 8, 16, 16)
    
    y = layer(x)
    assert y.shape == (2, 16, 32, 32), f"Expected (2, 16, 32, 32), got {y.shape}"


def test_bitconvtranspose2d_from_conv_transpose2d_clones_weights():
    """Test from_conv_transpose2d preserves weights and bias."""
    torch.manual_seed(2)
    conv = nn.ConvTranspose2d(8, 16, kernel_size=3, padding=1, bias=True)
    bit_layer = BitConvTranspose2d.from_conv_transpose2d(conv)

    assert torch.allclose(bit_layer.weight, conv.weight)
    assert bit_layer.bias is not None
    assert torch.allclose(bit_layer.bias, conv.bias)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_no_bias_cpu(quant_type):
    """Test BitConvTranspose2d without bias."""
    torch.manual_seed(3)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, bias=False, quant_type=quant_type)
    
    assert layer.bias is None
    
    x = torch.randn(4, 8, 32, 32)
    y = layer(x)
    assert y.shape == (4, 16, 32, 32)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_various_kernel_sizes_cpu(quant_type):
    """Test various kernel sizes."""
    test_configs = [
        (2, 2, 0, 0, 64),   # 2x2 kernel, stride 2, no padding -> 64x64
        (3, 1, 1, 0, 32),   # 3x3 kernel, stride 1, padding 1 -> 32x32
        (4, 2, 1, 0, 64),   # 4x4 kernel, stride 2, padding 1 -> 64x64
    ]
    
    for kernel_size, stride, padding, output_padding, expected_size in test_configs:
        torch.manual_seed(10)
        layer = BitConvTranspose2d(
            in_channels=8,
            out_channels=16,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        , quant_type=quant_type)
        x = torch.randn(2, 8, 32, 32)
        
        y = layer(x)
        assert y.shape == (2, 16, expected_size, expected_size), \
            f"Failed for kernel_size={kernel_size}, expected {expected_size}, got {y.shape[2]}"


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_stride_cpu(quant_type):
    """Test various stride configurations for upsampling."""
    test_configs = [
        (1, 1, 33),  # stride 1, padding 1 -> 33x33
        (2, 1, 64),  # stride 2, padding 1 -> 64x64
    ]
    
    for stride, padding, expected_size in test_configs:
        torch.manual_seed(11)
        layer = BitConvTranspose2d(
            in_channels=8,
            out_channels=16,
            kernel_size=4,
            stride=stride,
            padding=padding
        , quant_type=quant_type)
        x = torch.randn(2, 8, 32, 32)
        
        y = layer(x)
        assert y.shape == (2, 16, expected_size, expected_size), \
            f"Failed for stride={stride}, expected {expected_size}, got {y.shape[2]}"


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_output_padding_cpu(quant_type):
    """Test output_padding parameter."""
    torch.manual_seed(12)
    # With stride=2, we can use output_padding to add 1 extra pixel
    layer = BitConvTranspose2d(
        in_channels=8,
        out_channels=16,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=1
    , quant_type=quant_type)
    x = torch.randn(2, 8, 32, 32)
    
    y = layer(x)
    assert y.shape == (2, 16, 65, 65), f"Expected (2, 16, 65, 65), got {y.shape}"


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_various_channel_dimensions_cpu(quant_type):
    """Test various channel dimensions."""
    test_configs = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
    ]
    
    for in_channels, out_channels in test_configs:
        torch.manual_seed(13)
        layer = BitConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        , quant_type=quant_type)
        x = torch.randn(2, in_channels, 16, 16)
        
        y = layer(x)
        assert y.shape == (2, out_channels, 16, 16), \
            f"Failed for {in_channels}x{out_channels}"


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_batch_dimensions_cpu(quant_type):
    """Test different batch dimensions."""
    torch.manual_seed(14)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type)
    
    for batch_size in [1, 4, 8, 16]:
        x = torch.randn(batch_size, 8, 32, 32)
        y = layer(x)
        assert y.shape == (batch_size, 16, 32, 32)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_spatial_dimensions_cpu(quant_type):
    """Test various spatial dimensions."""
    spatial_sizes = [4, 8, 16, 32, 64]
    
    torch.manual_seed(15)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type)
    
    for size in spatial_sizes:
        x = torch.randn(2, 8, size, size)
        y = layer(x)
        assert y.shape == (2, 16, size, size), f"Failed for spatial size {size}x{size}"


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_rectangular_input_cpu(quant_type):
    """Test non-square input dimensions."""
    torch.manual_seed(16)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type)
    
    x = torch.randn(2, 8, 16, 32)  # height=16, width=32
    y = layer(x)
    
    assert y.shape == (2, 16, 16, 32)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_gradient_flow_cpu(quant_type):
    """Test that gradients flow properly through quantization."""
    torch.manual_seed(17)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type)
    x = torch.randn(4, 8, 16, 16, requires_grad=True)
    
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_training_vs_eval_mode_cpu(quant_type):
    """Test that training and eval modes produce consistent behavior."""
    torch.manual_seed(18)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type)
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


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_multiple_forward_passes_cpu(quant_type):
    """Test multiple forward passes with same layer."""
    torch.manual_seed(19)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type)
    
    for i in range(5):
        x = torch.randn(4, 8, 32, 32)
        y = layer(x)
        assert y.shape == (4, 16, 32, 32)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_deploy_mode_cpu(quant_type):
    """Test deploy mode (currently same as eval)."""
    torch.manual_seed(20)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type)
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


def test_bitconvtranspose2d_from_conv_transpose2d_no_bias():
    """Test from_conv_transpose2d with no bias."""
    torch.manual_seed(21)
    conv = nn.ConvTranspose2d(8, 16, kernel_size=3, padding=1, bias=False)
    bit_layer = BitConvTranspose2d.from_conv_transpose2d(conv)
    
    assert torch.allclose(bit_layer.weight, conv.weight)
    assert bit_layer.bias is None


def test_bitconvtranspose2d_from_conv_transpose2d_stride_padding():
    """Test from_conv_transpose2d preserves stride and padding."""
    torch.manual_seed(22)
    conv = nn.ConvTranspose2d(8, 16, kernel_size=4, stride=2, padding=1, bias=True)
    bit_layer = BitConvTranspose2d.from_conv_transpose2d(conv)
    
    assert bit_layer.stride == conv.stride
    assert bit_layer.padding == conv.padding
    
    x = torch.randn(2, 8, 32, 32)
    y_conv = conv(x)
    y_bit = bit_layer(x)
    
    assert y_bit.shape == y_conv.shape


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_groups_cpu(quant_type):
    """Test grouped transposed convolution."""
    torch.manual_seed(23)
    # 8 input channels, 16 output channels, groups=4
    # Each group: 2 input channels -> 4 output channels
    layer = BitConvTranspose2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=1,
        groups=4
    , quant_type=quant_type)
    x = torch.randn(2, 8, 16, 16)
    
    y = layer(x)
    assert y.shape == (2, 16, 16, 16)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_dilation_cpu(quant_type):
    """Test dilated transposed convolution."""
    torch.manual_seed(24)
    layer = BitConvTranspose2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=2,
        dilation=2
    , quant_type=quant_type)
    x = torch.randn(2, 8, 32, 32)
    
    y = layer(x)
    assert y.shape == (2, 16, 32, 32)


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_deploy_idempotent_cpu(quant_type):
    """Test that calling _deploy multiple times is safe."""
    torch.manual_seed(25)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type)
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


@pytest.mark.parametrize("quant_type", ALL_QUANTIZERS)
def test_bitconvtranspose2d_encoder_decoder_pair_cpu(quant_type):
    """Test encoder-decoder pair with conv and conv_transpose."""
    torch.manual_seed(26)
    # Encoder: downsample
    from bitcore import BitConv2d
    encoder = BitConv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=4,
        stride=2,
        padding=1
    )
    
    # Decoder: upsample
    decoder = BitConvTranspose2d(
        in_channels=16,
        out_channels=3,
        kernel_size=4,
        stride=2,
        padding=1
    , quant_type=quant_type)
    
    x = torch.randn(2, 3, 64, 64)
    encoded = encoder(x)
    assert encoded.shape == (2, 16, 32, 32)
    
    decoded = decoder(encoded)
    assert decoded.shape == (2, 3, 64, 64)


# GPU Tests

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconvtranspose2d_forward_gpu():
    """Test basic forward pass on GPU."""
    torch.manual_seed(30)
    device = torch.device("cuda")
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type).to(device)
    x = torch.randn(4, 8, 32, 32, device=device, requires_grad=True)

    y = layer(x)

    assert y.device.type == "cuda"
    assert y.shape == (4, 16, 32, 32)
    y.sum().backward()
    assert layer.weight.grad is not None
    if layer.bias is not None:
        assert layer.bias.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconvtranspose2d_upsampling_gpu():
    """Test upsampling with stride > 1 on GPU."""
    torch.manual_seed(31)
    device = torch.device("cuda")
    layer = BitConvTranspose2d(
        in_channels=8,
        out_channels=16,
        kernel_size=4,
        stride=2,
        padding=1
    , quant_type=quant_type).to(device)
    x = torch.randn(2, 8, 16, 16, device=device)
    
    y = layer(x)
    assert y.shape == (2, 16, 32, 32)
    assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconvtranspose2d_various_kernel_sizes_gpu():
    """Test various kernel sizes on GPU."""
    device = torch.device("cuda")
    test_configs = [
        (2, 2, 0, 0, 64),   # 2x2 kernel, stride 2, no padding -> 64x64
        (3, 1, 1, 0, 32),   # 3x3 kernel, stride 1, padding 1 -> 32x32
        (4, 2, 1, 0, 64),   # 4x4 kernel, stride 2, padding 1 -> 64x64
    ]
    
    for kernel_size, stride, padding, output_padding, expected_size in test_configs:
        torch.manual_seed(32)
        layer = BitConvTranspose2d(
            in_channels=8,
            out_channels=16,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        , quant_type=quant_type).to(device)
        x = torch.randn(2, 8, 32, 32, device=device)
        
        y = layer(x)
        assert y.shape == (2, 16, expected_size, expected_size)
        assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconvtranspose2d_stride_gpu():
    """Test various stride configurations on GPU."""
    device = torch.device("cuda")
    test_configs = [
        (1, 1, 33),  # stride 1, padding 1 -> 33x33
        (2, 1, 64),  # stride 2, padding 1 -> 64x64
    ]
    
    for stride, padding, expected_size in test_configs:
        torch.manual_seed(33)
        layer = BitConvTranspose2d(
            in_channels=8,
            out_channels=16,
            kernel_size=4,
            stride=stride,
            padding=padding
        , quant_type=quant_type).to(device)
        x = torch.randn(2, 8, 32, 32, device=device)
        
        y = layer(x)
        assert y.shape == (2, 16, expected_size, expected_size)
        assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconvtranspose2d_batch_dimensions_gpu():
    """Test different batch dimensions on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(34)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type).to(device)
    
    for batch_size in [1, 4, 8, 16]:
        x = torch.randn(batch_size, 8, 32, 32, device=device)
        y = layer(x)
        assert y.shape == (batch_size, 16, 32, 32)
        assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconvtranspose2d_no_bias_gpu():
    """Test BitConvTranspose2d without bias on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(35)
    layer = BitConvTranspose2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=1,
        bias=False
    , quant_type=quant_type).to(device)
    
    assert layer.bias is None
    
    x = torch.randn(4, 8, 32, 32, device=device)
    y = layer(x)
    assert y.shape == (4, 16, 32, 32)
    assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconvtranspose2d_deploy_mode_gpu():
    """Test deploy mode on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(36)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type).to(device)
    x = torch.randn(4, 8, 32, 32, device=device)
    
    layer._deploy()
    assert layer._is_deployed is True
    
    y = layer(x)
    assert y.shape == (4, 16, 32, 32)
    assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconvtranspose2d_groups_gpu():
    """Test grouped transposed convolution on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(37)
    layer = BitConvTranspose2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=1,
        groups=4
    , quant_type=quant_type).to(device)
    x = torch.randn(2, 8, 16, 16, device=device)
    
    y = layer(x)
    assert y.shape == (2, 16, 16, 16)
    assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconvtranspose2d_various_channel_dimensions_gpu():
    """Test various channel dimensions on GPU."""
    device = torch.device("cuda")
    test_configs = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
    ]
    
    for in_channels, out_channels in test_configs:
        torch.manual_seed(38)
        layer = BitConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        , quant_type=quant_type).to(device)
        x = torch.randn(2, in_channels, 16, 16, device=device)
        
        y = layer(x)
        assert y.shape == (2, out_channels, 16, 16)
        assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconvtranspose2d_spatial_dimensions_gpu():
    """Test various spatial dimensions on GPU."""
    device = torch.device("cuda")
    spatial_sizes = [4, 8, 16, 32, 64]
    
    torch.manual_seed(39)
    layer = BitConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, quant_type=quant_type).to(device)
    
    for size in spatial_sizes:
        x = torch.randn(2, 8, size, size, device=device)
        y = layer(x)
        assert y.shape == (2, 16, size, size)
        assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitconvtranspose2d_encoder_decoder_pair_gpu():
    """Test encoder-decoder pair with conv and conv_transpose on GPU."""
    device = torch.device("cuda")
    torch.manual_seed(40)
    from bitcore import BitConv2d
    
    encoder = BitConv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=4,
        stride=2,
        padding=1
    ).to(device)
    
    decoder = BitConvTranspose2d(
        in_channels=16,
        out_channels=3,
        kernel_size=4,
        stride=2,
        padding=1
    , quant_type=quant_type).to(device)
    
    x = torch.randn(2, 3, 64, 64, device=device)
    encoded = encoder(x)
    assert encoded.shape == (2, 16, 32, 32)
    assert encoded.device.type == "cuda"
    
    decoded = decoder(encoded)
    assert decoded.shape == (2, 3, 64, 64)
    assert decoded.device.type == "cuda"
