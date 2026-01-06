"""
Deployment examples for BitLinear layers.

This example demonstrates:
- Switching to deployment mode
- Inference in deployment mode
- Comparison between eval and deploy modes
- Optimized inference path
- Latency measurements and benchmarking
- Throughput measurements for standard LLM matmul sizes
"""

import torch
import time
from bitcore import BitLinear

# Check if bitops is available
try:
    import bitops
    HAS_BITOPS = True
except ImportError:
    HAS_BITOPS = False


def check_bitops_usage(layer):
    """Check if bitops is being used in a deployed layer."""
    if not layer._is_deployed or not HAS_BITOPS:
        return False
    
    # Check if the inference function is the bitops one
    # The bitops inference function will have '_bitops_inference_fn' in its name
    if hasattr(layer, 'inference_fn'):
        fn_name = str(layer.inference_fn)
        return '_bitops_inference_fn' in fn_name or 'bitops' in fn_name.lower()
    
    return False


def get_layer_size(layer):
    """Get the size of a layer in bytes."""
    total_size = 0
    for param in layer.parameters():
        total_size += param.numel() * param.element_size()
    for buffer in layer.buffers():
        total_size += buffer.numel() * buffer.element_size()
    return total_size


def format_size(size_bytes):
    """Format size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def example_deployment_basic():
    """Example: Basic deployment usage."""
    print("=" * 60)
    print("Basic Deployment")
    print("=" * 60)
    
    # Create and train a layer (simulated)
    layer = BitLinear(in_features=64, out_features=32, bias=True)
    x = torch.randn(8, 64)
    
    # Training mode
    layer.train()
    y_train = layer(x)
    print(f"Training mode output shape: {y_train.shape}")
    print(f"Training mode - requires_grad: {y_train.requires_grad}")
    print(f"Is deployed: {layer._is_deployed}")
    print()
    
    # Switch to deployment mode
    print("Switching to deployment mode...")
    layer.deploy()
    print(f"Is deployed: {layer._is_deployed}")
    print()
    
    # Inference in deployment mode
    layer.eval()
    with torch.no_grad():
        y_deploy = layer(x)
    print(f"Deployment mode output shape: {y_deploy.shape}")
    print()


def example_eval_vs_deploy():
    """Example: Compare eval and deploy modes - equivalence check."""
    print("=" * 60)
    print("Eval vs Deploy Equivalence")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = BitLinear(in_features=64, out_features=32, bias=True).to(device)
    x = torch.randn(8, 64, device=device)
    
    # Evaluation mode (before deployment)
    layer.eval()
    size_before = get_layer_size(layer)
    with torch.no_grad():
        y_eval = layer(x)
    
    # Deploy mode
    layer.deploy()
    size_after = get_layer_size(layer)
    using_bitops = check_bitops_usage(layer)
    
    with torch.no_grad():
        y_deploy = layer(x)
    
    # Check equivalence
    max_diff = (y_eval - y_deploy).abs().max().item()
    
    print(f"Bitops available: {HAS_BITOPS}")
    print(f"Using bitops: {using_bitops}")
    print(f"Layer size before: {format_size(size_before)}")
    print(f"Layer size after:  {format_size(size_after)}")
    print(f"Size reduction:    {format_size(size_before - size_after)} ({100 * (1 - size_after/size_before):.1f}%)")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Equivalence check: {'✓ PASS' if max_diff < 0.5 else '✗ FAIL'}")
    print()


def example_llm_throughput():
    """Example: Measure throughput for standard small LLM matmul sizes."""
    print("=" * 60)
    print("LLM Matmul Throughput Benchmark")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Bitops available: {HAS_BITOPS}")
    print()
    
    # Standard small LLM matmul sizes: (batch, seq_len, hidden_size)
    # Common hidden sizes: 1024, 2048, 4096
    # Common sequence lengths: 512, 1024, 2048
    test_configs = [
        (1, 512, 1024, 1024),    # Single token, small hidden
        (1, 512, 2048, 2048),    # Single token, medium hidden
        (1, 512, 4096, 4096),    # Single token, large hidden
        (8, 512, 1024, 1024),    # Small batch
        (8, 1024, 2048, 2048),   # Medium batch, longer sequence
        (16, 512, 2048, 2048),   # Larger batch
    ]
    
    num_warmup = 10
    num_iterations = 100
    
    print(f"{'Config':<20} {'Mode':<8} {'Throughput (kilo tok/s)':<25} {'Using Bitops':<12}")
    print("-" * 75)
    
    for batch_size, seq_len, in_features, out_features in test_configs:
        layer = BitLinear(in_features=in_features, out_features=out_features, bias=True).to(device)
        x = torch.randn(batch_size, seq_len, in_features, device=device)
        
        # Eval mode - get output for equivalence check before deploying
        layer.eval()
        with torch.no_grad():
            y_eval = layer(x)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = layer(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark eval
        start_time = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = layer(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        eval_time = time.perf_counter() - start_time
        
        total_tokens = batch_size * seq_len * num_iterations
        eval_throughput_tok_s = total_tokens / eval_time
        eval_throughput_kilo_tok_s = eval_throughput_tok_s / 1000.0
        
        config_str = f"B{batch_size}_S{seq_len}_H{in_features}"
        print(f"{config_str:<20} {'Eval':<8} {eval_throughput_kilo_tok_s:>23.2f} {'N/A':<12}")
        
        # Deploy mode
        layer.deploy()
        using_bitops = check_bitops_usage(layer)
        
        # Get deploy output for equivalence check
        with torch.no_grad():
            y_deploy = layer(x)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = layer(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark deploy
        start_time = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = layer(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        deploy_time = time.perf_counter() - start_time
        
        deploy_throughput_tok_s = total_tokens / deploy_time
        deploy_throughput_kilo_tok_s = deploy_throughput_tok_s / 1000.0
        bitops_str = "✓ Yes" if using_bitops else "✗ No"
        
        print(f"{config_str:<20} {'Deploy':<8} {deploy_throughput_kilo_tok_s:>23.2f} {bitops_str:<12}")
        
        # Equivalence check (using outputs from same layer before/after deployment)
        max_diff = (y_eval - y_deploy).abs().max().item()
        equiv_str = "✓" if max_diff < 0.5 else "✗"
        print(f"  {equiv_str} Equivalence: max_diff={max_diff:.6f}")
        print()
    
    print()


def example_latency_model():
    """Example: Benchmark latency of a complete model."""
    print("=" * 60)
    print("Latency Benchmark: Complete Model")
    print("=" * 60)
    
    import torch.nn as nn
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create a model with multiple BitLinear layers
    model = nn.Sequential(
        BitLinear(784, 256, quant_type="bitnet"),
        nn.ReLU(),
        BitLinear(256, 128, quant_type="bitnet"),
        nn.ReLU(),
        BitLinear(128, 10, quant_type="bitnet"),
    ).to(device)
    
    batch_size = 64
    x = torch.randn(batch_size, 784, device=device)
    
    num_warmup = 10
    num_iterations = 100
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark before deployment
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    before_time = time.perf_counter() - start_time
    before_latency_ms = (before_time / num_iterations) * 1000
    
    # Deploy all BitLinear layers
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.deploy()
    
    # Warmup after deployment
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark after deployment
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    after_time = time.perf_counter() - start_time
    after_latency_ms = (after_time / num_iterations) * 1000
    
    speedup = before_latency_ms / after_latency_ms
    
    print(f"\nModel benchmark results ({num_iterations} iterations, batch_size={batch_size}):")
    print(f"  Before deployment: {before_latency_ms:.4f} ms")
    print(f"  After deployment:  {after_latency_ms:.4f} ms")
    print(f"  Speedup:          {speedup:.2f}x")
    print()


def example_latency_batch_sizes():
    """Example: Compare latency across different batch sizes."""
    print("=" * 60)
    print("Latency Benchmark: Different Batch Sizes")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    layer = BitLinear(in_features=256, out_features=128, bias=True).to(device)
    layer.deploy()
    layer.eval()
    
    batch_sizes = [1, 8, 16, 32, 64, 128]
    num_warmup = 10
    num_iterations = 100
    
    print(f"\nBatch Size | Latency (ms) | Throughput (samples/s)")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 256, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = layer(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = layer(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        
        latency_ms = (elapsed / num_iterations) * 1000
        throughput = (batch_size * num_iterations) / elapsed
        
        print(f"  {batch_size:9d} | {latency_ms:11.4f} | {throughput:18.2f}")
    
    print()


def example_deploy_model():
    """Example: Deploy an entire model - equivalence check."""
    print("=" * 60)
    print("Deploying a Model")
    print("=" * 60)
    
    import torch.nn as nn
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a model with multiple BitLinear layers
    model = nn.Sequential(
        BitLinear(128, 64, quant_type="bitnet"),
        nn.ReLU(),
        BitLinear(64, 32, quant_type="bitnet"),
        nn.ReLU(),
        BitLinear(32, 10, quant_type="bitnet"),
    ).to(device)
    
    x = torch.randn(8, 128, device=device)
    
    # Before deployment
    model.eval()
    size_before = sum(get_layer_size(m) for m in model.modules() if isinstance(m, BitLinear))
    with torch.no_grad():
        y_before = model(x)
    
    # Deploy all BitLinear layers
    bitlinear_count = sum(1 for m in model.modules() if isinstance(m, BitLinear))
    using_bitops_list = []
    
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.deploy()
            using_bitops_list.append(check_bitops_usage(module))
    
    # After deployment
    size_after = sum(get_layer_size(m) for m in model.modules() if isinstance(m, BitLinear))
    with torch.no_grad():
        y_after = model(x)
    
    # Check equivalence
    max_diff = (y_before - y_after).abs().max().item()
    
    print(f"Bitops available: {HAS_BITOPS}")
    print(f"Number of BitLinear layers: {bitlinear_count}")
    print(f"Layers using bitops: {sum(using_bitops_list)}/{bitlinear_count}")
    print(f"Model size before: {format_size(size_before)}")
    print(f"Model size after:  {format_size(size_after)}")
    print(f"Size reduction:    {format_size(size_before - size_after)} ({100 * (1 - size_after/size_before):.1f}%)")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Equivalence check: {'✓ PASS' if max_diff < 0.5 else '✗ FAIL'}")
    print()


def example_deploy_idempotent():
    """Example: Deploy is idempotent (safe to call multiple times)."""
    print("=" * 60)
    print("Deploy Idempotency")
    print("=" * 60)
    
    layer = BitLinear(in_features=32, out_features=16, bias=True)
    x = torch.randn(4, 32)
    
    # First deployment
    layer.deploy()
    layer.eval()
    with torch.no_grad():
        y1 = layer(x)
    
    print(f"After first deploy - is_deployed: {layer._is_deployed}")
    print(f"Output shape: {y1.shape}")
    
    # Deploy again (should be safe)
    layer.deploy()
    with torch.no_grad():
        y2 = layer(x)
    
    print(f"After second deploy - is_deployed: {layer._is_deployed}")
    print(f"Output shape: {y2.shape}")
    
    # Verify outputs are identical
    assert torch.allclose(y1, y2), "Deploy should be idempotent!"
    print("✓ Deploy is idempotent (safe to call multiple times)")
    print()


def example_deployment_without_bias():
    """Example: Deployment without bias."""
    print("=" * 60)
    print("Deployment Without Bias")
    print("=" * 60)
    
    layer = BitLinear(in_features=64, out_features=32, bias=False)
    x = torch.randn(8, 64)
    
    print(f"Layer has bias: {layer.bias is not None}")
    
    # Deploy
    layer.deploy()
    layer.eval()
    
    with torch.no_grad():
        y = layer(x)
    
    print(f"Deployed output shape: {y.shape}")
    print(f"Is deployed: {layer._is_deployed}")
    print()


if __name__ == "__main__":
    example_deployment_basic()
    example_eval_vs_deploy()
    example_deploy_model()
    example_deploy_idempotent()
    example_deployment_without_bias()
    
    # LLM throughput benchmark
    print("\n" + "=" * 60)
    print("LLM Throughput Benchmarks")
    print("=" * 60 + "\n")
    example_llm_throughput()

