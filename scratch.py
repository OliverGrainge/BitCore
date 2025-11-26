
import bitops 
import torch 

M = 128
K = 24
N = 64

x = torch.randn(M, K, dtype=torch.float32) * 0.5
    
# Weight matrix - quantize to 2-bit and pack
w = torch.randn(N, K, dtype=torch.float32) * 0.3
w_scale, w_quant = bitops.quant_t2spg(w, group_size=N*K)  # Per-tensor (single scale)

# Pack weights to 2-bit (row by row)
w_packed = torch.stack([bitops.pack_t2s(w_quant[i]) for i in range(N)])

# Bias
bias = torch.zeros(N, dtype=torch.float32)

# Perform matrix multiplication - activation is quantized internally per-row
y = bitops.matmul_f32_i8spr_t2spt(x, w_scale, w_packed, bias)
print(y.shape, y.dtype)