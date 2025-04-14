import triton
import triton.language as tl
import torch

# Kernel
@triton.jit
def add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # block index
    block_start = pid * BLOCK_SIZE # block address
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # block_start + int[0, 1, .... BLOCK_SIZE - 1]

    # if idx >= N: return
    mask = offsets < N  # [true, true, true ,,,..... false, false, false]
    x = tl.load(X_ptr + offsets, mask=mask) # [X_ptr + offsets[0], X_ptr + offsets[1], ....]
    y = tl.load(Y_ptr + offsets, mask=mask) # [Y_ptr + offsets[0], Y_ptr + offsets[1], ....]
    tl.store(Z_ptr + offsets, x + y, mask=mask) # Z_ptr + offsets = [...] + [...]


# Launch
N = 1024
x = torch.randn(N, device='cuda')
y = torch.randn(N, device='cuda')
z = torch.empty(N, device='cuda')

BLOCK_SIZE = 256
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

add_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)

# Serial
def add_serial(X, Y, Z, N):
    for i in range(N):
        Z[i] = X[i] + Y[i]
