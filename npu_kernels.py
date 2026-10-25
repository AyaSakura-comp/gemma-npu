"""int8 x int8 -> int32 GEMM on the XDNA2 NPU, written as ordinary Triton.

`matmul_i8_k` loops over K in BLOCK_K steps (Triton's tutorial shape), so K
only has to be a multiple of BLOCK_K -- Gemma's 1536 / 6144 / 12288 are not
powers of two. The schedule is generated for that loop (k_loop, l2_k=BLOCK_K).

Nothing here is NPU-specific beyond the driver and the schedule: the launch is
a plain `kernel[grid](...)`, and passing tensors from
`triton.backends.amd_triton_npu.resident` is what makes it copy-free.
"""
import os

import torch
import triton
import triton.language as tl
from triton.backends.amd_triton_npu.driver import NPUDriver
from triton.backends.amd_triton_npu.matmul_transform import generate_matmul_transform

DRIVER = NPUDriver()
# Triton's active driver is per thread here (driver.py); a thread that
# launches NPU kernels calls use_npu() first.
triton.runtime.driver.set_active(DRIVER)


def use_npu():
    triton.runtime.driver.set_active(DRIVER)

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK_M = BLOCK_N = 512
BLOCK_K = 128
L1_M, L1_N = 64, 128
SCHEDULE = os.path.join(HERE, f"transform_i8_k{BLOCK_K}_{L1_M}x{L1_N}.mlir")
if not os.path.exists(SCHEDULE):
    with open(SCHEDULE, "w") as f:
        f.write(generate_matmul_transform(l1_m=L1_M, l1_n=L1_N, l2_k=BLOCK_K, pack_sizes=(8, 8, 8),
                                          accum_type="i32", k_loop=True))
os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = SCHEDULE


@triton.jit
def matmul_i8_k(A, B, C, N: tl.constexpr, K: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """C[M, N] = A[M, K] @ B[K, N]; M is the grid's, row-major everything."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K // BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)


def mm_i8(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """a [M, K] int8, b [K, N] int8, c [M, N] int32; M, N multiples of 512, K of 128."""
    M, K = a.shape
    N = b.shape[1]
    matmul_i8_k[(M // BLOCK_M, N // BLOCK_N)](a, b, c, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c
