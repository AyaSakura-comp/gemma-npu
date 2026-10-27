"""One decode-shaped int8 GEMM ([M_pad, K] x [K, N]) on the NPU with a small-M schedule.

usage: python gemv_sweep.py BM BN L1M L1N N K   (one config per process: the schedule is process-wide)
"""
import os
import sys
import time

import torch

BM, BN, L1M, L1N, N, K = map(int, sys.argv[1:7])
from triton.backends.amd_triton_npu.matmul_transform import generate_matmul_transform
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"t_gemv_{L1M}_{L1N}.mlir")
open(path, "w").write(generate_matmul_transform(l1_m=L1M, l1_n=L1N, l2_k=128, pack_sizes=(8, 8, 8),
                                                accum_type="i32", k_loop=True))
os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = path
import npu_kernels
os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = path
from triton.backends.amd_triton_npu import resident

a = resident.to_npu(torch.randint(-128, 128, (BM, K), dtype=torch.int8))
w = resident.to_npu(torch.randint(-128, 128, (K, N), dtype=torch.int8), static=True)
c = resident.empty((BM, N), dtype=torch.int32)
call = lambda: npu_kernels.matmul_i8_k[(1, N // BN)](a, w, c, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=128)
t = time.time()
try:
    call()
except Exception as e:
    print(f"BM{BM} BN{BN} l1 {L1M}x{L1N} N{N} K{K}: FAIL {type(e).__name__}: {str(e)[-1500:]}")
    sys.exit(0)
build = time.time() - t
ok = torch.equal(c, a.int() @ w.int())
ts = []
for _ in range(30):
    t = time.perf_counter(); call(); ts.append(time.perf_counter() - t)
ms = sorted(ts)[15] * 1e3
print(f"BM{BM} BN{BN} l1 {L1M}x{L1N} N{N} K{K}: exact={ok} {ms:.3f} ms, weights {K * N / ms / 1e6:.1f} GB/s (build {build:.0f}s)")
