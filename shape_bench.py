"""Per-call time of the K-loop int8 GEMM for Gemma 4 E2B shapes, plain vs resident tensors.

usage: python shape_bench.py M N K [M N K ...]
"""
import sys
import time

import torch

import npu_kernels
from triton.backends.amd_triton_npu import resident


def median_ms(fn, n=20):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[n // 2] * 1e3


args = list(map(int, sys.argv[1:]))
for M, N, K in zip(args[0::3], args[1::3], args[2::3]):
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8)
    c = torch.zeros((M, N), dtype=torch.int32)
    t0 = time.time()
    npu_kernels.mm_i8(a, b, c)
    build = time.time() - t0
    ok = torch.equal(c, a.int() @ b.int())
    plain = median_ms(lambda: npu_kernels.mm_i8(a, b, c))
    ra, rb, rc = resident.to_npu(a), resident.to_npu(b, static=True), resident.empty((M, N), dtype=torch.int32)
    ms = median_ms(lambda: npu_kernels.mm_i8(ra, rb, rc))
    ok2 = torch.equal(rc, c)
    ops = 2 * M * N * K
    print(f"M{M} N{N} K{K}: exact={ok}/{ok2} build {build:.0f}s | plain {plain:.2f} ms ({ops / plain / 1e9:.1f} TOPS) "
          f"| resident {ms:.2f} ms ({ops / ms / 1e9:.1f} TOPS)", flush=True)
