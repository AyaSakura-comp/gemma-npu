"""Resident K-loop GEMM time per row bucket for the two biggest Gemma 4 E2B shapes."""
import time

import torch

import npu_kernels
from triton.backends.amd_triton_npu import resident

for N, K in ((24576, 1536), (1536, 12288)):
    w = resident.to_npu(torch.randint(-128, 128, (K, N), dtype=torch.int8), static=True)
    for M in (512, 1024, 2048, 4096):
        a = resident.to_npu(torch.randint(-128, 128, (M, K), dtype=torch.int8))
        c = resident.empty((M, N), dtype=torch.int32)
        npu_kernels.mm_i8(a, w, c)
        ts = []
        for _ in range(10):
            t = time.perf_counter(); npu_kernels.mm_i8(a, w, c); ts.append(time.perf_counter() - t)
        ms = sorted(ts)[5] * 1e3
        print(f"N{N} K{K} M{M}: {ms:.2f} ms = {2 * M * N * K / ms / 1e9:.1f} TOPS, {ms / M * 1e3:.1f} us/row", flush=True)
