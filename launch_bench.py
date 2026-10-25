"""Per-call cost of a plain `kernel[grid](...)` int8 GEMM on the NPU.

Compares ordinary torch tensors with NPU-resident ones (amd_triton_npu.resident,
when it exists): A [M,K] int8, B [K,N] int8 weight, C [M,N] int32.

usage: python launch_bench.py M N K [rows]
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/home/chihmin/src/egemma-npu")
import npu_gemm  # noqa: F401  (writes the tuned 512x512 schedule)
os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = npu_gemm.SCHEDULE
from npu_i8 import matmul_i8

M, N, K = map(int, sys.argv[1:4])
R = int(sys.argv[4]) if len(sys.argv) > 4 else M
BM = BN = 512
a = torch.randint(-128, 128, (M, K), dtype=torch.int8)
b = torch.randint(-128, 128, (K, N), dtype=torch.int8)
c = torch.zeros((M, N), dtype=torch.int32)
ref = a.int() @ b.int()


def bench(a, b, c, label):
    call = lambda: matmul_i8[(M // BM, N // BN)](a, b, c, M, N, K, BLOCK_M=BM, BLOCK_N=BN)
    call()
    ok = torch.equal(c[:R], ref[:R])
    ts = []
    for _ in range(30):
        t = time.perf_counter(); call(); ts.append(time.perf_counter() - t)
    ts.sort()
    ms = ts[15] * 1e3
    print(f"{label}: exact={ok} {ms:.3f} ms/call = {2 * M * N * K / ms / 1e9:.0f} GOPS", flush=True)


bench(a, b, c, "torch tensors")
try:
    from triton.backends.amd_triton_npu import resident
except ImportError:
    sys.exit(0)
ra = resident.empty((M, K), dtype=torch.int8); ra.copy_(a)
rb = resident.to_npu(b, static=True)
rc = resident.empty((M, N), dtype=torch.int32)
bench(ra, rb, rc, "resident (B static)")
