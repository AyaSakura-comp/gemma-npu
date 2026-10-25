# Gemma 4 E2B on the XDNA2 NPU from PyTorch

HF transformers' `google/gemma-4-E2B-it`, text decoder, with every linear of the
prefill run as a W8A8 GEMM on the NPU through an ordinary Triton kernel
(`npu_kernels.py`, launched as `kernel[grid](...)` via Triton-XDNA), and decode
on the CPU. Needs the Triton-XDNA branch `feat/xrt-resident-tensors`
(`triton.backends.amd_triton_npu.resident`).

```bash
source ~/src/Triton-XDNA/npu-env.sh
python gemma_npu.py --backend npu            # TTFT + decode at 4 prompt lengths
python gemma_npu.py --backend cpu            # same model, HF bf16 on the CPU
./run_flm_bench.sh gemma4-it:e2b             # FastFlowLM, the same prompts
python eval_ppl.py npu ppl_npu.npz           # teacher-forced NLL (vs `cpu`)
python trace_pipe.py 3640 t.jsonl && python3 plot_trace.py t.jsonl gantt.png
```

## What runs where

| part | where | how |
|---|---|---|
| q/k/v, o, gate/up, down, PLE model projection | NPU | W8A8 (per-token / per-channel int8), int32 out; weights *static resident* tensors, activations and results in resident slot buffers -- nothing copied per launch |
| PLE gate (N=256) / projection (K=256) | CPU | `torch._int_mm`, same int8 weights (too small for a launch) |
| norms, RoPE, GELU, residuals, quant/dequant | CPU | `torch.compile`'d fused kernels between GEMMs (`pipeline.py`) |
| attention, prefill | CPU | sliding layers: only the window per 128-query block; full layers: causal flash, prefix + own part merged by logsumexp (`local_attn.py`, `pipeline.py`) |
| attention, decode | CPU | split-KV flash decode, 16-way (`local_attn.py`) |
| decode linears + LM head | CPU | `torch._weight_int8pack_mm` (bandwidth-bound GEMV; the CPU has ~2x the NPU's DRAM bandwidth) |

Prefill splits the prompt into 512-row micro-batches, each a generator over
the layers; GEMMs go to an NPU worker thread (the Triton launch releases the
GIL) while the main thread does other micro-batches' CPU work. Micro-batches
are kept one GEMM apart (`pipeline.LAG`): lockstep puts all of them in
attention, then all in the MLP, leaving one side idle in each phase; strict
earliest-first leaves a serial tail.

## Results (2026-10-25, same prompts, back to back)

| prompt | this, TTFT | FLM gemma4-it:e2b | this, decode | FLM decode |
|---|---|---|---|---|
| 207 | 362 ms | 840 ms | 30.0 tok/s | 24.3 |
| 858 | 721 ms | 1315 ms | 28.6 | 23.1 |
| 1878 | 1378 ms | 2022 ms | 28.3 | 22.2 |
| 3658 | 2911 ms | 3379 ms | 27.4 | 20.8 |

FLM's `gemma4e-flash:e2b` is 332 ms at 207 tokens but stops at ~1k tokens of
context. Perplexity, W8A8 NPU vs HF bf16: +2.3% (notes), -4.8% (wikitext-2),
i.e. within window-to-window noise.

## Against the CPU alone (same pipeline, GEMMs as `torch._int_mm`: `--backend cpu-int8`)

| prompt | NPU TTFT | CPU-int8 TTFT | HF bf16 CPU TTFT |
|---|---|---|---|
| 207 | 368 ms | **191 ms** | 203 ms |
| 858 | 680 ms | 701 ms | 2732 ms |
| 1878 | **1379 ms** | 1743 ms | 4028 ms |
| 3658 | **2739 ms** | 3806 ms | 5632 ms |

Decode is the same code on both (CPU), 28-30 tok/s; HF bf16 decodes at 13.5.
At 3658 tokens the GEMMs take 1.49 s as `_int_mm` on the CPU and keep the NPU
busy 2.38 s: the NPU is not the faster GEMM engine here, it is a second one
that works while the CPU does attention. Under ~1k tokens the 512-row padding
and 141+ launches make it slower than the CPU.

