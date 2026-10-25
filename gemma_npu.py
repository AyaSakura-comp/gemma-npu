"""Gemma 4 E2B (HF transformers, PyTorch on the CPU) with its linears on the NPU.

The model is the stock `Gemma4ForConditionalGeneration` forward; only the text
decoder's nn.Linear modules are swapped for W8A8 GEMMs that run the Triton
kernel in npu_kernels.py through a plain `kernel[grid](...)` launch:

  * weights: per-output-channel int8, quantized once into *static* resident
    tensors (amd_triton_npu.resident) -- never copied or synced again;
  * activations: per-token int8, written into one resident staging buffer;
    the int32 result is read straight out of one resident output buffer,
    synced only over the rows that were computed;
  * linears that read the same input run as one GEMM: q/k/v and gate/up. The
    first module called computes all of them, the others take their slice.

Prefill runs on the NPU in row chunks (CHUNKS); decode (one row) stays on the
CPU in bf16, where a GEMV is bandwidth-bound and the CPU has twice the NPU's
DRAM bandwidth.

usage: python gemma_npu.py [--backend npu|cpu-int8|cpu] [--prompt-tokens 190 840 1860 3640] [--decode 64]
"""
import argparse
import json
import os
import time

os.environ.setdefault("AMD_TRITON_NPU_XRT_MAX_SESSIONS", "24")

import torch
import torch.nn.functional as F

MODEL = "google/gemma-4-E2B-it"
MAX_ROWS = 4096
TIMES = {"npu": 0.0, "quant": 0.0, "dequant": 0.0, "launches": 0}
BY_SHAPE = {}  # (N_pad, K) -> [launches, seconds]


@torch.compile(dynamic=True)
def _quant_rows(x):
    """[T, K] float -> int8 rows and their scale [T]."""
    x = x.float()
    s = x.abs().amax(dim=1).clamp(min=1e-8) / 127
    return torch.round(x / s[:, None]).clamp(-127, 127).to(torch.int8), s


@torch.compile(dynamic=True)
def _dequant(acc, sx, sw, dtype: torch.dtype):
    return (acc.float() * sx[:, None] * sw[None, :]).to(dtype)


class NPUGemm:
    """Linears sharing one input, as a single resident W8A8 GEMM."""

    a_flat = None  # int8 activations, MAX_ROWS x max K
    c_flat = None  # int32 results, MAX_ROWS x max N
    slots = []     # per pipelined micro-batch: (int8 activations, int32 results), SLOT_ROWS each

    def __init__(self, linears, on_cpu=False):
        import npu_kernels
        from triton.backends.amd_triton_npu import resident

        self.k = npu_kernels
        w = torch.cat([l.weight.detach().float() for l in linears])
        self.sizes = [l.out_features for l in linears]
        n, self.K = w.shape
        assert self.K % npu_kernels.BLOCK_K == 0, self.K
        self.N = n
        self.n_pad = -(-n // npu_kernels.BLOCK_N) * npu_kernels.BLOCK_N
        sw = w.abs().amax(dim=1).clamp(min=1e-8) / 127
        q = torch.round(w / sw[:, None]).clamp(-127, 127).to(torch.int8)
        wt = torch.zeros((self.K, self.n_pad), dtype=torch.int8)
        wt[:, :n] = q.t()
        # A GEMM too small to be worth a launch runs as torch._int_mm on the
        # CPU, from the same int8 values.
        self.on_cpu = on_cpu
        self.w = None if on_cpu else resident.to_npu(wt, static=True)
        self.w_kn = wt[:, :n].contiguous() if on_cpu else None
        self.sw = sw
        self.dtype = linears[0].weight.dtype
        # Decode runs on the CPU from the same int8 values, [N, K] for
        # torch's weight-only int8 GEMV.
        self.w_cpu = q.contiguous()
        self.sw_cpu = sw.to(torch.bfloat16)
        self.pending = None
        self.name = ""  # set by to_npu

    @classmethod
    def allocate(cls, max_k, max_n):
        from triton.backends.amd_triton_npu import resident
        cls.a_flat = resident.empty(MAX_ROWS * max_k, dtype=torch.int8)
        cls.c_flat = resident.empty(MAX_ROWS * max_n, dtype=torch.int32)
        cls.slots = [(resident.empty(SLOT_ROWS * max_k, dtype=torch.int8),
                      resident.empty(SLOT_ROWS * max_n, dtype=torch.int32)) for _ in range(N_SLOTS)]

    def stage(self, slot, T):
        """The slot's int8 activation rows [T, K], to be filled before launch_slot."""
        assert T <= SLOT_ROWS, f"micro-batch of {T} rows > {SLOT_ROWS}"
        return self.slots[slot][0][: T * self.K].view(T, self.K)

    def launch_slot(self, T, slot):
        """Run the GEMM on a slot's staged rows; int32 [T, N], a view of the slot's results."""
        t = time.perf_counter()
        a_flat, c_flat = self.slots[slot]
        K, Np = self.K, self.n_pad
        rows = -(-T // SMALL_CHUNK) * SMALL_CHUNK
        for r0 in range(0, rows, SMALL_CHUNK):
            av = a_flat[r0 * K: (r0 + SMALL_CHUNK) * K].view(SMALL_CHUNK, K)
            cv = c_flat[r0 * Np: (r0 + SMALL_CHUNK) * Np].view(SMALL_CHUNK, Np)
            self.k.mm_i8(av, self.w, cv)
            TIMES["launches"] += 1
        dt = time.perf_counter() - t
        TIMES["npu"] += dt
        st = BY_SHAPE.setdefault((Np, K), [0, 0.0])
        st[0] += 1
        st[1] += dt
        return c_flat[: rows * Np].view(rows, Np)[:T, : self.N]

    @torch.compiler.disable
    def matmul(self, x2d):
        T = x2d.shape[0]
        if T < 16:  # decode: a GEMV, bandwidth-bound -> CPU
            return cpu_int8_linear(x2d, self.w_cpu, self.sw_cpu).to(self.dtype)
        if self.on_cpu:
            xq, sx = _quant_rows(x2d)
            return _dequant(torch._int_mm(xq, self.w_kn), sx, self.sw, self.dtype)
        assert T <= MAX_ROWS, f"prompt longer than {MAX_ROWS} tokens"
        t = time.perf_counter()
        xq, sx = _quant_rows(x2d)
        K, Np = self.K, self.n_pad
        rows = -(-T // self.k.BLOCK_M) * self.k.BLOCK_M
        a = self.a_flat[: rows * K].view(rows, K)
        a[:T].copy_(xq)
        t1 = time.perf_counter()
        TIMES["quant"] += t1 - t
        r0 = 0
        while r0 < rows:
            # A launch covers CHUNK or SMALL_CHUNK rows, past `rows` if need
            # be (the buffers hold MAX_ROWS, a multiple of both).
            bm = CHUNK if rows - r0 > CHUNK - SMALL_CHUNK else SMALL_CHUNK
            av = self.a_flat[r0 * K: (r0 + bm) * K].view(bm, K)
            cv = self.c_flat[r0 * Np: (r0 + bm) * Np].view(bm, Np)
            self.k.mm_i8(av, self.w, cv)
            TIMES["launches"] += 1
            r0 += bm
        t2 = time.perf_counter()
        TIMES["npu"] += t2 - t1
        st = BY_SHAPE.setdefault((Np, K), [0, 0.0])
        st[0] += 1
        st[1] += t2 - t1
        c = self.c_flat[: rows * Np].view(rows, Np)[:T, : self.N]
        y = _dequant(c, sx, self.sw, self.dtype)
        TIMES["dequant"] += time.perf_counter() - t2
        return y


def cpu_int8_linear(x2d, wq, sw):
    return torch._weight_int8pack_mm(x2d.to(torch.bfloat16).contiguous(), wq, sw)


class Int8Head(torch.nn.Module):
    """The LM head as a weight-only int8 GEMV on the CPU (prefill keeps one row)."""

    def __init__(self, linear):
        super().__init__()
        w = linear.weight.detach().float()
        sw = w.abs().amax(dim=1).clamp(min=1e-8) / 127
        self.wq = torch.round(w / sw[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
        self.sw = sw.to(torch.bfloat16)
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x):
        shape = x.shape[:-1]
        return cpu_int8_linear(x.reshape(-1, x.shape[-1]), self.wq, self.sw).reshape(*shape, -1)


class Slice(torch.nn.Module):
    """Stands in for one linear of an NPUGemm group."""

    def __init__(self, gemm, index, in_features, out_features):
        super().__init__()
        self.gemm, self.index = gemm, index
        self.in_features, self.out_features = in_features, out_features

    def forward(self, x):
        g = self.gemm
        shape = x.shape[:-1]
        if self.index == 0 or g.pending is None or g.pending[0] is not x:
            y = g.matmul(x.reshape(-1, x.shape[-1]))
            parts = torch.split(y, g.sizes, dim=-1) if len(g.sizes) > 1 else (y,)
            g.pending = (x, parts)
        out = g.pending[1][self.index]
        if self.index == len(g.sizes) - 1:
            g.pending = None
        return out.reshape(*shape, -1)


# Rows per launch. Past 1024 rows the big GEMMs gain little (10.2 -> 10.6
# TOPS at 4096), and every row bucket is another hw_context per shape -- of
# which the device has ~32, 10 held by the FLM service on this box.
CHUNK = SMALL_CHUNK = int(os.environ.get("GEMMA_NPU_ROWS", "512"))
SLOT_ROWS = SMALL_CHUNK
N_SLOTS = 4096 // SLOT_ROWS
COMPILE = os.environ.get("GEMMA_NPU_COMPILE", "1") == "1"


def to_npu(model, all_cpu=False):
    """Swap the text decoder's linears for W8A8 GEMMs; all_cpu runs every one as torch._int_mm."""
    lm = model.model.language_model
    groups = []
    per_layer = []
    for layer in lm.layers:
        a, mlp = layer.self_attn, layer.mlp
        qkv = [a.q_proj]
        if not a.is_kv_shared_layer:
            qkv += [a.k_proj] + ([a.v_proj] if a.v_proj is not None else [])
        groups += [(a, ["q_proj", "k_proj", "v_proj"][: len(qkv)], qkv),
                   (a, ["o_proj"], [a.o_proj]),
                   (mlp, ["gate_proj", "up_proj"], [mlp.gate_proj, mlp.up_proj]),
                   (mlp, ["down_proj"], [mlp.down_proj]),
                   (layer, ["per_layer_input_gate"], [layer.per_layer_input_gate]),
                   (layer, ["per_layer_projection"], [layer.per_layer_projection])]
        per_layer.append(["qkv", "o", "gate_up", "down", "ple_gate", "ple_proj"])
    groups.append((lm, ["per_layer_model_projection"], [lm.per_layer_model_projection]))
    gemms = []
    for parent, names, linears in groups:
        # The PLE gate (N=256) and projection (K=256) are 0.8 MOP/token each:
        # less than a launch costs.
        small = all_cpu or names[0] in ("per_layer_input_gate", "per_layer_projection")
        g = NPUGemm(linears, on_cpu=small)
        g.name = "+".join(names)
        gemms.append(g)
        for i, (name, l) in enumerate(zip(names, linears)):
            setattr(parent, name, Slice(g, i, l.in_features, l.out_features))
    layer_gemms = [dict(zip(keys, gemms[6 * i: 6 * i + 6])) for i, keys in enumerate(per_layer)]
    model.layer_gemms = layer_gemms
    if not all_cpu:
        NPUGemm.allocate(max(g.K for g in gemms), max(g.n_pad for g in gemms))
    # Tied to the input embedding, so this is a new int8 copy, not a swap.
    model.lm_head = Int8Head(model.lm_head)
    return gemms


def load(backend):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    import local_attn  # noqa: F401  (registers attn_implementation="local_sdpa")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    for name in ("audio_tower", "vision_tower", "embed_audio", "embed_vision"):
        if hasattr(model.model, name):
            setattr(model.model, name, None)
    # Only the text decoder: the audio tower would refuse a non-sdpa implementation.
    model.model.language_model.set_attn_implementation("local_sdpa")
    gemms = to_npu(model, all_cpu=backend == "cpu-int8") if backend in ("npu", "cpu-int8") else []
    if COMPILE:
        # Fuse the CPU-side elementwise work (norms, RoPE, GELU, residuals,
        # PLE) between the NPU calls, which are graph breaks.
        # Prefill only: decode is one row, where the guards and recompiles
        # cost more than the fusion saves.
        torch._dynamo.config.cache_size_limit = 256
        for layer in model.model.language_model.layers:
            eager = layer.forward
            compiled = torch.compile(eager, dynamic=True)
            layer.forward = (lambda eager, compiled: lambda hidden_states, *a, **k:
                             (compiled if hidden_states.shape[1] > 16 else eager)(hidden_states, *a, **k))(eager, compiled)
    return tok, model, gemms


def prompt_ids(tok, n_tokens):
    text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "filler.txt")).read()
    body = tok(text)["input_ids"][: n_tokens]
    msg = tok.decode(body) + "\n\nSummarize the text above in detail."
    ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"]
    return ids


@torch.inference_mode()
def run(model, ids, n_decode, prefill=None):
    """Prefill (timed) then greedy decode; returns (ttft, decode tok/s, tokens, last logits).

    `prefill` is a pipeline.Prefill; without one the stock HF forward prefills.
    """
    t = time.perf_counter()
    if prefill is not None:
        logits, past = prefill(ids)
    else:
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        logits, past = out.logits[0, -1].float(), out.past_key_values
    ttft = time.perf_counter() - t
    nxt = logits.argmax().view(1, 1)
    toks = [nxt.item()]
    t = time.perf_counter()
    for _ in range(n_decode - 1):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().view(1, 1)
        toks.append(nxt.item())
    dec = (n_decode - 1) / (time.perf_counter() - t) if n_decode > 1 else 0.0
    return ttft, dec, toks, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="npu", choices=["npu", "cpu-int8", "cpu"])
    ap.add_argument("--prompt-tokens", type=int, nargs="+", default=[190, 840, 1860, 3640])
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--shapes", action="store_true", help="print NPU time per GEMM shape")
    ap.add_argument("--no-pipeline", action="store_true", help="prefill through the HF forward")
    ap.add_argument("--save", default=None, help="write last-token logits and tokens per prompt (npz)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t = time.time()
    tok, model, gemms = load(args.backend)
    print(f"[{args.backend}] loaded in {time.time() - t:.0f}s, {len(gemms)} NPU GEMMs", flush=True)
    saved = {}
    prefill = None
    if gemms and not args.no_pipeline:
        import pipeline
        prefill = pipeline.Prefill(model, model.layer_gemms)
    for n in args.prompt_tokens:
        ids = prompt_ids(tok, n)
        run(model, ids, 2, prefill)  # warm-up: compiles kernels / torch.compile for this shape
        for k in TIMES:
            TIMES[k] = 0
        BY_SHAPE.clear()
        if prefill is not None:
            pipeline.STATS["wait"] = 0.0
            pipeline.STATS["cpu_gemm"] = 0.0
        ttft, dec, toks, logits = run(model, ids, args.decode, prefill)
        P = ids.shape[1]
        line = (f"[{args.backend}] prompt {P} tok: TTFT {ttft * 1e3:.0f} ms = {P / ttft:.0f} tok/s prefill; "
                f"decode {args.decode} tok at {dec:.1f} tok/s")
        if gemms:
            line += f" | prefill: NPU busy {TIMES['npu'] * 1e3:.0f} ms over {TIMES['launches']} launches"
            if prefill is not None:
                line += (f", CPU waited on NPU {pipeline.STATS['wait'] * 1e3:.0f} ms"
                         f", CPU int8 GEMMs {pipeline.STATS['cpu_gemm'] * 1e3:.0f} ms")
        print(line, flush=True)
        print("   ", json.dumps(tok.decode(toks[:24]))[:160], flush=True)
        if args.shapes:
            for (Np, K), (n_l, sec) in sorted(BY_SHAPE.items(), key=lambda kv: -kv[1][1]):
                print(f"      N{Np} K{K}: {n_l} launches, {sec * 1e3:.0f} ms ({sec / n_l * 1e3:.2f} ms each)")
        saved[f"logits_{P}"] = logits.numpy()
        saved[f"toks_{P}"] = toks
    if args.save:
        import numpy as np
        np.savez(args.save, **{k: np.asarray(v) for k, v in saved.items()})


if __name__ == "__main__":
    main()
