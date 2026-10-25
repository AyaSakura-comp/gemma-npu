"""Pipelined prefill for Gemma 4 E2B: the NPU and the CPU work at the same time.

In the HF forward the two take turns -- a layer's GEMM on the NPU, then its
norms, RoPE and attention on the CPU, then the next GEMM -- so each sits idle
while the other works. Here the prompt is split into two micro-batches of rows
(A = the first rows, B = the rest) and each runs as a generator over the
layers. A generator yields at every GEMM; the scheduler hands the GEMM to the
NPU worker thread and resumes whichever micro-batch's GEMM finished, so while
the NPU runs one micro-batch's GEMM the CPU does the other's attention and
elementwise work. (The Triton launch releases the GIL, so the worker thread's
launch and the main thread's torch ops really do overlap.)

Causality makes the split cheap: A's rows only attend to A, and B's rows attend
to A and B, so B needs A's keys and values for a layer -- which A, being one
GEMM ahead, always has by then.

Between GEMMs the CPU work is a handful of torch.compile'd kernels that fuse
the int32 -> float dequant, the norms / RoPE / GELU / residuals, and the
per-token int8 quant for the next GEMM, so no activation makes a separate pass
in float.

The cache this fills is HF's DynamicCache, so decode continues through the
stock forward.
"""
import queue
import os
import threading
import time

import torch
import torch.nn.functional as F
from transformers import DynamicCache

import local_attn

ROW_STEP = int(os.environ.get("GEMMA_NPU_ROWS", "512"))  # rows per NPU launch, and per micro-batch
ATTN_ROWS = 256  # full-attention query rows per CPU step
LAG = int(os.environ.get("GEMMA_NPU_LAG", "1"))  # target lag between micro-batches, in GEMMs
STATS = {"wait": 0.0}
TRACE = None  # set to a list to record (lane, label, start, end) spans


def _rms(x, w, eps):
    x = x.float()
    y = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return y * w.float() if w is not None else y


def _quant(x):
    x = x.float()
    s = x.abs().amax(dim=-1).clamp(min=1e-8) / 127
    return torch.round(x / s[:, None]).clamp(-127, 127).to(torch.int8), s


def _deq(acc, sx, sw):
    """int32 -> the bf16 an HF Linear would have produced."""
    return (acc.float() * sx[:, None] * sw[None, :]).to(torch.bfloat16)


def _rope(x, cos, sin):
    h = x.shape[-1] // 2
    rot = torch.cat((-x[..., h:], x[..., :h]), dim=-1)
    return x * cos + rot * sin


@torch.compile(dynamic=True)
def k_in(h, w, eps: float):
    """input_layernorm -> int8 rows for q/k/v."""
    return _quant(_rms(h, w, eps).to(torch.bfloat16))


@torch.compile(dynamic=True)
def k_qkv(acc, sx, sw, cos, sin, qw, kw, eps: float, H: int, KV: int, D: int, shared: bool):
    """q/k/v GEMM out -> normed, rotated q [H, T, D], k, v [KV, T, D] (bf16)."""
    y = _deq(acc, sx, sw)
    T = y.shape[0]
    q = y[:, : H * D].reshape(T, H, D)
    q = _rope(_rms(q, qw, eps).to(torch.bfloat16), cos[:, None], sin[:, None]).transpose(0, 1)
    if shared:
        return q, q, q
    k = y[:, H * D: (H + KV) * D].reshape(T, KV, D)
    v = y[:, (H + KV) * D:].reshape(T, KV, D)
    k = _rope(_rms(k, kw, eps).to(torch.bfloat16), cos[:, None], sin[:, None]).transpose(0, 1)
    v = _rms(v, None, eps).to(torch.bfloat16).transpose(0, 1)
    return q, k, v


@torch.compile(dynamic=True)
def k_attn_out(o):
    """attention output [H, T, D] -> int8 rows for o_proj."""
    H, T, D = o.shape
    return _quant(o.transpose(0, 1).reshape(T, H * D))


@torch.compile(dynamic=True)
def k_post_attn(acc, sx, sw, res, w_post, w_pre_ff, eps: float):
    """o_proj out -> residual add -> pre-FFN norm -> int8 rows for gate/up."""
    h = res + _rms(_deq(acc, sx, sw), w_post, eps).to(torch.bfloat16)
    return (h,) + _quant(_rms(h, w_pre_ff, eps).to(torch.bfloat16))


@torch.compile(dynamic=True)
def k_gelu(acc, sx, sw, I: int):
    """gate/up out -> gelu(gate) * up -> int8 rows for down_proj."""
    y = _deq(acc, sx, sw)
    a = (F.gelu(y[:, :I], approximate="tanh") * y[:, I:]).to(torch.bfloat16)
    return _quant(a)


@torch.compile(dynamic=True)
def k_post_ff(acc, sx, sw, res, w_post_ff, eps: float):
    """down_proj out -> residual add -> int8 rows for the PLE gate."""
    h = res + _rms(_deq(acc, sx, sw), w_post_ff, eps).to(torch.bfloat16)
    return (h,) + _quant(h)


@torch.compile(dynamic=True)
def k_ple_gate(acc, sx, sw, ple):
    """PLE gate out -> gelu * this layer's per-layer input -> int8 rows."""
    g = (F.gelu(_deq(acc, sx, sw), approximate="tanh") * ple).to(torch.bfloat16)
    return _quant(g)


@torch.compile(dynamic=True)
def k_ple_out(acc, sx, sw, h, w_post_ple, scalar, eps: float):
    """PLE projection out -> residual add -> layer scalar."""
    return ((h + _rms(_deq(acc, sx, sw), w_post_ple, eps).to(torch.bfloat16)) * scalar).to(torch.bfloat16)


_flash = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu


def _merge(o1, l1, o2, l2):
    """Combine two attention partials over disjoint key sets by logsumexp."""
    m = torch.maximum(l1, l2)
    w1, w2 = torch.exp(l1 - m)[..., None], torch.exp(l2 - m)[..., None]
    return ((o1.float() * w1 + o2.float() * w2) / (w1 + w2)).to(o1.dtype)


def full_attention(q, k_all, v_all, q_start):
    """q [H, Tq, D] at positions q_start.. ; k/v [KV, q_start + Tq, D] -> [H, Tq, D]."""
    H, Tq, D = q.shape
    KV = k_all.shape[0]
    q4 = q[None]

    def heads(x):
        return x[None].expand(1, H, x.shape[1], D) if KV != H else x[None]

    kb, vb = heads(k_all[:, q_start:]), heads(v_all[:, q_start:])
    ob, lb = _flash(q4, kb.contiguous(), vb.contiguous(), 0.0, True, scale=1.0)
    if q_start == 0:
        return ob[0]
    ka, va = heads(k_all[:, :q_start]), heads(v_all[:, :q_start])
    oa, la = _flash(q4, ka.contiguous(), va.contiguous(), 0.0, False, scale=1.0)
    return _merge(oa, la, ob, lb)[0]


def sliding_attention(q, k_all, v_all, q_start, W):
    """Sliding-window attention for queries at q_start.. against keys 0..q_start+Tq."""
    H, Tq, D = q.shape
    KV = k_all.shape[0]
    Bq = local_attn.BLOCK
    assert q_start % Bq == 0
    nb = -(-Tq // Bq)
    pad_q = nb * Bq - Tq
    qb = F.pad(q, (0, 0, 0, pad_q)).view(H, nb, Bq, D).transpose(0, 1)

    def windows(x):
        # keys for block b: global positions q_start + b*B - W .. q_start + b*B + B - 1
        lo = q_start - W
        x = x[:, max(lo, 0): q_start + Tq]
        x = F.pad(x, (0, 0, max(-lo, 0), pad_q))
        return x.unfold(1, W + Bq, Bq).permute(1, 0, 3, 2)

    qpos = q_start + torch.arange(nb)[:, None, None] * Bq + torch.arange(Bq)[None, :, None]
    kpos = qpos[:, :1, :1] - W + torch.arange(W + Bq)[None, None, :]
    mask = ((kpos >= 0) & (kpos <= qpos) & (qpos - kpos < W))[:, None]
    o = F.scaled_dot_product_attention(qb, windows(k_all), windows(v_all), attn_mask=mask, scale=1.0,
                                       enable_gqa=KV != H)
    return o.transpose(0, 1).reshape(H, nb * Bq, D)[:, :Tq]


class Job:
    def __init__(self, value=None, done=False):
        self.value, self.done = value, done


class Worker:
    """Runs NPU GEMMs on a thread of its own, lowest priority value first.

    `done` is notified after every job, so the scheduler can wait for
    whichever finishes rather than for one in particular.
    """

    def __init__(self):
        self.q = queue.PriorityQueue()
        self.done = threading.Condition()
        self.seq = 0
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        import npu_kernels
        npu_kernels.use_npu()
        while True:
            _, _, fn, job, label = self.q.get()
            t = time.perf_counter()
            try:
                value = fn()
                if TRACE is not None:
                    TRACE.append(("NPU", label, t, time.perf_counter()))
            except BaseException as e:  # noqa: BLE001 -- handed to the waiter
                value = e
            with self.done:
                job.value, job.done = value, True
                self.done.notify_all()

    def submit(self, fn, priority, label=""):
        job = Job()
        self.seq += 1
        self.q.put((priority, self.seq, fn, job, label))
        return job


class Prefill:
    def __init__(self, model, layer_gemms, worker=None):
        self.model = model
        self.lm = model.model.language_model
        self.cfg = self.lm.config
        self.layer_gemms = layer_gemms
        self.worker = worker or Worker()

    def gemm(self, g, xq, sx, slot):
        """Yieldable GEMM request: int8 rows are staged now, the result comes back on resume."""
        if g.on_cpu:
            return ("cpu", torch._int_mm(xq, g.w_kn), g.name)
        T = xq.shape[0]
        g.stage(slot, T).copy_(xq)
        return ("npu", lambda: g.launch_slot(T, slot), g.name)

    def steps(self, mb, ctx):
        """One micro-batch through every layer; yields GEMM requests, receives int32 results."""
        lm, cfg = self.lm, self.cfg
        s, e, slot = mb
        eps = cfg.rms_norm_eps
        h = ctx["h0"][s:e]
        for i, layer in enumerate(lm.layers):
            G = self.layer_gemms[i]
            a = layer.self_attn
            lt = cfg.layer_types[i]
            cos, sin = ctx["rope"][lt]
            cos, sin = cos[s:e], sin[s:e]
            H, D = cfg.num_attention_heads, a.head_dim
            KV = cfg.per_layer_config[i].num_key_value_heads
            shared = a.is_kv_shared_layer

            xq, sx = k_in(h, layer.input_layernorm.weight, eps)
            acc = yield self.gemm(G["qkv"], xq, sx, slot)
            q, k, v = k_qkv(acc, sx, G["qkv"].sw, cos, sin, a.q_norm.weight,
                            None if shared else a.k_norm.weight, eps, H, KV, D, shared)
            kv = ctx["kv"]
            if shared:
                k_all, v_all = kv[("shared", lt, slot)]
            else:
                # Every earlier micro-batch is ahead of this one, so its keys
                # and values for this layer are already here.
                kv[(i, slot)] = (k, v)
                if slot == 0:
                    k_all, v_all = k, v
                else:
                    prev = [kv[(i, j)] for j in range(slot + 1)]
                    k_all = torch.cat([p[0] for p in prev], 1)
                    v_all = torch.cat([p[1] for p in prev], 1)
                if slot == ctx["last"]:
                    ctx["cache"].update(k_all[None], v_all[None], i)
                if a.store_full_length_kv:
                    kv[("shared", lt, slot)] = (k_all, v_all)
            if a.sliding_window:
                o = sliding_attention(q, k_all, v_all, s, a.sliding_window)
            else:
                # A later micro-batch's full attention is the one long CPU
                # step (up to ~45 ms against every earlier key); done whole,
                # it holds the CPU while earlier micro-batches -- and so the
                # NPU -- wait on their next short step. In row chunks, with a
                # yield between, they can cut in.
                parts = []
                for r0 in range(0, e - s, ATTN_ROWS):
                    r1 = min(r0 + ATTN_ROWS, e - s)
                    parts.append(full_attention(q[:, r0:r1], k_all[:, : s + r1], v_all[:, : s + r1], s + r0))
                    if r1 < e - s:
                        yield ("yield", None, "attention")
                o = torch.cat(parts, 1)

            xq, sx = k_attn_out(o)
            acc = yield self.gemm(G["o"], xq, sx, slot)
            h, xq, sx = k_post_attn(acc, sx, G["o"].sw, h, layer.post_attention_layernorm.weight,
                                    layer.pre_feedforward_layernorm.weight, eps)
            acc = yield self.gemm(G["gate_up"], xq, sx, slot)
            xq, sx = k_gelu(acc, sx, G["gate_up"].sw, G["gate_up"].N // 2)
            acc = yield self.gemm(G["down"], xq, sx, slot)
            h, xq, sx = k_post_ff(acc, sx, G["down"].sw, h, layer.post_feedforward_layernorm.weight, eps)
            acc = yield self.gemm(G["ple_gate"], xq, sx, slot)
            xq, sx = k_ple_gate(acc, sx, G["ple_gate"].sw, ctx["ple"][s:e, i])
            acc = yield self.gemm(G["ple_proj"], xq, sx, slot)
            h = k_ple_out(acc, sx, G["ple_proj"].sw, h, layer.post_per_layer_input_norm.weight,
                          layer.layer_scalar, eps)
        return h

    @torch.inference_mode()
    def __call__(self, ids, all_hidden=False):
        """ids [1, T] -> (last-token logits [vocab], DynamicCache).

        With all_hidden, the final-norm hidden states [T, hidden] instead of
        the logits (for scoring every position).
        """
        lm, cfg = self.lm, self.cfg
        T = ids.shape[1]
        emb = lm.embed_tokens(ids)
        ple = lm.project_per_layer_inputs(emb, lm.get_per_layer_inputs(ids, emb))[0]
        pos = torch.arange(T)[None]
        rope = {lt: tuple(t[0] for t in lm.rotary_emb(emb, pos, lt)) for lt in set(cfg.layer_types)}
        cache = DynamicCache(config=cfg)
        # One micro-batch per launch's worth of rows: the more there are, the
        # more often the NPU has another one's GEMM queued while the CPU works.
        mbs = [(s, min(s + ROW_STEP, T), j) for j, s in enumerate(range(0, T, ROW_STEP))]
        ctx = dict(h0=emb[0], ple=ple, rope=rope, kv={}, cache=cache, last=len(mbs) - 1)
        out = self.run([self.steps(mb, ctx) for mb in mbs])
        if all_hidden:
            return lm.norm(torch.cat(out)[None])[0], cache
        h = out[-1][-1:]
        h = lm.norm(h[None])
        logits = self.model.lm_head(h)[0, -1].float()
        cap = cfg.final_logit_softcapping
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return logits, cache

    def run(self, gens):
        """Drive the generators, earliest micro-batch first on both sides.

        Earlier micro-batches get the NPU first and, among those whose GEMM is
        back, the CPU first. That staggers them -- micro-batch 0 runs ahead
        into its MLP (NPU-heavy) while later ones are still in attention
        (CPU-heavy) -- instead of all of them reaching attention, and then the
        MLP, together, which leaves one side idle in each phase. Running ahead
        is always allowed: a micro-batch only needs keys from earlier ones.
        """
        results = [None] * len(gens)
        inflight = {}  # generator -> Job
        submitted = [0] * len(gens)  # NPU GEMMs per generator so far
        worker = self.worker

        def priority(i):
            # How far micro-batch i is behind its place in the wavefront:
            # each one should trail the one before by LAG GEMMs. Strict
            # earliest-first (LAG = inf) lets the first ones race ahead and
            # leaves the last ones a serial tail; LAG = 0 is lockstep, which
            # puts every micro-batch in attention, then in the MLP, together.
            return submitted[i] + i * LAG

        def advance(i, value):
            t = time.perf_counter()
            try:
                kind, job, name = gens[i].send(value)
            except StopIteration as stop:
                results[i] = stop.value
                name = None
            if TRACE is not None:
                TRACE.append(("CPU", f"mb{i} -> {name or 'end'}", t, time.perf_counter()))
            if name is None:
                return
            if kind in ("cpu", "yield"):
                inflight[i] = Job(job, done=True)
            else:
                inflight[i] = worker.submit(job, priority(i), f"mb{i} {name}")
                submitted[i] += 1

        for i in range(len(gens)):
            advance(i, None)
        while inflight:
            t = time.perf_counter()
            with worker.done:
                while not any(j.done for j in inflight.values()):
                    worker.done.wait()
            STATS["wait"] += time.perf_counter() - t
            i = min((k for k, j in inflight.items() if j.done), key=lambda k: (priority(k), k))
            value = inflight.pop(i).value
            if isinstance(value, BaseException):
                raise value
            advance(i, value)
        return results
