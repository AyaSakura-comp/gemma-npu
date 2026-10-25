"""Prefill attention for Gemma 4 on the CPU that does not compute what the mask discards.

HF's sdpa path hands every layer a dense [T, T] mask. For the sliding-window
layers (28 of 35, window 512) that makes the CPU flash kernel compute all T x T
scores and throw most away -- at T = 3.6k that was 3.1 s of a 4.3 s prefill.
Here, during prefill:

  * full-attention layers use sdpa with is_causal (the kernel skips the upper
    triangle) and grouped-query heads, without repeating K/V;
  * sliding layers split the queries into blocks of BLOCK and give each block
    only the keys its window can reach -- the W keys before it and its own
    block -- as one batched sdpa call. BLOCK 128 measured fastest (T = 3658,
    head_dim 256: 14 ms against 26 ms at 512 and 94 ms dense).

Anything else (decode, or a prompt continuing a cache) goes to HF's sdpa.
Registered as attn_implementation "local_sdpa".
"""
import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

BLOCK = 128
_window_masks = {}


def _window_mask(nb, B, W, device):
    """[nb, 1, B, W + B] bool: block b's query i (position b*B + i) may see key j
    (position b*B - W + j) iff 0 <= key position, key <= query, query - key < W."""
    key = (nb, B, W)
    m = _window_masks.get(key)
    if m is None:
        q = torch.arange(nb)[:, None, None] * B + torch.arange(B)[None, :, None]
        k = torch.arange(nb)[:, None, None] * B - W + torch.arange(W + B)[None, None, :]
        m = ((k >= 0) & (k <= q) & (q - k < W))[:, None].to(device)
        _window_masks[key] = m
    return m


_flash_cpu = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu


def _split_kv_decode(query, key, value, scale):
    """One query per head against one KV head: [1, H, 1, D] -> [1, 1, H, D].

    The CPU flash kernel parallelises over (batch, head, query block), which
    for a single query is 8 tasks on a 16-core part -- 2.6 ms at 3.6k keys.
    With one KV head the heads are just 8 queries against the same keys, so
    they go in as rows, and the keys are split S ways into the batch
    (flash-decoding); the S partial results are merged by their logsumexp.
    0.18 ms at 3.6k keys, same error against fp32.
    """
    _, H, _, D = query.shape
    T = key.shape[2]
    S = max(1, min(16, T // 256))
    L = -(-T // S)
    pad = S * L - T
    q = query.view(1, 1, H, D).expand(S, 1, H, D)
    k = F.pad(key, (0, 0, 0, pad)).view(S, 1, L, D)
    v = F.pad(value, (0, 0, 0, pad)).view(S, 1, L, D)
    mask = None
    if pad:
        mask = torch.zeros(S, 1, 1, L, dtype=query.dtype)
        mask[-1, ..., L - pad:] = float("-inf")
    out, lse = _flash_cpu(q, k, v, 0.0, False, attn_mask=mask, scale=scale)
    w = torch.softmax(lse.float(), dim=0)[..., None]  # [S, 1, H, 1]
    return (out.float() * w).sum(0, keepdim=True).to(query.dtype)


def local_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
               sliding_window=None, **kwargs):
    B_, H, T, D = query.shape
    if B_ == 1 and T == 1:
        # Decode: the one query sees the whole cache, or its last W entries.
        # No mask, so no K/V repeated per head (HF's path copies the cache
        # 8x per layer per token whenever it passes a mask).
        if sliding_window and key.shape[2] > sliding_window:
            key, value = key[:, :, -sliding_window:], value[:, :, -sliding_window:]
        if key.shape[1] == 1:
            return _split_kv_decode(query, key, value, scaling), None
        out = F.scaled_dot_product_attention(query, key, value, scale=scaling, enable_gqa=True)
        return out.transpose(1, 2).contiguous(), None
    if B_ != 1 or key.shape[2] != T:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                      scaling=scaling, **kwargs)
    if not sliding_window or T <= sliding_window:
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scaling, enable_gqa=True)
        return out.transpose(1, 2).contiguous(), None

    W, Bq = sliding_window, BLOCK
    nb = -(-T // Bq)
    KV = key.shape[1]
    pad_q = nb * Bq - T
    q = F.pad(query, (0, 0, 0, pad_q))  # [1, H, nb*B, D]
    q = q.view(H, nb, Bq, D).transpose(0, 1)  # [nb, H, B, D]

    def windows(x):
        # [1, KV, T, D] -> [nb, KV, W + B, D]: block b gets keys b*B - W .. b*B + B - 1.
        x = F.pad(x, (0, 0, W, pad_q))[0]  # [KV, W + nb*B, D]
        return x.unfold(1, W + Bq, Bq).permute(1, 0, 3, 2)

    k, v = windows(key), windows(value)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=_window_mask(nb, Bq, W, q.device),
                                         scale=scaling, enable_gqa=KV != H)
    out = out.transpose(0, 1).reshape(1, H, nb * Bq, D)[:, :, :T]
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("local_sdpa", local_sdpa)
ALL_MASK_ATTENTION_FUNCTIONS.register("local_sdpa", sdpa_mask)
