"""Prefill (TTFT) and decode speed of an OpenAI-compatible server, streaming.

usage: python bench_flm.py PORT MODEL [prompt_tokens ...]
"""
import json
import sys
import time
import urllib.request

PORT, MODEL = sys.argv[1], sys.argv[2]
LENGTHS = [int(x) for x in sys.argv[3:]] or [190, 840, 1860, 3640]
TEXT = open("/home/chihmin/src/gemma-npu/filler.txt").read()
_tok = None


def prompt_text(n_tokens):
    """The same user message gemma_npu.prompt_ids builds: the first n tokens of the filler."""
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    return _tok.decode(_tok(TEXT)["input_ids"][:n_tokens]) + "\n\nSummarize the text above in detail."


def run(n_tokens, max_tokens=128):
    prompt = prompt_text(n_tokens)
    body = json.dumps({"model": MODEL, "stream": True, "max_tokens": max_tokens, "temperature": 0,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    n = 0
    usage = None
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                n += 1
                if first is None:
                    first = time.perf_counter()
    t1 = time.perf_counter()
    p = usage["prompt_tokens"] if usage else None
    c = usage["completion_tokens"] if usage else n
    return p, c, first - t0, (c - 1) / (t1 - first)


for L in LENGTHS:
    run(L, 8)  # warm-up
    p, c, ttft, dec = run(L, 65)
    print(f"prompt {p} tok: TTFT {ttft * 1e3:.0f} ms = {p / ttft:.0f} tok/s prefill; decode {c} tok at {dec:.1f} tok/s",
          flush=True)
