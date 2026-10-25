"""Gemma 4 E2B on vLLM's CPU backend, the same prompts as gemma_npu.py.

TTFT = a 1-token generate; decode = (n - 1) tokens / (time(n) - time(1)).
Prefix caching is off so every run prefills the whole prompt.

usage: VLLM_CPU_KVCACHE_SPACE=4 python vllm_bench.py [prompt_tokens ...]
"""
import os
import sys
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "google/gemma-4-E2B-it"
LENGTHS = [int(x) for x in sys.argv[1:]] or [190, 840, 1860, 3640]
N_DECODE = 32
TEXT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "filler.txt")).read()


def prompt_text(tok, n_tokens):
    """The user message gemma_npu.prompt_ids builds: the first n tokens of the filler."""
    return tok.decode(tok(TEXT)["input_ids"][:n_tokens]) + "\n\nSummarize the text above in detail."


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=4608, enable_prefix_caching=False,
              limit_mm_per_prompt={"image": 0, "audio": 0})

    def gen(ids, n):
        p = SamplingParams(max_tokens=n, min_tokens=n, temperature=0)
        t = time.perf_counter()
        out = llm.generate([{"prompt_token_ids": ids}], p, use_tqdm=False)
        return time.perf_counter() - t, out[0].outputs[0].token_ids

    for n in LENGTHS:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt_text(tok, n)}],
                                      add_generation_prompt=True, return_dict=True)["input_ids"]
        ids = list(ids)
        gen(ids, 2)  # warm-up
        t1 = min(gen(ids, 1)[0] for _ in range(2))
        tn, toks = gen(ids, N_DECODE)
        dec = (N_DECODE - 1) / (tn - t1)
        print(f"prompt {len(ids)}: TTFT {t1 * 1e3:.0f} ms ({len(ids) / t1:.0f} tok/s), "
              f"decode {dec:.1f} tok/s, first tokens {list(toks[:8])}", flush=True)


if __name__ == "__main__":
    main()
