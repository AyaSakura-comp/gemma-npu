"""Teacher-forced NLL of Gemma 4 E2B on fixed token windows: HF bf16 on the CPU vs the NPU W8A8 path.

usage: python eval_ppl.py cpu|npu OUT.npz
Windows: the first 3 x 2048 tokens of filler.txt (technical notes, zh/en) and of
wikitext-2 test (en).
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

import gemma_npu as G

backend, out = sys.argv[1], sys.argv[2]
WIN, N_WIN = 2048, 3
torch.set_num_threads(16)
tok, model, gemms = G.load(backend)
lm = model.model.language_model
cap = lm.config.final_logit_softcapping


def windows():
    # wikitext2_test.txt: the raw wikitext-2 test split, joined with newlines.
    texts = {"notes": open("filler.txt").read(), "wikitext": open("wikitext2_test.txt").read()}
    for name, text in texts.items():
        ids = tok(text)["input_ids"]
        for w in range(N_WIN):
            chunk = ids[w * WIN: (w + 1) * WIN]
            if len(chunk) == WIN:
                yield name, torch.tensor([[tok.bos_token_id] + chunk])


@torch.inference_mode()
def hidden(ids):
    if backend == "npu":
        import pipeline
        if not hasattr(hidden, "pf"):
            hidden.pf = pipeline.Prefill(model, model.layer_gemms)
        return hidden.pf(ids, all_hidden=True)[0]
    return lm(input_ids=ids).last_hidden_state[0]


@torch.inference_mode()
def nll(h, ids):
    tgt = ids[0, 1:]
    out = []
    for r in range(0, tgt.shape[0], 256):
        logits = model.lm_head(h[r: r + 256][None])[0].float()
        logits = torch.tanh(logits / cap) * cap
        out.append(F.cross_entropy(logits, tgt[r: r + 256], reduction="none"))
    return torch.cat(out).numpy(), None


res = {}
for name, ids in windows():
    h = hidden(ids)[:-1]
    l, _ = nll(h, ids)
    key = f"{name}_{len([k for k in res if k.startswith(name)])}"
    res[key] = l
    print(f"[{backend}] {key}: mean NLL {l.mean():.4f}, ppl {np.exp(l.mean()):.3f}", flush=True)
np.savez(out, **res)
