"""NLL of one window with HF's own sdpa vs local_sdpa (both bf16 CPU), to validate local_attn."""
import torch
import torch.nn.functional as F
import gemma_npu as G

torch.set_num_threads(16)
tok, model, _ = G.load("cpu")
lm = model.model.language_model
cap = lm.config.final_logit_softcapping
ids = torch.tensor([[tok.bos_token_id] + tok(open("wikitext2_test.txt").read())["input_ids"][:2048]])


@torch.inference_mode()
def nll():
    h = lm(input_ids=ids).last_hidden_state[0, :-1]
    tgt = ids[0, 1:]
    tot = 0.0
    for r in range(0, tgt.shape[0], 256):
        lg = torch.tanh(model.lm_head(h[r:r + 256]).float() / cap) * cap
        tot += F.cross_entropy(lg, tgt[r:r + 256], reduction="sum").item()
    return tot / tgt.shape[0]


a = nll()
lm.set_attn_implementation("sdpa")
b = nll()
lm.set_attn_implementation("eager")
c = nll()
print(f"local_sdpa {a:.4f}  hf sdpa {b:.4f}  hf eager {c:.4f}")
print(repr(open("wikitext2_test.txt").read()[:300]))
