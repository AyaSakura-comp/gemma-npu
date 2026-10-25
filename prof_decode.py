"""torch.profiler breakdown of decode steps after a long prompt."""
import sys
import torch
import gemma_npu as G

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3640
torch.set_num_threads(16)
tok, model, gemms = G.load("npu")
ids = G.prompt_ids(tok, n)
with torch.inference_mode():
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    nxt = out.logits[0, -1].argmax().view(1, 1)
    for _ in range(3):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        for _ in range(8):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
print(type(past).__name__, [type(l).__name__ for l in getattr(past, "layers", [])][:6])
print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=18, max_name_column_width=50))
