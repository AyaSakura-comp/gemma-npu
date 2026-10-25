"""torch.profiler breakdown of one NPU prefill (CPU ops by self time)."""
import sys
import torch
import gemma_npu as G

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3640
torch.set_num_threads(16)
tok, model, gemms = G.load("npu")
ids = G.prompt_ids(tok, n)
G.run(model, ids, 1)
with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
    model(input_ids=ids, use_cache=True, logits_to_keep=1)
print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=25, max_name_column_width=60))
