"""torch.profiler breakdown of one pipelined NPU prefill (main-thread CPU ops)."""
import sys
import torch
import gemma_npu as G
import pipeline

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3640
torch.set_num_threads(16)
tok, model, gemms = G.load("npu")
pf = pipeline.Prefill(model, model.layer_gemms)
ids = G.prompt_ids(tok, n)
pf(ids); pf(ids)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
    pf(ids)
print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=22, max_name_column_width=60))
