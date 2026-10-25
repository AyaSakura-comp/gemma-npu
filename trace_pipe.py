"""Record a timeline of one pipelined prefill (CPU steps per micro-batch, NPU GEMMs) as JSONL."""
import json
import sys
import time

import torch
import gemma_npu as G
import pipeline

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3640
out = sys.argv[2] if len(sys.argv) > 2 else "trace_pipe.jsonl"
torch.set_num_threads(16)
tok, model, gemms = G.load("npu")
pf = pipeline.Prefill(model, model.layer_gemms)
ids = G.prompt_ids(tok, n)
pf(ids); pf(ids)
pipeline.TRACE = []
t0 = time.perf_counter()
pf(ids)
total = time.perf_counter() - t0
with open(out, "w") as f:
    for lane, label, s, e in pipeline.TRACE:
        f.write(json.dumps({"lane": lane, "label": label, "start": s - t0, "end": e - t0}) + "\n")
busy = {}
for lane, label, s, e in pipeline.TRACE:
    busy[lane] = busy.get(lane, 0) + e - s
print(f"prefill {ids.shape[1]} tok in {total * 1e3:.0f} ms; busy: " +
      ", ".join(f"{k} {v * 1e3:.0f} ms" for k, v in busy.items()))
