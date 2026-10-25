"""Overlap summary of a pipeline trace (trace_pipe.py output)."""
import collections
import json
import sys

ev = [json.loads(l) for l in open(sys.argv[1])]
cpu = [e for e in ev if e["lane"] == "CPU"]
npu = [e for e in ev if e["lane"] == "NPU"]
T = max(e["end"] for e in ev)


def gaps(spans):
    spans = sorted((s["start"], s["end"]) for s in spans)
    t, out = 0.0, []
    for s, e in spans:
        if s > t:
            out.append((t, s))
        t = max(t, e)
    if T > t:
        out.append((t, T))
    return out


def by_label(spans, key, within=None):
    d = collections.defaultdict(float)
    for e in spans:
        if within is None:
            d[key(e)] += e["end"] - e["start"]
        else:
            for a, b in within:
                o = min(b, e["end"]) - max(a, e["start"])
                if o > 0:
                    d[key(e)] += o
    return {k: round(v * 1e3) for k, v in sorted(d.items(), key=lambda kv: -kv[1])}


cpu_key = lambda e: e["label"].split("-> ")[1]
npu_key = lambda e: e["label"].split(" ", 1)[1]
ng, cg = gaps(npu), gaps(cpu)
print(f"total {T * 1e3:.0f} ms; NPU idle {sum(b - a for a, b in ng) * 1e3:.0f} ms, "
      f"CPU idle {sum(b - a for a, b in cg) * 1e3:.0f} ms, both idle {sum(b - a for a, b in gaps(cpu + npu)) * 1e3:.0f} ms")
print("CPU work (by the GEMM it leads to):", by_label(cpu, cpu_key))
print("  of which while the NPU idles:   ", by_label(cpu, cpu_key, ng))
print("NPU work:", by_label(npu, npu_key))
print("  of which while the CPU idles:", by_label(npu, npu_key, cg))
