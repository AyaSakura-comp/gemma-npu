"""Two-lane Gantt (CPU main thread, NPU worker) of a pipeline trace, coloured by micro-batch."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

src, out = sys.argv[1], sys.argv[2]
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
t1 = float(sys.argv[4]) if len(sys.argv) > 4 else None
title = sys.argv[5] if len(sys.argv) > 5 else src
ev = [json.loads(l) for l in open(src)]
t1 = t1 or max(e["end"] for e in ev)
colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#9D755D"]
lanes = {"CPU": 1, "NPU": 0}
fig, ax = plt.subplots(figsize=(14, 3.8))
for e in ev:
    if e["end"] < t0 or e["start"] > t1:
        continue
    mb = int(e["label"].split()[0][2:])
    attn = e["lane"] == "CPU" and e["label"].endswith("o_proj")
    ax.barh(lanes[e["lane"]], (e["end"] - e["start"]) * 1e3, left=e["start"] * 1e3, height=0.6,
            color=colors[mb % len(colors)], edgecolor="black" if attn else "none", linewidth=0.6,
            hatch="//" if attn else None)
ax.set_yticks([0, 1], ["NPU (GEMMs)", "CPU (attention, norms, quant)"])
ax.set_xlim(t0 * 1e3, t1 * 1e3)
ax.set_xlabel("ms since prefill start")
n_mb = 1 + max(int(e["label"].split()[0][2:]) for e in ev)
handles = [Patch(color=colors[i], label=f"micro-batch {i}") for i in range(n_mb)]
handles.append(Patch(facecolor="white", edgecolor="black", hatch="//", label="CPU: attention"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=5, frameon=False)
ax.set_title(title)
fig.tight_layout()
fig.savefig(out, dpi=130)
