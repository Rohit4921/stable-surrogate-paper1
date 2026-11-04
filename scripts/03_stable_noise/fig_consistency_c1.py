"""Figure 1 (phase C): fitted alpha from the unified Laplace-ML study (ml_unified_a*_c1.json), n = 30 ... 1,920."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('stable_noise'))
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figures_style import *
rows = [o for f in sorted(glob.glob("ml_unified_a*_c1.json")) for o in json.load(open(f))]
A0 = [1.3, 1.7, 2.0]; NS = [30, 60, 120, 240, 1920]
fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
cols = {1.3: C_TP, 1.7: C_T, 2.0: C_GP}
for k, a0 in enumerate(A0):
    for j, n in enumerate(NS):
        v = np.array([r["ml"]["alpha"] for r in rows if r["a0"] == a0 and r["n"] == n])
        if not len(v): continue
        xx = j + (k - 1) * 0.24 + np.random.default_rng(k * 10 + j).uniform(-0.06, 0.06, len(v))
        ax.plot(xx, v, "o", ms=2.4, color=cols[a0], alpha=0.45, mew=0, label=f"true α = {a0}" if j == 0 else None)
        ax.plot([j + (k - 1) * 0.24 - 0.09, j + (k - 1) * 0.24 + 0.09], [np.median(v)] * 2, color=cols[a0], lw=2.4)
    ax.axhline(min(a0, 1.98), color=cols[a0], lw=0.8, ls=":")
ax.set_xticks(range(len(NS))); ax.set_xticklabels([f"n = {n:,}" for n in NS])
ax.set_ylabel("fitted α (Laplace ML)"); ax.set_ylim(0.75, 2.02)
None
ax.legend(fontsize=7.5, loc="lower right")
fig.savefig(_paths.paper("fig_consistency.pdf")); fig.savefig(_paths.fig("fig_consistency.png"), dpi=150); print("ok")
