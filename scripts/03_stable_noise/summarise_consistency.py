import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('stable_noise'))   # result files are read and written here
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figures_style import *
rows = []
for f in glob.glob("consistency_a*_small.json") + glob.glob("consistency_a*_big.json") + glob.glob("consistency_a*_big2.json"):
    rows += json.load(open(f))
A0 = [1.3, 1.7, 2.0]; NS = [30, 60, 120, 240]
tab = []
for a0 in A0:
    for n in NS:
        v = np.array([r["a_hat"] for r in rows if r["a0"] == a0 and r["n"] == n])
        if len(v): tab.append((a0, n, len(v), np.median(v), np.mean(v), np.sqrt(np.mean((v-a0)**2)), np.mean(v >= 1.999)))
print("a0    n  reps  median  mean   RMSE   P(a_hat=2)")
for t in tab: print("%.1f %4d %4d   %.3f  %.3f  %.3f  %.2f" % t)
json.dump([dict(a0=t[0], n=t[1], reps=t[2], median=t[3], mean=t[4], rmse=t[5], p2=t[6]) for t in tab], open("consistency_table.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
cols = {1.3: C_TP, 1.7: C_T, 2.0: C_GP}
for k, a0 in enumerate(A0):
    for j, n in enumerate(NS):
        v = np.array([r["a_hat"] for r in rows if r["a0"] == a0 and r["n"] == n])
        if not len(v): continue
        xx = j + (k-1)*0.22 + np.random.default_rng(k*10+j).uniform(-0.05, 0.05, len(v))
        ax.plot(xx, v, "o", ms=3, color=cols[a0], alpha=0.55, label=f"true α = {a0}" if j == 0 else None)
        ax.plot([j+(k-1)*0.22-0.08, j+(k-1)*0.22+0.08], [np.median(v)]*2, color=cols[a0], lw=2.4)
    ax.axhline(a0, color=cols[a0], lw=0.8, ls=":")
ax.set_xticks(range(len(NS))); ax.set_xticklabels([f"n = {n}" for n in NS])
ax.set_ylabel("fitted α (noise level, M2)"); ax.set_ylim(0.75, 2.08)
ax.set_title("GP with symmetric α-stable noise: α̂ concentrates on the truth", loc="left", fontsize=9.5, color=INK)
ax.legend(fontsize=7.5, loc="lower right")
fig.savefig(_paths.fig("fig_consistency_v1.png"), dpi=150)   # superseded by fig_consistency_c1.py (phase C)
