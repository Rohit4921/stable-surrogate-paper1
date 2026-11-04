import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('erp_demo'))   # result files are read and written here
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figures_style import *

f = lambda x: x*np.sin(10*x)+np.cos(2*x); FST = f(np.linspace(0, 1.2, 401)).max()
S = json.load(open("results_synthetic_validation.json")); N = json.load(open("results_nifty.json"))
R2 = lambda r: "GP-t" if (r["GP-t"]["a"] and r["GP-t"]["b"]) else "GP"
scen = ["clean", "contaminated", "t-noise"]; lab = ["Gaussian noise", "4 random outliers", "$t_{1.5}$ noise"]
fig, axs = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True, gridspec_kw={"width_ratios": [0.8, 1.35, 0.9]})

ax = axs[0]
for i, s in enumerate(scen):
    rs = [r for r in S if r["scen"] == s]
    for j, (rule, c, h) in enumerate((("R1", "#9fc3ee", None), ("R2", C_T, None))):
        ch = [r["choice"] for r in rs] if rule == "R1" else [R2(r) for r in rs]
        v = 100*ch.count("GP-t")/len(rs)
        ax.bar(i + (j-0.5)*0.34, v, 0.32, color=c, label=("R1: fit + next-query veto" if rule == "R1" else "R2: fit only") if i == 0 else None)
        ax.text(i + (j-0.5)*0.34, v + 2, f"{v:.0f}", ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(range(3)); ax.set_xticklabels(["Gaussian\nnoise", "4 random\noutliers", "$t_{1.5}$\nnoise"])
ax.set_ylabel("% of 30 data sets choosing GP-t"); ax.set_ylim(0, 100)
ax.set_title("(a) How often GP-t is chosen", loc="left", fontsize=9.5, color=INK); ax.legend(fontsize=7, loc="upper left")

ax = axs[1]
groups = (("always GP", C_GP, lambda r: "GP"), ("always GP-t", C_T, lambda r: "GP-t"),
          ("rule R1", "#9fc3ee", lambda r: r["choice"]), ("rule R2", INK2, R2))
for i, s in enumerate(scen):
    rs = [r for r in S if r["scen"] == s]
    for j, (nm, c, pick) in enumerate(groups):
        v = np.array([FST - r[pick(r)]["f_argmax_mean"] for r in rs]); pos = i + (j-1.5)*0.2
        xx = pos + np.random.default_rng(10*i+j).uniform(-0.05, 0.05, len(v))
        ax.plot(xx, np.maximum(v, 1e-4), "o", ms=2.4, color=c, alpha=0.55, label=nm if i == 0 else None)
        ax.plot([pos - 0.08, pos + 0.08], [max(np.median(v), 1e-4)]*2, color=c, lw=2.4)
ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(lab)
ax.set_ylabel("regret  f* − f(recommended x)")
ax.set_title("(b) Quality of the final recommendation (bars = medians)", loc="left", fontsize=9.5, color=INK)
ax.legend(fontsize=7, loc="upper left", ncol=4, columnspacing=0.8, handletextpad=0.2)
lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi*8)

ax = axs[2]
for i, n in enumerate((15, 30, 60)):
    rs = [r for r in N if r["n"] == n]
    nu = np.array([r["GP-t"]["nu"] for r in rs]); ch = np.array([R2(r) for r in rs])
    xx = i + np.random.default_rng(i).uniform(-0.13, 0.13, len(nu))
    ax.plot(xx[ch == "GP"], nu[ch == "GP"], "o", ms=4, color=C_GP, alpha=0.8, label="R2 keeps GP" if i == 0 else None)
    ax.plot(xx[ch == "GP-t"], nu[ch == "GP-t"], "D", ms=4.5, color=C_T, label="R2 picks GP-t" if i == 0 else None)
ax.axhline(30, color=INK2, lw=0.8, ls=":"); ax.text(-0.35, 34, "ν = 30", fontsize=7, color=INK2)
ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(["n = 15", "n = 30", "n = 60"])
ax.set_ylabel("GP-t fitted ν"); ax.set_xlim(-0.45, 2.45)
ax.set_title("(c) NIFTY 50: 20 data sets per n", loc="left", fontsize=9.5, color=INK); ax.legend(fontsize=7, loc="lower right")
fig.savefig(_paths.fig("fig4_replicates.pdf")); fig.savefig(_paths.fig("fig4_replicates.png"), dpi=150)
