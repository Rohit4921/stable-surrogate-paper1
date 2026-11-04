import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise/fold"))   # result files are read and written here
import json, numpy as np, matplotlib, sys
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figures_style import *
T = json.load(open("toy_exact.json")); S = json.load(open("slice_toy.json")); N = json.load(open("nofold_check.json"))
fig, axs = plt.subplots(1, 3, figsize=(11.6, 3.5), constrained_layout=True)
ax = axs[0]
for fo, mk in zip(T["folds"], ("o", "s")):
    R = fo["rows"]; x = np.array([-0.5*np.log(q["H"]) for q in R]); g = np.array([q["laplace"] - q["exact"] for q in R])
    ax.plot(x, g, mk, ms=4, color=C_GP, label=f"mode at f ≈ {fo['f_star']:.1f}")
xx = np.linspace(0, 7, 2); ax.plot(xx, xx - 1.65, ":", color=INK2, lw=1, label="slope 1")
ax.set_xlabel("−½ log H (curvature at the mode)"); ax.set_ylabel("Laplace − exact log evidence (quadrature)")
ax.set_title("A   One observation: Laplace diverges at a fold", loc="left", fontsize=9.5, color=INK); ax.legend(fontsize=7)
ax = axs[1]
for sl, c in zip(S["slices"], (C_TP, C_T, C_GP)):
    # at the fold endpoint lam_min is zero to working precision (machine-dependent): plot it at 1e-16 for every curve
    R = sl["rows"]; x = np.array([-0.5*np.log(1e-16 if q.get("machine_dependent", q["lam"] < 1e-6) else q["lam"]) for q in R])
    g = np.array([q["laplace"] - q["exact"] for q in R])
    ax.plot(x, g, "o-", ms=3, lw=1.2, color=c, label=f"α₀={sl['a0']}, n={sl['n']}, data set {sl['rep']}")
xx = np.linspace(0, 18, 2); ax.plot(xx, xx, ":", color=INK2, lw=1, label="gap = −½ log λ_min")
ax.set_xlabel("−½ log λ_min(I + K^½ W K^½)"); ax.set_ylabel("Laplace − IS log ML")
ax.set_title("B   The three failures: path to the fold", loc="left", fontsize=9.5, color=INK); ax.legend(fontsize=6.8, loc="upper left")
ax = axs[2]
rows = N["rows"]; ns = sorted({q["n"] for q in rows})
for i, n in enumerate(ns):
    R = [q for q in rows if q["n"] == n]; rho = np.array([q["rho"] for q in R]); fold = np.array([q["lam"] < 1e-3 for q in R])
    xx = i + np.random.default_rng(i).uniform(-0.15, 0.15, len(R))
    ax.plot(xx[~fold], rho[~fold], "o", ms=3.5, color=C_GP, alpha=0.7, label="no fold" if i == 0 else None)
    if fold.any(): ax.plot(xx[fold], rho[fold], "D", ms=6, color=C_TP, label="fold (Laplace failure)" if i == 0 else None)
ax.axhline(1, color=INK2, lw=0.9, ls=":"); ax.text(len(ns) - 0.6, 1.03, "ρ = 1", fontsize=7, color=INK2, ha="right")
ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"n = {n}" for n in ns]); ax.set_ylabel("no-fold ratio ρ at the Laplace optimum")
ax.set_title("C   Folds need ρ ≥ 1; ρ falls with n", loc="left", fontsize=9.5, color=INK); ax.legend(fontsize=7, loc="upper right")
fig.savefig(_paths.paper("fig_fold.pdf")); fig.savefig(_paths.fig("fig_fold.pdf")); fig.savefig(_paths.fig("fig_fold.png"), dpi=150); print("ok")
