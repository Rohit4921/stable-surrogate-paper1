import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('stable_noise'))   # result files are read and written here
import json, glob, numpy as np, matplotlib, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figures_style import *
R = [o for f in sorted(glob.glob("exact3_a*_main.json")) for o in json.load(open(f))]
M = [("laplace", "Laplace", C_GP, "o"), ("vb", "VB bound", C_TP, "s"), ("exact", "IS likelihood", C_T, "D")]
cells = sorted({(o["a0"], o["n"]) for o in R})
tab = []
for a0, n in cells:
    rs = [o for o in R if o["a0"] == a0 and o["n"] == n]
    row = dict(a0=a0, n=n, reps=len(rs))
    for k, *_ in M:
        a = np.array([o[k]["alpha"] for o in rs])
        row[k] = dict(median=float(np.median(a)), mean=float(a.mean()), rmse=float(np.sqrt(np.mean((a - a0) ** 2))),
                      p2=float(np.mean(a > 1.995)))
    # approximation error at the method's own optimum (own value - exact value at the same point)
    row["lap_err"] = [o["laplace"]["own"] - o["laplace"]["exact"] for o in rs]
    row["vb_err"] = [o["vb"]["own"] - o["vb"]["exact"] for o in rs]
    # how much exact likelihood is lost by using each method's optimum instead of the exact optimum
    row["lap_loss"] = [o["exact"]["exact"] - o["laplace"]["exact"] for o in rs]
    row["vb_loss"] = [o["exact"]["exact"] - o["vb"]["exact"] for o in rs]
    G = json.load(open("gp_null.json"))
    for k in ("laplace", "exact", "vb"):
        lr = np.array([2 * (o[k]["own"] - G[f"{a0}_{n}_{o['rep']}"]) for o in rs])
        row[f"lr_{k}"] = lr.tolist(); row[f"rej_{k}"] = float(np.mean(lr > 2.71))
    row["min_ess"] = float(min(o["exact"]["ess"] for o in rs))
    tab.append(row)
    print(f"a0={a0} n={n:3d} reps={len(rs):2d} | " + " | ".join(
        f"{lab}: med {row[k]['median']:.2f} mean {row[k]['mean']:.2f} RMSE {row[k]['rmse']:.2f} P2 {row[k]['p2']:.2f}" for k, lab, *_ in M)
        + f" | reject a=2 (LR>2.71): Laplace {row['rej_laplace']:.2f} VB {row['rej_vb']:.2f} exact {row['rej_exact']:.2f}"
        + f" | Laplace err max {max(row['lap_err']):+.1f}, #>3: {sum(e > 3 for e in row['lap_err'])} | VB err max {max(row['vb_err']):+.2f} | min ESS {row['min_ess']:.0f}")
allL = np.concatenate([t["lap_err"] for t in tab]); allV = np.concatenate([t["vb_err"] for t in tab])
print(f"Laplace error: >3 nats in {np.sum(allL > 3)}/{len(allL)}, >0 in {np.sum(allL > 0.1)}; median {np.median(allL):.2f}; "
      f"VB error: max {allV.max():.3f} (should be <= 0 up to MC), median {np.median(allV):.2f}")
json.dump(tab, open("exact_table.json", "w"), indent=1)

fig, axs = plt.subplots(1, 2, figsize=(10.6, 3.7), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1]})
ax = axs[0]
lab_cells = [c for c in cells if c[0] < 2] + [c for c in cells if c[0] == 2]
for i, (a0, n) in enumerate(lab_cells):
    rs = [o for o in R if o["a0"] == a0 and o["n"] == n]
    for j, (k, lab, c, mk) in enumerate(M):
        a = np.array([o[k]["alpha"] for o in rs]); pos = i + (j - 1) * 0.25
        xx = pos + np.random.default_rng(10 * i + j).uniform(-0.06, 0.06, len(a))
        ax.plot(xx, a, mk, ms=3.6, color=c, alpha=0.75, mew=0, label=lab if i == 0 else None)
        ax.plot([pos - 0.1, pos + 0.1], [np.median(a)] * 2, color=c, lw=2.4)
    ax.plot([i - 0.42, i + 0.42], [a0, a0], color=INK, lw=0.9, ls=":")
ax.set_xticks(range(len(lab_cells))); ax.set_xticklabels([f"α₀={a0}\nn={n}" for a0, n in lab_cells])
ax.set_ylabel("fitted α"); ax.set_ylim(0.75, 2.08)
ax.set_title("(a) Fitted α by method (bars = medians, dotted = truth)", loc="left", fontsize=9.5, color=INK)
ax.legend(fontsize=7.5, loc="lower right", ncol=3, markerscale=1.4)
ax = axs[1]
for j, (key, lab, c, mk) in enumerate((("lap_err", "Laplace", C_GP, "o"), ("vb_err", "VB bound", C_TP, "s"))):
    for i, n in enumerate((30, 60)):
        e = np.concatenate([t[key] for t in tab if t["n"] == n and t["a0"] < 2])
        pos = i + (j - 0.5) * 0.36
        xx = pos + np.random.default_rng(7 * i + j).uniform(-0.08, 0.08, len(e))
        ax.plot(xx, e, mk, ms=4, color=c, alpha=0.8, mew=0, label=lab if i == 0 else None)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_yscale("symlog", linthresh=1); ax.set_yticks([-3, -1, 0, 1, 3, 10, 20]); ax.set_yticklabels(["−3", "−1", "0", "1", "3", "10", "20"]); ax.set_ylim(-4, 25); ax.set_xticks([0, 1]); ax.set_xticklabels(["n = 30", "n = 60"]); ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("approx. − IS log ML at the fitted point")
ax.set_title("(b) Error of each approximation at its own optimum", loc="left", fontsize=9.5, color=INK)
ax.legend(fontsize=7.5, loc="upper left")
fig.savefig(_paths.fig("fig_laplace_exact_v1.png"), dpi=150)   # superseded by summarise_exact_c2.py (phase C)
print("saved")
