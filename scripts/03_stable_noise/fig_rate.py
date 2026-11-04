import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise"))   # result files are read and written here
import json, numpy as np, matplotlib, sys, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figures_style import *
R = json.load(open("rate_study.json")); ML = json.load(open("ml_unified_summary.json"))   # phase C unified Laplace-ML study (median/MAD, alpha <= 1.98, two starts)
cols = {1.3: C_TP, 1.7: C_T, 2.0: C_GP}
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True)
for a0 in (1.3, 1.7, 2.0):
    for design, ls in (("stratified", "-"), ("equispaced", "--")):
        rs = [r for r in R if r["a0"] == a0 and r["design"] == design]
        n = np.array([r["n"] for r in rs]); rm = np.array([r["rmse"] for r in rs]); sm = np.array([r["sqrt_m_rmse"] for r in rs])
        lab = f"α₀ = {a0}, " + ("first differences, stratified design" if design == "stratified" else "second differences, equispaced design")
        axs[0].plot(n, rm, ls, color=cols[a0], marker="o" if ls == "-" else "s", ms=3.5, label=lab)
        axs[1].plot(n, sm, ls, color=cols[a0], marker="o" if ls == "-" else "s", ms=3.5)
        if a0 < 2 and design == "stratified":
            axs[0].plot(n, [r["asym_sd"] for r in rs], ":", color=cols[a0], lw=1.2)
            axs[1].axhline(rs[-1]["asym_sd"] * np.sqrt(rs[-1]["m"]), color=cols[a0], lw=1.0, ls=":")
    if a0 < 2:
        ml = [r for r in ML if r["a0"] == a0]
        nm = np.array([r["n"] for r in ml])
        axs[0].plot(nm, [r["ml_rmse"] for r in ml], "-", color=cols[a0], marker="x", ms=7, mew=1.8, lw=0.8,
                    label=f"α₀ = {a0}, Laplace ML (100 data sets per n)")
        axs[2].errorbar(nm, [r["ml_sqrt_n_rmse"] for r in ml], yerr=np.abs(np.array([r["ml_sqrt_n_rmse_ci"] for r in ml]).T - [r["ml_sqrt_n_rmse"] for r in ml]),
                        fmt="-", color=cols[a0], marker="x", ms=7, mew=1.8, lw=1.2, elinewidth=0.8, capsize=2)
        axs[2].plot(nm, [r["ecf_sqrt_n_rmse"] for r in ml], "-", color=cols[a0], marker="o", ms=3.5, lw=1.0, alpha=0.6)
        axs[2].axhline(ml[0]["oracle_sqrt_n_sd"], color=cols[a0], lw=1.0, ls="-.")
        axs[2].axhline(ml[0]["ecf_delta_sqrt_n"], color=cols[a0], lw=1.0, ls=":")
for ax in axs: ax.set_xscale("log"); ax.set_xlabel("n"); ax.set_xlim(22, 10500)
axs[0].set_yscale("log"); axs[0].set_ylabel("RMSE of α̂")
axs[0].set_title("A   RMSE; dotted: delta-method sd", loc="left", fontsize=9.5, color=INK)
axs[1].set_ylabel("√m × RMSE"); axs[1].set_ylim(0, 2.4)
axs[1].set_title("B   √m × RMSE levels off: √n rate", loc="left", fontsize=9.5, color=INK)
axs[2].set_ylabel("√n × RMSE"); axs[2].set_ylim(0, 3.2)
axs[2].set_title("C   Laplace ML (×, 95% bootstrap bars) and difference\n     estimator (o), same data; dash-dot: known-f bound", loc="left", fontsize=9.5, color=INK)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=7)
fig.savefig(_paths.paper("fig_rate.pdf")); fig.savefig(_paths.fig("fig_rate.pdf")); fig.savefig(_paths.fig("fig_rate.png"), dpi=150); print("ok")
