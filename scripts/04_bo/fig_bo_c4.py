"""Figure 5 (phase C): median simple regret against the number of evaluations, per problem and surrogate
(bo_summary_c4.json, from summarise_bo_c4.py)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('bo'))
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figures_style import *
S = json.load(open("bo_summary_c4.json"))
ORDER = ["gauss", "outliers", "sas13", "t2", "skewout", "branin2", "gardner1", "gardner2", "hart6"]
TITLE = {"gauss": "Gaussian noise", "outliers": "15% gross outliers", "sas13": "SαS noise, α = 1.3", "t2": "t₂ noise (misspecified)",
         "skewout": "one-sided outliers (misspecified)", "nifty": "NIFTY 50 (illustrative)", "branin2": "Branin, 2-D, SαS(1.3)", "hart6": "Hartmann, 6-D, SαS(1.3)",
         "gardner1": "Gardner 1, 2-D, constrained, SαS(1.3)", "gardner2": "Gardner 2, 2-D, constrained, SαS(1.3)"}
STY = {"GP": (C_GP, "-"), "TP": (INK2, ":"), "GP-t": (C_TP, "--"), "RCGP": ("#8c6bb1", "-."), "SaS": (C_T, "-")}
N0 = {"branin2": 10, "gardner1": 10, "gardner2": 10, "hart6": 20}
TITLE.update({"gardner1": "Gardner 1 (constrained)", "gardner2": "Gardner 2 (constrained)", "skewout": "one-sided outliers",
              "t2": "t₂ noise", "branin2": "Branin, 2-D", "hart6": "Hartmann, 6-D", "sas13": "SαS(1.3) noise"})


def panel_figure(probs, nrow, ncol, size, out_pdf, out_png):
    """Figure 5 of the paper (six 1-D/2-D problems) and Figure S1 of the supplement (constrained and 6-D problems);
    split after review for legibility."""
    fig, axs = plt.subplots(nrow, ncol, figsize=size, constrained_layout=True, squeeze=False)
    for j, (ax, p) in enumerate(zip(axs.ravel(), probs)):
        P = S["problems"][p]
        for k, (c, ls) in STY.items():
            cur = np.array(P["surrogates"][k]["median_curve"]); n0 = N0.get(p, 8)
            ax.plot(np.arange(n0, n0 + len(cur)), np.maximum(cur, 1e-5), ls, color=c, lw=2.2 if k == "SaS" else 1.5,
                    label=k if k != "SaS" else "GP + SαS noise")
        ax.set_yscale("log"); ax.set_title(f"{'ABCDEF'[j]}   {TITLE[p]} ({P['reps']} reps)", loc="left", fontsize=11.5, color=INK)
        ax.set_xlabel("evaluations", fontsize=11); ax.tick_params(labelsize=10)
    for ax in axs.ravel()[len(probs):]: ax.axis("off")
    for a in axs[:, 0]: a.set_ylabel("median simple regret", fontsize=11)
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="outside upper center", ncol=5, fontsize=11, frameon=False)
    fig.savefig(_paths.paper(out_pdf)); fig.savefig(_paths.fig(out_png), dpi=150)


panel_figure(["gauss", "outliers", "skewout", "sas13", "t2", "branin2"], 2, 3, (12, 7.4), "fig_bo_study.pdf", "fig_bo_study.png")
panel_figure(["gardner1", "gardner2", "hart6"], 1, 3, (12, 4.0), "fig_bo_study_supp.pdf", "fig_bo_study_supp.png")
print("ok")
