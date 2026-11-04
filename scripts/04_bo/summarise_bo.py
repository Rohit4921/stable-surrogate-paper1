import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('bo'))   # result files are read and written here
import json, glob, numpy as np, matplotlib, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from figures_style import *
C_SAS = "#eda100"
KINDS = [("GP", C_GP, "-", "o"), ("TP", C_TP, "--", "s"), ("GP-t", C_T, "-", "D"), ("SaS", C_SAS, "-", "^")]
LAB = {"GP": "GP", "TP": "Student-t process", "GP-t": "GP + Student-t noise", "SaS": "GP + SαS noise"}
PROBS = [("gauss", "Gaussian noise"), ("outliers", "15% gross outliers"), ("sas13", "SαS noise, α = 1.3"), ("nifty", "NIFTY 50 look-back")]
R = {p: json.load(open(f"bo_{p}_main.json")) for p, _ in PROBS if glob.glob(f"bo_{p}_main.json")}
N0, FOLD_GAP, N_BANDS = 8, 5.0, ((8, 14), (15, 22), (23, 30))
summary = {}
for p, _ in PROBS:
    if p not in R: continue
    runs = R[p]; reps = sorted({o["rep"] for o in runs})
    by = {k: {o["rep"]: o for o in runs if o["kind"] == k} for k, *_ in KINDS}
    common = [r for r in reps if all(r in by[k] for k, *_ in KINDS)]
    scale = 100.0 if p == "nifty" else 1.0
    S = {"reps": len(common)}
    for k, *_ in KINDS:
        fin = np.array([by[k][r]["regret"][-1] for r in common]) * scale
        auc = np.array([np.mean(by[k][r]["regret"]) for r in common]) * scale
        S[k] = dict(final_median=float(np.median(fin)), final_mean=float(fin.mean()), auc_median=float(np.median(auc)),
                    tail_final=[by[k][r]["tail"][-1] for r in common])
    for k in ("TP", "GP-t", "SaS"):
        a = np.array([by[k][r]["regret"][-1] for r in common]); g = np.array([by["GP"][r]["regret"][-1] for r in common])
        S[k]["wins_vs_GP"] = int((a < g - 1e-12).sum()); S[k]["losses_vs_GP"] = int((a > g + 1e-12).sum())
        try: S[k]["wilcoxon_p"] = float(stats.wilcoxon(a, g).pvalue) if np.any(a != g) else 1.0
        except ValueError: S[k]["wilcoxon_p"] = 1.0
    a = np.array([by["SaS"][r]["regret"][-1] for r in common]); t = np.array([by["GP-t"][r]["regret"][-1] for r in common])
    S["SaS"]["wins_vs_GPt"] = int((a < t - 1e-12).sum()); S["SaS"]["losses_vs_GPt"] = int((a > t + 1e-12).sum())
    try: S["SaS"]["wilcoxon_p_vs_GPt"] = float(stats.wilcoxon(a, t).pvalue) if np.any(a != t) else 1.0
    except ValueError: S["SaS"]["wilcoxon_p_vs_GPt"] = 1.0
    logs = [e for r in common for e in by["SaS"][r].get("sas_log", [])]
    S["SaS"]["refit_rate"] = float(np.mean([e["refit"] for e in logs])); S["SaS"]["n_fits"] = len(logs)
    S["SaS"]["max_gap"] = float(max([e["gap"] for e in logs] or [np.nan]))
    # Laplace failures (folds) inside the BO loop, Section 4.4: share of SaS fits whose Laplace value exceeds the
    # exact check by more than 5 nats, overall and by number of evaluations n (fit k of a run is at n = N0 + k)
    gn = [(e["gap"], N0 + k) for r in common for k, e in enumerate(by["SaS"][r].get("sas_log", []))]
    big = [nn for g_, nn in gn if g_ > FOLD_GAP]
    S["SaS"]["fold_count"] = len(big); S["SaS"]["fold_share"] = float(len(big) / len(gn)) if gn else np.nan
    S["SaS"]["fold_share_by_n"] = {f"{lo}-{hi}": float(np.mean([g_ > FOLD_GAP for g_, nn in gn if lo <= nn <= hi]))
                                   for lo, hi in N_BANDS if any(lo <= nn <= hi for _, nn in gn)}
    S["SaS"]["fold_max_n"] = int(max(big)) if big else None
    S["TP"]["nu_at_bound"] = float(np.mean(np.array(S["TP"]["tail_final"]) > 999))
    S["GP-t"]["nu_lt30"] = float(np.mean(np.array(S["GP-t"]["tail_final"]) < 30))
    S["SaS"]["alpha_median"] = float(np.median(S["SaS"]["tail_final"])); S["SaS"]["alpha_eq2"] = float(np.mean(np.array(S["SaS"]["tail_final"]) > 1.995))
    summary[p] = S
    print(f"== {p} ({len(common)} reps; regret x{scale:g})")
    for k, *_ in KINDS:
        extra = "" if k == "GP" else f"  vs GP W/L {S[k]['wins_vs_GP']}/{S[k]['losses_vs_GP']} p={S[k]['wilcoxon_p']:.3f}"
        print(f"   {k:5s} final median {S[k]['final_median']:.4f} mean {S[k]['final_mean']:.4f}  mean-regret median {S[k]['auc_median']:.4f}{extra}")
    print(f"   TP nu at bound {S['TP']['nu_at_bound']:.2f}; GP-t nu<30 {S['GP-t']['nu_lt30']:.2f}; SaS alpha median {S['SaS']['alpha_median']:.2f}, "
          f"alpha=2 {S['SaS']['alpha_eq2']:.2f}; SaS vs GP-t W/L {S['SaS']['wins_vs_GPt']}/{S['SaS']['losses_vs_GPt']} p={S['SaS']['wilcoxon_p_vs_GPt']:.3f}; "
          f"safeguard refits {S['SaS']['refit_rate']:.3f} of {S['SaS']['n_fits']} fits (max gap {S['SaS']['max_gap']:.1f})")
nf = sum(summary[p]["SaS"]["n_fits"] for p in summary); nb = sum(summary[p]["SaS"]["fold_count"] for p in summary)
summary["fold_overall"] = dict(n_fits=nf, fold_count=nb, fold_share=float(nb / nf) if nf else np.nan, gap_threshold=FOLD_GAP)
print(f"== Laplace exceeded the exact check by > {FOLD_GAP:g} nats in {nb} of {nf} SaS fits ({100*nb/max(nf,1):.1f}%)")
for p in [q for q, _ in PROBS if q in R]:
    S = summary[p]["SaS"]
    print(f"   {p:8s} {100*S['fold_share']:.1f}%  by n: " + ", ".join(f"{k}: {100*v:.1f}%" for k, v in S["fold_share_by_n"].items())
          + f"  (largest n with a fold: {S['fold_max_n']})")
json.dump(summary, open("bo_summary.json", "w"), indent=1)

fig, axs = plt.subplots(1, 4, figsize=(12.8, 3.4), constrained_layout=True)
for ax, (p, title) in zip(axs, PROBS):
    if p not in R: ax.set_visible(False); continue
    runs = R[p]; common = summary[p]["reps"]; scale = 100.0 if p == "nifty" else 1.0
    for k, c, ls, mk in KINDS:
        M = np.array([o["regret"] for o in runs if o["kind"] == k]) * scale
        n = np.arange(8, 8 + M.shape[1])
        med = np.median(M, 0); q1, q3 = np.quantile(M, [0.25, 0.75], axis=0)
        if k in ("GP", "SaS"):
            ax.fill_between(n, np.maximum(q1, 1e-5), np.maximum(q3, 1e-5), color=c, alpha=0.13, lw=0)
        ax.plot(n, np.maximum(med, 1e-5), color=c, ls=ls, lw=2 if k in ("SaS", "GP") else 1.6, label=LAB[k])
        ax.plot(n[-1], max(med[-1], 1e-5), mk, color=c, ms=5)
    ax.set_yscale("linear" if p == "nifty" else "log"); ax.set_xlabel("evaluations n")
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
axs[0].set_ylabel("simple regret of recommendation\n(median; IQR band for GP and SαS)")
axs[3].set_ylabel("regret (% p.a.)")
axs[0].legend(fontsize=7, loc="lower left")
fig.savefig(_paths.fig("fig_bo_study_v1.png"), dpi=150)   # superseded by fig_bo_c4.py (phase C)
print("saved")
