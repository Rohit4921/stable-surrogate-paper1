"""Analysis of the phase-C BO study (bo_*_c4.json). Analysis plan fixed before the runs were summarised (6 Oct 2026):

Primary outcome: final simple regret (after the last BO step), per problem and surrogate.
Secondary: mean simple regret over all steps (area under the regret curve, divided by the number of steps), and the
tail of the final regret over repetitions: 90% quantile (VaR_0.9) and the mean of the worst 10% (CVaR_0.9; see cvar()).
Comparisons: SaS against each other surrogate (GP, TP, GP-t, RCGP) on every problem, paired by repetition
(common random numbers). For each pair: wins/ties/losses (tie if |difference| < 1e-9), the median paired difference
(SaS minus other; negative favours SaS) with a 95% paired-bootstrap percentile interval (5,000 resamples), and the
two-sided Wilcoxon signed-rank p-value (zero differences dropped). p-values are Holm-adjusted over all
problem x comparator pairs of the study (family-wise level 0.05).
Equivalence ("no cost when tails are absent"): on the gauss problem only, two one-sided tests at level 0.05 for the
mean paired difference SaS minus GP in final simple regret, margin delta = 0.025 (about 1% of the range of f on
[0, 1.2]): equivalence is declared if the 90% paired-bootstrap interval of the mean difference lies inside (-delta, delta).
Writes bo_summary_c4.json.

Change of 7 Oct 2026 (after external review; decided by the authors before the new runs): the NIFTY 50 benchmark is
dropped from the study and replaced by the constrained 2-D problems gardner1 and gardner2 (Gardner et al. 2014),
50 repetitions each. The Holm family is therefore the SaS-vs-other comparisons over the nine remaining problems
(36 tests). The NIFTY runs (bo_nifty_c4.json) stay in the package; with this script's analysis they gave no
significant difference (all Holm-adjusted p = 1 in the 32-test family).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('bo'))
import json, glob, numpy as np
from scipy.stats import wilcoxon
KINDS = ["GP", "TP", "GP-t", "RCGP", "SaS"]; DELTA = 0.025; B = 5000
rng = np.random.default_rng(20261006)


def cvar(x, level=0.9):
    """Mean of the worst (1 - level) fraction of x (upper tail; regret is a loss): the integrated empirical upper-tail
    quantile, (1/(1-level)) * integral of the empirical quantile function Q_m(u) over u in (level, 1); as a weighted
    sum of the order statistics this is (1/(m(1-level))) * sum of the m(1-level) largest values. When m(1-level) is an
    integer (100 or 50 repetitions here) it is the mean of the m(1-level) largest values; otherwise the boundary
    observation gets fractional weight. (Docstring formula corrected 4 Nov 2026, after review; the calculation is unchanged.) Corrected 4 Nov 2026: earlier versions averaged all x >= the sample 0.9-quantile,
    which includes more than 10% of the runs when there are ties at the cut-off."""
    x = np.sort(np.asarray(x, float))[::-1]; m = len(x); t = m * (1 - level)
    k = int(np.floor(t + 1e-9)); frac = t - k
    tot = x[:k].sum() + (frac * x[k] if frac > 1e-9 and k < m else 0.0)
    return float(tot / t)


def boot(d, stat, B=B):
    idx = rng.integers(0, len(d), (B, len(d))); return np.array([stat(d[i]) for i in idx])


out = dict(problems={}, tests=[])
EXCLUDE = {"bo_nifty_c4.json"}
for fn in sorted(f for f in glob.glob("bo_*_c4.json") if "summary" not in f and f not in EXCLUDE):
    R = json.load(open(fn)); prob = R[0]["problem"]
    reps = sorted({o["rep"] for o in R if o["kind"] == "SaS"})
    get = {(o["kind"], o["rep"]): o for o in R}
    reps = [r for r in reps if all((k, r) in get for k in KINDS)]
    P = dict(reps=len(reps), surrogates={})
    for k in KINDS:
        fin = np.array([get[(k, r)]["regret"][-1] for r in reps]); auc = np.array([np.mean(get[(k, r)]["regret"]) for r in reps])
        q = np.quantile(fin, 0.9)
        P["surrogates"][k] = dict(median_final=float(np.median(fin)), mean_final=float(fin.mean()), mean_auc=float(auc.mean()),
                                  var90=float(q), cvar90=cvar(fin, 0.9), median_curve=np.median([get[(k, r)]["regret"] for r in reps], 0).tolist())
    S = np.array([get[("SaS", r)]["regret"][-1] for r in reps])
    for k in KINDS[:-1]:
        O = np.array([get[(k, r)]["regret"][-1] for r in reps]); d = S - O
        nz = d[np.abs(d) >= 1e-9]
        p = float(wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
        bm = boot(d, np.median)
        out["tests"].append(dict(problem=prob, other=k, n=len(d), wins=int((d < -1e-9).sum()), ties=int((np.abs(d) < 1e-9).sum()),
                                 losses=int((d > 1e-9).sum()), median_diff=float(np.median(d)),
                                 ci95=[float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))], p=p))
    if prob == "gauss":
        O = np.array([get[("GP", r)]["regret"][-1] for r in reps]); d = S - O; bm = boot(d, np.mean)
        lo, hi = float(np.quantile(bm, 0.05)), float(np.quantile(bm, 0.95))
        P["equivalence_vs_GP"] = dict(delta=DELTA, mean_diff=float(d.mean()), ci90=[lo, hi], equivalent=bool(-DELTA < lo and hi < DELTA))
    sas_log = [g for r in reps for g in get[("SaS", r)].get("sas_log", [])]
    if "n_feasible" in R[0]:
        P["feasible_found"] = {k: float(np.mean([get[(k, r)]["n_feasible"] > 0 for r in reps])) for k in KINDS}
    P["sas_fits"] = len(sas_log); P["sas_refits"] = int(sum(g["refit"] for g in sas_log)); P["sas_gap_gt5"] = int(sum(g["gap"] > 5 for g in sas_log))
    # descriptive (not part of the pre-registered tests): tail estimates after the last step, TP = GP check
    a_end = np.array([get[("SaS", r)]["tail"][-1] for r in reps]); nu_t = np.array([get[("GP-t", r)]["tail"][-1] for r in reps])
    nu_tp = np.array([get[("TP", r)]["tail"][-1] for r in reps])
    P["tails"] = dict(sas_alpha_median=float(np.median(a_end)), sas_share_alpha2=float(np.mean(a_end > 1.975)),
                      gpt_share_nu_lt30=float(np.mean(nu_t < 30)), tp_share_nu_bound=float(np.mean(nu_tp > 990)),
                      tp_equals_gp=int(sum(np.allclose(get[("TP", r)]["regret"], get[("GP", r)]["regret"], rtol=0, atol=1e-12) for r in reps)))
    out["problems"][prob] = P
# Holm over all tests
ps = np.array([t["p"] for t in out["tests"]]); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
for t, a in zip(out["tests"], adj): t["p_holm"] = float(a); t["significant"] = bool(a < 0.05)
# Post hoc (added 7 Oct 2026 after review, not part of the pre-registered analysis; separate generator so the
# pre-registered numbers above are unchanged): smallest equivalence margin for SaS vs each comparator, i.e. the
# smallest delta for which the 90% paired-bootstrap interval of the mean difference in final regret lies in (-delta, delta).
rng2 = np.random.default_rng(20261007)
EXCLUDE = {"bo_nifty_c4.json"}
for fn in sorted(f for f in glob.glob("bo_*_c4.json") if "summary" not in f and f not in EXCLUDE):
    R = json.load(open(fn)); prob = R[0]["problem"]; get = {(o["kind"], o["rep"]): o for o in R}
    reps = [r for r in sorted({o["rep"] for o in R if o["kind"] == "SaS"}) if all((k, r) in get for k in KINDS)]
    S = np.array([get[("SaS", r)]["regret"][-1] for r in reps]); out["problems"][prob]["equivalence_margin_posthoc"] = {}
    for k in KINDS[:-1]:
        d = S - np.array([get[(k, r)]["regret"][-1] for r in reps]); idx = rng2.integers(0, len(d), (B, len(d)))
        bm = d[idx].mean(1); lo, hi = np.quantile(bm, [0.05, 0.95])
        out["problems"][prob]["equivalence_margin_posthoc"][k] = dict(mean_diff=float(d.mean()), ci90=[float(lo), float(hi)],
                                                                      delta_min=float(max(abs(lo), abs(hi))))
# Robustness check (added 8 Oct 2026 after review, not pre-registered): exact two-sided paired sign test on
# wins vs losses (ties dropped), Holm-adjusted over the same family, because final-regret differences are skewed
# and tied and need not be symmetric about their median, as the signed-rank test assumes.
from scipy.stats import binomtest
ps2 = np.array([binomtest(t["wins"], t["wins"] + t["losses"], 0.5).pvalue if t["wins"] + t["losses"] > 0 else 1.0 for t in out["tests"]])
order = np.argsort(ps2); adj2 = np.empty(m); run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * ps2[i])); adj2[i] = run
for t, p_, a_ in zip(out["tests"], ps2, adj2):
    t["sign_test"] = dict(p=float(p_), p_holm=float(a_), significant=bool(a_ < 0.05))
json.dump(out, open("bo_summary_c4.json", "w"), indent=1)
for prob, P in out["problems"].items():
    print(f"{prob} ({P['reps']} reps): " + "  ".join(f"{k} {v['median_final']:.4f}/{v['cvar90']:.3f}" for k, v in P["surrogates"].items())
          + (f"  | equivalence vs GP: {P['equivalence_vs_GP']}" if "equivalence_vs_GP" in P else ""))
    print(f"    tails {P['tails']}  SaS fits {P['sas_fits']} refits {P['sas_refits']} gap>5 {P['sas_gap_gt5']}")
    print("    post hoc smallest equivalence margin: " + "  ".join(f"vs {k} {v['delta_min']:.4f}" for k, v in P["equivalence_margin_posthoc"].items()))
for t in out["tests"]:
    print(f"  {t['problem']:8s} SaS vs {t['other']:5s}: W/T/L {t['wins']}/{t['ties']}/{t['losses']}  median diff {t['median_diff']:+.4f} "
          f"[{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]  p {t['p']:.2g}  Holm {t['p_holm']:.2g}{' *' if t['significant'] else ''}"
          f"  | sign test p {t['sign_test']['p']:.2g} Holm {t['sign_test']['p_holm']:.2g}{' *' if t['sign_test']['significant'] else ''}")
