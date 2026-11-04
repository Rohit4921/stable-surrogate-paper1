"""Fold diagnostic at each Laplace (and exact) optimum of the exact-likelihood study (Section 4.3).

At a fold (lam_min = 0 to working precision) the posterior mode, and hence the standard Laplace value, depends
on where the Newton iterate stops, which differs across machines (README note 6). So:
  * 'laplace' and 'gap' use the value stored by the study itself (exact3_*.json: the optimiser's own value at
    the Laplace optimum, and the Laplace value at the exact optimum), which is the number reported in the paper;
  * 'laplace_recomputed' is the value recomputed here from a cold start (NaN if det(I + K W) <= 0);
  * 'machine_dependent' flags lam_min < 1e-6, where the recomputed value (and lam_min itself) are not portable.
lam_min, n_neg and maxz classify the fits; the Fisher-type value is always defined.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise/fold"))   # result files are read and written here
import json, glob, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
import exact_m2 as X, fold_remedy as remedy
from stable_noise_gp import rsas, lik
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
rows = []
for fn in sorted(glob.glob("../exact3_a*_main.json")):
    for o in json.load(open(fn)):
        a0, n, rep = o["a0"], o["n"], o["rep"]
        r = np.random.default_rng(int(a0*1000)+10*n+rep)
        x = np.sort((np.arange(n)+r.uniform(size=n))/n*1.2); y = f(x)+0.05*rsas(a0, n, r)
        E = X.ExactM2(x, y, S=10); c = -n*np.log(E.ys)
        for which in ("laplace", "exact"):
            p = np.array(o[which]["p"]); ev = remedy.evaluate(E, p)
            sig, a = np.exp(p[2]), p[3]
            K, fm, _, _ = E.laplace(np.exp(p[0]), np.exp(p[1]), sig, a)
            stored = o[which]["own"] if which == "laplace" else o[which]["laplace"]
            rows.append(dict(a0=a0, n=n, rep=rep, at=which, alpha=float(a), lam_min=ev["lam_min"], n_neg=ev["n_neg"],
                             maxz=float(np.abs((E.y-fm)/sig).max()), laplace=float(stored),
                             laplace_recomputed=float(ev["standard"] + c), fisher=float(ev["fisher"] + c),
                             exact=o[which]["exact"], gap=float(stored - o[which]["exact"]),
                             gap_fisher=float(ev["fisher"] + c - o[which]["exact"]), machine_dependent=ev["machine_dependent"]))
json.dump(rows, open("diag_fold.json", "w"), indent=1)
R = [r for r in rows if r["at"] == "laplace"]
for r in sorted(R, key=lambda r: r["lam_min"])[:10]:
    rec = "undefined" if not np.isfinite(r["laplace_recomputed"]) else f"{r['laplace_recomputed'] - r['exact']:+.2f}"
    print(f"a0={r['a0']} n={r['n']} rep={r['rep']}: lam_min={r['lam_min']:.2e} negW={r['n_neg']} gap={r['gap']:+.2f} "
          f"(recomputed here: {rec}{', machine-dependent' if r['machine_dependent'] else ''}) fisher_gap={r['gap_fisher']:+.2f}")
lam = np.array([r["lam_min"] for r in R]); gap = np.array([r["gap"] for r in R]); fold = lam < remedy.LAM_MACHINE
print("fits with gap>3:", int((gap > 3).sum()), " their lam_min:", np.round(lam[gap > 3], 6))
print("folds (lam_min < 1e-6):", int(fold.sum()), "; lam_min of the other fits: min", round(lam[~fold].min(), 4), "median", round(np.median(lam[~fold]), 4))
from scipy.stats import spearmanr
print("Spearman(log lam_min, gap), non-fold fits =", round(spearmanr(np.log(lam[~fold]), gap[~fold]).statistic, 3))
F = np.array([r["gap_fisher"] for r in R]); print("Fisher-Laplace gap at Laplace optima: median", round(np.median(F), 2), "max", round(F.max(), 2), "min", round(F.min(), 2))
