"""Phase C: fold diagnostic lambda_min(I + K^{1/2} W K^{1/2}) at the Laplace optimum of every data set of the phase-C
Table 3 study (exact_c2_a*_n*_c2.json; median/MAD standardisation). Writes fold/diag_fold_c2.json and prints how the
gross Laplace failures (Laplace exceeds the IS value by more than 3 nats at its own optimum) relate to folds."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise/fold"))
import json, glob, numpy as np, warnings
warnings.filterwarnings("ignore")
import exact_m2 as X, fold_remedy as remedy
from stable_noise_gp import rsas
f = lambda x: x * np.sin(10 * x) + np.cos(2 * x)
rows = []
for fn in sorted(glob.glob("../exact_c2_a*_c2.json")):
    for o in json.load(open(fn)):
        a0, n, rep = o["a0"], o["n"], o["rep"]
        r = np.random.default_rng(int(a0 * 1000) + 10 * n + rep)
        x = np.sort((np.arange(n) + r.uniform(size=n)) / n * 1.2); y = f(x) + 0.05 * rsas(a0, n, r)
        E = X.ExactM2(x, y, S=10, standardise="robust")
        ev = remedy.evaluate(E, np.array(o["laplace"]["p"]))
        rows.append(dict(a0=a0, n=n, rep=rep, lam_min=float(ev["lam_min"]), gap=float(o["laplace"]["own"] - o["laplace"]["exact"]),
                         alpha=float(o["laplace"]["alpha"])))
json.dump(rows, open("diag_fold_c2.json", "w"), indent=1)
lam = np.array([r["lam_min"] for r in rows]); gap = np.array([r["gap"] for r in rows])
fail = gap > 3; fold = lam < remedy.LAM_MACHINE
print(f"{len(rows)} fits; gross failures (gap > 3 nats): {fail.sum()}; folds (lam_min < 1e-6): {fold.sum()}; failures that are folds: {(fail & fold).sum()}")
print("lam_min of failures:", np.round(np.sort(lam[fail]), 6))
print("lam_min of non-failures: min", round(lam[~fail].min(), 4), "5% quantile", round(np.quantile(lam[~fail], 0.05), 4), "median", round(np.median(lam[~fail]), 3))
for a0 in sorted({r["a0"] for r in rows}):
    for n in sorted({r["n"] for r in rows}):
        m = np.array([r["a0"] == a0 and r["n"] == n for r in rows])
        if m.any(): print(f"  a0={a0} n={n}: failures {int((fail & m).sum())}/{int(m.sum())}, folds {int((fold & m).sum())}")
