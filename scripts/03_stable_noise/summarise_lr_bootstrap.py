"""Summary of lr_bootstrap.py (lr_boot_a*_n*_c3.json) -> lr_bootstrap_summary.json.

Warp-speed Monte Carlo (Giacomini, Politis and White, 2013): in each (alpha0, n) cell the bootstrap critical value at
level a is the (1 - a) quantile of the one-per-data-set bootstrap statistics LR*; the rejection rate of the bootstrap
test is the share of data sets with LR above it (size for alpha0 = 2, power otherwise). For comparison: the rates of
the naive chi-square(1) cut-offs 2.71 and 3.84 (the 5% point of the 50:50 mixture of chi-square(0) and chi-square(1)
is 2.71). Monte Carlo standard errors are binomial.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise"))
import json, glob, numpy as np
rows = []
for fn in sorted(glob.glob("lr_boot_a*_n*_c3.json")):
    D = json.load(open(fn)); a0, n = D[0]["a0"], D[0]["n"]
    LR = np.array([o["LR"] for o in D]); LB = np.array([o["LR_boot"] for o in D]); m = len(D)
    row = dict(a0=a0, n=n, reps=m)
    for lev in (0.05, 0.10):
        q = float(np.quantile(LB, 1 - lev)); r = float(np.mean(LR > q))
        row[f"boot_crit_{int(lev*100)}"] = q; row[f"boot_reject_{int(lev*100)}"] = r; row[f"boot_reject_{int(lev*100)}_se"] = float(np.sqrt(r * (1 - r) / m))
    for c in (2.71, 3.84):
        r = float(np.mean(LR > c)); row[f"chi2_reject_{c}"] = r
    row["boot_at_bound"] = float(np.mean(np.array([o["alpha_boot"] for o in D]) > 1.975))
    rows.append(row)
    print(f"a0={a0} n={n:4d} reps={m:4d} | bootstrap: crit5 {row['boot_crit_5']:.2f} reject5 {row['boot_reject_5']:.3f} (se {row['boot_reject_5_se']:.3f}), "
          f"crit10 {row['boot_crit_10']:.2f} reject10 {row['boot_reject_10']:.3f} | chi2 2.71: {row['chi2_reject_2.71']:.3f}, 3.84: {row['chi2_reject_3.84']:.3f}"
          f" | bootstrap fits at bound {row['boot_at_bound']:.2f}")
json.dump(rows, open("lr_bootstrap_summary.json", "w"), indent=1)
