"""Summary of the phase-C unified Laplace-ML study (ml_unified_a*_c1.json) -> ml_unified_summary.json.

One row per (alpha0, n), in the format of ml_rate_summary.json (so fig_rate.py and summarise_crossfit.py can use it),
plus the Table 1 quantities: mean and median alpha-hat, RMSE, share of fits at the upper bound 1.98, rejection rate of
the LR test of alpha = 2 at the chi-square(1) 10% point 2.71 (calibration: lr_bootstrap.py), number of fits with a
fold at the optimum (det(I + K W) <= 0), median ||f_hat - f||_n, and the start check (largest log-likelihood gain
of the better start over the ECF start, and largest |difference| in alpha-hat).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise"))
import json, glob, numpy as np
from ecf_alpha import asym_sd
OR = json.load(open("oracle_fisher.json"))
rows = []
for fn in sorted(glob.glob("ml_unified_a*_c1.json")):
    D = json.load(open(fn)); a0 = D[0]["a0"]
    for n in sorted({o["n"] for o in D}):
        R = [o for o in D if o["n"] == n]
        ml = np.array([o["ml"]["alpha"] for o in R]); ec = np.array([min(o["alpha_tilde"], 2.0) if np.isfinite(o["alpha_tilde"]) else 2.0 for o in R])
        rm = float(np.sqrt(np.mean((ml - a0) ** 2))); rme = float(np.sqrt(np.mean((ec - a0) ** 2)))
        bs = np.random.default_rng(0).integers(0, len(ml), (2000, len(ml)))
        ci = np.quantile(np.sqrt(np.mean((ml[bs] - a0) ** 2, 1)) * np.sqrt(n), [0.025, 0.975])
        gain = [o["start_fixed"]["lml"] - o["start_ecf"]["lml"] for o in R]
        dal = [abs(o["start_fixed"]["alpha"] - o["start_ecf"]["alpha"]) for o in R]
        row = dict(a0=a0, n=n, reps=len(R), ml_median=float(np.median(ml)), ml_mean=float(ml.mean()), ml_bias=float(ml.mean() - a0),
                   ml_rmse=rm, ml_sqrt_n_rmse=rm * np.sqrt(n), ml_sqrt_n_rmse_ci=[float(ci[0]), float(ci[1])],
                   ml_at_bound=float(np.mean(ml > 1.975)), lr_reject_271=float(np.mean([o["LR"] > 2.71 for o in R])),
                   folds=int(sum(o["ml"]["sign"] <= 0 for o in R)), dn_median=float(np.median([o["ml"]["dn"] for o in R])),
                   ecf_rmse=rme, ecf_sqrt_n_rmse=rme * np.sqrt(n),
                   start_max_gain=float(max(gain)), start_max_dalpha=float(max(dal)),
                   start_better_fixed=int(sum(g > 1e-3 for g in gain)))
        if str(a0) in OR:
            row["oracle_sqrt_n_sd"] = OR[str(a0)]["sd_oracle"]; row["ml_rmse_over_oracle"] = row["ml_sqrt_n_rmse"] / row["oracle_sqrt_n_sd"]
            row["ecf_delta_sqrt_n"] = asym_sd(a0, 1) * np.sqrt(2); row["mse_ratio_ecf_over_ml"] = (rme / rm) ** 2
        rows.append(row)
        print(f"a0={a0} n={n:5d} reps={len(R):3d}: median {row['ml_median']:.3f} mean {row['ml_mean']:.3f} RMSE {rm:.4f} "
              f"sqrt(n)RMSE {row['ml_sqrt_n_rmse']:.2f} [{ci[0]:.2f},{ci[1]:.2f}]" + (f" (bound {row['oracle_sqrt_n_sd']:.2f})" if "oracle_sqrt_n_sd" in row else "")
              + f" | at bound {row['ml_at_bound']:.2f} | LR>2.71 {row['lr_reject_271']:.2f} | folds {row['folds']} | ||D||_n {row['dn_median']:.4f}"
              f" | ECF sqrt(n)RMSE {row['ecf_sqrt_n_rmse']:.2f} | fixed start better in {row['start_better_fixed']} (max gain {row['start_max_gain']:.2f}, max |da| {row['start_max_dalpha']:.3f})")
json.dump(rows, open("ml_unified_summary.json", "w"), indent=1)
