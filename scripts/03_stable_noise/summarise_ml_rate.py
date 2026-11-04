"""Summary of the ML rate study (ml_rate_a*_main.json) -> ml_rate_summary.json."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise"))   # result files are read and written here
import json, os, glob, numpy as np
HERE = os.getcwd()
OR = json.load(open(os.path.join(HERE, "oracle_fisher.json")))
from ecf_alpha import asym_sd
rows = []
for fn in sorted(glob.glob(os.path.join(HERE, "ml_rate_a*_main.json"))):
    D = json.load(open(fn)); a0 = D[0]["a0"]
    for n in sorted({o["n"] for o in D}):
        R = [o for o in D if o["n"] == n]
        ml = np.array([o["ml"]["alpha"] for o in R]); ec = np.array([o["alpha_tilde_clipped"] for o in R])
        rm_ml = float(np.sqrt(np.mean((ml - a0) ** 2))); rm_ec = float(np.sqrt(np.mean((ec - a0) ** 2)))
        chk = [o for o in R if "ml_fixed_start" in o]
        dl = [o["ml_fixed_start"]["lml"] - o["ml"]["lml"] for o in chk]
        da = [abs(o["ml_fixed_start"]["alpha"] - o["ml"]["alpha"]) for o in chk]
        sig = np.array([o["ml"]["sigma"] for o in R])
        bs = np.random.default_rng(0).integers(0, len(ml), (2000, len(ml)))
        ci = np.quantile(np.sqrt(np.mean((ml[bs] - a0) ** 2, 1)) * np.sqrt(n), [0.025, 0.975])
        rows.append(dict(a0=a0, n=n, reps=len(R), ml_sqrt_n_rmse_ci=[float(ci[0]), float(ci[1])],
                         ml_median=float(np.median(ml)), ml_bias=float(ml.mean() - a0), ml_sd=float(ml.std(ddof=1)), ml_rmse=rm_ml,
                         ml_sqrt_n_rmse=rm_ml * np.sqrt(n), ml_at_2=float(np.mean(ml > 1.995)),
                         ecf_rmse=rm_ec, ecf_sqrt_n_rmse=rm_ec * np.sqrt(n), ecf_delta_sqrt_n=asym_sd(a0, 1) * np.sqrt(2),
                         mse_ratio_ecf_over_ml=float((rm_ec / rm_ml) ** 2),
                         oracle_sqrt_n_sd=OR[str(a0)]["sd_oracle"], ml_rmse_over_oracle=float(rm_ml * np.sqrt(n) / OR[str(a0)]["sd_oracle"]),
                         sigma_median=float(np.median(sig)),
                         start_check_reps=len(chk), start_check_max_gain=float(max(dl)) if dl else None,
                         start_check_max_dalpha=float(max(da)) if da else None,
                         secs_median=float(np.median([o["secs"] for o in R if "ml_fixed_start" not in o] or [np.nan]))))
        r = rows[-1]
        print(f"a0={a0} n={n:5d} reps={len(R):3d}: ML median {r['ml_median']:.3f} bias {r['ml_bias']:+.3f} RMSE {rm_ml:.4f} "
              f"sqrt(n)RMSE {r['ml_sqrt_n_rmse']:.2f} (oracle {r['oracle_sqrt_n_sd']:.2f}, ratio {r['ml_rmse_over_oracle']:.2f}) | "
              f"ECF RMSE {rm_ec:.4f} sqrt(n)RMSE {r['ecf_sqrt_n_rmse']:.2f} (delta {r['ecf_delta_sqrt_n']:.2f}) | MSE ratio ECF/ML {r['mse_ratio_ecf_over_ml']:.2f} | "
              f"start check: max gain {r['start_check_max_gain']}, max |d alpha| {r['start_check_max_dalpha']} | {r['secs_median']:.1f}s")
for a0 in sorted({r["a0"] for r in rows}):
    big = [r for r in rows if r["a0"] == a0 and r["n"] >= 480]
    print(f"a0={a0}: n >= 480, mean ratio ML sqrt(n)RMSE / oracle {np.mean([r['ml_rmse_over_oracle'] for r in big]):.3f}, "
          f"MSE ratio ECF/ML range {min(r['mse_ratio_ecf_over_ml'] for r in big):.2f}-{max(r['mse_ratio_ecf_over_ml'] for r in big):.2f}, "
          f"max |d alpha| start check {max(r['start_check_max_dalpha'] for r in big):.1e}")
json.dump(rows, open(os.path.join(HERE, "ml_rate_summary.json"), "w"), indent=1)
