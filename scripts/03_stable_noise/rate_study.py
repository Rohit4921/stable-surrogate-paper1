"""Rate study for the difference-ECF estimator of alpha (Section 4.2 of Paper 1)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise"))   # result files are read and written here
import numpy as np, json, sys, os
from ecf_alpha import alpha_hat, asym_sd
from stable_noise_gp import rsas
C = (0.2, 1.0); SIG = 0.05; REPS = 500
f = lambda x: x*np.sin(10*x) + np.cos(2*x)
NS = [60, 120, 240, 480, 960, 1920, 3840, 7680]
out = []
for a0 in (1.3, 1.7, 2.0):
    for design, order in (("stratified", 1), ("equispaced", 2)):
        for n in NS:
            r = np.random.default_rng(int(a0*1000) + n + (0 if order == 1 else 7))
            est = []
            for rep in range(REPS):
                x = np.sort((np.arange(n) + r.uniform(size=n)) / n * 1.2) if design == "stratified" else (np.arange(n) + 0.5) / n * 1.2
                eps = rsas(a0, n, r) if a0 < 2 else np.sqrt(2) * r.standard_normal(n)
                est.append(alpha_hat(f(x) + SIG * eps, order, C))
            e = np.array(est); m = n // (2 if order == 1 else 3)
            row = dict(a0=a0, design=design, order=order, n=n, m=m, median=float(np.nanmedian(e)), mean=float(np.nanmean(e)),
                       bias=float(np.nanmean(e) - a0), sd=float(np.nanstd(e)), rmse=float(np.sqrt(np.nanmean((e - a0)**2))),
                       nan=int(np.isnan(e).sum()), asym_sd=asym_sd(min(a0, 1.999), order, C) / np.sqrt(m))
            row["sqrt_m_rmse"] = row["rmse"] * np.sqrt(m)
            out.append(row)
            print(f"a0={a0} {design:10s} order {order} n={n:5d}: median {row['median']:.3f} bias {row['bias']:+.3f} sd {row['sd']:.3f} "
                  f"RMSE {row['rmse']:.3f} (asym sd {row['asym_sd']:.3f})  sqrt(m)RMSE {row['sqrt_m_rmse']:.2f}", flush=True)
json.dump(out, open("rate_study.json", "w"), indent=1)
