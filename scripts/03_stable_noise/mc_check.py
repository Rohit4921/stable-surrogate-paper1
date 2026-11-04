"""Monte Carlo check of the delta-method sd (pure SaS noise, m = 20000 differences, 1000 replicates)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise"))   # result files are read and written here
import sys, os, json, numpy as np
from ecf_alpha import alpha_hat, asym_sd
from stable_noise_gp import rsas
r = np.random.default_rng(42); out = {}
for al in (1.3, 1.7, 1.95):
    est = np.array([alpha_hat(0.05 * rsas(al, 40000, r), 1, (0.2, 1.0), clip=False) for _ in range(1000)])
    mc, de = float(np.std(est) * np.sqrt(20000)), asym_sd(al, 1, (0.2, 1.0))
    out[str(al)] = dict(mc=mc, delta=de, ratio=mc / de, mean=float(est.mean())); print(al, out[str(al)], flush=True)
json.dump(out, open("mc_check.json", "w"), indent=1)
