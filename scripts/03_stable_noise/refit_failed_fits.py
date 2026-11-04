"""
Re-run the three large-n consistency fits that stopped at poor local optima with 2 restarts
(Section 4.2 of the paper), now with 8 restarts, and compare with refits.json.

The values used in the paper were substituted by hand into consistency_a*_big*.json at the time.
This script does NOT overwrite those files; it only prints the comparison. Note that the
consistency study was run with the earlier Fourier density table, while this script uses the
current scipy-based table (data/sas_tables.npz), so small differences in alpha-hat are expected.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('stable_noise'))
import json, numpy as np, warnings
warnings.filterwarnings("ignore")
from stable_noise_gp import StableNoiseGP, rsas
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
for r in json.load(open("refits.json")):
    a0, n, rep = r["a0"], r["n"], r["rep"]
    g = np.random.default_rng(int(a0*1000) + 10*n + rep)            # same data as consistency.py
    x = np.sort((np.arange(n)+g.uniform(size=n))/n*1.2); y = f(x) + 0.05*rsas(a0, n, g)
    m = StableNoiseGP().fit(x, y, restarts=8, seed=rep)
    print(f"a0={a0} n={n} rep={rep}: alpha-hat now {m.alpha:.3f} (recorded {r['a_hat']:.3f}); "
          f"log ML now {m.lml:.1f} (recorded: 8 restarts {r['lml8']:.1f}, 2 restarts {r['lml2']:.1f})", flush=True)
