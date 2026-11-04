"""Precompute log densities of standard SaS (ch.f. exp(-|t|^a)) from scipy.stats.levy_stable
(Nolan's integral representation) on an alpha grid 0.50..1.99 (step 0.01) and a z grid on [0, 60]."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
import numpy as np, time
from scipy.stats import levy_stable
Z = np.r_[np.linspace(0, 15, 1501), np.linspace(15.1, 60, 450)]
A = np.round(np.arange(0.50, 1.995, 0.01), 2)
L = np.empty((A.size, Z.size)); t = time.time()
for i, a in enumerate(A):
    L[i] = levy_stable.logpdf(Z, a, 0.0)
np.savez_compressed(_os.path.join(_paths.DATA, "sas_tables.npz"), alphas=A, z=Z, logp=L)
print("built", L.shape, "finite:", np.isfinite(L).all(), f"{time.time()-t:.0f}s")
