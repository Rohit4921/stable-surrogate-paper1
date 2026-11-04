"""Phase C: parametric-bootstrap calibration of the likelihood-ratio test of alpha = 2 (Gaussian noise) against
SaS noise with alpha in [0.8, 1.98], by the warp-speed Monte Carlo method (Giacomini, Politis and White, 2013):
for every Monte Carlo data set we compute the Laplace LR statistic and ONE bootstrap statistic LR*, from a data set
simulated under the GP fitted to it (Gaussian noise, ML hyperparameters). The bootstrap critical value at level a
is the (1-a) quantile of LR* over the data sets of a cell, and the rejection rate is P(LR > that quantile).
Laplace is exact for the GP, and for the SaS model it is the estimator of Theorem 28; the O(n) code makes the
fits cheap. Protocol as ml_unified.py (median/MAD standardisation, two starts for SaS).
Data: rng = default_rng(int(1000 alpha0) + 10 n + rep) as elsewhere; bootstrap: default_rng(5_000_000 + 1000 alpha0 + 10 n + rep).
Usage: python lr_bootstrap.py <alpha0> <n> <rep range a-b> <tag>
"""
import os as _os, sys as _sys
HERE_ = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.join(HERE_, "..", "..", "src")); _sys.path.insert(0, HERE_)
import _paths
import sys, os, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
import ml_unified as U          # changes directory to results/stable_noise
from surrogates import matern52
from crossfit_alpha import robust_scale


def lr_stat(x, y):
    m, sc = robust_scale(y); span = np.ptp(x)
    a_raw, a_c, sig_t = U.ecf_start(y)
    A = U.fit(x, y, [0.0, np.log(0.25 * span), np.log(sig_t / sc), a_c])
    B = U.fit(x, y, [0.0, np.log(0.25 * span), np.log(0.1), 1.95])
    S = A if A["lml"] >= B["lml"] else B
    G = U.fit(x, y, [0.0, np.log(0.25 * span), np.log(sig_t / sc)], gaussian=True)
    return 2 * (S["lml"] - G["lml"]), S, G


if __name__ == "__main__":
    a0 = float(sys.argv[1]); n = int(sys.argv[2]); lo, hi = [int(v) for v in sys.argv[3].split("-")]
    fn = f"lr_boot_a{a0}_n{n}_{sys.argv[4]}.json"
    out = json.load(open(fn)) if os.path.exists(fn) else []
    have = {o["rep"] for o in out}
    for rep in range(lo, hi):
        if rep in have: continue
        t0 = time.time(); x, y = U.data(a0, n, rep)
        LR, S, G = lr_stat(x, y)
        # one parametric-bootstrap data set from the fitted GP (on the standardised scale, then back)
        m, sc = robust_scale(y); sf, ell, sg = np.exp(G["p"][0]), np.exp(G["p"][1]), np.exp(G["p"][2])
        K = matern52(x, x, sf, ell) + (sg ** 2 + 1e-10) * np.eye(n)
        rb = np.random.default_rng(5_000_000 + int(1000 * a0) + 10 * n + rep)
        ys = m + sc * (np.linalg.cholesky(K) @ rb.standard_normal(n))
        LRb, Sb, _ = lr_stat(x, ys)
        o = dict(a0=a0, n=n, rep=rep, LR=float(LR), alpha=S["alpha"], LR_boot=float(LRb), alpha_boot=Sb["alpha"], secs=time.time() - t0)
        out.append(o); json.dump(out, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
        if rep % 25 == 0:
            print(f"a0={a0} n={n} rep={rep}: LR {LR:.2f} (alpha {S['alpha']:.2f}) LR* {LRb:.2f} {o['secs']:.1f}s", flush=True)
    print("done")
