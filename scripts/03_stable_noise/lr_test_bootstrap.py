"""Data-set-specific parametric-bootstrap test of H0: Gaussian-noise GP (SaS with alpha = 2) against the separated
working alternative H1: SaS noise with alpha in [0.8, 1.98] (Section 4.4 of the paper, Table 4).

    LR = 2 {max_H1 Laplace log-likelihood - max_H0 log-likelihood}

H1 does not contain H0, so LR can be negative. The null distribution of LR is estimated by B parametric-bootstrap
data sets drawn from the GP fitted under H0 (ML hyperparameters, on the median/MAD-standardised scale and mapped
back); both models are refitted to each bootstrap data set with the same protocol as the observed data
(lr_bootstrap.lr_stat: two starts for SaS, ECF start and alpha = 1.95). The p-value is (1 + #{LR* >= LR}) / (B + 1).
The validity of this bootstrap in this non-regular, separated comparison is supported by the Monte Carlo size
estimates of Table 4 (lr_bootstrap.py, warp-speed method), not by a theorem.

Library use:
    from lr_test_bootstrap import lr_test_bootstrap
    res = lr_test_bootstrap(x, y, B=499, seed=1)      # res["p_value"], res["LR"], res["crit"]
Command line (demonstration on one simulated data set of the paper's design):
    python lr_test_bootstrap.py <alpha0> <n> <rep> <B>
"""
import os as _os, sys as _sys
HERE_ = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.join(HERE_, "..", "..", "src")); _sys.path.insert(0, HERE_)
import numpy as np, time, json, warnings
warnings.filterwarnings("ignore")
from lr_bootstrap import lr_stat, U          # U = ml_unified (data generator and fitting protocol)
from surrogates import matern52
from crossfit_alpha import robust_scale


def lr_test_bootstrap(x, y, B=499, seed=0, level=0.05):
    """Parametric-bootstrap LR test of alpha = 2 for one data set. Returns a dict with LR, the bootstrap statistics,
    the bootstrap critical value at `level`, the p-value, the fitted alpha under H1 and the decision."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(y)
    LR, S, G = lr_stat(x, y)
    m, sc = robust_scale(y); sf, ell, sg = np.exp(G["p"][0]), np.exp(G["p"][1]), np.exp(G["p"][2])
    Lc = np.linalg.cholesky(matern52(x, x, sf, ell) + (sg ** 2 + 1e-10) * np.eye(n))
    rng = np.random.default_rng(seed); LRb = np.empty(B)
    for b in range(B):
        LRb[b] = lr_stat(x, m + sc * (Lc @ rng.standard_normal(n)))[0]
    p = (1 + np.sum(LRb >= LR)) / (B + 1)
    crit = float(np.quantile(LRb, 1 - level))
    return dict(LR=float(LR), alpha_hat=float(S["alpha"]), LR_boot=LRb.tolist(), crit=crit, p_value=float(p),
                reject=bool(p <= level), B=B, level=level)


if __name__ == "__main__":
    a0, n, rep, B = float(_sys.argv[1]), int(_sys.argv[2]), int(_sys.argv[3]), int(_sys.argv[4])
    x, y = U.data(a0, n, rep); t0 = time.time()
    r = lr_test_bootstrap(x, y, B=B, seed=rep)
    r["secs"] = time.time() - t0; r.update(a0=a0, n=n, rep=rep)
    fn = f"lr_test_demo_a{a0}_n{n}_rep{rep}_B{B}.json"
    json.dump(r, open(fn, "w"), indent=1)
    print(f"alpha0={a0} n={n} rep={rep}: LR {r['LR']:.2f} (alpha-hat {r['alpha_hat']:.2f}), bootstrap 5% critical value "
          f"{r['crit']:.2f}, p = {r['p_value']:.3f}, reject: {r['reject']}  ({r['secs']:.0f} s for B = {B})")
