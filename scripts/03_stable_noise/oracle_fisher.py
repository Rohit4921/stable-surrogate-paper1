"""Oracle efficiency bound for alpha: Fisher information of iid S-alpha-S(sigma) noise with known location
(f known), parameters (alpha, log sigma). sd_oracle(alpha) = sqrt([I^{-1}]_{alpha,alpha}); an efficient
estimator from n residuals has sd ~ sd_oracle / sqrt(n).
Checked by Monte Carlo: ML for (alpha, sigma) on iid samples of size 4000, 300 replicates."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise"))   # result files are read and written here
import sys, os, json, numpy as np
from scipy.optimize import minimize
HERE = os.getcwd()
from ssm_laplace import logpdf_derivs_cubic as L
from stable_noise_gp import rsas
ZMAX = 60.0


def fisher(a, h=1e-3):
    z = np.linspace(0, ZMAX, 120001)
    lp, d1, _ = L(z, a)
    sa = (L(z, a + h)[0] - L(z, a - h)[0]) / (2 * h)
    ss = -1 - z * d1
    p = np.exp(lp); wq = np.full(z.size, 2.0); wq[1::2] = 4.0; wq[0] = wq[-1] = 1.0; wq *= (z[1] - z[0]) / 3
    I = 2 * np.array([[np.sum(wq * p * sa * sa), np.sum(wq * p * sa * ss)], [0, np.sum(wq * p * ss * ss)]])
    # tails |z| > 60: p = e^{l60} (z/60)^{-1-a}, score_alpha = d - u, score_logsigma = a (u = log(z/60))
    l60 = L(np.array([ZMAX]), a)[0][0]; d = (L(np.array([ZMAX]), a + h)[0][0] - L(np.array([ZMAX]), a - h)[0][0]) / (2 * h)
    c = 2 * ZMAX * np.exp(l60)
    I[0, 0] += c * (d * d / a - 2 * d / a ** 2 + 2 / a ** 3)
    I[0, 1] += c * a * (d / a - 1 / a ** 2)
    I[1, 1] += c * a
    I[1, 0] = I[0, 1]
    mass = 2 * np.sum(wq * p) + c / a
    return I, mass


def sd_oracle(a):
    I, _ = fisher(a); return float(np.sqrt(np.linalg.inv(I)[0, 0]))


def mc_check(a, n=4000, reps=300, seed=1):
    r = np.random.default_rng(seed); est = []
    for _ in range(reps):
        e = rsas(a, n, r)
        nll = lambda p: -(L(e / np.exp(p[0]), p[1])[0].sum() - n * p[0])
        est.append(minimize(nll, [0.0, a], method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-6)).x[1])
    est = np.array(est); return float(np.sqrt(n) * est.std()), float(est.mean())


if __name__ == "__main__":
    out = {}
    for a in (1.0, 1.3, 1.5, 1.7, 1.9):
        I, mass = fisher(a); sd = float(np.sqrt(np.linalg.inv(I)[0, 0]))
        out[str(a)] = dict(sd_oracle=sd, sd_alpha_known_sigma=float(1 / np.sqrt(I[0, 0])), total_mass=float(mass))
        print(f"alpha {a}: sqrt(n) sd oracle {sd:.3f} (sigma known {1/np.sqrt(I[0,0]):.3f}); mass {mass:.6f}", flush=True)
    for a in (1.3, 1.7):
        mc, mean = mc_check(a); out[str(a)].update(mc_sd=mc, mc_mean=mean)
        print(f"alpha {a}: Monte Carlo sqrt(n) sd {mc:.3f} (mean {mean:.4f})", flush=True)
    json.dump(out, open(os.path.join(HERE, "oracle_fisher.json"), "w"), indent=1)
