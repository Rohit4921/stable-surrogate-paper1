"""Sensitivity of the computed fits to the tabulated S-alpha-S log-density (added 4 Nov 2026, pre-submission audit).
(i) Pointwise accuracy: the cubic-in-alpha table (O(n) code) and the linear-in-alpha table (dense code) against
    scipy.stats.levy_stable.logpdf (Nolan's integral; beta = 0, so S0 = S1) at random off-grid alpha in [0.8, 1.98] and
    z in [0, 80] (including the power-law extension beyond |z| = 60): log-density, its z-derivative and its
    alpha-derivative (central differences of each function).
(ii) Effect on estimates: the known-location ML estimator of (alpha, log sigma) (crossfit_alpha.fit, which uses the
    cubic table) refitted with the exact scipy log-density on 20 noise samples of size n = 480 for alpha0 = 1.3, 1.7
    (same generator as Tables 1-2). Reports the largest |alpha_table - alpha_exact|.
Writes results/checks/density_sensitivity.json.  Usage: python density_sensitivity.py [reps]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.stats import levy_stable
from scipy.optimize import minimize
from ssm_laplace import logpdf_derivs_cubic
from stable_noise_gp import logpdf_derivs as logpdf_derivs_linear, rsas
from crossfit_alpha import fit as fit_table, A_LO, A_HI
REPS = int(_sys.argv[1]) if len(_sys.argv) > 1 else 20
rng = np.random.default_rng(20261104)
out = {}
# (i) pointwise accuracy
A = rng.uniform(0.8, 1.98, 60); Z = np.r_[rng.uniform(0, 5, 20), rng.uniform(5, 60, 20), rng.uniform(60, 80, 10)]
h_a, h_z = 1e-4, 1e-4
res = {"cubic": [], "linear": []}
for a in A:
    ex = levy_stable.logpdf(Z, a, 0.0)
    exz = (levy_stable.logpdf(Z + h_z, a, 0.0) - levy_stable.logpdf(Z - h_z, a, 0.0)) / (2 * h_z)
    exa = (levy_stable.logpdf(Z, a + h_a, 0.0) - levy_stable.logpdf(Z, a - h_a, 0.0)) / (2 * h_a)
    for name, F in (("cubic", logpdf_derivs_cubic), ("linear", logpdf_derivs_linear)):
        lp, d1, _ = F(Z, a)
        da = (F(Z, a + h_a)[0] - F(Z, a - h_a)[0]) / (2 * h_a)
        res[name].append(np.c_[np.abs(lp - ex), np.abs(d1 - exz), np.abs(da - exa), Z])
for name in res:
    R = np.vstack(res[name]); core = R[:, 3] <= 60
    out[f"pointwise_{name}"] = dict(max_abs_logp=float(R[:, 0].max()), max_abs_logp_core=float(R[core, 0].max()),
                                    max_abs_dz=float(R[:, 1].max()), max_abs_dalpha=float(R[:, 2].max()),
                                    median_abs_dalpha=float(np.median(R[:, 2])), n_points=int(len(R)))
    print(name, out[f"pointwise_{name}"], flush=True)
# (ii) effect on the known-location ML estimate
def fit_exact(res_, start):
    def nll(p):
        a = float(np.clip(p[0], A_LO, A_HI)); s = np.exp(p[1])
        return -(levy_stable.logpdf(res_ / s, a, 0.0).sum() - len(res_) * p[1])
    r = minimize(nll, start, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-7, maxiter=400))
    return float(np.clip(r.x[0], A_LO, A_HI)), float(np.exp(r.x[1]))
rows = []
for a0 in (1.3, 1.7):
    for rep in range(REPS):
        r = np.random.default_rng(777000 + int(1000 * a0) + rep); e = 0.05 * rsas(a0, 480, r)
        a_t, s_t, _ = fit_table(e); t = time.time()
        a_e, s_e = fit_exact(e, [a_t, np.log(s_t)])
        rows.append(dict(a0=a0, rep=rep, alpha_table=a_t, alpha_exact=a_e, sigma_table=s_t, sigma_exact=s_e))
        print(a0, rep, round(a_t, 5), round(a_e, 5), f"{time.time()-t:.0f}s", flush=True)
d = np.array([abs(o["alpha_table"] - o["alpha_exact"]) for o in rows])
out["refit"] = dict(n=480, reps_per_alpha=REPS, max_abs_dalpha=float(d.max()), median_abs_dalpha=float(np.median(d)), rows=rows)
_os.makedirs(_paths.results("checks"), exist_ok=True)
json.dump(out, open(_os.path.join(_paths.results("checks"), "density_sensitivity.json"), "w"), indent=1)
print("refit: max |alpha_table - alpha_exact| =", d.max(), "median", np.median(d))
