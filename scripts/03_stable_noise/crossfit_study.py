"""Simulation for the efficiency theorem: cross-fitted plug-in ML of alpha with two pilots (running median, covered by
the Lemma; cross-fitted GP posterior mode, Appendix C) versus the oracle ML with f known, on the data generator of Table 1 / the ML rate study, n = 120 ... 7,680.
Also: pilot errors ||f_hat - f||_n and max|f_hat - f|, and for the GP pilot the fold sign and a
truth-start check of the computed mode (tag v2; the GP pilot uses median/MAD standardisation and alpha <= 1.98).
Also: coverage of the Wald interval alpha_hat +- 1.96 sqrt([I(alpha_hat)^{-1}]_aa / n).
Usage: python crossfit_study.py <alpha0> <n list> <reps> <tag>"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _paths
_os.chdir(_paths.results("stable_noise"))
import sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from crossfit_alpha import crossfit_alpha, fit
from stable_noise_gp import rsas
import importlib.util
spec = importlib.util.spec_from_file_location("of", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "oracle_fisher.py"))
OF = importlib.util.module_from_spec(spec); spec.loader.exec_module(OF)
f = lambda x: x * np.sin(10 * x) + np.cos(2 * x)
C_MED = 0.15   # window k = ceil(0.15 n^{2/3}); constant chosen on separate pilot runs (seeds 7000+)
a0 = float(sys.argv[1]); NS = [int(v) for v in sys.argv[2].split(",")]; REPS = int(sys.argv[3]); TAG = sys.argv[4]
fn = f"crossfit_a{a0}_{TAG}.json"
out = json.load(open(fn)) if _os.path.exists(fn) else []
have = {(o["n"], o["rep"]) for o in out}
sd_cache = {}
def sd_at(a):
    a = round(float(np.clip(a, 0.8, 1.98)), 3)
    if a not in sd_cache: sd_cache[a] = OF.sd_oracle(a)
    return sd_cache[a]
for n in NS:
    for rep in range(REPS):
        if (n, rep) in have: continue
        r = np.random.default_rng(900000 + int(1000 * a0) + 10 * n + rep)   # fresh seeds, disjoint from Tables 1-2
        x = np.sort((np.arange(n) + r.uniform(size=n)) / n * 1.2); e = 0.05 * rsas(a0, n, r); y = f(x) + e
        t = time.time()
        im, ig = {}, {}
        a_md, s_md, k = crossfit_alpha(y, c=C_MED, kind="median", info=im)  # running-median pilot (Lemma)
        a_gp, s_gp, _ = crossfit_alpha(y, x=x, kind="gp", info=ig, ftrue=f(x))  # cross-fitted GP pilot (Appendix C)
        a_or, s_or, _ = fit(e)                                               # oracle: f known
        o = dict(a0=a0, n=n, rep=rep, k=k, alpha_med=a_md, sigma_med=s_md, alpha_gp=a_gp, sigma_gp=s_gp,
                 alpha_oracle=a_or, sigma_oracle=s_or)
        for key, inf in (("med", im), ("gp", ig)):                           # pilot error: ||Delta||_n and sup |Delta_i|
            D = inf["fhat"] - f(x); o[f"dn_{key}"] = float(np.sqrt(np.mean(D ** 2))); o[f"dinf_{key}"] = float(np.abs(D).max())
        o["gp_folds"] = ig["folds"]
        for key in ("med", "gp", "oracle"):
            a = o[f"alpha_{key}"]; se = sd_at(a) / np.sqrt(n)
            o[f"se_{key}"] = float(se); o[f"cover_{key}"] = bool(abs(a - a0) <= 1.96 * se)
        o["secs"] = time.time() - t; out.append(o)
        json.dump(out, open(fn, "w"))
        if rep % 10 == 0: print(n, rep, round(a_md, 3), round(a_gp, 3), round(a_or, 3), f"{o['secs']:.1f}s", flush=True)
json.dump(out, open(fn, "w"), indent=0); print("done")
