"""Rerun of the running-median-pilot column of Table 2 with odd windows (4 Nov 2026, after the pre-submission audit).
The data sets are regenerated from the seeds of crossfit_study.py (identical x and noise); identity is checked by
refitting the oracle (f known) estimator and comparing with the stored crossfit_a*_v2.json value. Only the median
pilot changes: k_n = smallest odd integer >= max(5, ceil(0.15 n^{2/3})), so that the window median is one order
statistic, as the Lemma requires. Writes crossfit_median_odd_a<alpha0>.json.
Usage: python crossfit_median_odd.py <alpha0>"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise"))
import json, time, numpy as np, warnings, importlib.util
warnings.filterwarnings("ignore")
from crossfit_alpha import crossfit_alpha, fit
from stable_noise_gp import rsas
spec = importlib.util.spec_from_file_location("of", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "oracle_fisher.py"))
OF = importlib.util.module_from_spec(spec); spec.loader.exec_module(OF)
f = lambda x: x * np.sin(10 * x) + np.cos(2 * x)
C_MED = 0.15
a0 = float(_sys.argv[1])
old = {(o["n"], o["rep"]): o for o in json.load(open(f"crossfit_a{a0}_v2.json"))}
sd_cache = {}
def sd_at(a):
    a = round(float(np.clip(a, 0.8, 1.98)), 3)
    if a not in sd_cache: sd_cache[a] = OF.sd_oracle(a)
    return sd_cache[a]
out = []; t0 = time.time(); maxdiff = 0.0
for (n, rep) in sorted(old):
    r = np.random.default_rng(900000 + int(1000 * a0) + 10 * n + rep)          # same seeds as crossfit_study.py
    x = np.sort((np.arange(n) + r.uniform(size=n)) / n * 1.2); e = 0.05 * rsas(a0, n, r); y = f(x) + e
    a_or, _, _ = fit(e); maxdiff = max(maxdiff, abs(a_or - old[(n, rep)]["alpha_oracle"]))
    im = {}; a_md, s_md, k = crossfit_alpha(y, c=C_MED, kind="median", info=im)
    D = im["fhat"] - f(x); se = sd_at(a_md) / np.sqrt(n)
    out.append(dict(a0=a0, n=n, rep=rep, k=k, alpha_med=a_md, sigma_med=s_md, alpha_oracle_check=a_or,
                    dn_med=float(np.sqrt(np.mean(D ** 2))), dinf_med=float(np.abs(D).max()),
                    se_med=float(se), cover_med=bool(abs(a_md - a0) <= 1.96 * se)))
json.dump(dict(max_abs_oracle_diff=maxdiff, rows=out), open(f"crossfit_median_odd_a{a0}.json", "w"), indent=0)
print(f"a0={a0}: {len(out)} fits in {time.time()-t0:.0f}s; data identity check: max |oracle alpha - stored| = {maxdiff:.2e}")
