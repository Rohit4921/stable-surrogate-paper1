"""Recompute and save every auxiliary number quoted in Paper 1 (verification_log.json)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('stable_noise'))   # result files are read and written here
import json, glob, time, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
import exact_m2 as X
from stable_noise_gp import rsas, lik, logpdf_derivs, _single_fourier, _single
from scipy.stats import levy_stable
log = {}
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
def data(a0, n, rep):
    r = np.random.default_rng(int(a0*1000)+10*n+rep)
    x = np.sort((np.arange(n)+r.uniform(size=n))/n*1.2); return x, f(x)+0.05*rsas(a0, n, r)
T = json.load(open("exact_table.json")); G = json.load(open("gp_null.json"))
R = [o for fn in sorted(glob.glob("exact3_a*_main.json")) for o in json.load(open(fn))]
# (a) Laplace error / loss denominators
fail = [o for o in R if o["laplace"]["own"] - o["laplace"]["exact"] > 3]
nf = [o for o in R if o["a0"] < 2 and o not in fail]
log["n_fail"] = len(fail); log["fail_err"] = [round(o["laplace"]["own"] - o["laplace"]["exact"], 2) for o in fail]
log["lap_err_median_nonfail_a0lt2"] = float(np.median([o["laplace"]["own"] - o["laplace"]["exact"] for o in nf])); log["n_nonfail_a0lt2"] = len(nf)
L = [o["exact"]["exact"] - o["laplace"]["exact"] for o in R if o["a0"] < 2]
log["lap_loss_median_a0lt2"] = float(np.median(L)); log["lap_loss_max_a0lt2"] = float(np.max(L)); log["n_a0lt2"] = len(L)
# (b) Gaussian data: exact gains and MC offsets
gains = []; lr2 = []
for o in R:
    if o["a0"] == 2.0:
        g = o["exact"]["own"] - G[f"{o['a0']}_{o['n']}_{o['rep']}"]
        if o["exact"]["alpha"] > 1.995: lr2.append(2*g)
        elif o["exact"]["alpha"] > 1.5: gains.append(round(g, 3))
log["gauss_gains_alpha_lt2"] = sorted(gains); log["gauss_LR_at_alpha2_mean"] = float(np.mean(lr2))
log["trunc_minus_raw_at_exact_opt"] = [float(np.min([o["exact"]["exact"] - o["exact"]["exact_raw"] for o in R])), float(np.max([o["exact"]["exact"] - o["exact"]["exact_raw"] for o in R]))]
log["trunc_minus_raw_median"] = float(np.median([o["exact"]["exact"] - o["exact"]["exact_raw"] for o in R]))
# (c) Laplace failure diagnostics vs first replicate of the same cell
diag = []
for o in R:
    if o in fail or o["rep"] == 0:
        x, y = data(o["a0"], o["n"], o["rep"]); E = X.ExactM2(x, y, S=10)
        p = o["laplace"]["p"]; pp = (np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3])
        K, fm, _, _ = E.laplace(*pp); _, g, W, _ = lik(E.y, fm, pp[2], pp[3])
        ld = np.linalg.slogdet(np.eye(len(y)) + K*W[None, :])[1]
        diag.append(dict(a0=o["a0"], n=o["n"], rep=o["rep"], fail=o in fail, maxabsz=float(np.abs((E.y-fm)/pp[2]).max()),
                         n_negW=int((W < 0).sum()), logdet=float(ld)))
log["failure_diagnostics"] = diag
# (d) old-table jump and v1->v2 alpha change
z = np.array([7.9, 8.1]); log["old_table_jump"] = {str(a): float(np.diff(_single_fourier(z, int(round(a*1000)))[0])[0] - np.diff(levy_stable.logpdf(z, a, 0))[0]) for a in (1.3, 1.7, 1.9)}
d = []
for fn in sorted(glob.glob("exact3_a*_main.json")):
    new = {o["rep"]: o for o in json.load(open(fn))}; old = {o["rep"]: o for o in json.load(open("v1_tables/"+fn))}
    d += [abs(new[r]["laplace"]["alpha"] - old[r]["laplace"]["alpha"]) for r in new]
log["v1_v2_laplace_alpha_change_q90"] = float(np.quantile(d, .9)); log["v1_v2_laplace_alpha_change_max"] = float(np.max(d))
# (e) IS vs AIS agreement (new tables) and timing of one exact evaluation at n = 60
rows = []
for a0, n, rep in ((1.3, 30, 0), (1.3, 30, 3), (1.7, 60, 1)):
    x, y = data(a0, n, rep); E = X.ExactM2(x, y, S=40000, seed=9)
    for a in (1.7, 1.3, 1.0):
        pp = (1.0, 0.3, 0.05/E.ys, a); t = time.time(); v = E.is_mix(*pp); tt = time.time() - t
        ais = np.mean([X.ais_bridge(E, *pp, chains=128, T=1500, seed=s)[0] for s in (1, 2)])
        rows.append(dict(a0=a0, n=n, rep=rep, alpha=a, IS=float(v), AIS=float(ais), diff=float(v - ais), secs=tt))
log["is_vs_ais"] = rows; log["is_vs_ais_maxabs"] = float(max(abs(r["diff"]) for r in rows))
log["exact_eval_secs_n60"] = [r["secs"] for r in rows if r["n"] == 60]
json.dump(log, open("verification_log.json", "w"), indent=1)
for k, v in log.items():
    if k not in ("failure_diagnostics", "is_vs_ais"): print(k, v)
for r in log["failure_diagnostics"]: print(r)
for r in rows: print(r)
