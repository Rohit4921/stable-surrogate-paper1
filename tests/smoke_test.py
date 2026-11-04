"""
Smoke test (under a minute on one core). Run from the package root:  python tests/smoke_test.py

1. the stable density table agrees with scipy.stats.levy_stable;
2. a Student-t process with fixed nu reproduces the GP (Corollary 12 / check_pred.py);
3. the importance-sampling likelihood equals the closed form at alpha = 2;
4. a GP with SaS noise can be fitted;
5. a two-step BO run with the SaS surrogate completes;
6. the summary scripts regenerate the stored tables (consistency, Laplace-vs-exact, BO study) exactly;
7. the fold diagnostic lambda_min(B) flags a failing Laplace fit of Section 4.3 and passes a good one (Section 4.4);
8. the O(n) state-space Laplace agrees with the dense one, and refitting one data set of the ML rate study
   reproduces the stored alpha-hat (Section 4.2); summarise_ml_rate.py regenerates ml_rate_summary.json.
"""
import os, sys, json, glob, shutil, subprocess, tempfile, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
ok = True
def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True); ok &= bool(cond)

# Refits of stored alpha-hats. They agree to < 1e-6 (bit-for-bit arithmetic) on the reference platform (Linux x86-64;
# versions in MANIFEST.md). Elsewhere (other OS, C library, BLAS or Python build) last-bit rounding differences are
# amplified by the Nelder-Mead fits: the Laplace-ML fits stop at xatol = 1e-4 and the GP-pilot hyperparameter fits at
# xatol = 1e-3, so these alpha-hats are not determined more precisely than about 1e-4 on any platform. Such refits pass
# as [PASS*] when |difference| < 1e-4; larger differences fail.
EXACT, PLATFORM_TOL = 1e-6, 1e-4
n_platform = 0
def refit_check(name, got, stored, extra=True):
    global ok, n_platform
    d = abs(got - stored)
    if d < EXACT and extra:
        tag, note = "PASS", "exact"
    elif d < PLATFORM_TOL and extra:
        tag, note = "PASS*", "within the optimiser tolerance 1e-4, not exact: platform rounding, see README"; n_platform += 1
    else:
        tag, note = "FAIL", ""; ok = False
    print(f"[{tag}] {name} ({got:.7f} vs {stored:.7f}; |diff| {d:.1e}{'; ' + note if note else ''})", flush=True)

# 1 density
from scipy.stats import levy_stable
from stable_noise_gp import logpdf_derivs, StableNoiseGP, rsas
z = np.array([0.0, 0.7, 3.0, 8.05, 25.0])
err = max(np.abs(logpdf_derivs(z, a)[0] - levy_stable.logpdf(z, a, 0)).max() for a in (1.3, 1.7))
check("SaS log-density vs scipy", err < 1e-3, f"(max abs error {err:.1e})")

# 2 TP == GP with free amplitude
from surrogates import Surrogate
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
r = np.random.default_rng(7); n = 25
x = np.sort((np.arange(n)+r.uniform(size=n))/n*1.2); y = f(x) + 0.05*r.standard_normal(n)
gp = Surrogate("gp").fit(x, y, 6); tp = Surrogate("tp", nu_fixed=5.0).fit(x, y, 6)
g = np.linspace(0, 1.2, 7); sg = gp.predict(g)[1]; st = tp.predict(g)[1]
check("TP(nu=5) length-scale equals GP's", abs(tp.ell/gp.ell - 1) < 1e-3, f"(ratio {tp.ell/gp.ell:.5f})")
check("TP(nu=5) predictive scale equals GP's", np.abs(st/sg - 1).max() < 1e-3)
check("TP beta_S = n(nu-2)/nu", abs(tp.beta - n*3/5) < 1e-3, f"({tp.beta:.4f} vs {n*3/5})")

# 3 exact likelihood at alpha = 2
from exact_m2 import ExactM2
from scipy.stats import multivariate_normal as mvn
r = np.random.default_rng(1300+300); x = np.sort((np.arange(30)+r.uniform(size=30))/30*1.2); y = f(x)+0.05*rsas(1.3, 30, r)
E = ExactM2(x, y, S=4000, seed=0)
closed = mvn(np.zeros(30), E.K(1, 0.3) + 2*0.07**2*np.eye(30)).logpdf(E.y)
check("IS likelihood = closed form at alpha=2", abs(E.is_mix(1, 0.3, 0.07, 2.0) - closed) < 0.05)
check("VB bound exact at alpha=2", abs(E.vb(1, 0.3, 0.07, 2.0)[0] - closed) < 1e-6)

# 4 fit
m = StableNoiseGP().fit(x, y, restarts=1, seed=0)
check("GP + SaS noise fits", 0.8 <= m.alpha <= 2.0 and np.isfinite(m.lml), f"(alpha-hat {m.alpha:.2f})")

# 7 fold diagnostic
import fold_remedy
D = {(o["a0"], o["n"], o["rep"]): o for fn in ("1.7_n30",) for o in json.load(open(os.path.join(ROOT, f"results/stable_noise/exact3_a{fn}_main.json")))}
lam = {}
for rep in (0, 1):
    g = np.random.default_rng(1700 + 300 + rep); xx = np.sort((np.arange(30)+g.uniform(size=30))/30*1.2); yy = f(xx)+0.05*rsas(1.7, 30, g)
    lam[rep] = fold_remedy.evaluate(ExactM2(xx, yy, S=10, seed=0), np.array(D[(1.7, 30, rep)]["laplace"]["p"]))["lam_min"]
check("fold diagnostic flags failing fit, passes good fit", abs(lam[1]) < 1e-6 and lam[0] > 0.5, f"(lam_min {lam[1]:.1e} vs {lam[0]:.2f})")
ev = fold_remedy.evaluate(ExactM2(xx, yy, S=10, seed=0), np.array(D[(1.7, 30, 1)]["laplace"]["p"]))
check("fold point: no sentinel, Fisher value defined, flagged machine-dependent",
      (np.isnan(ev["standard"]) or ev["standard"] > -1e9) and np.isfinite(ev["fisher"]) and ev["machine_dependent"],
      f"(standard {ev['standard']:.2f}, fisher {ev['fisher']:.2f})")

# 8 O(n) Laplace and ML rate study
import ssm_laplace
from surrogates import matern52
ys = (y - y.mean()) / y.std(); d = 0
for hp in [(1.0, 0.3, 0.08, 1.3), (0.7, 0.15, 0.05, 1.6)]:
    dense = StableNoiseGP()._laplace(matern52(x, x, hp[0], hp[1]) + 1e-8*np.eye(30), ys, hp[2], hp[3])[1]
    d = max(d, abs(ssm_laplace.laplace(x, ys, *hp, interp="linear")[0] - dense))
check("O(n) Laplace = dense Laplace", d < 1e-3, f"(max diff {d:.1e} nats)")
sys.path.insert(0, os.path.join(ROOT, "scripts", "03_stable_noise"))
cwd = os.getcwd(); import ml_rate_study as MR; os.chdir(cwd)
from crossfit_alpha import robust_scale
stored = [o for o in json.load(open(os.path.join(ROOT, "results/stable_noise/ml_rate_a1.3_main.json"))) if o["n"] == 120 and o["rep"] == 3][0]
xx, yy = MR.data(1.3, 120, 3); _, ac, st = MR.ecf_start(yy)
refit = MR.fit(xx, yy, [0.0, np.log(0.25*np.ptp(xx)), np.log(st/yy.std()), ac])["alpha"]
refit_check("ML rate study refit reproduces stored alpha-hat", refit, stored["ml"]["alpha"])
cwd = os.getcwd(); import ml_unified as MU; os.chdir(cwd)
stU = [o for o in json.load(open(os.path.join(ROOT, "results/stable_noise/ml_unified_a1.3_c1.json"))) if o["n"] == 120 and o["rep"] == 3][0]
xx, yy = MU.data(1.3, 120, 3); m_, sc_ = robust_scale(yy); _, acU, stU_ = MU.ecf_start(yy)
A_ = MU.fit(xx, yy, [0.0, np.log(0.25*np.ptp(xx)), np.log(stU_/sc_), acU]); B_ = MU.fit(xx, yy, [0.0, np.log(0.25*np.ptp(xx)), np.log(0.1), 1.95])
aU = (A_ if A_["lml"] >= B_["lml"] else B_)["alpha"]
refit_check("phase-C unified Laplace-ML study reproduces stored alpha-hat", aU, stU["ml"]["alpha"])

# 9 cross-fitted plug-in estimator (Theorem on efficient estimation) reproduces a stored fit
from crossfit_alpha import crossfit_alpha, fit as cf_fit
CF = [o for o in json.load(open(os.path.join(ROOT, "results/stable_noise/crossfit_a1.3_v2.json"))) if o["n"] == 120 and o["rep"] == 0][0]
g = np.random.default_rng(900000 + 1300 + 1200 + 0); xx = np.sort((np.arange(120) + g.uniform(size=120)) / 120 * 1.2); ee = 0.05 * rsas(1.3, 120, g)
a_md = crossfit_alpha(f(xx) + ee, c=0.15, kind="median")[0]
refit_check("cross-fitted plug-in ML (running-median pilot) reproduces stored alpha-hat", a_md, CF["alpha_med"])
a_gp = crossfit_alpha(f(xx) + ee, x=xx, kind="gp")[0]
refit_check("cross-fitted plug-in ML (GP pilot, median/MAD standardisation) reproduces stored alpha-hat", a_gp, CF["alpha_gp"])

# 10 (4 Nov 2026) odd running-median windows, and the odd-window rerun reproduces a stored fit at n = 480 (k = 11)
from crossfit_alpha import window
check("running-median windows are odd (Lemma)", [window(n_, 0.15) for n_ in (120, 480, 1920, 7680)] == [5, 11, 25, 59])
OD = [o for o in json.load(open(os.path.join(ROOT, "results/stable_noise/crossfit_median_odd_a1.3.json")))["rows"] if o["n"] == 480 and o["rep"] == 0][0]
g = np.random.default_rng(900000 + 1300 + 4800 + 0); xx = np.sort((np.arange(480) + g.uniform(size=480)) / 480 * 1.2); ee = 0.05 * rsas(1.3, 480, g)
a_o = crossfit_alpha(f(xx) + ee, c=0.15, kind="median")[0]
refit_check("odd-window median pilot reproduces stored alpha-hat (n = 480, k = 11)", a_o, OD["alpha_med"], extra=OD["k"] == 11)

# 5 and 6 in a fresh copy of the package
tmp = tempfile.mkdtemp(); pkg = os.path.join(tmp, "pkg")
shutil.copytree(ROOT, pkg, ignore=shutil.ignore_patterns("__pycache__"))
env = dict(os.environ, OMP_NUM_THREADS="1", BO_NIT="2")
def run(script, *args):
    p = subprocess.run([sys.executable, os.path.join(pkg, "scripts", script), *args], env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr
code, out = run("04_bo/bo_study.py", "sas13", "0", "smoketest", "GP,SaS")
res = json.load(open(os.path.join(pkg, "results/bo/bo_sas13_smoketest.json"))) if code == 0 else []
check("two-step BO run (GP, SaS)", code == 0 and len(res) == 2 and len(res[1]["regret"]) == 3, out[-300:] if code else "")
code, out = run("04_bo/bo_study_c4.py", "outliers", "0-1", "smoketest", "GP,RCGP,SaS")
res = json.load(open(os.path.join(pkg, "results/bo/bo_outliers_smoketest.json"))) if code == 0 else []
check("two-step phase-C BO run (GP, RCGP, SaS)", code == 0 and len(res) == 3 and all(len(r["regret"]) == 3 for r in res), out[-300:] if code else "")
code, out = run("04_bo/bo_study_c4.py", "gardner2", "0-1", "smoketest", "GP,SaS")
res = json.load(open(os.path.join(pkg, "results/bo/bo_gardner2_smoketest.json"))) if code == 0 else []
check("two-step constrained BO run (Gardner 2; GP, SaS)", code == 0 and len(res) == 2 and all(len(r["regret"]) == 3 and "n_feasible" in r for r in res), out[-300:] if code else "")
code, out = run("03_stable_noise/lr_test_bootstrap.py", "1.7", "30", "0", "3")
fn = os.path.join(pkg, "results/stable_noise/lr_test_demo_a1.7_n30_rep0_B3.json")
res = json.load(open(fn)) if code == 0 and os.path.exists(fn) else {}
check("data-set-specific bootstrap LR test runs (B = 3)", code == 0 and len(res.get("LR_boot", [])) == 3 and 0 < res.get("p_value", 0) <= 1, out[-300:] if code else "")

def same(a, b, tol=1e-9):
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(same(u, v, tol) for u, v in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol * max(1, abs(a))
    return a == b
for script, stage, table in [("03_stable_noise/summarise_consistency.py", "stable_noise", "consistency_table.json"),
                             ("03_stable_noise/summarise_exact.py", "stable_noise", "exact_table.json"),
                             ("03_stable_noise/summarise_ml_rate.py", "stable_noise", "ml_rate_summary.json"),
                             ("03_stable_noise/summarise_ml_unified.py", "stable_noise", "ml_unified_summary.json"),
                             ("03_stable_noise/summarise_crossfit.py", "stable_noise", "crossfit_summary.json"),
                             ("03_stable_noise/summarise_exact_c2.py", "stable_noise", "exact_c2_table.json"),
                             ("03_stable_noise/summarise_lr_bootstrap.py", "stable_noise", "lr_bootstrap_summary.json"),
                             ("04_bo/summarise_bo.py", "bo", "bo_summary.json"),
                             ("04_bo/summarise_bo_c4.py", "bo", "bo_summary_c4.json")]:
    before = json.load(open(os.path.join(ROOT, "results", stage, table)))
    code, out = run(script)
    after = json.load(open(os.path.join(pkg, "results", stage, table))) if code == 0 else None
    check(f"{script} regenerates {table}", code == 0 and same(before, after), out[-300:] if code else "")
    if table == "bo_summary.json" and after:
        fo = after.get("fold_overall", {})
        check("BO fold rate (Sec. 4.5): 100 of 1,840 SaS fits", fo.get("fold_count") == 100 and fo.get("n_fits") == 1840,
              f"({fo.get('fold_count')} of {fo.get('n_fits')})")
    if table == "bo_summary_c4.json" and after:
        nf = sum(P["sas_fits"] for P in after["problems"].values()); ng = sum(P["sas_gap_gt5"] for P in after["problems"].values())
        nsig = sum(t["significant"] for t in after["tests"])
        nagree = sum(t["significant"] == t["sign_test"]["significant"] for t in after["tests"])
        check("phase-C BO (Sec. 5.3): 3,164 of 18,200 SaS fits with Laplace gap > 5; 7 of 36 Holm-significant; sign test agrees in 35 of 36",
              nf == 18200 and ng == 3164 and nsig == 7 and len(after["tests"]) == 36 and nagree == 35, f"({ng} of {nf}; {nsig} significant; sign test agrees {nagree})")
# 11 (4 Nov 2026) Tables 1 and 2 are generated from the stored summaries (no hand transcription)
code, out = run("03_stable_noise/make_tables_12.py")
same_tab = code == 0 and all(open(os.path.join(pkg, "paper", t)).read() == open(os.path.join(ROOT, "paper", t)).read() for t in ("tab_cons.tex", "tab_cf.tex"))
check("make_tables_12.py regenerates paper/tab_cons.tex and paper/tab_cf.tex", same_tab, out[-300:] if code else "")
# 12 (4 Nov 2026) CVaR_0.9 is the mean of the worst 10% of runs, also with ties at the 0.9-quantile (outliers/GP: 22 runs
# are >= the quantile), and the BO table panel (a) is generated from the summary
Rb = json.load(open(os.path.join(ROOT, "results/bo/bo_outliers_c4.json"))); Sb = json.load(open(os.path.join(ROOT, "results/bo/bo_summary_c4.json")))
fin = np.sort([o["regret"][-1] for o in Rb if o["kind"] == "GP"])[::-1]
nq = int((fin >= np.quantile(fin, 0.9)).sum())
check("CVaR_0.9 = mean of the worst 10% (tied cut-off)", abs(Sb["problems"]["outliers"]["surrogates"]["GP"]["cvar90"] - fin[:len(fin) // 10].mean()) < 1e-12 and nq > len(fin) // 10,
      f"(stored {Sb['problems']['outliers']['surrogates']['GP']['cvar90']:.6f}, worst-10% {fin[:len(fin) // 10].mean():.6f}, runs >= quantile {nq})")
code, out = run("04_bo/make_table_bo.py")
check("make_table_bo.py regenerates paper/tab_bo_a.tex", code == 0 and open(os.path.join(pkg, "paper", "tab_bo_a.tex")).read() == open(os.path.join(ROOT, "paper", "tab_bo_a.tex")).read(), out[-300:] if code else "")
shutil.rmtree(tmp)
import ssm_laplace as _S
print(f"\nO(n) Laplace kernel used: {_S.BACKEND}")
if ok and n_platform:
    print(f"ALL PASSED ({n_platform} refit(s) [PASS*] agree within the optimiser tolerance 1e-4, not exactly; "
          "exact agreement is expected only on the reference platform; see README)")
else:
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
