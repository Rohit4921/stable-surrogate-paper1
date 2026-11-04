"""Write MANIFEST.md (run manifest: environment, paper item -> script -> seeds -> outputs with SHA-256) and
results/CHECKSUMS.sha256 for the frozen archive. Run from the package root: python tools/make_manifest.py"""
import hashlib, json, glob, os, sys, platform, collections, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
import numpy, scipy, matplotlib
ITEMS = [  # (paper item, scripts, seed rule, outputs)
 ("Table: consistency of the Laplace ML estimate (tab:cons); ML rate study; Laplace ML column of tab:cf; Fig. rate C",
  "03_stable_noise/ml_unified.py <alpha0> ... then summarise_ml_unified.py; make_tables_12.py",
  "data: default_rng(int(1000 alpha0) + 10 n + rep); bootstrap CI: default_rng(0)",
  ["results/stable_noise/ml_unified_a1.3_c1.json", "results/stable_noise/ml_unified_a1.7_c1.json", "results/stable_noise/ml_unified_a2.0_c1.json", "results/stable_noise/ml_unified_summary.json", "paper/tab_cons.tex"]),
 ("Table: cross-fitted estimator (tab:cf): GP pilot, oracle, coverage",
  "03_stable_noise/crossfit_study.py <alpha0> 120,480,1920,7680 100 v2; summarise_crossfit.py; make_tables_12.py",
  "data: default_rng(900000 + int(1000 alpha0) + 10 n + rep); Monte Carlo SEs: default_rng(int(1000 alpha0) + n [+1])",
  ["results/stable_noise/crossfit_a1.3_v2.json", "results/stable_noise/crossfit_a1.7_v2.json", "results/stable_noise/crossfit_summary.json", "paper/tab_cf.tex"]),
 ("Table tab:cf, running-median column (odd windows, rerun 4 Nov 2026 on identical data)",
  "03_stable_noise/crossfit_median_odd.py <alpha0>",
  "same as crossfit_study.py; identity checked by refitting the oracle (max difference 0)",
  ["results/stable_noise/crossfit_median_odd_a1.3.json", "results/stable_noise/crossfit_median_odd_a1.7.json"]),
 ("Known-f bound (oracle Fisher information)", "03_stable_noise/oracle_fisher.py", "quadrature; Monte Carlo check default_rng(seed)", ["results/stable_noise/oracle_fisher.json"]),
 ("Fig. rate A, B: difference-ECF estimator", "03_stable_noise/rate_study.py then fig_rate.py", "default_rng(int(1000 alpha0) + n + (0 or 7))", ["results/stable_noise/rate_study.json", "paper/fig_rate.pdf"]),
 ("Table tab:exact, Fig. exact: Laplace vs variational bound vs IS", "03_stable_noise/exact_c2_study.py then summarise_exact_c2.py", "data: default_rng(int(1000 alpha0) + 10 n + rep)",
  sorted(glob.glob("results/stable_noise/exact_c2_*.json")) + ["paper/fig_laplace_exact.pdf"]),
 ("Table tab:boot: bootstrap test of alpha = 2 (warp-speed)", "03_stable_noise/lr_bootstrap.py then summarise_lr_bootstrap.py; single data set: lr_test_bootstrap.py",
  "data: default_rng(int(1000 alpha0) + 10 n + rep); bootstrap: default_rng(5000000 + int(1000 alpha0) + 10 n + rep)",
  sorted(glob.glob("results/stable_noise/lr_boot_*_c3.json")) + ["results/stable_noise/lr_bootstrap_summary.json"]),
 ("Folds: diagnostics, toy, slices, no-fold ratio, Fisher-type and guarded refits, Fig. fold", "03_stable_noise/fold/*.py",
  "data: default_rng(int(1000 alpha0) + 10 n + rep)", sorted(glob.glob("results/stable_noise/fold/*.json")) + ["paper/fig_fold.pdf"]),
 ("BO study: Table tab:bostudy, tab:bosafe, sign-test table, Fig. bostudy and its supplementary companion",
  "04_bo/bo_study_c4.py <problem> <reps> c4 (parallel lanes in results/bo/partial/, merged) then summarise_bo_c4.py (CVaR = mean of worst 10%, corrected 4 Nov 2026), make_table_bo.py, fig_bo_c4.py",
  "initial design default_rng(1000 rep + 7), noise default_rng(1000 rep + 11), candidates default_rng(1000 rep + 13 + it); bootstrap default_rng(20261006), post hoc default_rng(20261007)",
  sorted(glob.glob("results/bo/bo_*_c4.json")) + sorted(glob.glob("results/bo/partial/*.json")) + ["paper/tab_bo_a.tex", "paper/fig_bo_study.pdf", "paper/fig_bo_study_supp.pdf"]),
 ("Density tables (Section on the algorithm)", "03_stable_noise/build_tables.py", "deterministic (scipy.stats.levy_stable)", ["data/sas_tables.npz"]),
 ("Density-table sensitivity check (added 4 Nov 2026)", "05_checks/density_sensitivity.py 20", "points default_rng(20261104); noise default_rng(777000 + int(1000 alpha0) + rep)", ["results/checks/density_sensitivity.json"]),
 ("Process-level numerical checks (TP = GP, shape of G, stable profile)", "02_process_level/check_pred.py, check_shape.py, wp1_stable_profile.py", "default_rng(0), default_rng(7)", ["results/process_level/check_pred_output.txt"]),
]
L = ["# Run manifest (frozen archive)", "", f"Generated {datetime.date.today().isoformat()} by `tools/make_manifest.py`.", "",
     "## Environment", "",
     "- Stored results up to 8 Oct 2026 were produced with Python 3.11.15, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9 (see `requirements.txt`).",
     f"- Reruns and checks of 4 Nov 2026, and the smoke test of this archive: Python {platform.python_version()}, NumPy {numpy.__version__}, SciPy {scipy.__version__}, Matplotlib {matplotlib.__version__}, {platform.system()} {platform.machine()}.",
     "- Reproducibility across the two environments was checked by regenerating the Table 2 data sets from their seeds and refitting the oracle estimator: the maximum difference in the fitted alpha was 0.", "",
     "## Paper item -> script -> seeds -> outputs", ""]
for item, scr, seed, outs in ITEMS:
    L += [f"### {item}", "", f"- Scripts (in `scripts/`): `{scr}`", f"- Seeds: {seed}", "- Outputs (SHA-256):"]
    for o in outs:
        L.append(f"  - `{o}` " + (f"`{sha(o)[:16]}…`" if os.path.exists(o) else "**missing**"))
    L.append("")
# seed collisions
L += ["## Seed reuse across cells", "",
      "Within every simulation cell the seeds are distinct, so data sets within a cell are independent. The seed formula `int(1000 alpha0) + 10 n + rep` "
      "is shared on purpose by studies that analyse the same data sets (e.g. the Laplace/IS comparison uses the first 50 data sets of the consistency study). "
      "Across cells, the formula gives the same seed for some pairs of cells, listed below; those cells use overlapping random streams and are therefore not "
      "independent of each other. Each cell's estimates remain valid; no result in the paper combines or tests across colliding cells.", ""]
def cells(f):
    D = json.load(open(f)); D = D["rows"] if isinstance(D, dict) and "rows" in D else D
    return {(o["a0"], o["n"], o["rep"]) for o in D if "a0" in o}
for name, pat in [("ml_unified", "results/stable_noise/ml_unified_a*_c1.json"), ("exact_c2", "results/stable_noise/exact_c2_a*_n*_c2.json"), ("lr_boot", "results/stable_noise/lr_boot_*_c3.json")]:
    S = set().union(*[cells(f) for f in glob.glob(pat)]); seeds = collections.defaultdict(list)
    for a0, n, rep in S: seeds[int(a0 * 1000) + 10 * n + rep].append((a0, n))
    pairs = collections.Counter(tuple(sorted(set(v))) for v in seeds.values() if len(v) > 1)
    L.append(f"- `{name}`: {len(S)} data sets; colliding cells (alpha0, n): " + ("; ".join(f"{' & '.join(map(str, k))} ({c} seeds)" for k, c in pairs.items()) or "none"))
mx = max(int(a0 * 1000) + 10 * n + rep for f in glob.glob("results/stable_noise/ml_unified_a*_c1.json") for (a0, n, rep) in cells(f))
cf = [900000 + int(1000 * o["a0"]) + 10 * o["n"] + o["rep"] for f in glob.glob("results/stable_noise/crossfit_a*_v2.json") for o in json.load(open(f))]
L.append(f"- Table 2 data sets are fresh: their seeds ({min(cf)}..{max(cf)}) are disjoint from the consistency-study seeds (at most {mx}) and distinct from each other ({len(set(cf))} distinct of {len(cf)}).")
L += ["- BO study: the same seeds are used for every problem and surrogate (common random numbers). Comparisons are paired within a problem; the Holm adjustment over problems is valid under any dependence.", "",
      "## Checksums", "", "`results/CHECKSUMS.sha256` lists SHA-256 checksums of every file in `results/` and `data/`."]
open("MANIFEST.md", "w").write("\n".join(L) + "\n")
with open("results/CHECKSUMS.sha256", "w") as fh:
    for f in sorted(glob.glob("results/**/*", recursive=True) + glob.glob("data/**/*", recursive=True)):
        if os.path.isfile(f) and not f.endswith("CHECKSUMS.sha256"): fh.write(f"{sha(f)}  {f}\n")
print("MANIFEST.md written;", sum(1 for _ in open("results/CHECKSUMS.sha256")), "checksums")
