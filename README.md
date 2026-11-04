# Heavy tails in Gaussian-process surrogates: code, data and results

Code, stored results and a smoke test for

> N. S. Upadhye and R. K. Mani. *Heavy tails in Gaussian-process surrogates: process-level profiling degeneracy
> versus observation-level identifiability.* Submitted to *Statistica Sinica*.

Every simulation, table and figure of the article can be regenerated with the scripts here. The per-replicate results of
every study are stored, so the tables can be checked without rerunning the long simulations.

This work was supported by the Anusandhan National Research Foundation (ANRF), Government of India, under its
ARG-MATRICS scheme (grant ANRF/ARGM/2025/000532/MTR).

## Quick start

```bash
pip install -r requirements.txt      # numpy, scipy, pandas, matplotlib (versions of the stored results)
python tests/smoke_test.py           # 34 checks, about 10 minutes on one core
```

On a newer Python for which those exact versions are not available, `pip install numpy scipy pandas matplotlib`
also works (see *Tested environments*).

A C compiler (`gcc`) is optional. The O(n) Laplace approximation uses the C kernel `src/ssm_kernel.c`, compiled on
first import. Without `gcc` the same algorithm runs in pure Python, with identical results and a warning; it is
about 45 times slower per linear solve, which matters only for the long studies. On Windows, `gcc` is available
through MinGW-w64.

The smoke test checks, among other things, that

- the symmetric α-stable (SαS) density matches `scipy.stats.levy_stable`;
- a Student-t process with fixed ν reproduces the GP;
- the importance-sampling likelihood equals the closed form at α = 2;
- the fold diagnostic flags a failing Laplace fit and passes a good one;
- the O(n) Laplace approximation agrees with the dense one;
- refitting stored data sets reproduces the stored estimates of α (five checks);
- short BO runs (unconstrained and constrained) and the bootstrap test of α = 2 complete;
- the summary scripts regenerate every stored summary table exactly, and the table generators regenerate the
  table bodies in `paper/`.

It ends with `ALL PASSED`, and prints which Laplace kernel (C or Python) was used.

## Reproducing on other platforms

The stored results were computed on Linux x86-64 (versions in `MANIFEST.md`). There, the five refit checks agree
with the stored estimates to better than 1e-6.

On other platforms (another operating system, C library, BLAS or Python build) floating-point results can differ in
the last bits, and the Nelder–Mead fits amplify such differences. These fits stop at a tolerance of 1e-4 (Laplace ML)
or 1e-3 (GP-pilot hyperparameters), so the estimates are not determined more precisely than about 1e-4 on any
platform. A refit that differs from the stored value by less than 1e-4 is therefore reported as `[PASS*]`, with
the difference printed; a larger difference fails. On Windows (x64, Python 3.14, pure-Python kernel), two of the five
refits gave `[PASS*]`, with differences of 2.9e-5 and 6.7e-6; the other three agreed to better than 1e-12.

## Layout

```
src/        library code (imported by all scripts)
  surrogates.py        GP, Student-t process (TP), GP with Student-t noise (GP-t), RCGP-type; Matérn-5/2; EI
  stable_noise_gp.py   GP with SαS noise: density tables, Laplace fit, prediction, SαS sampler
  ssm_laplace.py       O(n) Laplace approximation for the Matérn-5/2 kernel on one input (Kalman filter/smoother)
  ssm_kernel.c         C kernel for ssm_laplace.py (pure-Python fallback included in ssm_laplace.py)
  exact_m2.py          exact likelihood: importance sampling, annealed-IS check, variational bound
  crossfit_alpha.py    cross-fitted plug-in estimator of α (running-median and GP pilots)
  ecf_alpha.py         difference-based empirical-characteristic-function estimator of α
  fold_remedy.py       Laplace variants (standard, Fisher-type, guarded) and the fold diagnostic
  procedure.py         surrogate-selection rule of the companion demonstration
  nifty_objective.py   NIFTY 50 moving-average objective of the companion demonstration
  figures_style.py, _paths.py
scripts/    one folder per part of the article; every script runs from any folder
  01_erp_demo/         companion demonstration (TP vs GP vs GP-t on fixed data sets)
  02_process_level/    numerical checks for the process-level results
  03_stable_noise/     observation-level results: estimation of α, Laplace vs exact likelihood, bootstrap test
    fold/              folds of the posterior mode
  04_bo/               repeated sequential BO study
  05_checks/           density-table sensitivity check
  get_nifty50.py       downloads the NIFTY 50 data (see Data)
data/       sas_tables.npz (SαS log-density tables, built by scripts/03_stable_noise/build_tables.py)
results/    stored outputs (JSON) of every study; CHECKSUMS.sha256
figures/    figures (PNG and PDF)
paper/      table bodies (tab_*.tex) and figures (PDF) used by the article, as written by the scripts
tests/      smoke_test.py
tools/      make_manifest.py (writes MANIFEST.md and results/CHECKSUMS.sha256)
```

## Which script produces which result

`MANIFEST.md` maps each table and figure of the article to its scripts, seed rule and output files, with SHA-256
checksums; `results/CHECKSUMS.sha256` lists checksums of every file in `results/` and `data/`. Files in `results/`
that `MANIFEST.md` does not list are earlier runs, kept for the record and used by some checks (for example
`results/stable_noise/v1_tables/`, `*_main.json`, `crossfit_summary_evenk_2026-10.json`,
`bo_summary_c4_threshold-cvar_2026-10.json`). `bo_nifty_c4.json` is a BO run on the NIFTY 50 objective that is not
part of the analysis in the article.

Command lines for the main studies (one process per line; set `OMP_NUM_THREADS=1`, since BLAS over-subscription
makes the fits about ten times slower):

```bash
# estimation of alpha: Laplace ML (100 data sets per cell), then the summary and table bodies
python scripts/03_stable_noise/ml_unified.py 1.3 30,60,120,240,480,960,1920,3840,7680 100 c1   # likewise 1.7
python scripts/03_stable_noise/ml_unified.py 2.0 30,60,120,240,480,960,1920 100 c1
python scripts/03_stable_noise/summarise_ml_unified.py

# cross-fitted plug-in estimator (GP and running-median pilots, oracle), then the odd-window median pilot
python scripts/03_stable_noise/crossfit_study.py 1.3 120,480,1920,7680 100 v2               # likewise 1.7
python scripts/03_stable_noise/crossfit_median_odd.py 1.3                                    # likewise 1.7
python scripts/03_stable_noise/summarise_crossfit.py
python scripts/03_stable_noise/make_tables_12.py

# difference-ECF estimator and the known-f bound
python scripts/03_stable_noise/rate_study.py
python scripts/03_stable_noise/oracle_fisher.py
python scripts/03_stable_noise/fig_rate.py

# Laplace vs importance sampling vs variational bound (n = 30, 60; alpha0 = 1.3, 1.7, 2.0)
python scripts/03_stable_noise/exact_c2_study.py 1.3 30 0-50 c2
python scripts/03_stable_noise/summarise_exact_c2.py

# parametric-bootstrap test of alpha = 2 (n = 30, 60, 120, 240; 0-300 data sets for alpha0 = 1.3, 1.7)
python scripts/03_stable_noise/lr_bootstrap.py 2.0 60 0-1000 c3
python scripts/03_stable_noise/summarise_lr_bootstrap.py

# folds of the posterior mode
python scripts/03_stable_noise/fold/diag_fold.py
python scripts/03_stable_noise/fold/toy_exact.py
python scripts/03_stable_noise/fold/slice_toy.py
python scripts/03_stable_noise/fold/fold_study.py small
python scripts/03_stable_noise/fold/fold_study.py large 120 _120                             # likewise 240 _240
python scripts/03_stable_noise/fold/guard_small.py
python scripts/03_stable_noise/fold/nofold_check.py                                          # after fold_study.py large
python scripts/03_stable_noise/fold/fig_fold.py

# repeated BO study (problems: gauss, outliers, sas13, t2, skewout, branin2, hart6, gardner1, gardner2)
python scripts/04_bo/bo_study_c4.py outliers 0-100 c4                                       # 0-50 for branin2, hart6, gardner1, gardner2
python scripts/04_bo/summarise_bo_c4.py
python scripts/04_bo/make_table_bo.py
python scripts/04_bo/fig_bo_c4.py

# density-table sensitivity check
python scripts/05_checks/density_sensitivity.py 20

# companion demonstration (needs the NIFTY 50 data for the NIFTY part)
python scripts/01_erp_demo/figures.py
python scripts/01_erp_demo/study.py synthetic 9000 _validation
python scripts/01_erp_demo/study.py nifty
python scripts/01_erp_demo/fig4.py
```

Run times on one core: minutes for the checks and summaries, about 1 h per (α₀, n) cell of the Laplace/IS
comparison, about 4 h per α₀ for the Laplace-ML study, and about 4 min per repetition of a one-dimensional BO problem
(longer in two and six dimensions).

All studies use fixed seeds; the seed rule of each study is given in `MANIFEST.md`.

## Data

**Simulated data.** All data in the estimation studies and the BO study are simulated by the scripts from fixed
seeds.

**SαS density tables.** `data/sas_tables.npz` holds log-density tables of the standard SαS law for
α = 0.50–1.99, built from `scipy.stats.levy_stable` by `scripts/03_stable_noise/build_tables.py`.
`scripts/05_checks/density_sensitivity.py` quantifies the table error (at most 1.8e-4 nats for |z| ≤ 60) and its
effect on the fitted α (at most 1.1e-4).

**NIFTY 50 index (companion demonstration only).** The companion demonstration uses daily opening and closing values
of the NIFTY 50 index (Yahoo Finance, ticker `^NSEI`), 14 September 2015 to 21 September 2026 (2,712 trading days).
Yahoo Finance does not permit redistribution, so the file is not included. To download it:

```bash
pip install yfinance
python scripts/get_nifty50.py
```

This writes `data/nifty50_daily_open_close.csv` and checks it against the SHA-256 checksum of the file used for the
article (`6bd750bd6be1c1f49c4481ffaf95ebb5859ce065b8b526ef5ad018410e93e3af`). If Yahoo has revised past values, the
script says so, and results that use these data may differ slightly from the stored ones. Only the scripts of the
companion demonstration and the `nifty` BO problem need this file; everything else, including the smoke test, runs
without it.

## Notes for reproduction

1. **Parallel runs.** Some BO problems were run in parallel pieces (files in `results/bo/partial/`) and merged,
   keeping the first record of each (surrogate, repetition). All seeds depend only on the repetition index, so the
   split does not affect the results.
2. **Values at a fold are machine-dependent.** At the failing Laplace optima of the Laplace/IS comparison, the
   smallest eigenvalue of I + K^½ W K^½ is zero to working precision. There, the posterior mode and the standard
   Laplace value depend on where the Newton iteration stops, which can differ between machines and library builds.
   `fold_remedy.evaluate` returns NaN for an undefined Laplace value and flags `machine_dependent`; the Fisher-type
   value is always defined. The classification of these points and every number quoted in the article are
   unaffected.
3. **O(n) vs dense Laplace.** `ssm_laplace.laplace(..., interp="linear")` and the dense `StableNoiseGP._laplace`
   differ by up to about 1e-3 nats when K is ill-conditioned, because the dense code adds a jitter 1e-8 I to K.
4. **Exact likelihood.** The importance-sampling estimate (`ExactM2.is_mix`) uses truncated weights and common random
   numbers. The scripts report the effective sample size, and `ais_bridge` gives an independent annealed-IS check.

## Tested environments

- Linux x86-64, Python 3.11.15, NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.2, Matplotlib 3.10.9: stored results up to
  8 October 2026 (`requirements.txt`).
- Linux x86-64, Python 3.13, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2: reruns and checks of November 2026; smoke
  test passes with both the C and the pure-Python kernel, all refits exact.
- Windows x64, Python 3.14, NumPy 2.5.3, pandas 3.0.5, no C compiler (pure-Python kernel): smoke test passes, two
  refits `[PASS*]` (see *Reproducing on other platforms*).

## Licence

The code is released under the MIT licence (`LICENSE`). The stored results (`results/`), figures (`figures/`) and
density tables (`data/sas_tables.npz`) are released under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); this includes the generated table bodies and figures in
`paper/`. The text of the article is not part of this repository.

## Citation

If you use this code, please cite the article above. Citation metadata for this software are in `CITATION.cff`.
