# Run manifest (frozen archive)

Generated 2026-11-04 by `tools/make_manifest.py`.

## Environment

- Stored results up to 8 Oct 2026 were produced with Python 3.11.15, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9 (see `requirements.txt`).
- Reruns and checks of 4 Nov 2026, and the smoke test of this archive: Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, Linux x86_64.
- Reproducibility across the two environments was checked by regenerating the Table 2 data sets from their seeds and refitting the oracle estimator: the maximum difference in the fitted alpha was 0.

## Paper item -> script -> seeds -> outputs

### Table: consistency of the Laplace ML estimate (tab:cons); ML rate study; Laplace ML column of tab:cf; Fig. rate C

- Scripts (in `scripts/`): `03_stable_noise/ml_unified.py <alpha0> ... then summarise_ml_unified.py; make_tables_12.py`
- Seeds: data: default_rng(int(1000 alpha0) + 10 n + rep); bootstrap CI: default_rng(0)
- Outputs (SHA-256):
  - `results/stable_noise/ml_unified_a1.3_c1.json` `3d58ef0f4a478309…`
  - `results/stable_noise/ml_unified_a1.7_c1.json` `f59cead8067f471e…`
  - `results/stable_noise/ml_unified_a2.0_c1.json` `f0b1cd3a598ec6ec…`
  - `results/stable_noise/ml_unified_summary.json` `335f885a3e462ac0…`
  - `paper/tab_cons.tex` `085d355f30a8beb7…`

### Table: cross-fitted estimator (tab:cf): GP pilot, oracle, coverage

- Scripts (in `scripts/`): `03_stable_noise/crossfit_study.py <alpha0> 120,480,1920,7680 100 v2; summarise_crossfit.py; make_tables_12.py`
- Seeds: data: default_rng(900000 + int(1000 alpha0) + 10 n + rep); Monte Carlo SEs: default_rng(int(1000 alpha0) + n [+1])
- Outputs (SHA-256):
  - `results/stable_noise/crossfit_a1.3_v2.json` `a569f9b82fc47921…`
  - `results/stable_noise/crossfit_a1.7_v2.json` `502338daf9bfd790…`
  - `results/stable_noise/crossfit_summary.json` `dfad0df8d735a78f…`
  - `paper/tab_cf.tex` `ace60f4053d8924e…`

### Table tab:cf, running-median column (odd windows, rerun 4 Nov 2026 on identical data)

- Scripts (in `scripts/`): `03_stable_noise/crossfit_median_odd.py <alpha0>`
- Seeds: same as crossfit_study.py; identity checked by refitting the oracle (max difference 0)
- Outputs (SHA-256):
  - `results/stable_noise/crossfit_median_odd_a1.3.json` `8e22d82f7b035bde…`
  - `results/stable_noise/crossfit_median_odd_a1.7.json` `51799f8d48ffd4fb…`

### Known-f bound (oracle Fisher information)

- Scripts (in `scripts/`): `03_stable_noise/oracle_fisher.py`
- Seeds: quadrature; Monte Carlo check default_rng(seed)
- Outputs (SHA-256):
  - `results/stable_noise/oracle_fisher.json` `5e4997e5f3396d25…`

### Fig. rate A, B: difference-ECF estimator

- Scripts (in `scripts/`): `03_stable_noise/rate_study.py then fig_rate.py`
- Seeds: default_rng(int(1000 alpha0) + n + (0 or 7))
- Outputs (SHA-256):
  - `results/stable_noise/rate_study.json` `de04dcd97844e665…`
  - `paper/fig_rate.pdf` `db6b60d484638308…`

### Table tab:exact, Fig. exact: Laplace vs variational bound vs IS

- Scripts (in `scripts/`): `03_stable_noise/exact_c2_study.py then summarise_exact_c2.py`
- Seeds: data: default_rng(int(1000 alpha0) + 10 n + rep)
- Outputs (SHA-256):
  - `results/stable_noise/exact_c2_a1.3_n30_c2.json` `15d4f379d181d10e…`
  - `results/stable_noise/exact_c2_a1.3_n60_c2.json` `6758518ccb0364e9…`
  - `results/stable_noise/exact_c2_a1.7_n30_c2.json` `8284d1cb1e6a8b08…`
  - `results/stable_noise/exact_c2_a1.7_n60_c2.json` `94eb4072bbae14b6…`
  - `results/stable_noise/exact_c2_a2.0_n30_c2.json` `fa31039f80cceb23…`
  - `results/stable_noise/exact_c2_a2.0_n60_c2.json` `329db5dc4b26f8be…`
  - `results/stable_noise/exact_c2_table.json` `e8a2c7e1485e805a…`
  - `paper/fig_laplace_exact.pdf` `b5a9dc31320056a9…`

### Table tab:boot: bootstrap test of alpha = 2 (warp-speed)

- Scripts (in `scripts/`): `03_stable_noise/lr_bootstrap.py then summarise_lr_bootstrap.py; single data set: lr_test_bootstrap.py`
- Seeds: data: default_rng(int(1000 alpha0) + 10 n + rep); bootstrap: default_rng(5000000 + int(1000 alpha0) + 10 n + rep)
- Outputs (SHA-256):
  - `results/stable_noise/lr_boot_a1.3_n120_c3.json` `be63a214833c2078…`
  - `results/stable_noise/lr_boot_a1.3_n240_c3.json` `51456aacf5bafa8a…`
  - `results/stable_noise/lr_boot_a1.3_n30_c3.json` `a295c268c426558d…`
  - `results/stable_noise/lr_boot_a1.3_n60_c3.json` `7af0d375610bf7b8…`
  - `results/stable_noise/lr_boot_a1.7_n120_c3.json` `fd61af7c7883c5db…`
  - `results/stable_noise/lr_boot_a1.7_n240_c3.json` `93b2f1b4970b7598…`
  - `results/stable_noise/lr_boot_a1.7_n30_c3.json` `f5d41811868fa8bb…`
  - `results/stable_noise/lr_boot_a1.7_n60_c3.json` `9296564b7f9c5dd9…`
  - `results/stable_noise/lr_boot_a2.0_n120_c3.json` `ef641b44e7b13d36…`
  - `results/stable_noise/lr_boot_a2.0_n240_c3.json` `b1c9935299983a8d…`
  - `results/stable_noise/lr_boot_a2.0_n30_c3.json` `1c374264bb9989be…`
  - `results/stable_noise/lr_boot_a2.0_n60_c3.json` `f64437ec8feb9bc1…`
  - `results/stable_noise/lr_bootstrap_summary.json` `85f78d56bafbf32a…`

### Folds: diagnostics, toy, slices, no-fold ratio, Fisher-type and guarded refits, Fig. fold

- Scripts (in `scripts/`): `03_stable_noise/fold/*.py`
- Seeds: data: default_rng(int(1000 alpha0) + 10 n + rep)
- Outputs (SHA-256):
  - `results/stable_noise/fold/diag_fold.json` `3701a1db488a15e8…`
  - `results/stable_noise/fold/diag_fold_c2.json` `b9705649a72fcbcd…`
  - `results/stable_noise/fold/fold_large_120.json` `124aa22167db168b…`
  - `results/stable_noise/fold/fold_large_240.json` `c91dc206b67910f1…`
  - `results/stable_noise/fold/fold_small.json` `750948839d8b7f4f…`
  - `results/stable_noise/fold/guard_small.json` `38413b7544bbafd8…`
  - `results/stable_noise/fold/nofold_check.json` `d30f8792efe0e0cf…`
  - `results/stable_noise/fold/slice_toy.json` `89dc434c762c8b34…`
  - `results/stable_noise/fold/toy_exact.json` `9ba2dd3476fddc09…`
  - `paper/fig_fold.pdf` `c043b8c253ded9de…`

### BO study: Table tab:bostudy, tab:bosafe, sign-test table, Fig. bostudy and its supplementary companion

- Scripts (in `scripts/`): `04_bo/bo_study_c4.py <problem> <reps> c4 (parallel lanes in results/bo/partial/, merged) then summarise_bo_c4.py (CVaR = mean of worst 10%, corrected 4 Nov 2026), make_table_bo.py, fig_bo_c4.py`
- Seeds: initial design default_rng(1000 rep + 7), noise default_rng(1000 rep + 11), candidates default_rng(1000 rep + 13 + it); bootstrap default_rng(20261006), post hoc default_rng(20261007)
- Outputs (SHA-256):
  - `results/bo/bo_branin2_c4.json` `a9e508069551c779…`
  - `results/bo/bo_gardner1_c4.json` `f316932eb40ac7da…`
  - `results/bo/bo_gardner2_c4.json` `57a8d6cb6b1090c5…`
  - `results/bo/bo_gauss_c4.json` `dfd3112f6bdfbe13…`
  - `results/bo/bo_hart6_c4.json` `af0226cb8ebf1ffd…`
  - `results/bo/bo_nifty_c4.json` `15c1dbf563ddb616…`
  - `results/bo/bo_outliers_c4.json` `5252573582b7f128…`
  - `results/bo/bo_sas13_c4.json` `9ac0b54ed3e40af3…`
  - `results/bo/bo_skewout_c4.json` `ec4239de536e2441…`
  - `results/bo/bo_summary_c4.json` `a8ed796ed9100bd8…`
  - `results/bo/bo_t2_c4.json` `23ff332250c2b5c2…`
  - `results/bo/partial/bo_branin2_c4b.json` `5081cd68f8616fc9…`
  - `results/bo/partial/bo_branin2_c4c.json` `eebfad7992cbcf97…`
  - `results/bo/partial/bo_gardner2_c4b.json` `bd7772e42f29cde6…`
  - `paper/tab_bo_a.tex` `47adc8a30a0a05f3…`
  - `paper/fig_bo_study.pdf` `c941464013f840fe…`
  - `paper/fig_bo_study_supp.pdf` `1f47e6b938af00fa…`

### Density tables (Section on the algorithm)

- Scripts (in `scripts/`): `03_stable_noise/build_tables.py`
- Seeds: deterministic (scipy.stats.levy_stable)
- Outputs (SHA-256):
  - `data/sas_tables.npz` `aa9f7a1ffb9494c6…`

### Density-table sensitivity check (added 4 Nov 2026)

- Scripts (in `scripts/`): `05_checks/density_sensitivity.py 20`
- Seeds: points default_rng(20261104); noise default_rng(777000 + int(1000 alpha0) + rep)
- Outputs (SHA-256):
  - `results/checks/density_sensitivity.json` `ea5700689d0b4643…`

### Process-level numerical checks (TP = GP, shape of G, stable profile)

- Scripts (in `scripts/`): `02_process_level/check_pred.py, check_shape.py, wp1_stable_profile.py`
- Seeds: default_rng(0), default_rng(7)
- Outputs (SHA-256):
  - `results/process_level/check_pred_output.txt` `a52f86d1c01fe09c…`

## Seed reuse across cells

Within every simulation cell the seeds are distinct, so data sets within a cell are independent. The seed formula `int(1000 alpha0) + 10 n + rep` is shared on purpose by studies that analyse the same data sets (e.g. the Laplace/IS comparison uses the first 50 data sets of the consistency study). Across cells, the formula gives the same seed for some pairs of cells, listed below; those cells use overlapping random streams and are therefore not independent of each other. Each cell's estimates remain valid; no result in the paper combines or tests across colliding cells.

- `ml_unified`: 2500 data sets; colliding cells (alpha0, n): (1.7, 60) & (2.0, 30) (100 seeds)
- `exact_c2`: 300 data sets; colliding cells (alpha0, n): (1.7, 60) & (2.0, 30) (50 seeds)
- `lr_boot`: 4400 data sets; colliding cells (alpha0, n): (1.3, 120) & (2.0, 30) & (2.0, 60) (200 seeds); (1.7, 120) & (2.0, 60) (200 seeds); (1.3, 120) & (1.7, 60) & (2.0, 30) (100 seeds); (1.7, 60) & (2.0, 30) (200 seeds); (1.3, 60) & (1.7, 30) (200 seeds)
- Table 2 data sets are fresh: their seeds (902500..978599) are disjoint from the consistency-study seeds (at most 78599) and distinct from each other (800 distinct of 800).
- BO study: the same seeds are used for every problem and surrogate (common random numbers). Comparisons are paired within a problem; the Holm adjustment over problems is valid under any dependence.

## Checksums

`results/CHECKSUMS.sha256` lists SHA-256 checksums of every file in `results/` and `data/`.
