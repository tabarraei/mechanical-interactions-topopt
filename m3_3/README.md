# M3.3A controlled OC pilot

Run from the repository root using the existing environment:

```bash
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/m3-3-mpl
PYTHON=/users/atabarra/.venv/bin/python
$PYTHON run_m3_3a.py all
```

The FEM solver keeps its unchanged MKL default of 16 threads. The command creates a fresh timestamped directory under `m3_3/generations/`; it refuses to overwrite an explicitly selected directory. It runs 20 deterministic specified problems, compares `baseline_v1`, `tight_midpoint_v1` and `feasible_upper_v1`, verifies relaxed early stopping, saves selected-state replay results and plots, and runs focused/full tests. The maximum budget is 500 updates with no automatic extensions. Existing global optimizer defaults and M1–M3.2 sources/results remain unchanged.

`latest_generation.json` identifies the completed report and manifests. Each generation preserves executable source bytes and checks they remain unchanged during execution. Test archives have `.py.txt` suffixes to prevent pytest from collecting archived test copies. The outer `m3.3a.problem.v1` specification includes the exact initial x and effective versioned optimizer options; the nested canonical M3.1 specification describes the unchanged mechanical core.

Fresh-process numerical replay, without optimization:

```bash
$PYTHON run_m3_3a.py validate --output m3_3/generations/GENERATION_ID
```

The command reconstructs selected saved x/rho/u/C/gradients, independent stiffness actions and versioned OC transitions, writing a new timestamped replay table. A capped trajectory is never relabeled converged.

Matched-state production-call timing, without FEM:

```bash
PYTHONPATH=. $PYTHON m3_3/benchmark_oc.py --generation m3_3/generations/GENERATION_ID
```

This compares a single unchanged baseline helper call with the guarded new options on identical saved states, rotating order with one warmup and nine repetitions. The main baseline pilot includes a duplicated bisection audit, so its wall ratios alone are not clean production speed comparisons.

Tests can be run independently:

```bash
$PYTHON -m pytest -q -p no:cacheprovider tests/test_robust_oc.py
$PYTHON -m pytest -q -p no:cacheprovider
```

Executable source snapshots preserve package paths. To replay archived generation code, run its `source_snapshots/run_m3_3a.py` from the original repository root; it imports the archived package and creates a fresh generation. Existing trajectory specifications and array fingerprints bind runs to their complete problem identity, not mesh dimensions.

The fixed multiplier bracket is checked before feasible-upper selection. Unattainable move-limited budgets, nonfinite inputs and an infeasible fixed upper endpoint fail explicitly; no bracket expansion, density rescaling or numerical-core changes are performed. Tight multiplier precision is not a guarantee of KKT convergence or optimality.

The completed pilot is indexed by [M3_3A_OC_VALIDATION_REPORT.md](M3_3A_OC_VALIDATION_REPORT.md). Its full generation contains `final_validation.json`, `capped_cases.csv`, `initial_feasibility.csv`, raw matched-state repetitions and separately fingerprinted report/benchmark supplements.

Refresh the read-only final assessment after generating the matched-state timings:

```bash
PYTHONPATH=. $PYTHON m3_3/finish_report.py --generation m3_3/generations/GENERATION_ID
```

Recorded observations: 20 focused tests passed in 3.30 s and 103 regression tests passed in 13.75 s. All 835 protected files are unchanged. Main replay checked 418 states and 358 transitions plus 120 independent early-stop runs; fresh-process replay passed 3,780 comparisons. The matched timing measured 60 identical states with nine rotating repetitions per option.
