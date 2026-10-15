# M3.4 monitored production framework

The completed generation, numerical evidence, and remaining conditions are in `generations/20261014T000500998078Z/`. Its readiness report and `final_validation.json` are authoritative after completion.

Use the repository environment:

```bash
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/m3-4-mpl
PYTHON=/users/atabarra/.venv/bin/python
```

MKL/PARDISO retains its default 16 threads. Execution is sequential: a fresh monitored process generates or independently replays one problem. Replay owns and explicitly closes a reused solver context. The original numerical methods and journal implementation are unchanged.

Create a new immutable plan and source archive:

```bash
$PYTHON run_m3_4.py plan --output m3_4/generations/NEW_ID
$PYTHON run_m3_4.py dry-run --output m3_4/generations/NEW_ID
$PYTHON run_m3_4.py pilot --output m3_4/generations/NEW_ID
```

`--output` names a fresh directory for `plan`, or an existing generation for other commands. The default cap is 500. `--cap` applies only to planning and is bounded by 500; the recorded historical equivalence experiment is supported for the default 500-cap pilot. The pilot has seven predefined configurations. The separate 164-configuration proposal is planning-only and cannot be executed by this launcher. A future larger launcher requires separate authorization and review.

The dry run prints estimates and gates without writing trajectories or running FEM. Limits are immutable: worker RSS 1024 MiB, parent RSS 768 MiB, combined RSS 1792 MiB, and 900 seconds per worker. The parent samples RSS every 50 ms and terminates a worker that crosses a gate. Measurements and short-peak limitations are documented in the report. A failed gate is not automatically relaxed.

Resume an incomplete generation with its original source/configuration bytes:

```bash
$PYTHON run_m3_4.py resume --output m3_4/generations/GENERATION_ID
```

Completed generations cannot resume. Committed journal frames remain immutable; orphan frames are quarantined by the native storage implementation. Recomputed uncommitted work is allowed, so exactly-once FEM execution is not promised.

Fresh independent replay, report refresh, and provenance audit:

```bash
$PYTHON run_m3_4.py validate --output m3_4/generations/GENERATION_ID
$PYTHON run_m3_4.py report --output m3_4/generations/GENERATION_ID
$PYTHON run_m3_4.py audit --output m3_4/generations/GENERATION_ID
```

Replay launches one worker per saved problem and writes new timestamped results without optimization. Report reads saved summaries only, performs zero FEM, and writes a timestamped refresh after the first report. Audit checks all immutable specifications, executing sources, and protected prior files. Original numerical observations remain intact.

Checks:

```bash
$PYTHON -m pytest -q -p no:cacheprovider tests/test_production_dataset.py
$PYTHON -m pytest -q -p no:cacheprovider
```

See [MANIFEST_SCHEMA.md](MANIFEST_SCHEMA.md) for manifests, family grouping, sampling, quality labels, and source restoration. All native arrays remain at their original resolution. Capped endpoints are dynamics examples after independent replay; native reference eligibility stays false. The provisional reference screen is a sidecar diagnostic, never an optimality certificate. No 192/384 FEM, full dataset, neural training, HPC, commit, or push belongs to this stage.
