# M3.3B pipeline commands

The completed report is indexed by [M3_3B_DATASET_PIPELINE_REPORT.md](M3_3B_DATASET_PIPELINE_REPORT.md). [DATASET_SCHEMA.md](DATASET_SCHEMA.md) defines features, storage, normalization, windows, splits and restart semantics. `latest_generation.json` points to the exact completed generation.

Use the existing environment from the repository root:

```bash
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/m3-3b-mpl
PYTHON=/users/atabarra/.venv/bin/python
$PYTHON run_m3_3b.py all
```

This creates a fresh timestamped `m3_3b/generations/` directory, generates six small problems, deliberately pauses after two updates, resumes, independently validates every state/transition, checks windows/features/splits, reproduces the first trajectory uninterrupted in temporary storage, and runs focused/full tests. Existing generations are preserved. MKL keeps its unchanged default16; OpenBLAS uses1. No neural network or large dataset is generated.

Generate from the default six-problem configuration, or an explicit JSON configuration:

```bash
$PYTHON run_m3_3b.py generate --output m3_3b/generations/new_demo
$PYTHON run_m3_3b.py generate --config config.json --output m3_3b/generations/custom_demo
```

The stage CLI intentionally accepts at most eight problems and caps at500 updates. The demonstrated cap is40. A 100–200-problem execution entry point requires a later reviewed configuration/authorization; it is not enabled implicitly. The module APIs are configuration driven and support the validated mesh, load, support and initial-state choices. Off-grid physical point loads fail instead of being interpolated. Explicit initial slack/perturbations are recorded; density is never repaired after OC.

Exercise interruption and resume without touching completed samples:

```bash
$PYTHON run_m3_3b.py generate --output m3_3b/generations/interrupt_demo --stop-after-updates 2
$PYTHON run_m3_3b.py resume --output m3_3b/generations/interrupt_demo
```

Resuming requires the same executing sources and configuration. Committed checkpoint hashes are checked; uncommitted orphan files are quarantined. Final `.npy` arrays use read-only mmap and contain each state field once. `Trajectory.transition` reconstructs adjacent updates and verifies their saved fingerprints. Completed samples are immutable and reused after integrity checks.

Validate in a fresh process without rerunning optimization:

```bash
$PYTHON run_m3_3b.py validate --output m3_3b/generations/GENERATION_ID
```

This writes timestamped `fresh_validation_*.json` and comparison CSV files. It preserves the original `validation.json`, including the separately run reproducibility audit. An earlier development generation used an overwriting validation command; its source bytes are preserved and the final generation fixes that behavior.

Execute the exact archived code when the working tree has moved on:

```bash
$PYTHON m3_3b/replay_snapshot.py --generation m3_3b/generations/GENERATION_ID validate
$PYTHON m3_3b/replay_snapshot.py --generation m3_3b/generations/GENERATION_ID resume
```

The helper verifies every source digest, restores the package and `.py.txt` test archives to a temporary source tree, executes that version from the original repository working directory, then removes the temporary restoration. It does not replace repository sources or completed dataset samples. Recorded numerical dependencies still need to be compatible.

Refresh the report from saved evidence only:

```bash
$PYTHON run_m3_3b.py report --output m3_3b/generations/GENERATION_ID
```

Tests:

```bash
$PYTHON -m pytest -q -p no:cacheprovider tests/test_trajectory_dataset.py
$PYTHON -m pytest -q -p no:cacheprovider
```

Final observed results:19 focused tests passed in3.74s;122 regression tests passed in15.01s, both exit zero. The demo contains229 states,223 transitions and1,146 windows. Every state/transition passed independent reconstruction; all1,193 protected M1–M3.3A hashes are unchanged. Two trajectories reached the change threshold; no final label is certified optimal.

The train/validation/test_id/test_ood assignments cover whole related groups. OOD withholds bottom-clamped problems. ID denotes declared sampler-domain membership, not empirical category coverage: the tiny validation pin/roller group is absent from the demo training subset. A later pilot must audit support/load coverage and family grouping before using the splits as a generalization benchmark.

Fresh FEM gradients/displacements/energy and no-FEM features have separate profiles. Full problem conditioning is preserved in the problem specification; models must not treat labels, target termination or realized trajectory length as input channels. No training, HPC job, commit or push is performed.
