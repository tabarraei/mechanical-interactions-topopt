# M3.3C ten-problem completion

This adds the diverse pilot, force controls, quality/sampling/prediction analyses and readiness decisions to the completed scaling extension. It preserves every previous numerical source and result; the scaling experiment is reused after archive and dataset checksum checks.

From repository root:

```bash
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/m3-3c-completion-mpl
PYTHON=/users/atabarra/.venv/bin/python
$PYTHON run_m3_3c_completion.py all
```

The entry point creates a new `m3_3c/completion_generations/` directory, archives executing sources, generates ten primary problems and two small force controls, independently replays all states, checks restart, evaluates non-neural predictions and quality/sampling/resource analyses, then runs focused/full regression and a fresh replay. Before the full pilot it measures and preserves a20-update production prefix. Remaining runs proceed only if estimated runtime stays below one hour and process peak below2GiB. Each trajectory retains the explicit500 cap and.001 change target; caps are not silently extended.

Resume an interrupted generation with unchanged original sources and configuration:

```bash
$PYTHON run_m3_3c_completion.py resume --output m3_3c/completion_generations/GENERATION_ID
```

Fresh numerical replay, with no optimization and timestamped outputs:

```bash
$PYTHON run_m3_3c_completion.py validate --output m3_3c/completion_generations/GENERATION_ID
$PYTHON m3_3c/completion_replay_snapshot.py --generation m3_3c/completion_generations/GENERATION_ID
```

The second command restores the source archive to temporary storage and uses the original workspace only for protected-artifact checks. It writes a new replay record, leaving the original audit and completed arrays intact.

Report refresh from saved tables only:

```bash
$PYTHON run_m3_3c_completion.py report --output m3_3c/completion_generations/GENERATION_ID
```

This produces a timestamped report. The original completed report is retained.

Focused checks and full regression:

```bash
$PYTHON -m pytest -q -p no:cacheprovider tests/test_dataset_readiness.py tests/test_mesh_scaling.py
$PYTHON -m pytest -q -p no:cacheprovider
```

`dataset_readiness.balanced_training_indices` optionally samples only training windows, uniformly by problem then uniformly within that problem. It does not modify arrays, splits or targets. `project_prediction` is an explicitly versioned prediction-only Euclidean box/volume projection, with no FEM. Raw extrapolation stays separate. Native trajectory reference flags stay false; provisional endpoint eligibility is a sidecar screen and never certifies an optimum.

Read `M3_3C_FULL_PILOT_REPORT.md` inside the generation for actual results and the three separate readiness decisions. No neural models,100–200 trajectories,192×96 FEM, HPC jobs, Git commit or push are executed.
