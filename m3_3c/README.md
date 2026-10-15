# M3.3C scaling extension

Run from the repository root with the existing environment:

```bash
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/m3-3c-mpl
PYTHON=/users/atabarra/.venv/bin/python
$PYTHON run_m3_3c.py all
```

Each run selects a new timestamped generation. `--output` must name a new directory. The executing sources are archived before numerical work. No previous sources or results are changed. MKL/OMP keep the solver's default 16 threads.

The CLI measures one 48×24 baseline, fixed-domain 96×48 and domain-growth 96×48 in separate processes. Only after each worker stays below 2 GiB peak and 10 seconds warm FEM API time does it run up to 20 feasible-upper OC updates on each of those three specifications. These are resource screens, not hardware-independent guarantees. It independently replays every saved state and transition and runs focused/full regression. 192×96 and 384×192 specifications contain complete coordinates and controls, but perform no FEM or optimization at those sizes. A and B share the physical 48×24 baseline; larger cases remain separate.

Standalone supported benchmark:

```bash
$PYTHON run_m3_3c.py benchmark --mesh 96 --experiment A --output /tmp/m3-3c-benchmark.json
```

Fresh replay without optimization, preserving the original audit:

```bash
$PYTHON run_m3_3c.py validate --output m3_3c/generations/GENERATION_ID
$PYTHON m3_3c/replay_snapshot.py --generation m3_3c/generations/GENERATION_ID validate
```

The second command restores exact archived source bytes to temporary storage. This validates larger native M3.3B samples; it does not invoke the historical M3.3B CLI or change its guarded sampler.

Checks:

```bash
$PYTHON -m pytest -q -p no:cacheprovider tests/test_mesh_scaling.py
$PYTHON -m pytest -q -p no:cacheprovider
```

`latest_generation.json` identifies the completed report. Complete configurations, source hashes, primitive benchmarks, finite differences, native trajectories, comparison rows, conservative field comparisons, plots, larger-mesh planning scenarios and test logs are stored within the generation. `protected_before.json` fingerprints the pre-existing working tree including ignored research artifacts. All committed samples remain dynamics-only, with no certified optimum/reference label.

A 192×96 single-state command is prepared in the scientific report and requires separate explicit approval before execution. No 192/384 trajectories, HPC submission, training, commit or push belongs to this extension.
