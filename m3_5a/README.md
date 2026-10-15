# M3.5A read-only ML pilot

The authoritative release is `m3_4br4approved/results/20261015_approved_continuation/accepted_dataset_index.json`. This package uses NumPy/CPU PyTorch and refuses physics/production imports. It never calls FEM or OC.

```python
from m3_5a.loader import Release, TrajectoryDataset
from m3_5a.normalization import Normalization
from m3_5a.features import collate

with Release() as release:
    normalization = Normalization.fit(release)  # state0 of official train only
    dataset = TrajectoryDataset(release, mode="A", history=3,
                                normalization=normalization)
    sample = dataset[0]
    dataset.validate_sample(sample)
    batch = collate([dataset[0], dataset[1]])
```

Examples use design delta targets, distinct elemental/nodal tensors and causal padding/masks. ModeB additionally reads validated current mechanics and requires evaluated mechanics at deployment. Low-level `Release.open()` arrays are borrowed maps valid only until the next trajectory switch/close; sample tensors are independent copies. Avoid retaining these low-level maps. Config pins release and artifact-manifest SHA256; optional constructor hash arguments allow the same pinning in other applications.

The bounded reproducible configuration is in `results/20261015_initial_pilot/config.json`: CPU, seed3501, K3, width24,400 steps/mode, two training-only early transitions,600s training budget. No official held-out split is made. A new reproduction runs only the explicitly authorized ML pilot:

```bash
# Choose a NEW nonexistent root; never rerun against the completed results root.
PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m m3_5a.prepare m3_5a/results/NEW_EMPTY_RUN
PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m m3_5a.snapshot m3_5a/results/NEW_EMPTY_RUN
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m m3_5a.run m3_5a/results/NEW_EMPTY_RUN
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m m3_5a.final_checks m3_5a/results/NEW_EMPTY_RUN
PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m m3_5a.report m3_5a/results/NEW_EMPTY_RUN
```

`report.py` expects an initial source manifest; the executed initial pilot retains both initial and final revisions. To archive a new reproduction before workers, use `snapshot.py NEW_EMPTY_RUN` after preparation, then run the workers and `report.py`. Final65 correctness/leakage tests run only this module; no scientific/full-generation test suite is invoked:

```bash
PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider m3_5a/test_pipeline.py
```

No generalization, reference optimality, feasibility of neural predictions or FEM acceleration is established. The exact released train assignments, source/validation records and original twelve trajectories remain immutable.
