# M3.6A source checkpoint

This is an architecture/infrastructure checkpoint on m2-spatial-interactions, not a trained model or new numerical acceptance. The source-file list is source_files.json; the final exact committed list is available with `git show --format= --name-only m3.6a-architecture-v1` after tagging. Earlier M3 generation/validation source is preserved for inspection; running it remains subject to its original authority and numerical/resource gates. A Git tag grants no scientific execution permission.

The default model has10,772 parameters. One shared untrained instance supports48×24,48×32,96×48; the12 accepted trajectories contain3709 states/3697 transitions. See m3_6a/M3_6A_ARCHITECTURE_SPEC.md and its companion interface/operator/temporal/target/visualization documents. Reports under m3_6a describe the original local milestone; their excluded evidence-file references remain local artifact dependencies rather than files distributed by this checkpoint.

This is a source-only checkout. Native datasets, saved FEM operands, original source archives, acceptance/evidence/visualization bundles and normalization artifacts are deliberately external. Restoring those exact hash-bound local artifacts is required for accepted-data examples/tests. Missing inputs must fail closed; never fabricate an accepted index or regenerate data to make a test pass. Pure synthetic architecture tests do not require those artifacts:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3-checkpoint-mpl python -m pytest -q -p no:cacheprovider m3_6a/test_model.py
```

Safe local complete coverage, when original accepted artifacts are available, is the same command with m3_6a/test_data.py and m3_6a/test_visualization.py appended. Do not invoke a repository-wide test suite casually: historical tests include FEM, OC and temporary trajectory generation. No optimizer/training step is needed for either architecture command.

Dependencies are recorded in environment.json as observed versions, not a portable lockfile or instruction to install cluster-specific PARDISO. Architecture-only requirements are Python, NumPy, Torch and pytest; saved-data visualization also needs Matplotlib and Pillow, and notebook display needs IPython. Earlier numerical modules use SciPy/pypardiso/MKL but were not imported or executed for checkpoint tests.

Known notebook exception: old m3_5v/visualizations/20261015_accepted12_v1/trajectory_viewer.ipynb changed during the earlier milestone, from SHA2562a5c3f5296e6ae9a3f7375d9ee416fd9065fd204c741853dad61a498014da93d to b3c9ceb16ca92a9923dedeb7b50937d6e98a645ba9f81919c84da343a2625c70. It contains executed outputs; attribution was not established. This checkpoint leaves it untouched and excludes it. The modified root03_load_and_plot.ipynb is also excluded and preserved.

Source authorization logic is included; real authority books, issued receipts, worker tickets and private request attachments are excluded. The112-byte m3_5b2/approved_receipts.json is an immutable EMPTY, issuance-disabled schema fixture required by the hash-bound source. It has no approvals and is not the external protected authority book. It does not grant permission. No actual approval record is included. Attachment paths embedded in historical audit scripts refer to unavailable external provenance; those scripts are not portable execution entry points.

Frozen30 planning artifacts, historical plan10 and all acceptance/authorization controls remain unchanged. No dataset is admitted or published by this checkpoint. No FEM, OC, numerical replay, dataset generation, training or SLURM occurs as part of the Git operation.
