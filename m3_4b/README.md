# M3.4B streamed production orchestration

The controlled generation is `generations/20261014T013017Z`. Its implementation plan precedes numerical work. Completed reports, validation, trace checksums and original-versus-revised design tables are stored there. The old M3.4 launcher and verified mathematics are unchanged.

From the repository root:

```bash
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/m3-4b-mpl
PYTHON=/users/atabarra/.venv/bin/python
```

Solver MKL/PARDISO retains its historical default16 threads. The new launcher accepts `plan`, `dry-run`, `stress`, `batch`, `resume`, `validate`, `report`, and `audit`.

```bash
$PYTHON run_m3_4b.py plan --output m3_4b/generations/NEW_ID
$PYTHON run_m3_4b.py dry-run --output m3_4b/generations/NEW_ID
$PYTHON run_m3_4b.py stress --output m3_4b/generations/NEW_ID
$PYTHON run_m3_4b.py batch --output m3_4b/generations/NEW_ID
```

`plan` requires a new output generation and archives executing bytes before numerical work. It serializes all164 planning configurations, deterministic specifications, grouped splits, quotas/near-neighbor audits, and twelve authorized batch jobs. It never authorizes the planning population. `dry-run` performs package/source/native-linkage/configuration/disk checks and records preflight; it performs no optimization or trajectory FEM. Estimates retain twofold disk safety and checkpoint/recovery/source/temp allowances. `stress` uses100 inexpensive synthetic subprocesses plus two descriptor-retention comparison processes, zero FEM.

**One real batch is authorized for this stage.** `m3_4b/authorization.json` reserves every ID and the original generation/source identity. Maximum16 is a code constant; this experiment reserves12. Completed and resumed jobs count toward the same total. Another generation cannot execute by changing `--output`, and no flag disables the cap. A future larger execution needs separately reviewed code/configuration and explicit authorization. Do not delete/edit the authorization ledger to repeat a batch.

```bash
$PYTHON run_m3_4b.py resume --output m3_4b/generations/INCOMPLETE_ID
$PYTHON run_m3_4b.py validate --output m3_4b/generations/GENERATION_ID
$PYTHON run_m3_4b.py report --output m3_4b/generations/GENERATION_ID
$PYTHON run_m3_4b.py audit --output m3_4b/generations/GENERATION_ID
```

`resume` uses exact source/configuration/package identities and never samples new jobs. A completed generation refuses batch/resume. `validate` independently replays each saved state/transition in a new process without optimization, adding timestamped evidence. `report` uses saved summaries and streaming monitor reducers only, zero FEM. Report refreshes receive new names and derived-output directories. `audit` rechecks original protected files and all executing/archive/configuration hashes. Native data remain immutable.

Each worker is monitored every50ms with unchanged worker1024MiB,parent768MiB,combined1792MiB and900s limits. Workers are sequential. A failure is recorded and stops closed; resources/validation are never silently relaxed. Completed output is safely reused after a worker exit preceding progress publication; no committed state or FEM execution is duplicated in that case. Uncommitted work after a crash can be recomputed, so exactly-once FEM execution is not claimed.

Raw monitor samples are append-only JSONL; one unique attempt ID and immutable problem ID accompany every record. fsync occurs every20samples and on close. Compact summary files contain timing/exit/RSS maxima/sample count/path/checksum; no raw list is returned to the parent. Partial interrupted traces are retained. Report aggregation reduces one file/record at a time and verifies checksums/attribution. Kernel VmHWM and worker-reported high water supplement interval samples, which can miss short peaks.

Native storage, `feasible_upper_v1`, .001 change target, 500cap, SIMP/filter/gradient/OC formulas and conservative labels are unchanged. Every saved state/transition is independently replayed; cached feature profiles preserve original grids/units. Fresh-FEM feature cost at inference is disclosed. Capped endpoints are valid dynamics after replay, not reference labels. No trained model, certified optimum or acceleration result is produced.

Tests:

```bash
$PYTHON -m pytest -q -p no:cacheprovider tests/test_scalable_production.py
$PYTHON -m pytest -q -p no:cacheprovider
```

Read `MANIFEST_SCHEMA.md` and generation `REPRODUCE.md` for source restoration and dataset-design definitions. No192/384 FEM, full dataset, parallel workers, training, HPC,commit/push belongs to this stage.

## Recorded stopping condition

The recorded batch stopped on the tenth configuration's independent OC diagnostic comparison. Ten configurations generated; nine passed full replay; two reserved configurations remain unexecuted. Production status is `failed_closed`, not complete. Read the generation's `M3_4B_PRODUCTION_READINESS_REPORT.md` and `M3_4B_NEXT_REVIEW.md`.

The completed-batch `report` path expects all results and recovery evidence. This stopped assessment was instead produced by its preserved metadata-only `stopped_report_source.py.txt`, which verifies hashes, streams monitor records, and audits saved constraints without FEM. It refuses to overwrite its original stopped report. Do not run `resume` or retry the batch without the explicit continuation/policy review required by the recorded stop. Full tests passed181 cases, but that does not override a production validation failure.
