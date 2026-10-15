# M3.5B-2E decision and nonnumerical preflight evidence

The user's scientific D2 decision is recorded; protected policy issuance and execution authorization are separate and remain absent. Read the final report in `evidence/20261015_preflight_v1/`. No numerical worker or issuer exists in this package.

New tests use only in-memory/stamped temporary fixtures and read-only hash/metadata inspection. Existing reviewed production sources are untouched.

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider m3_5b2e/test_preflight.py m3_5b2d/test_candidate.py m3_5b2s/test_alias_schema.py
```
