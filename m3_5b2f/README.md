# Protected D2 policy receipt review

Evidence only; no issuer or numerical runner. See `evidence/20261015_receipt_review_v1/M3_5B2F_FINAL_REPORT.md` and the exact non-executable request. The scientific D2 decision is verified; receipt identity/validity and genuine protected publication remain outstanding. No D2 scientific reapproval is requested.

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider m3_5b2f/test_receipt_boundary.py m3_5b2e/test_preflight.py
```

Tests are synthetic/parser/read-only checks, never protected issuance or FEM/OC. A documented test records the existing absence of authenticated issuer-name verification; protected publication is the actual authority boundary.
