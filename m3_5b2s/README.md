# M3.5B-2S historical reuse protocol

Read [the final report](reports/20261015_protocol_v1/M3_5B2S_FINAL_REPORT.md) and policy audit. The proposal, alias and schema confer no execution, adoption or release authority. Existing numerical and production sources are unchanged. This package contains only new design/evidence files and a pure standard-library schema validator.

Synthetic/read-only checks:

```bash
PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider m3_5b2s/test_alias_schema.py
```

`validate_reference_design` uses hypothetical hash maps only to test reference consistency; it always reports numerically_verified=false/release_eligible=false. It is neither a certificate verifier nor a trust/approval mechanism. No runtime imports this package, and no production integration was enabled.
