# M3.5B-2D disabled candidate

Read [the final report](reports/20261015_candidate_v1/M3_5B2D_FINAL_REPORT.md). Candidate policy, alias, crosswalk, numerical acceptance and release defaults are false. No executable authorization or approval receipt is supplied. The protected external authority store was not created; no numerical work or runtime adoption is enabled.

Tests are synthetic/read-only and never invoke FEM, OC, numerical replay or dispatch:

```bash
PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider m3_5b2d/test_candidate.py m3_5b2s/test_alias_schema.py
```

The new validation-only runner delegates original scientific implementations after all live gates. The release adapter returns staging metadata only after all distinct approvals/raw evidence; it does not publish an ML index. Existing production runners, restart policy, data and ML source are unchanged.
