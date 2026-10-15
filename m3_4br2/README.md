# M3.4B-R2 isolated validation

No production generation entry point. The audit policy is selected explicitly with `--policy m3.4b.oc_diagnostic_audit.v2`; `historical` retains unchanged comparator behavior. Read the generation reports and machine-readable prerequisites before execution. One real saved-trajectory replay is conditionally authorized in R2, and must not be repeated merely because a worker fails. Production continuation remains separately authorized.

Use OPENBLAS_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1, MPLCONFIGDIR=/tmp/r2-mpl and the existing /users/atabarra/.venv/bin/python; keep MKL solver default16. Run focused checks with `python -m pytest -q -p no:cacheprovider m3_4br2/test_audit.py m3_4br2/test_infrastructure.py`. Full module-isolated regression runs original temporary generation fixtures, explicitly authorized by the user during R2. Source archives use .txt to prevent duplicate pytest collection.
