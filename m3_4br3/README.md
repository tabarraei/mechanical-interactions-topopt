# M3.4B-R3 validation only

Separate v3 derived-change audit; R2 and numerical source remain untouched. Read results reports and machine-readable outcomes. The authorized continuation uses the same saved 501-state trajectory and original cached independent states 0..41. No production generation is performed.

Reproduce focused tests with OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/r3-mpl /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider m3_4br3/test_audit.py. Regression runs each original and R2/R3 test module in fresh processes via python -m m3_4br3.regression NEW_EMPTY_DIRECTORY. Existing temporary regression fixtures are authorized by the R3 task.

Do not rerun launch against results: its exclusive reservation and immutable attempt paths prevent repeated replay. A new validation attempt must explicitly account for cache/source dependencies and previously evaluated states. See results/CONTINUATION_PROTOCOL.md for subsequent production prerequisites.
