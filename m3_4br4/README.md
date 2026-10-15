# M3.4B-R4 controlled continuation

Read latest_results.json for the versioned results root. R4 preserves the original twelve-ID manifest/authorization ledger and all historical outputs. Ten existing trajectories passed independent replay; the original R3 complete evidence is hash/source-bound and reused, the other nine have fresh R4 full replays. Native96 replay and resource tests passed. Four real recovery fixtures preserved checkpoint prefixes and passed independent replay, but three failed the unchanged control-equivalence gate. No production continuation occurred; indices10/11 remain unexecuted.

Use production_ledger.json, validation_ledger.json, accepted_dataset_index.json and BLOCKER_REPORT.md. Scratch recovery samples are isolated test evidence, not additional production/training samples. No reference-optimum labels are granted. Existing capped histories remain capped.

Do not rerun pipeline or finish_preflight against this root. Immutable reservations and logs retain every failed/interrupted attempt. Any subsequent recovery policy change needs explicit scientific authorization; original numerical/resource gates remain unchanged. Original R4 authority permits the two reserved production jobs only after mandatory preflight passes; scaled164 generation is not authorized.

coupled_recovery.py is an unexecuted, fail-closed diagnostic helper checking association with existing OC audits. It was not installed as an acceptance exception and cannot override the displacement comparison. Its presence is not validation evidence.

Focused tests: OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/r4-mpl /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider m3_4br2/test_audit.py m3_4br2/test_infrastructure.py m3_4br3/test_audit.py m3_4br4/test_control.py. Full module-isolated regression: python -m m3_4br4.regression NEW_EMPTY_OUTPUT_DIRECTORY, same environment. Only existing temporary test fixtures are included. Never run generators into existing historical roots.
