"""Human-readable evidence reports; no execution authority."""
from .common import *

def documents(out,s,registry,conflicts,real):
 c=conflicts[0]
 shared="""Implementation/testing only. No FEM, OC, trajectory generation, neural training/inference, qualification, SLURM, commit or push occurred. Historical comparator failures remain retained. The original production runner, M3.5A implementation, frozen selection and prior data were not changed.
"""
 files={
 'M3_5B2_IMPLEMENTATION_REPORT.md':f"""# M3.5B-2 implementation

Implemented a separate guarded runner and read-only release adapter in `m3_5b2/`. All60 focused tests passed; zero failed. See `final_tests.log` and `test_outcomes.json`. Numerical execution was deliberately not tested end to end because it is outside this task's authorization.

`common.py` binds immutable specifications and source/environment identities; `authorization.py` implements strict semantic validation and issuance checks; `registry.py` inventories historical identities/equivalences; `checkpoint.py` verifies committed bytes; `budget.py` persists work accounting; `runner.py` enforces sequential monitored dispatch; `worker.py` delegates to original numerical implementations; `validation.py` and `accept.py` require independent acceptance; `release.py` supplies bounded-memory loading and train-only normalization. `test_controls.py` and `test_release.py` contain synthetic/read-only checks. `audit.py` creates this evidence package; `documentation.py` writes reports. `approved_receipts.json` is an empty inert template.

All created source files and exact SHA256 values are in `source_manifest.json`, with preserved source snapshots. All report/artifact hashes are in `artifact_manifest.json`; its own hash is not recursively included. `protected_before.json` records the pre-change historical baseline and `source_artifact_integrity.json` records the final comparison. Existing working-tree changes were preserved; this task added only `m3_5b2/`.

Frozen manifest: `{PLAN_SHA}`. Original release index: `{ORIGINAL_SHA}`. Original77 production source/archive identities are rechecked. No scientific selection or partition changed.

A broader registry exposed one pre-existing provenance conflict: plan10, `{c['selected_id']}`, has mechanical-equivalence signature `{c['equivalence_sha256']}` already present under a different historical ID.29 selected cases pass identity eligibility, but this is not resource or execution approval. The conflict remains blocked; no automatic reuse, relabeling, new ID or regeneration is permitted.

{shared}""",
 'M3_5B2_AUTHORIZATION_CONTRACT.md':"""# Authorization contract

The frozen selection is a scientific proposal, never dispatch authority. The sidecar schema is `authorization.schema.json`; the stricter semantic parser is `authorization.py`. Duplicate JSON keys, nonfinite values, unexpected fields, unknown/duplicate/original12 IDs, wrong manifest/source/environment, root/stage mismatch, excessive limits/work, invalid dates and qualification escalation reject.

A syntactically valid sidecar cannot activate execution. The live path requires its exact SHA256 and matching approval metadata in `/users/atabarra/.codex/approvals/m3_5b2_issued_authorizations.json`, a protected authority book outside this writable workspace. This task did not create that file or publish any approval. The workspace `approved_receipts.json` is an empty hash-pinned inert template; editing it fails integrity and cannot issue authority. There is no issuer CLI, test permit, environment bypass or caller-selected trust path. Future deliberate publication requires a separately explicit scoped approval and the applicable filesystem permission. This is a local content-addressed approval workflow under protected-source/authority assumptions, not a cryptographic claim about arbitrary edited code or privileged users.

The sidecar binds manifest, exact IDs and stage, output root, production/validation/implementation hashes, environment, fixed resource/time limits, kind, experiment actions, explicit work ceilings, disk reservation, resume/qualification evidence, identity, timestamp, expiry and decision reference. The semantic parser additionally enforces full original stopping for production, non-release qualification and exact case-specific qualification prerequisites. No sidecar is issued here.

Live execution checks current expiry/environment/threads before numerical imports. Required threads remain OPENBLAS1/MKL16/OMP16. Historical accepted-evidence review uses its recorded execution time and environment; it confers no new run permission. Durable locked counters charge FEM/OC before entry, persist failures and cumulative wall time across renewals, and refuse exceeded budgets/deadlines. Existing output cannot be overwritten; leases and the global registry block competing or repeated work. Explicit resume requires exact prior authority, journal and work-ledger bindings. A renewal preserves prior receipts and consumed work.

Tests trap production imports and demonstrate that absent, malformed and otherwise valid-but-unissued authorizations reject before FEM/OC imports. Actual numerical workers also require a one-use parent-issued ticket. No authorized numerical integration run is claimed.
""",
 'M3_5B2_RELEASE_ADAPTER_REPORT.md':f"""# Verified release adapter

`ExpandedRelease` preserves the original12 through the unchanged M3.5A verifier and accepts only independently accepted selected expansion entries. It supports train, validation, test_id and test_ood, rejects incomplete/unvalidated/referenced-optimum/corrupt/count/layout/provenance/duplicate/equivalent/family-conflicting entries, preserves termination, and uses bounded memory-mapped trajectory access. The established design-x/physical-rho distinction, feature layout, causal history and variable-mesh crop behavior are reused without input semantic changes.

The actual original-only adapter view was verified: {real['trajectories']} trajectories, {real['states']} states, {real['transitions']} transitions. No42-trajectory dataset exists. The four-partition expanded view is tested using clearly marked synthetic fixtures. Its test-only verifier subclass is not available as a production dispatch option; default verification rejects fixture authority. A retained real192-state history's raw independent proof and checkpoint bytes were also checked without a solve.

Normalization uses accepted training trajectories only and the established state0-only RMS policy. Metadata binds exact training IDs, training-release identity, feature schema, fitting policy and statistics. Training identity excludes held-out content. Held-out mutation tests preserve fitted statistics and training identity; forged statistics reject on load. A cache test checks borrowed maps stay valid during feature extraction. No normalization has been fitted on nonexistent expansion data or exported as an expanded fitted release.

{shared}""",
 'M3_5B2_VALIDATION_INTEGRATION.md':"""# Independent acceptance integration

Numerical generation delegates to original `dataset_storage.generate_sample`, FEM solver and robust OC. Independent replay delegates to unchanged R3 replay; supplemental feature/finite-difference checks reuse R4 checks with only an in-memory selected-job lookup adapter. No second FEM, sensitivity, filter, material or OC formula was introduced.

Release acceptance requires complete raw independent certificates, equilibrium/boundary/displacement/compliance/sensitivities, OC action/multiplier, strict volume feasibility, incoming/stopping and original termination metadata. v2 residual and v3 incoming-change contracts remain mandatory. Full native hashes, initial design, source binding and action/delta evidence are checked. Maxiter histories remain capped learning histories with no reference-optimality label.

CategoryA exact checkpoint bytes and categoryB full independent correctness remain mandatory. Finalized native arrays reconstruct the original committed NPZ/JSON bytes for exact checkpoint verification. Partial recovery checks the original ordered journal; uncommitted orphans are handled by the original quarantine implementation. A failed mandatory gate preserves evidence and stops without automatic retry.

Actual resumed histories additionally require a real same-spec control and before-resume checkpoint evidence. CategoryC invokes the unchanged approved policy and retained evidence adapter, hash-bound to policy c74a81c5477622aaccce4c0866ebbda190dfe4d01fa74f985cd38d15cd31b120, adapter fd30fdb7eb9392ce77d6a7203d5ee162f62b8d7c1d66416b31863e7be2349761 and matrix417686f60b4e1462cf76cb84264d14ad07640c5bb259422f524b83827e1a4881. Missing real control evidence blocks acceptance. Historical failed comparisons are not overwritten or relabeled.

Tests establish integration guards and read-only evidence checking; no real new generation/replay/restart or resource qualification was authorized or run. Actual monitored end-to-end behavior remains to be established by a future scoped run.
""",
 'M3_5B2_RESOURCE_QUALIFICATION_PROPOSAL.md':"""# Non-executable resource qualification proposal

`resource_qualification_proposal.json` supplies exact original IDs, spec/source hashes, family/split, rationale, resource uncertainty, output-root proposals, evidence and stop/quarantine rules for all five cases. Every authorization flag is false.

| Plan | Case | Remaining qualification |
|---|---|---|
|98|dual-pin support|Exact support case generation/replay resource evidence|
|130|aspect ratio3|Exact elongated-domain generation/replay evidence|
|138|64x32|Exact mesh generation/replay evidence|
|145|96x48|Full-cap501-state residency, plus disclosed coarse plan10 provenance conflict|
|151|96x48|Full-cap501-state residency|

Recommend requesting plan98 alone first. Proposed maximum: one ID, four sequential monitored workers,1506 FEM calls,1000 OC calls,501 states,500 updates,3600 cumulative worker seconds; each worker900 seconds,1024MiB worker/768MiB parent/1792MiB combined. These are proposal ceilings, not permission. Disk minimums and exact output roots are case-specific in the JSON. Resource limits are unchanged.

Recommended experiment preserves the original500-update cap and first stopping point, followed by complete independent replay and supplemental checks. If original stopping completes, retained accepted qualification output can be considered for a separately authorized read-only production adoption, without regeneration. Qualification never automatically publishes an ML entry. No shortened run may be called a complete original trajectory.

An optional two-update/three-state prefix is strictly diagnostic, not full qualification, and cannot trigger automatic follow-up. Subsequent partial resume requires a new exact scoped receipt, prior journal/work accounting, real control evidence and unchanged categoryC; if that evidence is absent, remain blocked. A diagnostic does not authorize a fresh duplicate run.

For96, earlier193-state evidence and any early-stop history do not certify501-state residency. Do not override original stopping to manufacture501 states. Any alternative residency-only diagnostic needs separate scientific review and authority; these two gates therefore remain unresolved.

Before requesting the first run: review exact plan98 ID/spec, final frozen implementation/validation/production source hashes, current environment/thread identity, registry coverage/no-conflict check, disk reserve and clean proposed root; specify exact actions, maximum work, expiry, stop/quarantine rules and approval identity. A future explicit decision must deliberately publish its exact receipt through the protected authority workflow. Recheck all gates at dispatch. No executable request/authorization was created by this task.
""",
 'M3_5B2_FINAL_REPORT.md':f"""# M3.5B-2 final report

Infrastructure implementation and60 focused tests are complete, but the full30-case expansion is blocked by a newly detected historical configuration-equivalence conflict. No numerical execution is authorized or performed.

1. Created only the `m3_5b2/` implementation, tests and versioned evidence directory. Prior working-tree changes were preserved; original runner, M3.5A and research artifacts are unchanged.
2. Exact source SHA256 values and snapshots: `source_manifest.json`. Exact artifact hashes: `artifact_manifest.json`. Protected baseline/final checks: `protected_before.json` and `source_artifact_integrity.json`.
3. Frozen selected manifest verified as `{PLAN_SHA}`; all proposed authorization flags remain false. Original index `{ORIGINAL_SHA}` and77 original source/archive identities remain unchanged.
4.60 tests passed,0 failed. See `final_tests.log`; synthetic controls and read-only historical verification only.
5. Official unchanged runner rejects missing/invalid/unissued scope before numerical imports; syntactic validity alone is insufficient. The protected external issuer book is not created here. Tests verify this boundary; no authorized real numerical run is claimed.
6. Global inventory scans repository metadata, specifications, journals, leases/quarantines/incomplete outputs and original releases; scan failures reject.34 historical IDs are represented.29 selected identities are eligible; plan10 is blocked by the legacy alias detailed in `configuration_equivalence_conflicts.json`. Source/equivalence coverage is in `global_duplicate_registry.json`.
7. Adapter supports all four partitions with synthetic accepted-entry fixtures. The default real original-only view verifies12 trajectories/3709 states/3697 transitions. No prospective42 dataset is asserted to exist.
8. Normalization is training-only/state0-only, with held-out mutation, exact training identity/statistics and causal future-state tests. No nonexistent expansion statistics were fitted.
9. Full independent primary mechanics/sensitivity/OC/multiplier/feasibility, v2/v3, stopping, checkpoint and hash-bound restartC remain mandatory. Test fixtures cannot establish real new acceptance.
10. All five exact resource gates remain enforced:98,130,138,145,151.96 requires actual501-state/full-cap residency;193 is insufficient. No case qualifies another.
11. New FEM calls0; OC calls0; generated trajectories0; model training/inference0; qualification0; SLURM0; commit/push0.
12. Unresolved: selected plan10 `{c['selected_id']}` is mechanically equivalent to retained historical `m3_3b_d88cb10eb219fb432141045c` (501 states/500 updates, maxiter) at `m3_3c/completion_generations/20261013T203017221805Z/samples/m3_3b_d88cb10eb219fb432141045c`. Seed0 versus703 differs, but full initial design and optimizer/mechanical configuration agree after excluding seed. Equivalence SHA `{c['equivalence_sha256']}`. This is provenance evidence, not a new mechanical acceptance. Explicit provenance/selection review is required before this ID may proceed; no automatic adoption/relabeling/redesign is authorized. Plan145's related coarse case must disclose this conflict. Real new monitored execution and partial restart/control integration remain unexercised; missing mandatory evidence blocks them.
13. First narrow qualification proposal is plan98 alone, with exact ID/spec/output root and budgets in `resource_qualification_proposal.json`. It needs separately explicit scoped approval, protected receipt publication, final source/environment/disk/registry checks and unchanged mandatory validation/resource limits. No approval is requested as part of this report and none is inferred. The full30 cannot proceed while the provenance conflict and five resource blockers remain.

BLOCKED_BY_INTEGRITY_OR_PROVENANCE
"""}
 for name,body in files.items():
  p=out/name
  with p.open('x') as f:f.write(body)
 write(out/'final_status.json',dict(status='BLOCKED_BY_INTEGRITY_OR_PROVENANCE',implementation_complete=True,tests_passed=60,tests_failed=0,execution_authorized=False,issued_authorizations=0,new_numerical_work=0,provenance_conflicts=1,identity_eligible=29,resource_blockers=[98,130,138,145,151]))
