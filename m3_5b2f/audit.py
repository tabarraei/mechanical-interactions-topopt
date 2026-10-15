"""Read-only issuance audit; creates only non-executable workspace evidence."""
import ast,sys,shutil,json
from pathlib import Path
from datetime import datetime,timezone
from m3_5b2d.common import *
OUT=ROOT/'m3_5b2f/evidence/20261015_receipt_review_v1'
D=ROOT/'m3_5b2d/reports/20261015_candidate_v1'
E=ROOT/'m3_5b2e/evidence/20261015_preflight_v1'
REQUEST=Path('/users/atabarra/.codex/attachments/c2b5eedc-9887-4af8-ba90-7e71a76b2a98/Pasted text.txt')
def ref(p):return dict(path=str(p),sha256=digest(p))
def put(n,s):
 with (OUT/n).open('x') as f:f.write(s)
def prepare():
 b=target_binding();check(canonical(b)==read(E/'protected_policy_issuance_request.json')['binding_sha256'],'Exact E/D binding')
 for base in (D,E):
  for n,h in read(base/'artifact_manifest.json').items():check(digest(base/n)==h,'Prior artifact identity '+n)
 for n,e in read(D/'source_manifest.json').items():check(digest(ROOT/n)==e['sha256']==digest(D/e['archive']),'Reviewed source identity')
 check(not AUTHORITY.exists(),'Protected book changed; reassess actual issuance before proceeding')
 policy_hash=digest(ROOT/'m3_5b2d/policy.py');old_prov=read(E/'request_provenance.json');check(old_prov['policy_decision']=='D2' and old_prov['policy_decision_approved_by_user'] is True,'Existing scientific decision')
 check(digest(Path(old_prov['request']['path']))==old_prov['request']['sha256'],'Scientific approval provenance')
 write(OUT/'source_and_input_freeze.json',dict(binding=b,binding_sha256=canonical(b),reviewed_D_source_manifest=ref(D/'source_manifest.json'),reviewed_D_artifact_manifest=ref(D/'artifact_manifest.json'),reviewed_E_artifact_manifest=ref(E/'artifact_manifest.json'),policy=ref(ROOT/'m3_5b2d/policy.py'),authorization_validator=ref(ROOT/'m3_5b2d/authorization.py'),existing_D2_approval=ref(E/'request_provenance.json'),existing_D2_decision=ref(E/'M3_5B2E_D2_DECISION_RECORD.md'),current_user_request=ref(REQUEST)))
 shutil.copyfile(REQUEST,OUT/'user_request.txt')
 scope=dict(schema_version='m3.5b2f.policy_scope_provenance.NON_EXECUTABLE.v1',historical_id=HISTORICAL,selected_id=SELECTED,historical_path=str(INPUT),frozen_manifest_sha256=MANIFEST,mechanical_equivalence_sha256=EQUIVALENCE,policy='historical_saved_trajectory_admission.v1_candidate',policy_hash=policy_hash,full_binding_sha256=canonical(b),source_and_input_freeze=ref(OUT/'source_and_input_freeze.json'),D2_decision='Approved scientific choice verified from M3.5B-2E user request; not a protected receipt',D2_approval_provenance=ref(E/'request_provenance.json'),scope='One exact historical saved trajectory may be considered for training admission only after complete current independent numerical acceptance and later separate alias/member-crosswalk/release approvals. No new production trajectory or other historical ID.',original_interruption_certified=False,numerical_execution_authorized=False,numerical_validation_accepted=False,alias_approved=False,family_crosswalk_approved=False,release_eligible=False,production_restart_policy_unchanged=True,termination='maxiter',reference_optimum=False)
 write(OUT/'policy_scope_provenance.json',scope)
 intended=dict(kind='policy',sha256=policy_hash,binding_sha256=canonical(b),decision='D2',identity=None,approved_at=None,expires_at=None,reference=str(OUT/'policy_scope_provenance.json')+' SHA256 '+digest(OUT/'policy_scope_provenance.json'))
 req=dict(schema_version='m3.5b2f.issuance_request.NON_EXECUTABLE.v1',status='blocked_missing_receipt_metadata',protected_destination=str(AUTHORITY),expected_book_schema='m3.5b2d.reviewed_approvals.v1',book_format={'schema_version':'m3.5b2d.reviewed_approvals.v1','records':'Existing records preserved; exactly one new policy record only after authorized publication'},proposed_policy_record=intended,missing_fields=['identity','approved_at','expires_at'],field_interpretation={'identity':'Genuinely designated publisher/approval metadata identity, not invented by Codex','approved_at':'Timezone-aware actual authorized issuance/valid-from time, recorded only when legitimately available','expires_at':'Explicit authorized finite validity endpoint; must be later than current validation time'},outstanding_administrative_action='An actor genuinely authorized for protected publication must designate their receipt identity and policy-validity interval, then publish this exact hash-bound policy-only record through controlled protected-file publication. No new scientific D2 decision is requested.',external_issuer_required_by_code=False,cryptographic_signature_required_by_code=False,named_actor_or_key_designated_in_requests=False,publication_permission_proven=False,protected_publication_attempted=False,original_interruption_certified=False,numerical_execution_authorized=False,numerical_validation_accepted=False,alias_approved=False,family_crosswalk_approved=False,release_eligible=False,publication_timestamp=None,receipt_identity=None,immutable_receipt_hash=None,request_is_not_a_receipt=True)
 write(OUT/'M3_5B2F_NONEXECUTABLE_ISSUANCE_REQUEST.json',req)
 write(OUT/'M3_5B2F_POLICY_RECEIPT_STATUS.json',dict(status='BLOCKED_BY_MISSING_RECEIPT_METADATA',scientific_D2_verified=True,protected_destination=str(AUTHORITY),protected_book_exists=False,policy_hash=policy_hash,binding_sha256=canonical(b),protected_policy_issued=False,runtime_policy_receipt_recognized=False,publication_timestamp=None,receipt_identity=None,receipt_hash=None,missing_metadata=req['missing_fields'],numerical_execution_authorized=False,numerical_validation_accepted=False,alias_approved=False,family_crosswalk_approved=False,release_eligible=False))
 put('M3_5B2F_ISSUANCE_WORKFLOW_AUDIT.md',f'''# Existing protected issuance workflow

Scientific D2 approval is verified from the unchanged E decision/source record and the exact original user request hash. This F request authorizes examining and completing issuance only if actual authority/prerequisites exist; it explicitly forbids treating the prompt as proof of protected publication authority. No scientific reapproval is requested.

The runtime entry is m3_5b2d.authorization.approval_record, which reads `{AUTHORITY}`. The book is absent, including its parent directory. Its exact root fields are schema_version='m3.5b2d.reviewed_approvals.v1' and records. A record has exactly eight fields:kind,sha256,binding_sha256,decision,identity,approved_at,expires_at,reference. A policy record must have kind=policy, decision=D2, policy SHA256 `{policy_hash}` and full binding SHA256 `{canonical(b)}`. The binding includes exact historical/selected IDs, path/specs, manifest/equivalence, original input hashes, historical/selected generation sources and all reviewed validation/implementation sources.

The validator requires exactly one matching row, a nonempty identity/reference and timezone-aware approved_at<=now<expires_at. It rejects missing/duplicate/wrong-hash/wrong-binding/expired/schema-inconsistent records. It has no publisher API, registered signer identity, cryptographic signature verification, explicit receipt-ID field, authenticated-actor check or enforced append-only journal. Authority is deliberately assumed from genuine controlled publication to this fixed protected file, outside this session's writable roots. The code does not itself verify ownership/mode or authenticate an arbitrary nonempty identity string. No external issuer service, mandatory independent reviewer or signing key is invented by this audit.

A user-approved scientific decision is a valid policy-provenance input. It is not proof that a given process may create that protected file, nor does it fill receipt identity or validity metadata. The current requests designate neither a genuine publisher identity nor a policy receipt validity interval. No actual record is issued; approved_at/expires_at/identity and actual publication/receipt hash stay null in the non-executable request. The prior E timestamp is an observation/recording time, not silently reused as a protected issuance timestamp. The prior proposal's24-hour *execution* expiry is not policy validity and is not reused.

Exact reference scope/provenance is frozen separately so the strict eight-field runtime row need not gain unsupported keys or create a hash cycle. After genuine publication, canonical SHA256 of the exact complete row can serve as its content-derived external receipt identity, and the exact book bytes receive a separate file SHA256. No such final receipt identity/hash is claimed now. Hashes in this evidence package identify unissued request artifacts only.

There is no implementation-enforced append-only history. Any authorized publication must preserve unrelated existing rows byte-for-byte in value, record the before-book hash or absence, use reviewed controlled publication, avoid duplicate policy rows and retain exact post-publication row/book/hash/readback provenance. These are publication-preservation requirements, not a newly implemented issuer or rewrite of the validator. No protected record, policy source, production runner, numerical contract, resource ceiling or old evidence was modified.
''')
 put('M3_5B2F_AUTHORITY_AND_VALIDITY_REVIEW.md',f'''# Authority, metadata and the one remaining administrative action

CaseB applies. The scientific decision is already approved. The remaining administrative action is genuine protected policy-only issuance by an authorized publisher, supplying identity and a concrete timezone-aware policy validity interval. No independent issuer/key requirement exists in the code; a publisher cannot be invented merely because metadata accepts a string. The current prompt explicitly withholds such an inference. Neither actual protected publication authority nor required identity/validity metadata is supplied.

The exact missing record fields are identity,approved_at,expires_at. Publication timestamp, receipt identity and receipt hash can only be recorded after actual publication. Identity must describe the actual designated actor/provenance; approved_at must represent a legitimate issuance/valid-from time; expires_at must be an expressly selected live finite endpoint. They are not filled with synthetic test values, the current clock, an account name inferred from a filesystem path, an invented reviewer or a guessed duration.

Authorized publication procedure:

1. Designate the genuinely authorized protected publisher and approve the identity/validity metadata for this already approved D2 scope. Do not ask for a new D2 scientific choice or add numerical work.
2. Reverify source_and_input_freeze.json against current exact D sources, original inputs, scientific decision provenance, manifest/equivalence and full binding. Any drift stops; never silently substitute a newer policy/runner.
3. Review M3_5B2F_NONEXECUTABLE_ISSUANCE_REQUEST.json and the hash-bound policy_scope_provenance.json. Populate only the three missing fields with legitimate values. Confirm that the record grants no numerical execution/adoption/release permission.
4. Through actual authorized protected-file publication at `{AUTHORITY}`, preserve every unrelated existing record. If absent, create only the prescribed book schema with this single policy row. Do not add protocolA_execution, alias, family_crosswalk or numerical_acceptance rows. The fixed location is outside current writable roots: applicable filesystem permission must be obtained for any actual write. No such write or escalation was attempted here because the authority/metadata prerequisite remains unresolved; sandbox access alone would not create semantic authority.
5. Record actual publication time and exact canonical-row/file hashes in a genuine publication audit. Read back using unchanged approval_record('policy', policy_hash, target_binding()). Test wrong object/binding/expiry, then verify missing execution authorization still rejects before solver imports.
6. Retain before/after book evidence and immutable row hash; leave worker tickets, validation outputs, numerical certificates, alias/crosswalk/release approvals and all production files untouched.

This procedure describes the established protected-file trust model, not a substitute issuer command or bypass. It is intentionally non-executable. No new authorization system was designed. The smallest next input is designation of publisher identity and policy validity, followed by that actor's protected policy-only publication. ProtocolA would still require a separate exact numerical execution decision and protected execution issuance afterward.
''')
 put('M3_5B2F_AUTHORIZATION_SEPARATION_TESTS.md','''# Focused boundary tests

39 focused nonnumerical tests passed,0 failed:17 newF checks and22 retainedE checks. Actual pytest log is preserved. No synthetic object is proof of genuine issuance; none was written to the protected authority path. E's temporary ticket/ledger fixtures remain isolated test data, not live credentials.

Tests cover missing actual policy book, malformed/missing/duplicate/expired policy rows, wrong policy hash, wrong D1 decision, empty issuer metadata, wrong historical/selected ID/manifest/equivalence/input root and resulting binding hash, other-configuration reuse, policy-only rejection for execution/alias/crosswalk/numerical-acceptance kinds, new-production scope rejection and missing execution rejection before solver imports. E adds execution-issuer metadata mismatch, environment/source/input/root mismatch, policy-only/no-execution flow, ticket reuse, durable budget overflow, incomplete evidence, unapproved alias/crosswalk and absence of a publication API.

Important limitation discovered and retained: the pure policy-row parser accepts an arbitrary nonempty identity in an in-memory trusted-book fixture. It has no authenticated issuer registry/signature to decide that a named identity is wrong. A dedicated audit test records this behavior rather than falsely claiming arbitrary wrong-issuer authentication rejects. Empty identities and execution receipt/sidecar mismatches reject. Genuineness therefore depends on authorized protected publication, which is not established here. This limitation is reported, not repaired by weakening or redesigning the frozen validator.

ScientificD2, protected policy issuance, numerical execution approval, protected execution issuance, numerical acceptance, alias approval, family-crosswalk approval and ML publication approval remain eight separate stages. Policy metadata alone cannot confer any later approval. The staging adapter returns published=false and has no publication API; real ML release publication remains a separate future reviewed action. All actual downstream flags remain false.
''')
 put('M3_5B2F_FINAL_REPORT.md','''# M3.5B-2F protected policy receipt review

The existing scientific D2 approval is verified. Genuine protected policy publication did not occur because designated receipt identity and policy-validity metadata, together with actual authorized protected publication, remain unavailable. No new scientific decision is requested.

1. D2 approval verified against the exact unchanged E user-request provenance and decision record; exact one-history/one-selected-row scope retained.
2. Existing workflow is a fixed protected JSON book with exact eight-field records and policy/binding hash plus D2/identity/reference/validity checks. There is no issuer CLI, key or authenticated signer registry.
3. Genuine controlled publication to that protected book is required. The prompt is not proof of the publisher's authority; neither a new identity nor a validity interval is fabricated.
4. Required identity,approved_at,expires_at are missing. Exact policy/input/source/binding/provenance/scope fields are frozen in the complete non-executable request; publication-dependent timestamp/receipt hash remain null.
5. Protected publication:no. No protected file or parent directory was created, and no substitute receipt was issued.
6. Remaining action: designate an authorized publisher and receipt identity/validity interval, then have that actor perform this exact policy-only protected publication/readback procedure. Applicable filesystem permission is additionally required for the fixed outside-workspace path; no publication attempt was made with missing prerequisites.
7. Receipt ID/hash/validity:none issued. Only the non-executable request and evidence package have artifact hashes. Runtime has no explicit receipt_id field; a future genuine publication audit can identify exact row/book bytes by SHA256.
8. Runtime guard currently recognizes no policy receipt because the book is absent. No synthetic fixture is used as actual readback evidence.
9. Even a future genuine policy row is insufficient for FEM/OC. Separate numerical execution approval and protected execution record remain required; alias/crosswalk/numerical-acceptance/publication remain later distinct approvals.
10.39 focused tests passed,0 failed. Empty issuer/metadata mismatch rejects, but arbitrary nonempty policy identity is not authenticated by this implementation; that audit limitation is explicit rather than claimed as a passing wrong-issuer security guarantee.
11. No prior protected, production, dataset, visualization or source file changed. Original12/frozen30/historical arrays/journals/assignments/numerical contracts/restart policy and resource ceilings remain intact. Source and artifact hashes and protected baseline/final comparison accompany this report.
12. FEM0,OC0,replay0,generation0,restart0,ML0,SLURM0; no execution credentials, actual worker tickets, numerical acceptance, alias adoption, release publication, commit or push.
13. Next: complete the administrative policy-only issuance action for the already approvedD2. After genuine readback, separately review and approve the exact ProtocolA execution sidecar/expiry/binding and deliberate execution issuance. This task grants no such execution authority.

BLOCKED_BY_MISSING_RECEIPT_METADATA
''')
 check(not any(n.startswith(('topopt_utils','numpy','torch','pypardiso')) for n in sys.modules),'Solver-free audit required')
 print('CaseB: exact non-executable issuance package prepared; protected book untouched',flush=True)
def finalize():
 check('39 passed' in Path('/tmp/m3_5b2f_focused_tests.log').read_text(),'Focused tests');shutil.copyfile('/tmp/m3_5b2f_focused_tests.log',OUT/'focused_tests.log')
 old=read(OUT/'protected_before.json');changed=[n for n,h in old.items() if not (ROOT/n).is_file() or digest(ROOT/n)!=h];check(not changed,'Protected drift '+str(changed[:10]))
 check(not AUTHORITY.exists() and not (ROOT/'m3_5b2d/validation_runs').exists() and not (ROOT/'m3_5b2d/validation_attempts').exists(),'Protected publication/execution forbidden with missing prerequisites')
 write(OUT/'protected_final_integrity.json',dict(passed=True,protected_files=len(old),changed=changed,protected_policy_issued=False,execution_authorized=False,FEM=0,OC=0,replay=0,generation=0,restart=0,ML=0,SLURM=0,tests_passed=39,tests_failed=0))
 snap=OUT/'source_snapshots';snap.mkdir();sources={}
 for p in Path(__file__).parent.iterdir():
  if p.is_file() and p.suffix in ('.py','.md'):
   if p.suffix=='.py':ast.parse(p.read_text())
   d=snap/(p.name+'.txt');shutil.copyfile(p,d);sources[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(d.relative_to(OUT)))
 write(OUT/'source_manifest.json',sources);write(OUT/'artifact_manifest.json',{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()});print({'status':'BLOCKED_BY_MISSING_RECEIPT_METADATA','tests_passed':39,'protected_files_unchanged':len(old)},flush=True)
if __name__=='__main__':finalize() if sys.argv[1:]==['finalize'] else prepare()
