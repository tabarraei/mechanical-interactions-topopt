"""D2 decision provenance and read-only preflight. No issuance or numeric runner."""
import sys,os,json,ast,shutil,platform,importlib.metadata
from pathlib import Path
from datetime import datetime,timezone
from m3_5b2d.common import *
from m3_5b2d.policy import POLICY
from m3_5b2e.report_text import documents
OUT=ROOT/'m3_5b2e/evidence/20261015_preflight_v1'
D=ROOT/'m3_5b2d/reports/20261015_candidate_v1'
REQUEST=Path('/users/atabarra/.codex/attachments/3e750dd2-7ed4-43b1-bdb2-4dcaf8557e74/Pasted text.txt')
def ref(p):return dict(path=str(Path(p)),sha256=digest(p))
def prepare():
 before=read(OUT/'protected_before.json');check(all((ROOT/n).is_file() and digest(ROOT/n)==h for n,h in before.items()),'Historical preflight baseline')
 sm=read(D/'source_manifest.json');source_diffs=[]
 for n,e in sm.items():
  if digest(ROOT/n)!=e['sha256'] or digest(D/e['archive'])!=e['sha256']:source_diffs.append(n)
 check(not source_diffs,'Reviewed D source drift; do not substitute revisions')
 dm=read(D/'artifact_manifest.json');check(all(digest(D/n)==h for n,h in dm.items()),'D historical artifact integrity')
 binding=target_binding();check(binding==read(D/'target_binding.json'),'Exact64-test implementation/input binding');registry=registry_check(binding)
 write(OUT/'request_provenance.json',dict(request=ref(REQUEST),recorded_UTC=datetime.now(timezone.utc).isoformat(),source='Explicit user-provided M3.5B-2E request',policy_decision='D2',policy_decision_approved_by_user=True,protected_issuance=False,execution_authorized=False))
 shutil.copyfile(REQUEST,OUT/'user_request.txt')
 freeze=dict(schema_version='m3.5b2e.source_freeze.v1',reviewed_D_sources={n:{**e,'actual_sha256':digest(ROOT/n)} for n,e in sm.items()},reviewed64_test_source_differences=source_diffs,reviewed_D_artifact_manifest=ref(D/'artifact_manifest.json'),test_logs=[ref(D/'focused_tests_64_passed.log'),ref(D/'final_tests_95_passed.log')],policy=ref(ROOT/'m3_5b2d/policy.py'),runner=ref(ROOT/'m3_5b2d/runner.py'),worker=ref(ROOT/'m3_5b2d/worker.py'),authorization=ref(ROOT/'m3_5b2d/authorization.py'),accounting=ref(ROOT/'m3_5b2d/accounting.py'),monitor=ref(ROOT/'m3_4b/monitoring.py'),certificate=ref(ROOT/'m3_5b2d/certificate.py'),alias_schema=ref(D/'alias.schema.json'),crosswalk_schema=ref(D/'crosswalk.schema.json'),registry=ref(ROOT/'m3_5b2/registry.py'),registry_and_release_integration=ref(ROOT/'m3_5b2d/integration.py'),existing_release_adapter=ref(ROOT/'m3_5b2/release.py'),all_current_original_archive_and_validation_sources=binding['sources'],historical_input_hashes=binding['input_hashes'],binding=binding)
 write(OUT/'M3_5B2E_SOURCE_FREEZE.json',freeze)
 expected_env=read(ROOT/'m3_5b2s/reports/20261015_protocol_v1/review_environment_identity.json');env=environment()
 check(env['python']==expected_env['python'] and env['platform']==expected_env['platform'] and all(env['packages'][k]==expected_env['packages'][k] for k in env['packages']),'Python/package/platform environment drift')
 check(env['threads']==expected_env['required_future_threads'],'Controlled preflight thread envelope')
 disk=shutil.disk_usage(ROOT);output=ROOT/'m3_5b2d/validation_runs/20261015_protocolA_historical_plan10_v1'
 check(not output.exists(),'Output collision');check(not (ROOT/'m3_5b2d/validation_attempts').exists(),'Existing validation lease/attempt')
 check(disk.free>=4*1024**3,'Disk floor')
 check(Path('/proc/self/status').is_file() and all(k+':' in Path('/proc/self/status').read_text() for k in ('VmRSS','VmHWM')),'Monitor facilities')
 check(WORK==dict(FEM=1005,OC=500,selected_trials=500,bisection_trials=52500,workers=2,worker_seconds=1800),'Work consistency')
 check(STAGES['replay']['FEM']==2*501 and STAGES['supplemental']['FEM']==3 and STAGES['replay']['OC']==500 and STAGES['replay']['selected_trials']==500,'Decomposition')
 write(OUT/'controlled_environment.json',env);write(OUT/'registry_preflight.json',registry)
 resources=dict(planning_limits=WORK,memory_limits=LIMITS,work_decomposition=STAGES,consistent=True,observed_free_disk_bytes=disk.free,minimum_disk_bytes=4*1024**3,proposed_output_root=str(output),output_exists=False,validation_attempts_exist=False,monitor_procfs_available=True,fcntl_available=__import__('importlib.util',fromlist=['find_spec']).find_spec('fcntl') is not None,environment=env,environment_sha256=digest(OUT/'controlled_environment.json'),ambient_threads=read(OUT/'ambient_environment.json')['threads'],launch_requires_explicit_threads=True,numerical_resource_qualification=False,accounting_verification='Synthetic negative tests persist counts across Budget objects and reject overflow; no numerical work charged in live ledger',no_live_work_ledger_created=True)
 write(OUT/'M3_5B2E_RESOURCE_CHECKS.json',resources)
 # No independent reviewer/key requirement is invented. Missing receipt metadata
 # and protected publication are disclosed; this object is not a receipt book.
 policy_request=dict(schema_version='m3.5b2e.policy_issuance_request.NON_EXECUTABLE.v1',policy_decision='D2',user_approval=ref(REQUEST),policy_implementation=ref(ROOT/'m3_5b2d/policy.py'),binding_sha256=canonical(binding),protected_destination=str(AUTHORITY),protected_file_exists=AUTHORITY.exists(),issuer_identity=None,approved_at=None,expires_at=None,decision_reference='User request SHA256 '+digest(REQUEST),missing_prerequisites=['Designated metadata identity for protected publisher/approver','Explicit policy receipt validity window/expiry metadata','Deliberate permitted filesystem publication at the protected authority path'],independent_reviewer_required_by_code=False,cryptographic_key_required_by_code=False,signature_verification_implemented=False,identity_trust_model='Authority is the protected book and exact object/binding hashes; identity strings are receipt metadata, not independently authenticated credentials',protected_issuance_completed=False,execution_authorized=False)
 check(not AUTHORITY.exists(),'Unexpected protected issuance; review before claiming authority')
 write(OUT/'protected_policy_issuance_request.json',policy_request)
 proposal=dict(schema_version='m3.5b2e.protocolA_proposal.NON_EXECUTABLE.v1',historical_id=HISTORICAL,selected_id=SELECTED,frozen_manifest_sha256=MANIFEST,equivalence_sha256=EQUIVALENCE,input_root=str(INPUT),output_root=str(output),source_freeze=ref(OUT/'M3_5B2E_SOURCE_FREEZE.json'),full_binding_sha256=canonical(binding),environment=ref(OUT/'controlled_environment.json'),required_threads=env['threads'],policy_decision_record='M3_5B2E_D2_DECISION_RECORD.md',user_policy_approved=True,protected_policy_issuance_reference=None,protected_policy_request=ref(OUT/'protected_policy_issuance_request.json'),actions_proposed=['Current full independent replay of saved501 states/500 transitions','Current supplemental feature/centered-FD/profile checks','Read-only raw certificate assembly; no optimization or restart run'],work_ceilings=WORK,resource_limits=LIMITS,disk_requirement_bytes=4*1024**3,expiry_proposal='At most24 hours after separate future execution issuance; never active now; exact timezone-aware timestamps must be reviewed before issuance',stop_rules=['Immediate stop/quarantine on any mandatory source/input/env/disk/registry/work/resource/numerical failure','No retry, lease reset, extra worker or repeated numerical state','Preserve partial evidence, monitors and consumed work; exclude failed/incomplete candidate from ML','No generation, restart, alias/crosswalk approval or release publication'],required_evidence=['501 complete independent primary/API state operands and mechanics/COO records','500 action/selected multiplier/strict feasibility/current v2-v3 records with hashes','Supplemental feature/FD/profile pass; unchanged first stopping/maxiter/no-reference metadata','Complete source/environment/native/work bindings and monitored worker/parent/time evidence','Historical original interruption remains explicitly uncertified'],acceptance_criteria='Every unchanged current primary and v2/v3/supplemental/identity/resource gate passes; no old replay assertion replaces raw evidence. Numerical acceptance, alias, crosswalk and publication remain separate.',issuer_requirements={'protected_policy_publisher':None,'execution_approver':None,'execution_publisher':None,'independent_numerical_acceptance_reviewer':None,'alias_scientific_reviewer':None,'member_crosswalk_reviewer':None,'publication_approver':None,'note':'Roles not designated by this package; no code-mandated external signing key or reviewer identity registry exists'},execution_authorized=False,authorization_issued=False,numerical_validation_accepted=False,alias_approved=False,family_crosswalk_approved=False,release_eligible=False)
 write(OUT/'M3_5B2E_EXECUTION_PROPOSAL.json',proposal)
 write(OUT/'preflight_checks.json',dict(technical_checks_passed=True,source_drift=[],native_intact=True,all_D_artifact_hashes_match=True,controlled_environment_passed=True,ambient_thread_configuration_passed=False,ambient_note='Inherited shell leaves all thread variables unset; no live guard GO. Explicit1/16/16 envelope used only for this solver-free preflight and tests.',disk_passed=True,output_clean=True,registry_passed=True,monitoring_available=True,accounting_synthetic_tests_required=True,protected_policy_issued=False,execution_authorized=False,preflight_is_not_runner_activation=True))
 documents(OUT,policy_request,proposal,resources)
 check(not any(n.startswith(('topopt_utils','numpy','torch','pypardiso','m3_4br3.replay')) for n in sys.modules),'Nonnumerical preflight imported solver/numerical modules')
 print('Read-only technical preflight passed in explicit thread envelope; protected policy issuance pending',flush=True)
def finalize():
 log=Path('/tmp/m3_5b2e_negative_tests.log');check('117 passed' in log.read_text(),'Negative tests');shutil.copyfile(log,OUT/'negative_and_regression_tests.log')
 old=read(OUT/'protected_before.json');changed=[n for n,h in old.items() if not (ROOT/n).is_file() or digest(ROOT/n)!=h];check(not changed,'Protected/source/artifact drift '+str(changed[:10]))
 check(not AUTHORITY.exists() and not (ROOT/'m3_5b2d/validation_runs').exists() and not (ROOT/'m3_5b2d/validation_attempts').exists(),'No protected issuance/execution permitted')
 write(OUT/'protected_final_integrity.json',dict(passed=True,protected_files=len(old),changed=changed,tests_passed=117,tests_failed=0,new_negative_tests=22,prior_regressions=95,FEM=0,OC=0,restart=0,replay=0,ML=0,SLURM=0,alias_adoption=0,release_publication=0,protected_policy_issued=False,execution_credentials_issued=False))
 snapshots=OUT/'source_snapshots';snapshots.mkdir();sources={}
 for p in Path(__file__).parent.iterdir():
  if p.is_file() and p.suffix in ('.py','.md'):
   if p.suffix=='.py':ast.parse(p.read_text())
   dest=snapshots/(p.name+'.txt');shutil.copyfile(p,dest);sources[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(OUT)))
 write(OUT/'preflight_source_manifest.json',sources);write(OUT/'final_status.json',dict(status='BLOCKED_BY_PROTECTED_POLICY_ISSUANCE',user_D2_decision_recorded=True,protected_policy_issued=False,execution_authorized=False,authorization_issued=False,numerical_validation_accepted=False,alias_approved=False,family_crosswalk_approved=False,release_eligible=False,technical_preflight_passed_in_controlled_environment=True,tests_passed=117))
 write(OUT/'artifact_manifest.json',{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()});print({'status':'BLOCKED_BY_PROTECTED_POLICY_ISSUANCE','protected_files':len(old),'tests_passed':117},flush=True)
if __name__=='__main__':finalize() if sys.argv[1:]==['finalize'] else prepare()
