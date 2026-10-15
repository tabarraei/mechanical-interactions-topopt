"""Create review-only artifacts and finalize protected integrity after tests."""
import sys,ast,shutil
from pathlib import Path
from .common import *
from .policy import DEFAULT,POLICY,decision_state
from .certificate import template,validate
from .integration import alias_template,validate_alias

def schema_of(v):
 if isinstance(v,dict):return dict(type='object',required=sorted(v),additionalProperties=False,properties={k:schema_of(x) for k,x in v.items()})
 return {'const':v}
def schema_doc(v,title):
 d=schema_of(v);d.update({'$schema':'https://json-schema.org/draft/2020-12/schema','title':title,'description':'Review-only schema, not authority or acceptance; semantic validators enforce source/input/count/evidence and protected approval requirements.'});return d

def prepare():
 b=target_binding();scope_registry=registry_check(b);write(REPORT/'target_binding.json',b);write(REPORT/'registry_validation_inventory.json',scope_registry)
 default=dict(schema_version=POLICY,decision=None,**DEFAULT,historical_interruption_status='uncertified_missing_original_receipt_and_control',execution_authorized=False)
 write(REPORT/'policy_candidate_state.json',default);write(REPORT/'policy_decision_outcomes.json',{k:decision_state(k) for k in ('D1','D2')})
 write(REPORT/'approval_book_empty_template.json',dict(schema_version='m3.5b2d.reviewed_approvals.v1',records=[]))
 cert=template(b);validate(cert,b);write(REPORT/'validation_certificate_unresolved_template.json',cert)
 cs=schema_doc(cert,'m3.5b2d.validation_certificate.v1')
 for k in ('states','transitions'):cs['properties'][k]={'type':'integer','minimum':0,'maximum':501 if k=='states' else 500}
 for k,v in WORK.items():cs['properties']['counts']['properties'][k]={'type':'number' if k=='worker_seconds' else 'integer','minimum':0,'maximum':v}
 for k in GATES:cs['properties']['gates']['properties'][k]={'type':'boolean'}
 for k in ('v2','v3','supplemental','raw_validation_complete'):cs['properties'][k]={'type':'boolean'}
 cs['properties']['resources']={'type':'array','maxItems':2};cs['properties']['raw_evidence']={'type':'object'};cs['properties']['approvals']={'type':'object','required':['policy','numerical_acceptance','alias','family_crosswalk'],'additionalProperties':False,'properties':{k:{'type':['object','null']} for k in ('policy','numerical_acceptance','alias','family_crosswalk')}};cs['properties']['certificate_hash']={'type':'string','pattern':'^[0-9a-f]{64}$'}
 write(REPORT/'validation_certificate.schema.json',cs)
 alias=alias_template(b);validate_alias(alias,b);write(REPORT/'alias_candidate.json',alias);als=schema_doc(alias,'m3.5b2d.provenance_alias.v1')
 for k in ('alias_approved','family_crosswalk_approved'):als['properties'][k]={'type':'boolean'}
 als['properties']['approval_metadata']={'type':'object','required':['alias','crosswalk'],'additionalProperties':False,'properties':{'alias':{'type':['object','null']},'crosswalk':{'type':['object','null']}}};als['properties']['certificate_reference']={'type':['object','null']};write(REPORT/'alias.schema.json',als)
 crosswalk=dict(schema_version='m3.5b2d.member_crosswalk.v1',historical_id=HISTORICAL,selected_id=SELECTED,families=alias['families'],partitions=alias['partitions'],equivalence_sha256=EQUIVALENCE,family_crosswalk_approved=False,approval_identity=None,approved_at=None,approval_reference=None)
 write(REPORT/'crosswalk_candidate.json',crosswalk);write(REPORT/'crosswalk.schema.json',schema_doc(crosswalk,'Member-specific unapproved crosswalk'))
 from .authorization import FIELDS
 props={k:{'type':'object' if k in ('binding','environment','limits','work','approval') else 'integer' if k=='disk_bytes' else 'array' if k=='actions' else 'string'} for k in FIELDS}
 props['binding']={'const':b};props['limits']={'const':LIMITS};props['work']={'const':WORK};props['actions']={'const':['replay','supplemental','stop_and_quarantine_no_retry_no_release']};props['schema_version']={'const':'m3.5b2d.protocolA.authorization.v1'};props['kind']={'const':'historical_saved_validation_only'}
 write(REPORT/'authorization.schema.json',{'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Unissued ProtocolA sidecar contract','type':'object','required':sorted(FIELDS),'additionalProperties':False,'properties':props,'description':'Schema plus strict semantic validator required; syntactic validity never confers authority. No executable example is created.'})
 from .report_text import documents
 documents(REPORT,64)
 for src,dst in [('/tmp/m3_5b2d_tests_initial.log','initial_tests_57_passed_1_failed.log'),('/tmp/m3_5b2d_tests_verified.log','focused_tests_64_passed.log'),('/tmp/m3_5b2d_tests_combined.log','intermediate_combined_95_passed.log')]:shutil.copyfile(src,REPORT/dst)
 write(REPORT/'review_input_hashes.json',{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'PROJECT_CONTEXT.md',ROOT/'AGENTS.md',ROOT/'m3_5b2s/reports/20261015_protocol_v1/M3_5B2S_FINAL_REPORT.md',ROOT/'m3_5b2s/reports/20261015_protocol_v1/M3_5B2S_EVIDENCE_GAP_MATRIX.json',ROOT/'m3_5b2r/reports/20261015_provenance_v1/M3_5B2R_FINAL_REPORT.md',ROOT/'m3_4br4r1/results/20261015T020242Z/R4_R1_RESTART_ACCEPTANCE_SPEC.md',ROOT/'m3_4br4approved/results/20261015_approved_continuation/accepted_dataset_index.json']})
 print('Prepared disabled templates/schemas and review documents',flush=True)

def finalize():
 log=Path('/tmp/m3_5b2d_tests_FINAL.log');check('95 passed' in log.read_text(),'Final tests');shutil.copyfile(log,REPORT/'final_tests_95_passed.log')
 old=read(REPORT/'protected_before.json');changed=[n for n,h in old.items() if not (ROOT/n).is_file() or digest(ROOT/n)!=h];check(not changed,'Protected files changed '+str(changed[:10]))
 b=target_binding();check(b==read(REPORT/'target_binding.json'),'Frozen candidate target binding changed')
 check(not (ROOT/'m3_5b2d/validation_runs').exists() and not (ROOT/'m3_5b2d/validation_attempts').exists(),'No validation output or lease permitted here')
 write(REPORT/'protected_final_integrity.json',dict(passed=True,protected_files=len(old),changed=changed,manifest_sha256=MANIFEST,approved_restart_policy_sha256=digest(ROOT/'m3_4br4r1/policy.py'),authority_created=False,numerical_runs=0,FEM=0,OC=0,restart=0,ML=0,SLURM=0,production_modified=False,alias_adopted=False,release_published=False))
 snap=REPORT/'source_snapshots';snap.mkdir();sources={}
 for p in Path(__file__).parent.iterdir():
  if not p.is_file() or p.suffix not in ('.py','.md'):continue
  if p.suffix=='.py':ast.parse(p.read_text())
  dest=snap/(p.name+'.txt');shutil.copyfile(p,dest);sources[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(REPORT)))
 write(REPORT/'source_manifest.json',sources);write(REPORT/'test_outcomes.json',dict(new_focused_passed=64,prior_schema_regression_passed=31,combined_passed=95,combined_failed=0,initial_development_passed=57,initial_development_failed=1,initial_failure='Test fixture changed threads after environment binding; environment gate correctly fired before missing approval gate; corrected fixture, retained original log',scope='Synthetic counters/schemas/import tripwires and read-only metadata/hash checks only; no numerical work'))
 write(REPORT/'final_status.json',dict(status='READY_FOR_HISTORICAL_POLICY_REVIEW',**DEFAULT,execution_authorized=False,tests_passed=95,new_focused_tests=64,tests_failed=0))
 write(REPORT/'artifact_manifest.json',{str(p.relative_to(REPORT)):digest(p) for p in REPORT.rglob('*') if p.is_file()})
 print({'status':'READY_FOR_HISTORICAL_POLICY_REVIEW','protected_files':len(old),'tests_passed':95,'new_FEM':0,'new_OC':0},flush=True)
if __name__=='__main__':finalize() if sys.argv[1:]==['finalize'] else prepare()
