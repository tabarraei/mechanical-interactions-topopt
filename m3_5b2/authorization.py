"""Strict inert-sidecar parser plus a separately pinned, reviewed approval book.
No issuer/activation command is provided. Parsing never confers authority.
"""
from datetime import datetime,timezone
from pathlib import Path
from .common import *
SCHEMA='m3.5b2.authorization.v1'
# The workspace book is an immutable EMPTY schema/example, never an issuer.
# Runtime issuance is read ONLY from a fixed authority path outside writable
# workspace roots; publication requires a future explicit approval workflow.
AUTHORITY_BOOK=Path('/users/atabarra/.codex/approvals/m3_5b2_issued_authorizations.json')
# Pinning the empty example does not make implementation hashes depend on
# future issued receipts (avoids a circular code/sidecar/book SHA dependency).
APPROVAL_BOOK_SHA='fbd820d4c268d555b8652ecf6cc8b13f9f82d1d78c6367dbd16f73e1f40750a4'
ACTIONS={'generate','resume','replay','supplemental','adopt'}
FIELDS={'schema_version','manifest_sha256','authorization_id','kind','stage','approved_ids','output_root','production_sources','validation_sources','implementation_sources','environment','limits','stop_rules','maximum_work','approval','experiment','resume_bindings','qualification_evidence','disk_requirement_bytes'}
WORK={'max_ids','max_worker_jobs','max_FEM_calls','max_OC_calls','max_states_per_id','max_updates_per_id','max_total_worker_seconds'}
STOP=dict(on_failure='stop_and_quarantine',on_resource='stop_and_quarantine',on_incomplete='exclude_from_release',overwrite=False,reference_labels=False,concurrency=1)
def timestamp(s):
 check(isinstance(s,str),'Timestamp type');t=datetime.fromisoformat(s.replace('Z','+00:00'));check(t.tzinfo is not None,'Timezone required');return t

def validate_candidate(p,selection,*,kind,stage,ids,output_root,current_environment=None,verification_time=None):
 """Pure structure/scope validation, deliberately NOT an execution permit."""
 check(isinstance(p,dict) and set(p)==FIELDS,'Missing/unknown authorization fields');check(p['schema_version']==SCHEMA and p['manifest_sha256']==PLAN_SHA,'Authorization schema/manifest')
 check(isinstance(p['authorization_id'],str) and p['authorization_id'].isalnum() and len(p['authorization_id'])>=8,'Authorization ID')
 check(p['kind'] in ('production','resource_qualification') and p['kind']==kind,'Qualification-to-production escalation');check(type(p['stage']) is int and p['stage']==stage and 1<=stage<=5,'Unauthorized stage escalation')
 allowed=p['approved_ids'];check(isinstance(allowed,list) and allowed and len(set(allowed))==len(allowed),'Duplicate/empty approved IDs');check(set(allowed).issubset(selection.jobs) and not set(allowed)&selection.original_ids,'Unauthorized/original12 ID')
 check(isinstance(ids,list) and ids and len(set(ids))==len(ids) and set(ids).issubset(allowed),'Requested IDs outside approval')
 check(all(selection.jobs[pid]['resource']['stage']==stage for pid in allowed),'Stage/ID mismatch')
 output=Path(p['output_root']);check(output.is_absolute() and str(output.resolve())==p['output_root']==str(Path(output_root).resolve()) and output.is_relative_to(ROOT/'m3_5b2/executions'),'Incorrect output root')
 check(p['production_sources']==selection.production_sources and p['validation_sources']==selection.validation_sources and p['implementation_sources']==selection.implementation_sources(),'Incorrect/missing source fingerprints')
 check(p['environment']==(current_environment if current_environment is not None else environment()),'Invalid environment identity');check(p['limits']==LIMITS and all(type(p['limits'][k]) is type(v) for k,v in LIMITS.items()),'Unauthorized resource/timeout increase/change');check(p['stop_rules']==STOP and all(type(p['stop_rules'][k]) is type(v) for k,v in STOP.items()),'Stop/quarantine rules')
 check(type(p['disk_requirement_bytes']) is int and p['disk_requirement_bytes']>=minimum_disk_bytes(selection,allowed),'Insufficient/unbound disk reserve')
 w=p['maximum_work'];check(isinstance(w,dict) and set(w)==WORK and all(type(v) is int and v>=(0 if k in ('max_FEM_calls','max_OC_calls') else 1) for k,v in w.items()),'Maximum work bounds');check(w['max_ids']==len(allowed) and w['max_states_per_id']<=501 and w['max_updates_per_id']<=500 and w['max_worker_jobs']<=6*len(allowed),'Maximum work scope');check(w['max_FEM_calls']<=4000*len(allowed) and w['max_OC_calls']<=2000*len(allowed) and w['max_total_worker_seconds']<=900*w['max_worker_jobs'],'Unbounded authorization work')
 a=p['approval'];check(isinstance(a,dict) and set(a)=={'identity','approved_at','expires_at','decision_reference'} and all(isinstance(a[k],str) and a[k] for k in a),'Approval metadata');now=verification_time if verification_time is not None else datetime.now(timezone.utc);check(timestamp(a['approved_at'])<=now<timestamp(a['expires_at']),'Approval date/expiry')
 e=p['experiment'];check(isinstance(e,dict) and set(e)=={'actions','max_new_updates','release_allowed','original_spec_unchanged'} and set(e['actions']).issubset(ACTIONS) and e['actions'] and e['original_spec_unchanged'] is True,'Experiment actions')
 check(e['max_new_updates'] is None or type(e['max_new_updates']) is int and 0<e['max_new_updates']<=500,'Diagnostic update bound');check(type(e['release_allowed']) is bool,'Release flag')
 if e['max_new_updates'] is None:check(w['max_states_per_id']==501 and w['max_updates_per_id']==500,'Original-cap work budget required')
 else:check(e['max_new_updates']<=w['max_updates_per_id'] and e['max_new_updates']+1<=w['max_states_per_id'],'Diagnostic work budget')
 if kind=='resource_qualification':check(e['release_allowed'] is False and 'adopt' not in e['actions'],'Qualification cannot publish production')
 else:check(e['max_new_updates'] is None and e['release_allowed'] is True,'Production cannot use shortened diagnostic stopping')
 check(isinstance(p['resume_bindings'],dict) and set(p['resume_bindings']).issubset(allowed),'Resume scope')
 for pid,binding in p['resume_bindings'].items():
  check(set(binding)=={'purpose','journal_sha256','prior_authorization_sha256','work_ledger_sha256','restart_control'} and binding['purpose'] in ('resume','adopt'),'Resume/adoption binding schema')
  if binding['purpose']=='resume':check('resume' in e['actions'] and isinstance(binding['restart_control'],dict),'Resume requires independently validated control provenance')
  else:check(kind=='production' and 'adopt' in e['actions'],'Qualification cannot adopt production')
 check(isinstance(p['qualification_evidence'],dict),'Qualification evidence')
 if kind=='production':
  for pid in allowed:
   if selection.jobs[pid]['plan_index'] in BLOCKERS:
    q=p['qualification_evidence'].get(pid);check(isinstance(q,dict) and set(q)=={'path','sha256'} and digest(Path(q['path']))==q['sha256'],'Missing case-specific qualification')
    q=read(q['path']);check(q['problem_id']==pid and q['spec_sha256']==selection.jobs[pid]['spec_sha256'] and q['passed'] is True and q['limits']==LIMITS and q['completed_original_trajectory'] is True,'Unqualified case cannot escalate')
    if selection.jobs[pid]['plan_index'] in (145,151):check(q['full_cap_history_residency_qualified'] is True and q['states']==501 and q['completed_updates']==500,'193-state96 does not qualify full-cap residency')
 return p

def review_book(book,payload_sha,payload):
 check(isinstance(book,dict) and set(book)=={'schema_version','issuance_enabled','approved_authorizations'} and book['schema_version']=='m3.5b2.approval_book.v1','Approval book schema')
 check(book['issuance_enabled'] is True,'No authorization has been issued')
 found=[r for r in book['approved_authorizations'] if r.get('sidecar_sha256')==payload_sha];check(len(found)==1,'Sidecar syntax is not approval; digest not deliberately issued')
 r=found[0];check(set(r)=={'sidecar_sha256','authorization_id','approval_identity','decision_reference','approved_at'} and r['authorization_id']==payload['authorization_id'] and r['approval_identity']==payload['approval']['identity'] and r['decision_reference']==payload['approval']['decision_reference'] and r['approved_at']==payload['approval']['approved_at'],'Approval receipt metadata binding')
 return r

def verify_execution_threads():
 check({k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}==dict(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='16',OMP_NUM_THREADS='16'),'Controlled execution thread environment must preserve OPENBLAS1/MKL16/OMP16')

def issued_authorization(path,selection,*,kind,stage,ids,output_root):
 check(path is not None,'Missing authorization');path=Path(path);check(path.is_file(),'Missing authorization file')
 payload=read(path);validate_candidate(payload,selection,kind=kind,stage=stage,ids=ids,output_root=output_root)
 # No caller-supplied trust store, test permit, environment toggle or bypass.
 book=Path(__file__).with_name('approved_receipts.json');check(digest(book)==APPROVAL_BOOK_SHA,'Reviewed approval-book hash changed')
 check(AUTHORITY_BOOK.is_file(),'No authorization has been issued through the protected approval workflow')
 review_book(read(AUTHORITY_BOOK),digest(path),payload);verify_execution_threads();selection.verify_sources();return payload

def historical_authorization(path,selection,*,kind,stage,ids,output_root,executed_at):
 """Read-only provenance check at recorded execution time, NEVER a run permit."""
 path=Path(path);p=read(path);validate_candidate(p,selection,kind=kind,stage=stage,ids=ids,output_root=output_root,current_environment=p['environment'],verification_time=timestamp(executed_at))
 template=Path(__file__).with_name('approved_receipts.json');check(digest(template)==APPROVAL_BOOK_SHA and AUTHORITY_BOOK.is_file(),'Missing protected historical issuance record')
 review_book(read(AUTHORITY_BOOK),digest(path),p);selection.verify_sources();return p
