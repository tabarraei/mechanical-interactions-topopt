"""No-solver authorization, registry and checkpoint adversarial tests."""
import copy,json,sys,importlib.abc
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
import pytest
from .common import *
from .authorization import validate_candidate,issued_authorization,review_book,SCHEMA,STOP,APPROVAL_BOOK_SHA
from .registry import Registry
from .checkpoint import verify_checkpoint
from .runner import dispatch,verify_owned_root
from .budget import Budget
@pytest.fixture(scope='session')
def selection():return Selection()
def pid_at(s,i):return next(pid for pid,j in s.jobs.items() if j['plan_index']==i)
def candidate(s,kind='resource_qualification',index=98):
 pid=pid_at(s,index);stage=s.jobs[pid]['resource']['stage'];now=datetime.now(timezone.utc)
 return dict(schema_version=SCHEMA,manifest_sha256=PLAN_SHA,authorization_id='SYNTHETICONLY',kind=kind,stage=stage,approved_ids=[pid],output_root=str(ROOT/'m3_5b2/executions/NEVER_CREATED'),production_sources=copy.deepcopy(s.production_sources),validation_sources=copy.deepcopy(s.validation_sources),implementation_sources=s.implementation_sources(),environment=environment(),limits=LIMITS.copy(),stop_rules=STOP.copy(),maximum_work=dict(max_ids=1,max_worker_jobs=4,max_FEM_calls=1600,max_OC_calls=1000,max_states_per_id=501,max_updates_per_id=500,max_total_worker_seconds=3600),approval=dict(identity='synthetic-test-NOT-approval',approved_at=(now-timedelta(hours=1)).isoformat(),expires_at=(now+timedelta(days=1)).isoformat(),decision_reference='unit-test-only-no-issued-authority'),experiment=dict(actions=['generate','replay','supplemental']+(['adopt'] if kind=='production' else []),max_new_updates=None,release_allowed=kind=='production',original_spec_unchanged=True),resume_bindings={},qualification_evidence={},disk_requirement_bytes=minimum_disk_bytes(s,[pid]))
def validate(s,p,**kw):return validate_candidate(p,s,kind=kw.get('kind',p['kind']),stage=kw.get('stage',p['stage']),ids=kw.get('ids',p['approved_ids']),output_root=kw.get('output_root',p['output_root']))
@pytest.mark.parametrize('attack',['missing','extra','manifest','source','implementation','ID','original12','duplicate','stage','root','limits','work','environment','stop','shorten_production','qualification_to_production','qual_release','fake_receipt_metadata','disk'])
def test_adversarial_authorization(selection,attack):
 p=candidate(selection)
 if attack=='disk':p['disk_requirement_bytes']=0
 elif attack=='missing':p.pop('environment')
 elif attack=='extra':p['bypass']=True
 elif attack=='manifest':p['manifest_sha256']='bad'
 elif attack=='source':p['production_sources'][next(iter(p['production_sources']))]='bad'
 elif attack=='implementation':p['implementation_sources']={}
 elif attack=='ID':p['approved_ids']=['unknown']
 elif attack=='original12':p['approved_ids']=[next(iter(selection.original_ids))]
 elif attack=='duplicate':p['approved_ids']*=2
 elif attack=='stage':p['stage']=5
 elif attack=='root':p['output_root']='/tmp/unauthorized'
 elif attack=='limits':p['limits']['worker_rss_mib']=1025
 elif attack=='work':p['maximum_work']['max_FEM_calls']=100000000
 elif attack=='environment':p['environment']['python']='fake interpreter'
 elif attack=='stop':p['stop_rules']['overwrite']=True
 elif attack=='shorten_production':p=candidate(selection,'production',0);p['experiment']['max_new_updates']=2
 elif attack=='qualification_to_production':
  with pytest.raises(ValueError,match='escalation'):validate(selection,p,kind='production')
  return
 elif attack=='qual_release':p['experiment']['release_allowed']=True
 elif attack=='fake_receipt_metadata':p['approval']['identity']=''
 with pytest.raises((ValueError,KeyError,TypeError)):validate(selection,p)
def test_valid_candidate_is_inert_not_issued(selection,tmp_path):
 p=candidate(selection);assert validate(selection,p)==p
 path=tmp_path/'UNISSUED_SYNTHETIC_CANDIDATE.json';write(path,p)
 with pytest.raises(ValueError,match='issued'):issued_authorization(path,selection,kind=p['kind'],stage=p['stage'],ids=p['approved_ids'],output_root=p['output_root'])
 assert read(Path(__file__).with_name('approved_receipts.json'))['approved_authorizations']==[]
def test_receipt_metadata_binds_exact_digest_only_in_memory(selection):
 p=candidate(selection);a=p['approval'];record=dict(sidecar_sha256='synthetic-memory-digest',authorization_id=p['authorization_id'],approval_identity=a['identity'],decision_reference=a['decision_reference'],approved_at=a['approved_at']);book=dict(schema_version='m3.5b2.approval_book.v1',issuance_enabled=True,approved_authorizations=[record]);assert review_book(book,'synthetic-memory-digest',p)==record
 with pytest.raises(ValueError):review_book(book,'different-digest',p)
def test_missing_authorization_before_production_import(selection,tmp_path):
 class Trap(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0] in {'topopt_utils','pypardiso','m3_4b','m3_4br3','m3_4br4','torch'}:raise AssertionError('Production import before authorization')
 trap=Trap();sys.meta_path.insert(0,trap)
 try:
  with pytest.raises(ValueError,match='Missing authorization'):dispatch(None,kind='resource_qualification',stage=3,ids=[pid_at(selection,98)],output_root=str(tmp_path/'never'))
  with pytest.raises(ValueError):dispatch(PLAN,kind='resource_qualification',stage=3,ids=[pid_at(selection,98)],output_root=str(tmp_path/'never'))
 finally:sys.meta_path.remove(trap)
 assert not (tmp_path/'never').exists()
@pytest.mark.parametrize('i',[98,130,138,145,151])
def test_five_qualification_blockers_cannot_be_promoted(selection,i):
 p=candidate(selection,'production',i)
 with pytest.raises(ValueError,match='qualification'):validate(selection,p)
def test_193_state_claim_cannot_qualify_96(selection,tmp_path):
 p=candidate(selection,'production',145);pid=p['approved_ids'][0];q=tmp_path/'SYNTHETIC_qualification.json';write(q,dict(problem_id=pid,spec_sha256=selection.jobs[pid]['spec_sha256'],passed=True,limits=LIMITS,completed_original_trajectory=True,full_cap_history_residency_qualified=True,states=193,completed_updates=192));p['qualification_evidence'][pid]=dict(path=str(q),sha256=digest(q))
 with pytest.raises(ValueError,match='193-state'):validate(selection,p)
@pytest.fixture(scope='session')
def registry(selection):return Registry(ROOT,selection)
def test_real_global_inventory_and_new_equivalence_conflict(selection,registry):
 r=registry.report();assert r['coverage_complete'];assert all(p in registry.records for p in selection.original_ids);assert len(registry.records)>=34
 assert sum(x['selected'] and x['identity_eligible_for_future_authorization'] for x in r['rows'])==29
 with pytest.raises(ValueError,match='equivalent'):registry.eligible(pid_at(selection,10))
 for pid in selection.original_ids:
  with pytest.raises(ValueError):registry.eligible(pid)
def test_configuration_equivalence_detects_seed_only_alias(selection,tmp_path):
 pid=pid_at(selection,0);spec=copy.deepcopy(selection.specs[pid]);alias='m3_3b_SYNTHETIC_ALIAS';spec['problem_id']=alias;spec['mechanical']['seed']=999;folder=tmp_path/'samples'/alias;write(folder/'problem.json',spec)
 s=SimpleNamespace(original=[],original_ids=set(),jobs={pid:selection.jobs[pid]},specs={pid:selection.specs[pid]});r=Registry(tmp_path,s)
 with pytest.raises(ValueError,match='equivalent'):r.eligible(pid)
def make_checkpoint(tmp_path,spec):
 p=tmp_path/'incomplete'/spec['problem_id'];write(p/'problem.json',spec);(p/'frames').mkdir();(p/'updates').mkdir();(p/'frames/00000.npz').write_bytes(b'synthetic-byte-integrity-fixture-NOT-FEM')
 j=dict(schema_version='m3.3b.checkpoint.v1',binding=dict(problem_id=spec['problem_id'],spec_sha=canonical(spec),source_id='synthetic-source'),states=[dict(file='frames/00000.npz',sha256=digest(p/'frames/00000.npz'))],updates=[]);write(p/'journal.json',j);return p
def test_invalid_checkpoint_partial_recovery_and_orphan_handling(selection,tmp_path):
 spec=selection.specs[pid_at(selection,98)];p=make_checkpoint(tmp_path,spec);proof=verify_checkpoint(p,spec,'synthetic-source');assert proof['committed_states']==1
 (p/'frames/orphan.tmp').write_bytes(b'uncommitted');assert verify_checkpoint(p,spec,'synthetic-source')['orphans']==['frames/orphan.tmp']
 (p/'frames/00000.npz').write_bytes(b'corrupt')
 with pytest.raises(ValueError,match='byte integrity'):verify_checkpoint(p,spec,'synthetic-source')
def test_owned_root_overwrite_and_explicit_recovery_binding(selection,tmp_path):
 root=tmp_path/'owned';root.mkdir()
 with pytest.raises(ValueError,match='overwrite'):verify_owned_root(root,{},'new',[])
 write(root/'authorization_binding.json',dict(authorization_sha256='prior'));write(root/'work_ledger.json',dict(lifetime='retained'))
 pid=pid_at(selection,98);spec=selection.specs[pid];p=make_checkpoint(root/'native',spec)
 binding=dict(purpose='resume',prior_authorization_sha256='prior',work_ledger_sha256=digest(root/'work_ledger.json'),journal_sha256=digest(p/'journal.json'),restart_control={})
 assert verify_owned_root(root,dict(resume_bindings={pid:binding}),'new',[pid])=='prior'
 binding['journal_sha256']='wrong'
 with pytest.raises(ValueError):verify_owned_root(root,dict(resume_bindings={pid:binding}),'new',[pid])
def test_budget_refuses_before_work_and_preserves_lifetime(tmp_path):
 a=dict(maximum_work=dict(max_FEM_calls=1,max_OC_calls=1,max_worker_jobs=1,max_total_worker_seconds=1));b=Budget(tmp_path,a,'synthetic');b.charge('FEM')
 with pytest.raises(ValueError):b.charge('FEM')
 assert read(tmp_path/'work_ledger.json')['lifetime']['FEM']==1
 with pytest.raises(ValueError):b.charge('worker_seconds',2)
 assert read(tmp_path/'work_ledger.json')['lifetime']['worker_seconds']==2

def test_valid_unissued_dispatch_cannot_import_or_start_work(selection,tmp_path):
 p=candidate(selection);file=tmp_path/'INERT_UNISSUED_QUALIFICATION_CANDIDATE.json';write(file,p)
 class Trap(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0] in {'topopt_utils','pypardiso','m3_4b','m3_4br3','m3_4br4','torch'}:raise AssertionError('Unauthorized production import')
 trap=Trap();sys.meta_path.insert(0,trap)
 try:
  with pytest.raises(ValueError,match='issued'):dispatch(file,kind=p['kind'],stage=p['stage'],ids=p['approved_ids'],output_root=p['output_root'])
 finally:sys.meta_path.remove(trap)
 assert not Path(p['output_root']).exists()
def test_coverage_failure_is_not_silently_ignored(selection,monkeypatch):
 import m3_5b2.registry as module
 monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=2,stderr='permission denied',stdout=''))
 with pytest.raises(ValueError,match='coverage'):Registry(ROOT,selection)

def test_released_identity_does_not_erase_earlier_inventory_roles(selection,registry):
 for e in selection.original:
  records=registry.records[e['problem_id']];assert any(r.get('is_released_location') and r['status']=='already_released' for r in records)
  assert all('historical_inventory_status' in r for r in records)

@pytest.mark.parametrize('mkl',['16','8',None])
def test_execution_thread_contract_is_preserved(monkeypatch,mkl):
 from .authorization import verify_execution_threads
 monkeypatch.setenv('OPENBLAS_NUM_THREADS','1');monkeypatch.setenv('OMP_NUM_THREADS','16')
 if mkl is None:monkeypatch.delenv('MKL_NUM_THREADS',raising=False)
 else:monkeypatch.setenv('MKL_NUM_THREADS',mkl)
 if mkl=='16':verify_execution_threads()
 else:
  with pytest.raises(ValueError,match='thread environment'):verify_execution_threads()

def test_expired_execution_permission_does_not_erase_historical_parser_provenance(selection):
 from .authorization import timestamp
 p=candidate(selection);then=datetime.now(timezone.utc)-timedelta(days=5);p['approval']['approved_at']=then.isoformat();p['approval']['expires_at']=(then+timedelta(days=1)).isoformat()
 with pytest.raises(ValueError,match='expiry'):validate(selection,p)
 assert validate_candidate(p,selection,kind=p['kind'],stage=p['stage'],ids=p['approved_ids'],output_root=p['output_root'],current_environment=p['environment'],verification_time=then+timedelta(hours=1))==p
 with pytest.raises(TypeError):issued_authorization(None,selection,kind=p['kind'],stage=p['stage'],ids=p['approved_ids'],output_root=p['output_root'],verification_time=then)

def test_expiry_deadline_stops_before_numerical_counter_entry(tmp_path):
 a=dict(maximum_work=dict(max_FEM_calls=5,max_OC_calls=5,max_worker_jobs=1,max_total_worker_seconds=900),approval=dict(expires_at=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()))
 b=Budget(tmp_path,a,'synthetic-expired');assert b.remaining_seconds()<0
 with pytest.raises(ValueError,match='time window'):b.charge('FEM')
 assert read(tmp_path/'work_ledger.json')['lifetime']['FEM']==0
