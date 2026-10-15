"""Synthetic negative checks only; no protected issuance or numerical dispatch."""
import copy,sys,json,os
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
from m3_5b2d.common import *
from m3_5b2d.authorization import validate_candidate,review_book
from m3_5b2d.certificate import template,validate
from m3_5b2d.integration import alias_template,validate_alias,HistoricalReleaseAdapter
from m3_5b2d.accounting import Budget

@pytest.fixture(scope='module')
def b():return target_binding()
@pytest.fixture
def p(b):
 now=datetime.now(timezone.utc)
 return dict(schema_version='m3.5b2d.protocolA.authorization.v1',authorization_id='SYNTHETICNOTISSUED',kind='historical_saved_validation_only',binding=copy.deepcopy(b),output_root=str(ROOT/'m3_5b2d/validation_runs/NOT_AUTHORIZED_FIXTURE'),environment=environment(),limits=copy.deepcopy(LIMITS),work=copy.deepcopy(WORK),actions=['replay','supplemental','stop_and_quarantine_no_retry_no_release'],disk_bytes=4*1024**3,approval=dict(identity='SYNTHETIC_NOT_ISSUER',approved_at=(now-timedelta(seconds=1)).isoformat(),expires_at=(now+timedelta(minutes=5)).isoformat(),decision_reference='SYNTHETIC_NOT_APPROVAL'))

def test_missing_execution_authorization(tmp_path):
 from m3_5b2d.runner import dispatch
 with pytest.raises(ValueError):dispatch(None,tmp_path/'not_created')
 assert not (tmp_path/'not_created').exists()

def test_policy_decision_does_not_issue_execution(p,b,tmp_path,monkeypatch):
 import m3_5b2d.authorization as a
 monkeypatch.setattr(a,'target_binding',lambda:b);monkeypatch.setattr(a,'environment',lambda:p['environment'])
 for k,v in dict(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='16',OMP_NUM_THREADS='16').items():monkeypatch.setenv(k,v)
 calls=[]
 def synthetic_approval(kind,*args):
  calls.append(kind)
  if kind=='policy':return {'fixture':'in_memory_policy_only_NOT_ISSUANCE'}
  raise ValueError('Synthetic missing execution authorization')
 monkeypatch.setattr(a,'approval_record',synthetic_approval)
 path=tmp_path/'SYNTHETIC_UNISSUED_AUTH.fixture.json';write(path,p)
 with pytest.raises(ValueError,match='missing execution'):a.live_guard(path,p['output_root'])
 assert calls==['policy','protocolA_execution']

@pytest.mark.parametrize('attack',['historical','selected','source','input','environment','output','expiry'])
def test_scope_rejection(p,b,attack):
 q=copy.deepcopy(p)
 if attack=='historical':q['binding']['historical_id']='wrong'
 elif attack=='selected':q['binding']['selected_id']='wrong'
 elif attack=='source':q['binding']['sources']['m3_4br3/replay.py']='0'*64
 elif attack=='input':q['binding']['input_hashes'][next(iter(q['binding']['input_hashes']))]='0'*64
 elif attack=='environment':q['environment']={}
 elif attack=='output':q['output_root']='/tmp/wrong-root'
 else:q['approval']['expires_at']='2000-01-01T00:00:00Z'
 with pytest.raises(ValueError):validate_candidate(q,b,p['output_root'],p['environment'],datetime.now(timezone.utc))

def test_invalid_empty_issuer_rejected():
 now=datetime.now(timezone.utc)
 r=dict(kind='policy',sha256='0'*64,binding_sha256='1'*64,decision='D2',identity='',approved_at=(now-timedelta(seconds=1)).isoformat(),expires_at=(now+timedelta(minutes=5)).isoformat(),reference='SYNTHETIC')
 with pytest.raises(ValueError):review_book(dict(schema_version='m3.5b2d.reviewed_approvals.v1',records=[r]),kind='policy',object_hash='0'*64,binding_hash='1'*64)

def test_execution_issuer_metadata_mismatch(p,b,tmp_path,monkeypatch):
 import m3_5b2d.authorization as a
 monkeypatch.setattr(a,'target_binding',lambda:b);monkeypatch.setattr(a,'environment',lambda:p['environment'])
 for k,v in dict(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='16',OMP_NUM_THREADS='16').items():monkeypatch.setenv(k,v)
 monkeypatch.setattr(a,'approval_record',lambda kind,*args:dict(identity='WRONG_SYNTHETIC_IDENTITY',reference=p['approval']['decision_reference'],approved_at=p['approval']['approved_at'],expires_at=p['approval']['expires_at']))
 path=tmp_path/'SYNTHETIC.fixture.json';write(path,p)
 with pytest.raises(ValueError,match='metadata'):a.live_guard(path,p['output_root'])

def test_reused_worker_ticket_is_rejected(p,b,tmp_path,monkeypatch):
 # Fake ticket/lease entirely in pytest temp directory. Never a valid live ticket.
 import m3_5b2d.runner as r
 monkeypatch.setattr(r,'live_guard',lambda *args:p);monkeypatch.setattr(r,'ROOT',tmp_path)
 root=tmp_path/'SYNTHETIC-output';auth=tmp_path/'SYNTHETIC.fixture.json';write(auth,{'fixture':True});ticket=root/'reservations/replay.json'
 write(ticket,dict(authorization_sha256=digest(auth),binding_sha256=canonical(b),stage='replay',root=str(root),parent_pid=os.getppid()))
 write(root/'authorization_binding.json',dict(authorization_sha256=digest(auth),binding_sha256=canonical(b)))
 write(tmp_path/'m3_5b2d/validation_attempts'/f'{HISTORICAL}.json',dict(output_root=str(root),authorization_sha256=digest(auth)))
 r.worker_guard(auth,root,'replay',ticket)
 with pytest.raises(FileExistsError):r.worker_guard(auth,root,'replay',ticket)

@pytest.mark.parametrize('kind',['FEM','OC','selected_trials','bisection_trials','workers'])
def test_persistent_budget_refuses_overflow(p,tmp_path,kind):
 budget=Budget(tmp_path,p);budget.charge(kind,amount=WORK[kind])
 with pytest.raises(ValueError):Budget(tmp_path,p).charge(kind)
 assert read(budget.path)['counts'][kind]==WORK[kind]

def test_incomplete_evidence(b):
 with pytest.raises(ValueError):validate(template(b),b,complete=True)

@pytest.mark.parametrize('missing',['alias','crosswalk'])
def test_independent_alias_and_crosswalk(b,missing):
 a=alias_template(b);a['alias_approved']=missing!='alias';a['family_crosswalk_approved']=missing!='crosswalk';a['approval_metadata']={'alias':{'fixture':True},'crosswalk':{'fixture':True}}
 with pytest.raises(ValueError):validate_alias(a,b,approved=True)

def test_no_publication_api_or_unapproved_staging(b,tmp_path,monkeypatch):
 import m3_5b2d.integration as i
 assert not hasattr(HistoricalReleaseAdapter,'publish')
 monkeypatch.setattr(i,'target_binding',lambda:b);a=tmp_path/'UNAPPROVED.fixture.json';write(a,alias_template(b))
 with pytest.raises(ValueError):HistoricalReleaseAdapter.stage_entry(a,tmp_path/'no-certificate.json',[])

def test_no_solver_modules_imported():
 assert not any(n.startswith(('topopt_utils','numpy','torch','pypardiso','m3_4br3.replay')) for n in sys.modules)
