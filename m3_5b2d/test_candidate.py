"""Only synthetic controls, immutable metadata/hashes, no solver invocation."""
import copy,json,sys,subprocess
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
from m3_5b2d.common import *
from m3_5b2d.policy import decision_state,DEFAULT
from m3_5b2d.authorization import validate_candidate,live_guard,review_book
from m3_5b2d.certificate import template,seal,validate,UNCERTIFIED
from m3_5b2d.integration import alias_template,validate_alias,ensure_single_history,HistoricalReleaseAdapter
from m3_5b2d.accounting import Budget,verify_resources

@pytest.fixture(scope='module')
def binding():return target_binding()
@pytest.fixture
def candidate(binding,tmp_path):
 now=datetime.now(timezone.utc)
 return dict(schema_version='m3.5b2d.protocolA.authorization.v1',authorization_id='SyntheticOnly123',kind='historical_saved_validation_only',binding=copy.deepcopy(binding),output_root=str(ROOT/'m3_5b2d/validation_runs/NOT_AUTHORIZED_TEST'),environment=environment(),limits=copy.deepcopy(LIMITS),work=copy.deepcopy(WORK),actions=['replay','supplemental','stop_and_quarantine_no_retry_no_release'],disk_bytes=4*1024**3,approval=dict(identity='SYNTHETIC_FIXTURE',approved_at=(now-timedelta(seconds=2)).isoformat(),expires_at=(now+timedelta(hours=1)).isoformat(),decision_reference='SYNTHETIC_NOT_APPROVAL'))

def complete_fixture(b):
 c=template(b);c.update(states=501,transitions=500,gates={k:True for k in GATES},v2=True,v3=True,supplemental=True,raw_validation_complete=True,counts={**WORK,'bisection_trials':24500,'worker_seconds':20},raw_evidence=dict(root='SYNTHETIC',replay_summary_sha256='0'*64,supplemental_sha256='0'*64,ledger_sha256='0'*64,parent_memory_sha256='0'*64,checkpoint={'fixture':True}),resources=[dict(exit_code=0,resource_event=None,worker_exited=True,wall_seconds=10,worker_peak_rss_mib=700,worker_sampled_hwm_mib=700,parent_peak_mib=100,combined_peak_mib=800)]*2)
 return seal(c)

def test_defaults_and_decisions():
 assert not any(DEFAULT.values())
 assert not decision_state('D1',policy_approved=True,numerical=True,provenance=True,alias=True,crosswalk=True)['release_eligible']
 r=decision_state('D2',policy_approved=True,numerical=True,provenance=True,alias=True,crosswalk=True);assert r['release_eligible'] and r['historical_interruption_status'].startswith('uncertified') and not r['production_restart_policy_changed']
 for missing in ('policy_approved','numerical','provenance','alias','crosswalk'):
  kw=dict(policy_approved=True,numerical=True,provenance=True,alias=True,crosswalk=True);kw[missing]=False;assert not decision_state('D2',**kw)['release_eligible']

def test_missing_authorization_before_imports(tmp_path):
 from m3_5b2d.runner import dispatch
 with pytest.raises(ValueError,match='Missing authorization'):dispatch(None,tmp_path/'unused')
 assert not (tmp_path/'unused').exists()

@pytest.mark.parametrize('attack',['historical','selected','equivalence','manifest','source','input','output','resource','work','environment','production','actions','expiry','unknown'])
def test_authorization_adversaries(candidate,binding,attack):
 p=copy.deepcopy(candidate)
 if attack=='historical':p['binding']['historical_id']='wrong'
 elif attack=='selected':p['binding']['selected_id']='wrong'
 elif attack=='equivalence':p['binding']['equivalence_sha256']='0'*64
 elif attack=='manifest':p['binding']['manifest_sha256']='0'*64
 elif attack=='source':p['binding']['sources']['m3_4br3/replay.py']='0'*64
 elif attack=='input':p['binding']['input_hashes'][next(iter(p['binding']['input_hashes']))]='0'*64
 elif attack=='output':p['output_root']=str(ROOT/'wrong')
 elif attack=='resource':p['limits']['worker_rss_mib']=2048
 elif attack=='work':p['work']['FEM']=1006
 elif attack=='environment':p['environment']={}
 elif attack=='production':p['kind']='new_production'
 elif attack=='actions':p['actions'].append('generate')
 elif attack=='expiry':p['approval']['expires_at']='2000-01-01T00:00:00+00:00'
 else:p['permit']=True
 with pytest.raises(ValueError):validate_candidate(p,binding,candidate['output_root'],candidate['environment'],datetime.now(timezone.utc))

def test_syntax_not_approval(candidate,binding,tmp_path,monkeypatch):
 validate_candidate(candidate,binding,candidate['output_root'],candidate['environment'],datetime.now(timezone.utc))
 auth=tmp_path/'unissued.json';write(auth,candidate)
 import m3_5b2d.authorization as a
 monkeypatch.setattr(a,'target_binding',lambda:binding);monkeypatch.setattr(a,'environment',lambda:candidate['environment'])
 for k,v in dict(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='16',OMP_NUM_THREADS='16').items():monkeypatch.setenv(k,v)
 with pytest.raises(ValueError,match='No protected human policy approval'):a.live_guard(auth,candidate['output_root'])
 assert not Path(candidate['output_root']).exists()

@pytest.mark.parametrize('kind',['policy','protocolA_execution','alias','family_crosswalk','numerical_acceptance'])
def test_empty_issuance_never_approves(kind):
 with pytest.raises(ValueError):review_book(dict(schema_version='m3.5b2d.reviewed_approvals.v1',records=[]),kind=kind,object_hash='0'*64,binding_hash='1'*64)

@pytest.mark.parametrize('attack',['missing','incomplete','other_target','fixture','event','reference','v2','v3','supplemental','FEM','OC','trials','hash','timeout','partial','source'])
def test_certificate_adversaries(binding,attack):
 c=complete_fixture(binding)
 if attack=='missing':c.pop('gates')
 elif attack=='incomplete':c['gates']['primary_mechanics']=False
 elif attack=='other_target':c['binding']=dict(c['binding'],historical_id='other')
 elif attack=='fixture':c['fixture']=True
 elif attack=='event':c['historical_interruption_status']='certified'
 elif attack=='reference':c['reference_optimum']=True
 elif attack in ('v2','v3','supplemental'):c[attack]=False
 elif attack=='FEM':c['counts']['FEM']=1006
 elif attack=='OC':c['counts']['OC']=501
 elif attack=='trials':c['counts']['selected_trials']=501
 elif attack=='hash':c['certificate_hash']='0'*64
 elif attack=='timeout':c['resources']=[dict(c['resources'][0],wall_seconds=901)]*2
 elif attack=='partial':c['transitions']=499
 elif attack=='source':c['binding']=dict(c['binding'],historical_source_id='0'*64)
 if attack!='hash':c=seal(c)
 with pytest.raises(ValueError):validate(c,binding,complete=True)

def test_certificate_default_incomplete_and_fixture_reject(binding):
 validate(template(binding),binding)
 with pytest.raises(ValueError):validate(template(binding),binding,complete=True)
 c=complete_fixture(binding);validate(c,binding,complete=True);assert not c['numerical_validation_accepted'] and not c['release_eligible']

@pytest.mark.parametrize('attack',['alias','crosswalk','partition','global_family','original_seed','event'])
def test_alias_rejects(binding,attack):
 a=alias_template(binding)
 if attack=='alias':a['family_crosswalk_approved']=True
 elif attack=='crosswalk':a['alias_approved']=True
 elif attack=='partition':a['partitions']['historical']='test_id'
 elif attack=='global_family':a['families']['scope']='global_union'
 elif attack=='original_seed':a['seeds']['historical']=0
 else:a['historical_interruption_status']='certified'
 with pytest.raises(ValueError):validate_alias(a,binding,approved=True)

@pytest.mark.parametrize('field,value',[('problem_id',HISTORICAL),('selected_id',SELECTED),('equivalence_sha256',EQUIVALENCE)])
def test_double_counting(binding,field,value):
 with pytest.raises(ValueError):ensure_single_history([{field:value}],binding)
 assert ensure_single_history([],binding)['transitions_added']==500

def test_staging_rejects_unapproved_before_raw_evidence(binding,tmp_path,monkeypatch):
 import m3_5b2d.integration as i
 monkeypatch.setattr(i,'target_binding',lambda:binding);a=tmp_path/'alias.json';write(a,alias_template(binding))
 with pytest.raises(ValueError,match='Unapproved'):HistoricalReleaseAdapter.stage_entry(a,tmp_path/'missing_certificate.json',[])

@pytest.mark.parametrize('kind',['FEM','OC','selected_trials','bisection_trials','workers','worker_seconds'])
def test_work_charge_refuses_overrun(candidate,tmp_path,kind):
 b=Budget(tmp_path,candidate)
 if kind=='worker_seconds':
  with pytest.raises(ValueError):b.charge(kind,amount=WORK[kind]+1)
  assert read(b.path)['counts'][kind]==WORK[kind]+1
 else:
  b.charge(kind,amount=WORK[kind])
  with pytest.raises(ValueError):b.charge(kind)
  assert read(b.path)['counts'][kind]==WORK[kind]

def test_stage_ceiling(candidate,tmp_path):
 b=Budget(tmp_path,candidate);b.charge('FEM','supplemental',3)
 with pytest.raises(ValueError):b.charge('FEM','supplemental')

def test_overwrite_and_disk_guard(candidate,tmp_path,monkeypatch):
 import m3_5b2d.runner as r
 root=tmp_path/'existing';root.mkdir();monkeypatch.setattr(r,'live_guard',lambda *a:candidate)
 with pytest.raises(ValueError,match='overwrite'):r.preflight('synthetic',root)

def test_no_solver_import_in_clean_process():
 code="import sys; from m3_5b2d.runner import dispatch;\ntry: dispatch(None,'/tmp/not-created-m3-5b2d')\nexcept ValueError: pass\nassert not any(x.startswith(('topopt_utils','torch','pypardiso','numpy')) for x in sys.modules)"
 r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True);assert r.returncode==0,r.stderr


def test_missing_numerical_certificate_rejects(binding,tmp_path,monkeypatch):
 import m3_5b2d.integration as i
 monkeypatch.setattr(i,'target_binding',lambda:binding)
 a=alias_template(binding);a.update(alias_approved=True,family_crosswalk_approved=True,approval_metadata={'alias':{'fixture':True},'crosswalk':{'fixture':True}})
 path=tmp_path/'hypothetical_alias.json';write(path,a)
 with pytest.raises(FileNotFoundError):HistoricalReleaseAdapter.stage_entry(path,tmp_path/'missing_certificate.json',[])

@pytest.mark.parametrize('key,value',[('worker_sampled_hwm_mib',1025),('parent_peak_mib',769),('combined_peak_mib',1793),('wall_seconds',901)])
def test_each_resource_limit(key,value):
 r=dict(exit_code=0,resource_event=None,worker_exited=True,wall_seconds=1,worker_peak_rss_mib=1,worker_sampled_hwm_mib=1,parent_peak_mib=1,combined_peak_mib=2);r[key]=value
 with pytest.raises(ValueError):verify_resources(r)

def test_invalid_authorization_json_before_imports(tmp_path):
 from m3_5b2d.runner import dispatch
 p=tmp_path/'invalid.json';p.write_text('{')
 with pytest.raises(ValueError):dispatch(p,tmp_path/'unused')
 assert not (tmp_path/'unused').exists()
