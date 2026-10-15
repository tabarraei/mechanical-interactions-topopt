import copy,json,os
from pathlib import Path
import numpy as np
import pytest
from m3_4br2.test_audit import fixture
from m3_4br2.audit import historical
from .audit import audit_transition,change_audit
CASES=['identical','perturb','near_zero','move','threshold_safe','changed_max','tied_max','redistribution','wrong_action','previous_index','next_index','reversed','missing_state','missing_operand','nonfinite','shape','wrong_scalar','primary_fail_equal_change','threshold_cross','mechanical_fail','missing_incoming','wrong_historical']
OUTCOMES=[]
def data(case):
 s,r,target=fixture(value=.3);s['action'][:]=.32;r=copy.deepcopy(s);threshold=.001
 if case=='near_zero':s['action'][:]=s['x'];r=copy.deepcopy(s);r['action'][0]=np.nextafter(.3,1)
 if case=='move':s['action'][:]=.4;r=copy.deepcopy(s)
 if case in ('threshold_safe','threshold_cross'):s['action'][:]=.301;r=copy.deepcopy(s);threshold=float(np.max(abs(s['action']-s['x'])))
 if case=='threshold_safe':threshold+=1e-6
 if case in ('perturb','threshold_safe','threshold_cross'):r['action'][0]+=2e-13 if case!='threshold_cross' else -2e-13
 if case=='threshold_cross':r['action'][:]-=2e-13
 if case=='changed_max':s['action'][0]+=1e-13;r=copy.deepcopy(s);r['action'][1]+=2e-13
 if case=='tied_max':r['action'][0]+=2e-13
 if case=='redistribution':r['action'][0]+=.0001;r['action'][1]-=.0001
 if case=='wrong_action':r['action'][0]+=.01
 if case=='primary_fail_equal_change':r['x']+=.001;r['action']+=.001
 for side in (s,r):
  side['diagnostic']['volume_excess']=float(side['action'].mean()-target)
  side['diagnostic']['historical_sum_excess']=float(sum(side['action'])-len(side['action'])*target)
 sc=float(np.max(abs(s['action']-s['x'])));rc=float(np.max(abs(r['action']-r['x'])))
 e=dict(transition=2,indices=[2,3],saved_change=sc,outgoing_change=sc,threshold=threshold)
 p={'mechanics':dict(passed=True),'OC':historical(r['action'],s['action'],1e-8),'incoming':historical(rc,sc,1e-12)}
 if case=='previous_index':e['indices']=[1,3]
 if case=='next_index':e['indices']=[2,4]
 if case=='reversed':e['indices']=[3,2]
 if case=='missing_state':e['indices']=[2]
 if case=='missing_operand':r['action']=None
 if case=='nonfinite':r['action'][0]=np.nan
 if case=='shape':r['action']=r['action'].reshape(8,8)
 if case=='wrong_scalar':e['saved_change']+=1e-13
 if case=='mechanical_fail':p['mechanics']['passed']=False
 if case=='missing_incoming':del p['incoming']
 if case=='wrong_historical':p['incoming']['passed']=False
 return s,r,target,e,p
@pytest.mark.parametrize('case',CASES)
def test_cases(case):
 s,r,t,e,p=data(case);expected=case in CASES[:7]
 result=audit_transition(s,r,t,p,change_evidence=e)
 OUTCOMES.append(dict(case=case,expected=expected,actual=result['passed'],reason=result['reason'],change=result['change_audit']))
 assert result['passed']==expected,result
 if case in ('perturb','near_zero','changed_max','tied_max'):assert not result['change_audit']['original']['passed']
 if case=='threshold_cross':assert result['reason']=='Stopping classification differs'
def test_missing_evidence():
 s,r,t,e,p=data('identical');assert not audit_transition(s,r,t,p)['passed']
def test_retained_transition41():
 from m3_4br2.memory import load_trajectory
 from m3_4br2.preflight import OLD,FAILED
 tr=load_trajectory(OLD/'native/samples'/FAILED)
 folder=Path('m3_4br2/generations/20261014T124054Z/independent_replay_final')
 with np.load(folder.parent/'transition41_reconstructed_from_retained_operands.npz') as z:
  s={k:z['saved_'+k].copy() for k in ('x','action')};r={k:z['replay_'+k].copy() for k in ('x','action')}
 e=dict(transition=41,indices=[41,42],saved_change=float(tr.arrays['previous_change'][42]),outgoing_change=float(tr.arrays['update_change'][41]),threshold=tr.problem.tolerance)
 result=change_audit(s,r,e);assert result['passed'];assert not result['original']['passed'];assert result['saved_max_indices']==[1233]
 if os.environ.get('R3_RESULTS'):(Path(os.environ['R3_RESULTS'])/'transition41_decomposition.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()))
def teardown_module():
 if os.environ.get('R3_RESULTS'):(Path(os.environ['R3_RESULTS'])/'adversarial_outcomes.json').write_text(json.dumps(OUTCOMES,indent=2,default=lambda x:x.item()))

def test_actual_api_operand_capture():
 from .operands import compliance_with_displacement
 class Dummy:
  def __init__(self):self.calls=0
  def solve(self,rho):self.calls+=1;return np.asarray(rho)+self.calls*1e-13
  def compliance(self,rho):
   u=self.solve(rho);return float(u.sum()),-u*u
 d=Dummy();before=d.solve
 (c,g),u=compliance_with_displacement(d,np.ones(3));assert d.calls==1;assert 'solve' not in d.__dict__;assert d.solve==before
 np.testing.assert_array_equal(g,-u*u);assert c==u.sum()

def test_recorder_restores_on_failure():
 from .operands import compliance_with_displacement
 class Dummy:
  def solve(self,*args):raise RuntimeError('solve failed')
  def compliance(self,*args):return self.solve()
 d=Dummy()
 with pytest.raises(RuntimeError):compliance_with_displacement(d)
 assert 'solve' not in d.__dict__

def test_cached_original_evidence():
 from .audit import cached_primary,POLICY
 original={'incoming':{'passed':False}};adapted={'incoming':{'passed':True,'policy':POLICY}}
 assert cached_primary(dict(policy=POLICY,primary=adapted,original_primary=original)) is original
 assert cached_primary(dict(primary=original)) is original

def test_missing_original_cached_evidence_rejects():
 from .audit import cached_primary,POLICY
 with pytest.raises(ValueError):cached_primary(dict(policy=POLICY,primary={'incoming':{'passed':True}}))
