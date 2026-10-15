import copy,json,os
from pathlib import Path
import numpy as np
import pytest
from m3_4br2.audit import audit_transition,POLICY,bounds
from m3_4br2.memory import frame_allclose,digest
MATRIX=[]
def fixture(n=64,value=.3,target=.4):
 x=np.full(n,value,dtype=np.float64);g=np.full(n,-1.,dtype=np.float64)
 d=dict(multiplier_lower=1.-5e-9,multiplier_upper=1.,bracket_width=5e-9,upper_feasible=True,strict_feasible=True,lower_above_budget=True,root_status='active_root',volume_excess=float(x.mean()-target),historical_sum_excess=float(sum(x)-n*target),seconds=0.)
 s=dict(x=x.copy(),rho=x.copy(),g_rho=g.copy(),g_x=g.copy(),action=x.copy(),selected_multiplier=1.,diagnostic=d)
 return s,copy.deepcopy(s),target

CASES=['equal','redistribution','small_design','physical_gradient','design_gradient','action','multiplier','lower_endpoint','upper_endpoint','one_ulp_volume','box','move','nan','float32','missing','shape','ordering','mechanics','wrong_residual','coherent_shift','nonzero_residual']
@pytest.mark.parametrize('case',CASES)
def test_discrimination(case):
 s,r,target=fixture();primary={'mechanics':dict(passed=True)};expected=case in ('equal','nonzero_residual')
 if case=='redistribution':r['x'][0]+=1e-4;r['x'][1]-=1e-4
 elif case=='small_design':r['x'][0]+=1e-7
 elif case=='physical_gradient':r['g_rho'][0]+=.001
 elif case=='design_gradient':r['g_x'][0]+=.001
 elif case=='action':r['action'][0]+=.0001;r['action'][1]-=.0001
 elif case=='multiplier':r['selected_multiplier']=.9
 elif case=='lower_endpoint':r['diagnostic']['multiplier_lower']=.99
 elif case=='upper_endpoint':r['diagnostic']['multiplier_upper']=1.1
 elif case=='one_ulp_volume':
  s,r,target=fixture(n=1,value=np.nextafter(.4,np.inf))
 elif case=='box':r['action'][0]=.0009
 elif case=='move':
  s,r,target=fixture(value=.1);s['action'][:]=r['action'][:]=.3
 elif case=='nan':r['g_x'][0]=np.nan
 elif case=='float32':r['g_x']=r['g_x'].astype('float32')
 elif case=='missing':r['g_x']=None
 elif case=='shape':r['g_x']=r['g_x'].reshape(8,8)
 elif case=='ordering':r['g_x']=np.ones(128,dtype=float)[::2]
 elif case=='mechanics':primary['mechanics']['passed']=False
 elif case=='wrong_residual':r['diagnostic']['volume_excess']+=.01
 elif case=='coherent_shift':
  r['action']+=2e-13;r['diagnostic']['volume_excess']=float(r['action'].mean()-target);r['diagnostic']['historical_sum_excess']=float(sum(r['action'])-64*target)
 old=audit_transition(s,r,target,primary,policy='historical');new=audit_transition(s,r,target,primary)
 MATRIX.append(dict(case=case,expected=expected,actual=new['passed'],old_policy=old['passed'],new_policy=new['passed'],reason=new['reason']))
 assert new['passed']==expected,new

@pytest.mark.parametrize('n',[1,7,64,1536,4096])
@pytest.mark.parametrize('v',[.001,.01,.3,.8])
def test_sizes(n,v):
 s,r,t=fixture(n,v,min(1.,v+.05));a=audit_transition(s,r,t,{'m':dict(passed=True)});assert a['passed'];assert bounds(s['action'],r['action'],t)['volume_excess']>=0

@pytest.mark.parametrize('mode',['equal','last_frame','inside','outside','nan_equal','nan_mismatch','inf_equal','inf_wrong','empty'])
def test_frame_semantics(mode):
 a=np.full((13,31),.3);b=a.copy()
 if mode in ('last_frame','outside'):b[-1,-1]+=.001
 elif mode=='inside':b[-1,-1]+=1e-12
 elif mode=='nan_equal':a[0,0]=b[0,0]=np.nan
 elif mode=='nan_mismatch':a[0,0]=np.nan
 elif mode=='inf_equal':a[0,0]=b[0,0]=np.inf
 elif mode=='inf_wrong':a[0,0]=np.inf;b[0,0]=-np.inf
 elif mode=='empty':a=b=np.zeros((0,31))
 answers=[]
 for f in (lambda a,b:np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12),frame_allclose):
  try:f(a,b);answers.append(True)
  except AssertionError as e:
   answers.append(False)
   if f==frame_allclose and mode=='last_frame':assert 'Global frame 12' in str(e)
 assert answers[0]==answers[1]

def test_shape_dtype_hash(tmp_path):
 with pytest.raises(AssertionError):frame_allclose(np.zeros((1,3)),np.zeros((2,3)))
 with pytest.raises(AssertionError):frame_allclose(np.zeros(3),np.zeros(3,dtype='float32'))
 p=tmp_path/'data';p.write_bytes(b'x'*3000000)
 import hashlib
 assert digest(p)==hashlib.sha256(p.read_bytes()).hexdigest()

def teardown_module():
 p=os.environ.get('R2_MATRIX')
 if p:Path(p).write_text(json.dumps(MATRIX,indent=2))

def test_demonstrable_reduction_only():
 root=Path(__file__).resolve().parents[1];p=root/'m3_4b/generations/20261014T013017Z/native/samples/m3_3b_d7d625d32faa15712edf39c5/arrays/x.npy'
 x=np.array(np.load(p,mmap_mode='r')[7]);s,r,t=fixture(len(x));s['x']=x.copy();r['x']=x.copy();s['action']=x.copy();r['action']=x.copy()
 for side in (s,r):
  side['diagnostic']['volume_excess']=float(x.mean()-t);side['diagnostic']['historical_sum_excess']=float(sum(x)-len(x)*t)
 r['mean_reduction']='sequential_sum_divide';r['diagnostic']['volume_excess']=float(sum(x)/len(x)-t)
 old=audit_transition(s,r,t,{'m':dict(passed=True)},policy='historical');new=audit_transition(s,r,t,{'m':dict(passed=True)})
 MATRIX.append(dict(case='identical operands demonstrably different reductions',expected=True,actual=new['passed'],old_policy=old['passed'],new_policy=new['passed'],reason=new['reason']))
 assert not old['passed'];assert new['passed'],new
 r['mean_reduction']='unknown';assert not audit_transition(s,r,t,{'m':dict(passed=True)})['passed']

def test_saved_diagnostics_omit_seconds_and_extra_numerical_keys_fail():
 s,r,t=fixture();del s['diagnostic']['seconds']
 assert audit_transition(s,r,t,{'m':dict(passed=True)})['passed']
 r['diagnostic']['unrecognized_numerical_field']=1.
 assert not audit_transition(s,r,t,{'m':dict(passed=True)})['passed']

def test_failed_historical_incoming_change_is_never_overridden():
 s,r,t=fixture();primary={'mechanics':dict(passed=True),'incoming_change':dict(passed=False,absolute_error=1.8463008899516353e-13,absolute_tolerance=6.291607216967392e-14)}
 result=audit_transition(s,r,t,primary)
 assert not result['passed'];assert result['unapplied_diagnostic_observations']['volume_excess']['candidate_passed']
 MATRIX.append(dict(case='historical incoming-change gate overrides passing residuals',expected=False,actual=result['passed'],old_policy=False,new_policy=False,reason=result['reason']))
