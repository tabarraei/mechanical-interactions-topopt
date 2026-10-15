"""Comparator-only synthetic tests and sealed retained artifacts. No FEM/OC."""
import copy,json,os
from pathlib import Path
import numpy as np
import pytest
from .policy import compare,displacement,residual,array_digest,canonical
OUTCOMES=[]

def seal(s):
 T=len(s['arrays']['x']);s['certificate']=dict(status='verified_complete',states=T,transitions=T-1,source_id=s['metadata']['source_id'],problem_id=s['spec']['problem_id'],configuration_digest=canonical(s['spec']),policy='m3.4b.transition_change_audit.v3',source_fingerprints_verified=True,retained_records_verified=True,native_digests={k:array_digest(v) for k,v in s['arrays'].items()})
 return s

def bookkeeping(s):
 a=s['arrays'];T=len(a['x']);target=s['spec']['mechanical']['volume_fraction'];a['volume_excess']=np.asarray([float(x.mean()-target) for x in a['x']]);a['historical_sum_excess']=np.asarray([float(sum(x)-len(x)*target) for x in a['x']]);a['update_change']=np.max(abs(np.diff(a['x'],axis=0)),axis=1);a['previous_change']=np.r_[0.,a['update_change']]
 s['transitions']=[]
 for t in range(T-1):
  d=dict(multiplier_upper=1.,strict_feasible=True,upper_feasible=True,volume_excess=float(a['volume_excess'][t+1]),historical_sum_excess=float(a['historical_sum_excess'][t+1]))
  s['transitions'].append(dict(iteration=t,next_iteration=t+1,valid=True,optimizer=s['spec']['optimizer'],delta_digest=array_digest(a['x'][t+1]-a['x'][t]),maximum_change=float(a['update_change'][t]),selected_multiplier=1.,diagnostic=d))
 return s

def fixture():
 x=np.array([[.3,.3,.3],[.31,.30,.30],[.32,.30,.30]],dtype=float);u=np.tile([0.,0.,2.,.001,1.,.001],(3,1));a=dict(x=x,rho=x.copy(),u=u,g_rho=-np.ones((3,3)),g_x=-np.ones((3,3)),compliance=np.ones(3),iteration=np.arange(3),has_incoming_update=np.array([False,True,True]))
 for k in ('residual_relative_force','residual_backward_error','energy_compliance_relative','force_balance_relative','moment_balance_relative'):a[k]=np.zeros(3)
 spec=dict(problem_id='synthetic',mechanical=dict(volume_fraction=.4,force_vector=[0.,0.,1.,0.,0.,0.],fixed_dofs=[0,1]),optimizer=dict(tolerance=.001,maxiter=2,move=.1,option='feasible_upper_v1'))
 s=seal(bookkeeping(dict(arrays=a,spec=spec,metadata=dict(source_id='original',termination_reason='maxiter',state_count=3,completed_updates=2))))
 checkpoint=dict(status='verified_exact',source_identity=True,configuration_identity=True,source_id='original',configuration_digest=canonical(spec),committed_states=1,committed_transitions=0,committed_native_digests={k:array_digest(v[:0 if k=='update_change' else 1]) for k,v in s['arrays'].items()})
 return s,copy.deepcopy(s),checkpoint

CASES=['equal','small_design','small_distributed_displacements','near_zero_component','modified_committed_byte','state_order','transition_order','wrong_source','wrong_configuration','large_design','mean_preserving_redistribution','wrong_action','wrong_sensitivity','wrong_displacement','localized_displacement_global_pass','load_identity','boundary_identity','strict_volume','altered_stop','missing_certificate','missing_displacement','nonfinite','shape','wrong_sum','missing_checkpoint','missing_certificate_binding','bad_mechanical_certificate','wrong_multiplier','mechanical_invariant','unsupported_dtype']
@pytest.mark.parametrize('case',CASES)
def test_adversaries(case):
 s,r,c=fixture();expected='Accepted' if case in CASES[:4] else ('Indeterminate' if case in ('missing_certificate','missing_displacement','missing_checkpoint','missing_certificate_binding') else 'Rejected')
 if case=='small_design':r['arrays']['x'][1,0]+=1e-13;bookkeeping(r);seal(r)
 elif case=='small_distributed_displacements':r['arrays']['u'][1:,2:]+=1e-13;seal(r)
 elif case=='near_zero_component':r['arrays']['u'][1:,3]+=1.2e-12;seal(r)
 elif case=='modified_committed_byte':c['status']='hash_mismatch'
 elif case=='state_order':r['arrays']['iteration'][1]=2;seal(r)
 elif case=='transition_order':r['transitions'][0]['next_iteration']=2
 elif case=='wrong_source':r['metadata']['source_id']='wrong'
 elif case=='wrong_configuration':r['spec']['optimizer']['move']=.2
 elif case=='large_design':r['arrays']['x'][1,0]+=.001;bookkeeping(r);seal(r)
 elif case=='mean_preserving_redistribution':r['arrays']['x'][1,0]+=.001;r['arrays']['x'][1,1]-=.001;bookkeeping(r);seal(r)
 elif case=='wrong_action':r['arrays']['x'][2,0]+=.01;bookkeeping(r);seal(r)
 elif case=='wrong_sensitivity':r['arrays']['g_rho'][1,0]+=.001;seal(r)
 elif case=='wrong_displacement':r['arrays']['u'][1,3]+=.01;seal(r)
 elif case=='localized_displacement_global_pass':r['arrays']['u'][1,3]+=1e-9;seal(r)
 elif case=='load_identity':r['spec']['mechanical']['force_vector'][2]=2.
 elif case=='boundary_identity':r['spec']['mechanical']['fixed_dofs']=[0]
 elif case=='strict_volume':
  s['arrays']['x'][:]=np.nextafter(.4,np.inf);r=copy.deepcopy(s);bookkeeping(s);bookkeeping(r);seal(s);seal(r);c['committed_native_digests']={k:array_digest(v[:0 if k=='update_change' else 1]) for k,v in r['arrays'].items()}
 elif case=='altered_stop':r['metadata']['termination_reason']='change_tolerance'
 elif case=='missing_certificate':del r['certificate']
 elif case=='missing_displacement':del r['arrays']['u']
 elif case=='nonfinite':r['arrays']['u'][1,3]=np.nan;seal(r)
 elif case=='shape':r['arrays']['u']=r['arrays']['u'][:,:4]
 elif case=='wrong_sum':r['arrays']['historical_sum_excess'][1]+=1e-13;seal(r)
 elif case=='missing_checkpoint':c=None
 elif case=='missing_certificate_binding':r['certificate']['native_digests'].pop('u')
 elif case=='bad_mechanical_certificate':r['certificate']['status']='failed'
 elif case=='wrong_multiplier':r['transitions'][0]['selected_multiplier']=2.
 elif case=='mechanical_invariant':r['arrays']['residual_relative_force'][1]=1e-5;seal(r)
 elif case=='unsupported_dtype':r['arrays']['u']=r['arrays']['u'].astype('float32')
 result=compare(s,r,c);OUTCOMES.append(dict(case=case,expected=expected,actual=result['status'],reason=result['reason']))
 assert result['status']==expected,result
 if case=='near_zero_component':assert not result['historical_comparator']
 if case=='localized_displacement_global_pass':
  from .policy import global_comparison
  assert global_comparison(r['arrays']['u'][1],s['arrays']['u'][1],1e-9)['passed']

def test_displacement_rotation_invariance():
 a=np.array([2.,.001,1.,3.]);b=a.copy();b[1]+=1.2e-12
 theta=.7;R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
 base=displacement(b,a,np.array([],int));rotated=displacement((b.reshape(-1,2)@R).ravel(),(a.reshape(-1,2)@R).ravel(),np.array([],int))
 assert base['passed'] and rotated['passed']

def test_zero_node_localized_error():
 a=np.array([0.,0.,100.,100.]);b=a.copy();b[0]=2e-12
 assert not displacement(b,a,np.array([],int))['passed']

@pytest.mark.parametrize('n',[1,3,64,1536])
def test_near_zero_residual_cancellation(n):
 a=np.full(n,np.nextafter(.3,0));b=a.copy();b[0]=np.nextafter(b[0],0)
 ra=float(sum(a)-n*.3);rb=float(sum(b)-n*.3)
 result=residual(a,b,.3,ra,rb,'historical_sum_excess');assert result['passed'];assert abs(result['observed_difference'])<=result['total_bound']

def test_bad_residual_exceeds_bound():
 a=np.full(3,.3);r=float(sum(a)-3*.4)
 from .policy import EvidenceError
 with pytest.raises(EvidenceError):residual(a,a,.4,r+1e-4,r,'historical_sum_excess')

def test_sensitivity_payload_not_covered_by_old_certificate():
 s,r,c=fixture();r['arrays']['g_x'][1,0]+=1e-13
 result=compare(s,r,c);assert result['status']=='Rejected';assert 'independently validated' in result['reason']

def test_fixed_support_corruption():
 s,r,c=fixture();r['arrays']['u'][1,0]=1e-14;seal(r)
 assert compare(s,r,c)['status']=='Rejected'

def test_exact_configuration_certificate_binding():
 s,r,c=fixture();s['spec']['mechanical']['force_vector'][2]=2.;r['spec']=copy.deepcopy(s['spec']);c['configuration_digest']=canonical(s['spec'])
 assert compare(s,r,c)['status']=='Rejected'

def teardown_module():
 if os.environ.get('R4_R1_OUT'):(Path(os.environ['R4_R1_OUT'])/'synthetic_outcomes.json').write_text(json.dumps(OUTCOMES,indent=2))

def test_strict_volume_cannot_use_derived_residual_allowance():
 s,r,c=fixture()
 for side in (s,r):
  side['arrays']['x'][:]=np.nextafter(.4,np.inf);bookkeeping(side);seal(side)
 c['committed_native_digests']={k:array_digest(v[:0 if k=='update_change' else 1]) for k,v in r['arrays'].items()}
 result=compare(s,r,c);assert result['status']=='Rejected';assert result['reason']=='Strict volume/box infeasibility'

def test_unchanged_mechanics_limit_even_if_histories_agree():
 s,r,c=fixture()
 for side in (s,r):side['arrays']['residual_backward_error'][1]=2e-12;seal(side)
 result=compare(s,r,c);assert result['status']=='Rejected';assert result['reason']=='Unchanged mechanical invariant residual_backward_error'

def test_global_norm_cannot_cover_error_at_small_displacement_node():
 control=np.array([100.,100.,1e-8,1e-8]);restart=control.copy();restart[3]+=2e-12
 from .policy import global_comparison
 assert global_comparison(restart,control,1e-9)['passed']
 assert not displacement(restart,control,np.array([],int))['passed']
