"""PROPOSED ONLY: m3.4b.restart_acceptance.v2. No FEM/OC imports or calls.
Checkpoint integrity, independent correctness and control equivalence are separate.
The production gate does not import or activate this module.
"""
import hashlib,json,math
from fractions import Fraction as F
import numpy as np
POLICY='m3.4b.restart_acceptance.v2'
RTOL=1e-10
ATOL=1e-12
RESIDUALS={'volume_excess','historical_sum_excess'}
MECHANICAL_LIMITS={'residual_relative_force':1e-7,'residual_backward_error':1e-12,'energy_compliance_relative':1e-8,'force_balance_relative':1e-7,'moment_balance_relative':1e-7}

class EvidenceError(Exception):
 def __init__(self,status,message):self.status=status;super().__init__(message)
def reject(message):raise EvidenceError('Rejected',message)
def missing(message):raise EvidenceError('Indeterminate',message)
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array_digest(value):
 a=np.ascontiguousarray(value);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def upward(value):
 r=float(value)
 return math.nextafter(r,math.inf) if F(r)<value else r

def fixed(actual,control):
 a=np.asarray(actual);b=np.asarray(control)
 if a.shape!=b.shape or a.dtype!=b.dtype:reject('Native shape/dtype mismatch')
 if not np.isfinite(a).all() or not np.isfinite(b).all():reject('Nonfinite native operand')
 return bool(np.all(np.abs(a-b)<=ATOL+RTOL*np.abs(b))) if a.dtype!=bool else bool(np.array_equal(a,b))

def global_comparison(actual,control,rtol):
 a=np.asarray(actual);b=np.asarray(control);error=float(np.linalg.norm(a-b));norm=float(np.linalg.norm(b));scale=max(norm,float(np.linalg.norm(a)))
 tolerance=float(rtol*norm+100*max(a.size,1)*np.finfo(float).eps*scale)
 return dict(passed=error<=tolerance,absolute_error=error,relative_error=error/norm if norm else (0. if error==0 else None),allowance=tolerance)

def displacement(actual,control,fixed_dofs):
 a=np.asarray(actual);b=np.asarray(control)
 if a.shape!=b.shape or a.ndim!=1 or not len(a) or len(a)%2 or a.dtype!=np.float64 or b.dtype!=np.float64:reject('Displacement shape/dtype')
 if not np.isfinite(a).all() or not np.isfinite(b).all():reject('Nonfinite displacement')
 if np.any(a[fixed_dofs]!=0) or np.any(b[fixed_dofs]!=0):reject('Nonzero homogeneous constrained displacement')
 an=a.reshape(-1,2);bn=b.reshape(-1,2);d=an-bn
 local_error=np.hypot(d[:,0],d[:,1]);scale=np.hypot(bn[:,0],bn[:,1]);budget=ATOL+RTOL*scale
 global_result=global_comparison(a,b,1e-9)
 bad=np.flatnonzero(local_error>budget);ratios=local_error/budget
 return dict(passed=not len(bad) and global_result['passed'],global_check=global_result,local_maximum_ratio=float(np.max(ratios)),maximum_absolute_component_error=float(np.max(abs(a-b))),maximum_nodal_error=float(np.max(local_error)),failing_nodes=bad.tolist(),local_rule='||u_restart(node)-u_control(node)||2 <= 1e-12 + 1e-10*||u_control(node)||2',affected_DOFs=int(np.count_nonzero(a!=b)),affected_nodes=int(np.count_nonzero(np.any(an!=bn,axis=1))))

def residual(a,b,target,ra,rb,kind):
 a=np.asarray(a);b=np.asarray(b)
 if a.shape!=b.shape or a.ndim!=1 or a.dtype!=np.float64 or b.dtype!=np.float64 or not len(a):reject('Residual operand shape/dtype')
 if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite([ra,rb,target]).all():reject('Nonfinite residual operand')
 if np.any(a<.001) or np.any(a>1) or np.any(b<.001) or np.any(b>1):reject('Residual design box')
 n=len(a);u=F(1,2**53)
 if n*u>=1:missing('Unsupported reduction length')
 gamma=lambda k:k*u/(1-k*u)
 sa=sum(map(lambda x:F(float(x)),a),F(0));sb=sum(map(lambda x:F(float(x)),b),F(0))
 if kind=='historical_sum_excess':
  suma=float(sum(a));sumb=float(sum(b));budget=float(n*target);exact_a=sa;exact_b=sb;reduction_bound=gamma(n-1)*(sa+sb)
 elif kind=='volume_excess':
  suma=float(a.mean());sumb=float(b.mean());budget=float(target);exact_a=sa/n;exact_b=sb/n;reduction_bound=gamma(n)*(sa+sb)/n
 else:reject('Unsupported derived residual')
 if ra!=float(suma-budget) or rb!=float(sumb-budget):reject('Incorrect recorded '+kind)
 ea=F(suma)-exact_a;eb=F(sumb)-exact_b
 suba=F(float(ra))-(F(suma)-F(budget));subb=F(float(rb))-(F(sumb)-F(budget))
 # Binary64 positive normal inputs: <=n-1 additions, plus mean division.
 # Common rounded target cancels; subtraction errors use input-magnitude bounds.
 subtraction_bound=u*(abs(F(suma))+abs(F(sumb))+2*abs(F(budget)))
 induced=abs(exact_a-exact_b);bound=induced+reduction_bound+subtraction_bound;error=abs(F(float(ra))-F(float(rb)))
 if error>bound or abs(ea)+abs(eb)>reduction_bound or abs(suba)+abs(subb)>subtraction_bound:reject('Derived residual exceeds justified bound')
 return dict(passed=True,historical_passed=bool(abs(ra-rb)<=ATOL+RTOL*abs(rb)),observed_difference=float(F(float(ra))-F(float(rb))),exact_operand_difference=float(exact_a-exact_b),actual_reduction_error=float(ea),control_reduction_error=float(eb),actual_subtraction_error=float(suba),control_subtraction_error=float(subb),induced_bound=upward(induced),reduction_bound=upward(reduction_bound),subtraction_bound=upward(subtraction_bound),total_bound=upward(bound),cancellation_amplification=abs(float(exact_a-exact_b))/max(abs(ra),abs(rb)) if max(abs(ra),abs(rb)) else 0.)

def check_proof(side):
 proof=side.get('certificate')
 if proof is None:missing('Missing independent validation certificate')
 if proof.get('status')!='verified_complete':
  if proof.get('status')=='failed':reject('Independent mechanical/OC validation failed')
  missing('Independent validation evidence incomplete')
 a=side['arrays'];T=len(a['x'])
 if proof.get('states')!=T or proof.get('transitions')!=T-1:reject('Certificate state/transition coverage')
 if proof.get('problem_id')!=side['spec'].get('problem_id') or proof.get('source_id')!=side['metadata'].get('source_id'):reject('Independent certificate identity')
 if proof.get('configuration_digest')!=canonical(side['spec']):reject('Independent configuration/load/boundary binding')
 if proof.get('policy')!='m3.4b.transition_change_audit.v3':missing('Established v2/v3 independent audit unavailable')
 if set(proof.get('native_digests',{}))!=set(a):missing('Incomplete independent operand binding')
 for key,value in a.items():
  if array_digest(value)!=proof['native_digests'][key]:reject('Native '+key+' changed from independently validated operands')
 if not proof.get('source_fingerprints_verified') or not proof.get('retained_records_verified'):missing('Unverified independent source/record provenance')

def compare(control,restart,checkpoint):
 result=dict(policy=POLICY,status='Indeterminate',checkpoint_integrity=None,independent_correctness=None,numerical_equivalence=None,historical_comparator=None,production_gate_activated=False,checks={})
 try:
  if checkpoint is None or checkpoint.get('status')=='missing':missing('Missing exact committed checkpoint proof')
  if checkpoint.get('status')!='verified_exact':reject('Committed checkpoint byte/hash/order changed')
  if not checkpoint.get('source_identity') or not checkpoint.get('configuration_identity'):reject('Checkpoint source/configuration identity')
  result['checkpoint_integrity']='Accepted'
  for side in (control,restart):
   if 'arrays' not in side or 'u' not in side['arrays']:missing('Missing displacement/native operands')
   if 'spec' not in side or 'metadata' not in side or 'transitions' not in side:missing('Missing trajectory identity/bookkeeping')
  if control['spec']!=restart['spec']:reject('Wrong configuration/load/boundary identity')
  if checkpoint.get('source_id')!=control['metadata']['source_id'] or checkpoint.get('configuration_digest')!=canonical(control['spec']):reject('Checkpoint exact source/configuration binding')
  if control['metadata']['source_id']!=restart['metadata']['source_id']:reject('Wrong source identity')
  aa=control['arrays'];bb=restart['arrays']
  if set(aa)!=set(bb):reject('Native field set differs')
  required={'x','rho','u','g_rho','g_x','compliance','iteration','previous_change','update_change','has_incoming_update','volume_excess','historical_sum_excess',*MECHANICAL_LIMITS}
  if not required.issubset(aa):missing('Missing mandatory native field')
  T=len(aa['x']);spec=control['spec'];target=spec['mechanical']['volume_fraction'];threshold=spec['optimizer']['tolerance'];cap=spec['optimizer']['maxiter'];fixed_dofs=np.asarray(spec['mechanical']['fixed_dofs'],dtype=int)
  if T<2 or not .001<=target<=1 or not threshold>0:reject('Unsupported trajectory/budget')
  historic=True
  for key in aa:
   a=np.asarray(aa[key]);b=np.asarray(bb[key])
   if a.shape!=b.shape or a.dtype!=b.dtype:reject('Native shape/dtype '+key)
   if not np.isfinite(a).all() or not np.isfinite(b).all():reject('Nonfinite '+key)
   old=fixed(b,a);historic=historic and old
   result['checks'][key]=dict(historical_passed=old)
   if key not in {'u',*RESIDUALS} and not old:reject('Primary native '+key+' differs')
  result['historical_comparator']=historic
  if 'committed_states' not in checkpoint or 'committed_transitions' not in checkpoint or 'committed_native_digests' not in checkpoint:missing('Missing exact committed operand binding')
  count=checkpoint['committed_states'];updates=checkpoint['committed_transitions']
  if not isinstance(count,int) or not 1<=count<=T or updates!=count-1:reject('Committed prefix counts/order')
  if set(checkpoint['committed_native_digests'])!=set(bb):missing('Incomplete exact committed field binding')
  for key,a in bb.items():
   length=updates if key=='update_change' else count
   if array_digest(a[:length])!=checkpoint['committed_native_digests'][key]:reject('Committed native '+key+' modified')
  for side in (control,restart):
   a=side['arrays'];meta=side['metadata'];tr=side['transitions']
   if len(a['x'])!=T or len(a['iteration'])!=T or not np.array_equal(a['iteration'],np.arange(T)):reject('Incorrect state ordering')
   if len(tr)!=T-1 or len(a['update_change'])!=T-1:reject('Missing transition/state')
   if bool(a['has_incoming_update'][0]) or not np.all(a['has_incoming_update'][1:]) or a['previous_change'][0]!=0:reject('Incoming timing sentinel')
   if np.any(a['u'][:,fixed_dofs]!=0):reject('Homogeneous support displacement')
   if a['x'].dtype!=np.float64 or a['u'].dtype!=np.float64 or a['x'].ndim!=2 or a['u'].ndim!=2:reject('Primary shape/dtype')
   if a['u'].shape[1]!=len(spec['mechanical']['force_vector']):reject('Load/displacement dimension')
   for t,x in enumerate(a['x']):
    if np.any(x<.001) or np.any(x>1) or x.mean()>target or sum(x)>len(x)*target:reject('Strict volume/box infeasibility')
    for key,tol in MECHANICAL_LIMITS.items():
     if a[key][t]<0 or a[key][t]>tol:reject('Unchanged mechanical invariant '+key)
   for t,trans in enumerate(tr):
    if trans['iteration']!=t or trans['next_iteration']!=t+1 or not trans['valid'] or trans['optimizer']!=spec['optimizer']:reject('Incorrect transition ordering/optimizer')
    delta=a['x'][t+1]-a['x'][t];change=float(np.max(abs(delta)))
    if array_digest(delta)!=trans['delta_digest'] or change!=trans['maximum_change'] or change!=a['update_change'][t] or change!=a['previous_change'][t+1]:reject('Incorrect action/change bookkeeping')
    if np.any(a['x'][t+1]<np.maximum(.001,a['x'][t]-.1)) or np.any(a['x'][t+1]>np.minimum(1,a['x'][t]+.1)):reject('Move-limit infeasibility')
    if trans['selected_multiplier']!=trans['diagnostic']['multiplier_upper'] or not trans['diagnostic']['strict_feasible'] or not trans['diagnostic']['upper_feasible']:reject('OC selection/endpoint feasibility')
   reason='change_tolerance' if a['previous_change'][-1]<threshold else 'maxiter'
   if np.any(a['previous_change'][1:-1]<threshold) or meta['termination_reason']!=reason or (reason=='maxiter' and T-1!=cap):reject('Altered stopping classification')
   if meta['state_count']!=T or meta['completed_updates']!=T-1:reject('Native count metadata')
   check_proof(side)
  if control['metadata']['termination_reason']!=restart['metadata']['termination_reason'] or not np.array_equal(aa['previous_change'][1:]<threshold,bb['previous_change'][1:]<threshold):reject('Restart/control stopping decisions differ')
  result['independent_correctness']='Accepted'
  uchecks=[];rchecks={key:[] for key in RESIDUALS}
  for t in range(T):
   u=displacement(bb['u'][t],aa['u'][t],fixed_dofs);uchecks.append(u)
   if not u['passed']:reject('Localized/global displacement comparison failed at state '+str(t))
   # The original pointwise x comparison already passed, independently bound.
   for key in RESIDUALS:rchecks[key].append(residual(bb['x'][t],aa['x'][t],target,float(bb[key][t]),float(aa[key][t]),key))
  for ta,tb in zip(control['transitions'],restart['transitions']):
   if set(ta['diagnostic'])!=set(tb['diagnostic']):reject('OC diagnostic field set')
   for key,a in ta['diagnostic'].items():
    b=tb['diagnostic'][key]
    if key in RESIDUALS:
     t=ta['next_iteration']
     if a!=float(aa[key][t]) or b!=float(bb[key][t]):reject('Native/OC residual mismatch')
    elif isinstance(a,(str,bool)):
     if a!=b:reject('OC diagnostic '+key)
    elif not fixed(np.asarray(b),np.asarray(a)):reject('OC diagnostic '+key)
   if not fixed(np.asarray(tb['selected_multiplier']),np.asarray(ta['selected_multiplier'])):reject('Restart multiplier divergence')
  result.update(status='Accepted',numerical_equivalence='Accepted',reason='Exact checkpoint proof, unchanged independent correctness, strict feasibility/stopping, unchanged primary comparisons and local nodal/residual criteria passed',displacements=uchecks,residuals=rchecks,action_comparison=dict(passed=fixed(bb['x'][1:],aa['x'][1:]),rule='Unchanged native design criterion on every outgoing action'))
 except EvidenceError as e:result.update(status=e.status,reason=str(e))
 except (KeyError,TypeError,IndexError) as e:result.update(status='Indeterminate',reason='Missing/malformed mandatory evidence: '+str(e))
 return result
