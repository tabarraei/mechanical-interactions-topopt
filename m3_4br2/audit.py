"""Selectable validation only. Imports historical comparator without modifying it."""
import math
from fractions import Fraction as F
import numpy as np
POLICY='m3.4b.oc_diagnostic_audit.v2'
RESIDUALS=('volume_excess','historical_sum_excess')

def historical(a,b,rtol):
 from topopt_utils.fea.trajectory_validation import comparison
 return comparison(a,b,rtol,enforce=False)

def upper(q):
 r=float(q)
 return math.nextafter(r,math.inf) if F.from_float(r)<q else r

def bounds(a,b,target):
 # Conservative sequential error applies to binary addition trees with <=n-1
 # rounded additions, positive normal binary64 inputs, no overflow/underflow.
 n=a.size;u=F(1,2**53)
 if n*u>=1:raise ValueError('Unsupported reduction length')
 gamma=lambda k:k*u/(1-k*u)
 ma=F.from_float(float(np.max(a)));mb=F.from_float(float(np.max(b)))
 muA=F.from_float(float(a.mean()));muB=F.from_float(float(b.mean()));v=F.from_float(float(target))
 sa=F.from_float(float(sum(a)));sb=F.from_float(float(sum(b)));budget=F.from_float(float(n*target))
 return dict(mean=upper(gamma(n)*(ma+mb)),raw_sum=upper(gamma(n-1)*n*(ma+mb)),
  volume_excess=upper(gamma(n)*(ma+mb)+u*(abs(muA)+abs(muB)+2*abs(v))),
  historical_sum_excess=upper(gamma(n-1)*n*(ma+mb)+u*(abs(sa)+abs(sb)+2*abs(budget))))

def vectors(a,b):
 if a is None or b is None:raise ValueError('Missing independent operand')
 a=np.asarray(a);b=np.asarray(b)
 if a.dtype!=np.float64 or b.dtype!=np.float64:raise ValueError('Unsupported dtype')
 if a.shape!=b.shape or a.ndim!=1 or not a.size:raise ValueError('Shape/order contract')
 if not a.flags.c_contiguous or not b.flags.c_contiguous:raise ValueError('Unsupported reduction layout')
 if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite operand')
 return a,b

def feasibility(x,target):
 return bool(np.min(x)>=.001 and np.max(x)<=1 and x.mean()<=target and sum(x)<=len(x)*target)

def reduced_mean(side,z):
 method=side.get('mean_reduction','numpy_mean')
 if method=='numpy_mean':return float(z.mean())
 if method=='sequential_sum_divide':return float(sum(z)/len(z))
 raise ValueError('Unsupported reduction method')

def audit_transition(saved,replay,target,primary,policy=POLICY,identity=None):
 """Inputs contain actual actions, designs, both gradients and diagnostics.
 Primary is a nonempty mapping of independent mechanical comparison records.
 The caller also verifies full problem identity, stored hashes and exact trials.
 Every original diagnostic record remains present even on failure.
 """
 if policy not in (POLICY,'historical'):raise ValueError('Unknown selectable policy')
 result=dict(policy=policy,passed=False,identity=identity,primary=primary,checks={},diagnostics={},reduction='NumPy mean and historical Python sum of float64 scalars; unknown NumPy tree, sequential worst-case bound')
 result['raw_saved_diagnostics']={k:v for k,v in saved.get('diagnostic',{}).items() if k!='seconds'}
 result['raw_replayed_diagnostics']={k:v for k,v in replay.get('diagnostic',{}).items() if k!='seconds'}
 try:
  observed_a,observed_b=vectors(saved.get('action'),replay.get('action'))
  observed_bounds=bounds(observed_a,observed_b,target)
  result['observed_operands']=dict(saved_mean=float(observed_a.mean()),replay_mean=float(observed_b.mean()),saved_sum=float(sum(observed_a)),replay_sum=float(sum(observed_b)),target=target)
  result['unapplied_diagnostic_observations']={}
  for k in RESIDUALS:
   old=historical(replay['diagnostic'][k],saved['diagnostic'][k],1e-12)
   allowance=max(old['absolute_tolerance'],observed_bounds[k]) if policy==POLICY else old['absolute_tolerance']
   result['unapplied_diagnostic_observations'][k]=dict(original=old,arithmetic_bound=observed_bounds[k],candidate_passed=bool(old['absolute_error']<=allowance),applied_only_if_all_prerequisites_pass=True)
 except (ValueError,KeyError,TypeError) as e:result['operand_observation_error']=str(e)
 try:
  if not primary or not all(isinstance(r.get('passed'),(bool,np.bool_)) and bool(r['passed']) for r in primary.values()):raise ValueError('Independent primary evidence failed/missing')
  for side in (saved,replay):
   if 'gradient_certificate' in side and side['gradient_certificate'].get('passed') is not True:raise ValueError('Conditioned local gradient certificate')
  for name,tol in [('x',1e-12),('rho',1e-12),('g_rho',1e-9),('g_x',1e-9),('action',1e-8)]:
   a,b=vectors(saved.get(name),replay.get(name));r=historical(b,a,tol);result['checks'][name]=r
   if not r['passed']:raise ValueError('Primary '+name)
   # Supplement global norm; never replace it. Zero references get no atol.
   component=bool(np.all(abs(b-a)<=tol*abs(a)))
   result['checks'][name]['supplement_pure_relative_passed']=component
   if not component:
    certificates=[side.get('gradient_certificate',{}) for side in (saved,replay)]
    if name not in ('g_rho','g_x') or not all(c.get('passed') is True for c in certificates):raise ValueError('Componentwise '+name)
    result['checks'][name]['conditioned_local_certificates']=certificates
  a,b=vectors(saved['action'],replay['action']);x,y=vectors(saved['x'],replay['x'])
  if not (.001<=target<=1) or not np.isfinite(target):raise ValueError('Target')
  result['strict_feasibility']=dict(saved=feasibility(a,target),replay=feasibility(b,target))
  if not all(result['strict_feasibility'].values()):raise ValueError('Strict feasibility')
  for before,after in ((x,a),(y,b)):
   if np.min(before)<.001 or np.max(before)>1:raise ValueError('Initial box')
   if np.any(after<np.maximum(.001,before-.1)) or np.any(after>np.minimum(1,before+.1)):raise ValueError('Move box')
  bound=bounds(a,b,target);result['bound']=bound
  result['operands']=dict(saved_mean=float(a.mean()),replay_mean=float(b.mean()),saved_numpy_sum=float(a.sum()),replay_numpy_sum=float(b.sum()),saved_historical_sum=float(sum(a)),replay_historical_sum=float(sum(b)),target=target,n=a.size)
  for side,z in ((saved,a),(replay,b)):
   d=side['diagnostic'];lo=d['multiplier_lower'];hi=d['multiplier_upper']
   if not (np.isfinite(lo) and np.isfinite(hi) and 0<=lo<=hi<=1e5 and hi-lo<=1e-8):raise ValueError('Bracket invariant')
   if side['selected_multiplier']!=hi:raise ValueError('Exact selected upper')
   if not d['upper_feasible'] or not d['strict_feasible']:raise ValueError('Endpoint feasibility')
   if d['root_status']=='active_root' and not d['lower_above_budget']:raise ValueError('Lower bracket')
   if d['volume_excess']!=float(reduced_mean(side,z)-target) or d['historical_sum_excess']!=float(sum(z)-len(z)*target):raise ValueError('Diagnostic/action inconsistency')
  for label,aa,bb in [('raw_mean',a.mean(),b.mean()),('raw_sum',sum(a),sum(b)),('selected_multiplier',saved['selected_multiplier'],replay['selected_multiplier'])]:
   r=historical(bb,aa,1e-12);result['checks'][label]=r
   if not r['passed']:raise ValueError(label)
  failures=[]
  if (saved['diagnostic'].keys()-{'seconds'})!=(replay['diagnostic'].keys()-{'seconds'}):raise ValueError('Diagnostic keys')
  for k,aa in saved['diagnostic'].items():
   if k=='seconds':continue
   bb=replay['diagnostic'][k]
   if isinstance(aa,(str,bool)):
    old=dict(passed=aa==bb);new=old['passed'];allow=None
   else:
    old=historical(bb,aa,1e-12);allow=max(old['absolute_tolerance'],bound[k]) if k in RESIDUALS and policy==POLICY else old['absolute_tolerance'];new=old['absolute_error']<=allow
   result['diagnostics'][k]=dict(saved=aa,replay=bb,original=old,new_passed=bool(new),new_allowance=allow,old_failed_new_passed=not old['passed'] and bool(new))
   if not new:failures.append(k)
  if failures:raise ValueError('Diagnostics '+','.join(failures))
  result['passed']=True;result['reason']='All prerequisites and diagnostics passed'
 except (ValueError,KeyError,TypeError) as e:result['reason']=str(e)
 return result
