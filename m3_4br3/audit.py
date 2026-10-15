"""Coupled derived-change validation. Numerical core and R2 policy untouched."""
from fractions import Fraction as F
import numpy as np
from m3_4br2.audit import audit_transition as v2, POLICY as V2, vectors, historical, upper
POLICY='m3.4b.transition_change_audit.v3'

def change_audit(saved,replay,evidence):
 r=dict(passed=False,policy=POLICY)
 try:
  if evidence is None:raise ValueError('Missing transition evidence')
  k=evidence['transition'];indices=evidence['indices']
  if not isinstance(k,int) or k<0 or indices!=[k,k+1]:raise ValueError('Transition indexing')
  x,y=vectors(saved.get('x'),replay.get('x'));a,b=vectors(saved.get('action'),replay.get('action'))
  if a.shape!=x.shape:raise ValueError('Transition shape')
  for name,s,t,tol in [('x',x,y,1e-12),('action',a,b,1e-8)]:
   if not historical(t,s,tol)['passed'] or not np.all(abs(t-s)<=tol*abs(s)):raise ValueError('Primary '+name)
  da=a-x;db=b-y;sa=float(np.max(abs(da)));rb=float(np.max(abs(db)))
  sc=evidence['saved_change'];oc=evidence['outgoing_change'];threshold=evidence['threshold']
  if not np.isfinite([sc,oc,threshold]).all() or threshold<=0:raise ValueError('Nonfinite/invalid scalar')
  if sc!=sa or oc!=sa:raise ValueError('Saved diagnostic/transition inconsistency')
  # Exact binary-rational operand differences; exact measured subtraction errors
  # avoid unsupported assumptions about cancellation or subnormal arithmetic.
  exact_a=[F(float(aa))-F(float(xx)) for aa,xx in zip(a,x)]
  exact_b=[F(float(bb))-F(float(yy)) for bb,yy in zip(b,y)]
  induced=max(abs(aa-bb) for aa,bb in zip(exact_a,exact_b))
  ea=max(abs(F(float(dd))-ee) for dd,ee in zip(da,exact_a))
  eb=max(abs(F(float(dd))-ee) for dd,ee in zip(db,exact_b))
  bound=induced+ea+eb;error=abs(F(rb)-F(sc))
  old=historical(rb,sc,1e-12)
  r.update(saved_change=sc,replay_change=rb,original=old,induced_bound=upper(induced),saved_subtraction_error=upper(ea),replay_subtraction_error=upper(eb),bound=upper(bound),absolute_error=upper(error),saved_max_indices=np.flatnonzero(abs(da)==sa).tolist(),replay_max_indices=np.flatnonzero(abs(db)==rb).tolist(),stopping_threshold=threshold,saved_stop=sc<threshold,replay_stop=rb<threshold)
  if error>bound:raise ValueError('Lipschitz bound')
  if (sc<threshold)!=(rb<threshold):raise ValueError('Stopping classification differs')
  r.update(passed=True,reason='Own-operand maxima, induced bound and stopping classification passed',old_failed_new_passed=not old['passed'])
 except (ValueError,KeyError,TypeError) as exc:r['reason']=str(exc)
 return r

def audit_transition(saved,replay,target,primary,policy=POLICY,identity=None,change_evidence=None):
 if policy!=POLICY:raise ValueError('Unknown R3 policy')
 change=change_audit(saved,replay,change_evidence)
 corrected={k:dict(v) for k,v in primary.items()}
 if 'incoming' not in corrected:change.update(passed=False,reason='Missing primary incoming evidence')
 else:
  if change.get('original')!=corrected['incoming']:change.update(passed=False,reason='Historical incoming evidence mismatch')
  corrected['incoming']={**corrected['incoming'],'passed':change['passed'],'policy':POLICY}
 result=v2(saved,replay,target,corrected,V2,identity)
 result.update(policy=POLICY,change_audit=change,original_primary=primary)
 if not change['passed']:result.update(passed=False,reason=change['reason'])
 return result

def cached_primary(row):
 """Revalidate original evidence rather than the prior policy's adapted gate."""
 if row.get('policy')==POLICY:
  if 'original_primary' not in row:raise ValueError('Missing original cached primary evidence')
  return row['original_primary']
 return row['primary']
