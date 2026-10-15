"""Exhaust fixed native recovery comparisons; retain failures without changing gates."""
import json,sys
from pathlib import Path
import numpy as np
from .common import results,job
from m3_4br2.memory import load_trajectory
from m3_4b.monitoring import atomic

def run(out,point):
 out=Path(out);pid=job(10)[1]['problem_id'];a=load_trajectory(out/'recovery'/point/'samples'/pid);b=load_trajectory(out/'recovery/control/samples'/pid)
 rows=[]
 for key in a.arrays:
  av=a.arrays[key];bv=b.arrays[key]
  if av.shape!=bv.shape:rows.append(dict(field=key,passed=False,reason='Shape differs'));continue
  failing=[];max_error=0.;first=None
  for t in range(len(av)):
   if av.dtype==bool:error=np.not_equal(av[t],bv[t]).astype(float)
   else:error=np.abs(av[t]-bv[t])
   max_error=max(max_error,float(np.max(error)))
   allowance=1e-12+1e-10*np.abs(bv[t]);bad=error>allowance
   if np.any(bad):
    failing.append(t)
    if first is None:
     idx=int(np.flatnonzero(np.asarray(bad).ravel())[0]);aa=np.asarray(av[t]).ravel();bb=np.asarray(bv[t]).ravel();ee=np.asarray(error).ravel();tt=np.asarray(allowance).ravel();first=dict(state=t,index=idx,actual=float(aa[idx]),control=float(bb[idx]),absolute_error=float(ee[idx]),allowance=float(tt[idx]))
  rows.append(dict(field=key,passed=not failing,max_absolute_error=max_error,failing_states=failing,first_failure=first))
 diagnostics=[]
 for t,(ta,tb) in enumerate(zip(a.transitions,b.transitions)):
  for k,va in ta['diagnostic'].items():
   vb=tb['diagnostic'][k]
   passed=va==vb if isinstance(va,(str,bool)) else abs(va-vb)<=1e-12+1e-10*abs(vb)
   if not passed:diagnostics.append(dict(transition=t,quantity=k,actual=va,control=vb,error=abs(va-vb),allowance=1e-12+1e-10*abs(vb)))
 first_bad=next((r for r in rows if not r['passed']),None)
 decomposition=None
 if first_bad and first_bad['field']=='historical_sum_excess':
  from fractions import Fraction as F
  t=first_bad['first_failure']['state'];x=np.asarray(a.arrays['x'][t]);y=np.asarray(b.arrays['x'][t]);sx=sum(map(lambda z:F(float(z)),x),F(0));sy=sum(map(lambda z:F(float(z)),y),F(0))
  decomposition=dict(state=t,exact_design_sum_difference=float(sx-sy),stored_scalar_difference=float(a.arrays['historical_sum_excess'][t]-b.arrays['historical_sum_excess'][t]),actual_sum_reduction_error=float(F(float(sum(x)))-sx),control_sum_reduction_error=float(F(float(sum(y)))-sy),maximum_design_difference=float(np.max(abs(x-y))),strict_actual_feasible=bool(x.mean()<=a.problem.volume_fraction and sum(x)<=len(x)*a.problem.volume_fraction),strict_control_feasible=bool(y.mean()<=b.problem.volume_fraction and sum(y)<=len(y)*b.problem.volume_fraction))
 atomic(out/'checks'/('recovery_diagnosis_'+point+'.json'),dict(point=point,all_fixed_native_comparisons_passed=all(r['passed'] for r in rows),native_fields=rows,transition_diagnostic_failures=diagnostics,termination_actual=a.metadata['termination_reason'],termination_control=b.metadata['termination_reason'],source_id_actual=a.metadata['source_id'],source_id_control=b.metadata['source_id'],decomposition=decomposition,policy_changed=False,production_gate_pass=False if any(not r['passed'] for r in rows) else not diagnostics))
 print(point,[(r['field'],len(r.get('failing_states',[])),r.get('max_absolute_error')) for r in rows if not r['passed']],flush=True)
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
