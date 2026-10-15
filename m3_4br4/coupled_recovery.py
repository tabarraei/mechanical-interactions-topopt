"""Check documented OC v2/v3 conditions on fully replayed recovery histories.
Does not change any policy or native-array comparison. Only associates a failing
native incoming residual with the exact existing OC diagnostic/action audit.
"""
import sys,json
from pathlib import Path
import numpy as np
from .common import job,read
from m3_4br2.memory import load_trajectory
from m3_4br2.gradient_certificate import certify
from m3_4br3.audit import audit_transition
from m3_4br2.audit import historical
from m3_4b.monitoring import atomic

def audit(out,point):
 out=Path(out);pid=job(10)[1]['problem_id'];a=load_trajectory(out/'recovery/control/samples'/pid);b=load_trajectory(out/'recovery'/point/'samples'/pid)
 fixed=read(out/'checks'/('recovery_diagnosis_'+point+'.json'));ledger=read(out/'execution_ledger.json');control=next(r for r in ledger['recovery'] if r['kind']=='uninterrupted_control') if any(r.get('kind')=='uninterrupted_control' for r in ledger['recovery']) else ledger['recovery'][0]
 recovered=next(r for r in ledger['recovery'] if r.get('point')==point)
 if not control['validation']['passed'] or not recovered['validation']['passed'] or not recovered['committed_prefix_hashes_preserved']:raise ValueError('Independent replay/exact provenance prerequisite')
 allowed={'historical_sum_excess','volume_excess'}
 bad=[r['field'] for r in fixed['native_fields'] if not r['passed'] and r['field'] not in allowed]
 if bad:raise ValueError('Non-residual native comparison failed '+str(bad))
 if a.spec!=b.spec or a.metadata['source_id']!=b.metadata['source_id'] or a.metadata['termination_reason']!=b.metadata['termination_reason']:raise ValueError('Problem/source/stopping differs')
 if a.arrays['previous_change'][0]!=0 or b.arrays['previous_change'][0]!=0:raise ValueError('Initial sentinel')
 for key in allowed:np.testing.assert_array_equal(a.arrays[key][0],b.arrays[key][0])
 rows=[];allpass=True
 for t in range(len(a.transitions)):
  sa={k:np.asarray(a.arrays[k][t]) for k in ('x','rho','g_rho','g_x')};re={k:np.asarray(b.arrays[k][t]) for k in ('x','rho','g_rho','g_x')}
  sa.update(action=np.asarray(a.arrays['x'][t+1]),diagnostic=a.transitions[t]['diagnostic'],selected_multiplier=a.transitions[t]['selected_multiplier']);re.update(action=np.asarray(b.arrays['x'][t+1]),diagnostic=b.transitions[t]['diagnostic'],selected_multiplier=b.transitions[t]['selected_multiplier'])
  for tr,body in ((a,sa),(b,re)):
   for key in allowed:
    if float(tr.arrays[key][t+1])!=tr.transitions[t]['diagnostic'][key]:raise ValueError('Native residual not exact incoming OC diagnostic')
   body['gradient_certificate']=certify(tr.problem,body['rho'],tr.arrays['u'][t],body['g_rho'],body['g_x'])
  primary=dict(independent_control=dict(passed=True),independent_restart=dict(passed=True),unchanged_nonresidual_native_fields=dict(passed=True),exact_committed_prefix=dict(passed=True),OC=historical(re['action'],sa['action'],1e-8),incoming=historical(float(np.max(abs(re['action']-re['x']))),float(a.arrays['previous_change'][t+1]),1e-12))
  result=audit_transition(sa,re,a.problem.volume_fraction,primary,change_evidence=dict(transition=t,indices=[t,t+1],saved_change=float(a.arrays['previous_change'][t+1]),outgoing_change=float(a.arrays['update_change'][t]),threshold=a.problem.tolerance))
  rows.append(dict(transition=t,**result));allpass=allpass and result['passed']
 payload=dict(passed=allpass,policy='Unmodified OC v2/v3, exactly incoming-associated native residuals; unchanged non-residual pointwise criteria',fixed_native_result=fixed['all_fixed_native_comparisons_passed'],fixed_failed_fields=[r['field'] for r in fixed['native_fields'] if not r['passed']],independent_histories_passed=True,states=a.metadata['state_count'],transitions=len(rows),all_v2_v3_prerequisites_passed=allpass,new_tolerance_added=False,new_FEM=0,new_OC_searches=0,results=rows)
 atomic(out/'checks'/('coupled_recovery_'+point+'.json'),json.loads(json.dumps(payload,default=lambda x:x.item())))
 print(point,allpass,[(r['transition'],r['reason']) for r in rows if not r['passed']][:5],flush=True)
if __name__=='__main__':audit(sys.argv[1],sys.argv[2])
