"""Numerical adapter, unreachable without an issued sidecar AND one-use ticket.
All formulas remain in the frozen original implementation; wrappers only count.
"""
import sys,time
from .common import *
from .runner import worker_gate
from .budget import Budget

def main(auth_path,kind,stage,pid,root,task,ticket):
 s,a=worker_gate(auth_path,kind,int(stage),pid,root,task,ticket)
 # The gate above imports no production module or numerical dependency.
 from topopt_utils.fea.solver import FeaSolver
 from topopt_utils.fea import robust_oc
 from topopt_utils.fea.production_dataset import OwnedSolver
 from m3_4b.monitoring import memory
 root=Path(root);budget=Budget(root,a,digest(auth_path));counts=dict(FEM=0,OC=0);original_solve=FeaSolver.solve;original_update=robust_oc.update
 def solve(self,*args,**kwargs):budget.charge('FEM');counts['FEM']+=1;return original_solve(self,*args,**kwargs)
 def update(*args,**kwargs):budget.charge('OC');counts['OC']+=1;return original_update(*args,**kwargs)
 FeaSolver.solve=solve;robust_oc.update=update;spec=s.specs[pid];job=s.jobs[pid];source_id=s.manifest['source_binding']['original_execution_source_id'];start=time.monotonic();dest=root/'worker_results'/f'{a["authorization_id"]}_{pid}_{task}.json'
 try:
  if task in ('generate','resume'):
   from topopt_utils.fea import dataset_storage as ds
   owned=[]
   class ScopedSolver(OwnedSolver):
    def __init__(self,*args,**kw):super().__init__(*args,**kw);owned.append(self)
   ds.FeaSolver=ScopedSolver;ds.update=update
   original_enrich=ds.enrich_state
   def enrich(p,x,t,previous,solver):
    check(t<a['maximum_work']['max_states_per_id'] and t<=a['maximum_work']['max_updates_per_id'],'Per-ID state/update work bound')
    return original_enrich(p,x,t,previous,solver)
   ds.enrich_state=enrich
   try:result=ds.generate_sample(root/'native',spec,source_id,max_new_updates=a['experiment']['max_new_updates'])
   finally:
    for solver in owned:solver.close()
   result={**result,'folder':str(result['folder'])}
   if result['status']=='completed_reused':raise ValueError('No silent completed reuse during new generation')
  elif task=='replay':
   native_meta=read(root/'native/samples'/pid/'metadata.json');check(native_meta['state_count']<=a['maximum_work']['max_states_per_id'] and native_meta['completed_updates']<=a['maximum_work']['max_updates_per_id'],'Replay state/update work bound')
   from m3_4br3.replay import run
   run(root/'native/samples'/pid,root/'validation'/pid,'m3.4b.transition_change_audit.v3');result=read(root/'validation'/pid/'summary.json')
  elif task=='supplemental':
   import m3_4br4.checks as checks
   # Adapt only original lookup; unchanged feature/FD/matched-history function.
   checks.job=lambda index:({'source_id':source_id},job,spec)
   checks.features_fd(root,0,root/'native/samples'/pid,f'checks/{pid}.json');result=read(root/'checks'/f'{pid}.json')
  else:raise ValueError('Unknown numerical task')
  if 'states' in result:check(result['states']<=a['maximum_work']['max_states_per_id'],'State count work bound')
  if 'transitions' in result:check(result['transitions']<=a['maximum_work']['max_updates_per_id'],'Transition work bound')
  check(memory().get('VmHWM',0)<=1024,'Worker native HWM limit');write(dest,dict(passed=True,result=result,counts=counts,memory=memory(),wall_seconds=time.monotonic()-start,source_id=source_id,authorization_sha256=digest(auth_path)))
 finally:FeaSolver.solve=original_solve;robust_oc.update=original_update
if __name__=='__main__':main(*sys.argv[1:])
