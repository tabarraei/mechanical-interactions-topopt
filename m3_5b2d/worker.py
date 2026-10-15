"""Only numerical delegation; no generator or restart/control action exists."""
import sys,time
from .common import *
from .runner import worker_guard
from .accounting import Budget

def main(auth,root,stage,ticket):
 p=worker_guard(auth,root,stage,ticket)
 from topopt_utils.fea.solver import FeaSolver
 from topopt_utils.fea import robust_oc
 from m3_4b.monitoring import memory
 budget=Budget(root,p);solve0=FeaSolver.solve;update0=robust_oc.update;trial0=robust_oc.trial;in_update=0;iterations=[]
 def charge(kind):
  check(memory().get('VmHWM',0)<=1024,'Worker HWM');budget.charge(kind,stage)
 def solve(self,*a,**kw):charge('FEM');return solve0(self,*a,**kw)
 def trial(*a,**kw):charge('bisection_trials' if in_update else 'selected_trials');return trial0(*a,**kw)
 def update(*a,**kw):
  nonlocal in_update
  charge('OC');in_update+=1
  try:
   x,d=update0(*a,**kw);iterations.append(d['iterations']);return x,d
  finally:in_update-=1
 FeaSolver.solve=solve;robust_oc.update=update;robust_oc.trial=trial;root=Path(root)
 try:
  if stage=='replay':
   from m3_4br3.replay import run
   run(INPUT,root/'replay','m3.4b.transition_change_audit.v3')
  elif stage=='supplemental':
   import m3_4br4.checks as checks
   b=p['binding'];spec=read(INPUT/'problem.json');job=dict(problem_id=HISTORICAL,partition='train',family=spec['related_group'])
   # Lookup adaptation only: no original12 identity is overwritten.
   checks.job=lambda index:({'source_id':b['historical_source_id']},job,spec)
   checks.features_fd(root,0,INPUT,'supplemental.json')
  else:raise ValueError('Unknown action')
  check(memory().get('VmHWM',0)<=1024,'Final worker HWM');write(root/f'{stage}_work.json',dict(passed=True,bisection_iterations=iterations,memory=memory(),historical_source_id=p['binding']['historical_source_id']))
 finally:FeaSolver.solve=solve0;robust_oc.update=update0;robust_oc.trial=trial0
if __name__=='__main__':main(*sys.argv[1:])
