"""Separate validation-only runner; guards precede production imports."""
import os,sys,shutil
from .common import *
from .authorization import live_guard
from .accounting import Budget,verify_resources,parent_guard

def preflight(auth_path,output):
 p=live_guard(auth_path,output);root=Path(output).resolve();check(not root.exists(),'Output overwrite/partial validation: no automatic retries')
 registry=registry_check(p['binding']);check(not (ROOT/'m3_5b2d/validation_attempts'/f'{HISTORICAL}.json').exists(),'Prior validation attempt: no automatic duplicate/retry');parent=root.parent
 while not parent.exists():parent=parent.parent
 check(shutil.disk_usage(parent).free>=p['disk_bytes'],'Disk preflight')
 return p,registry

def worker_guard(auth,root,stage,ticket):
 p=live_guard(auth,root);check(stage in STAGES,'Unauthorized worker stage');t=read(ticket)
 check(Path(ticket).resolve()==Path(root).resolve()/'reservations'/f'{stage}.json','Worker ticket path')
 check(t==dict(authorization_sha256=digest(auth),binding_sha256=canonical(p['binding']),stage=stage,root=str(Path(root).resolve()),parent_pid=os.getppid()),'One-use worker ticket')
 check(read(Path(root)/'authorization_binding.json')==dict(authorization_sha256=digest(auth),binding_sha256=canonical(p['binding'])),'Owned output identity')
 check(read(ROOT/'m3_5b2d/validation_attempts'/f'{HISTORICAL}.json')==dict(output_root=str(Path(root).resolve()),authorization_sha256=digest(auth)),'Validation attempt lease')
 write(Path(str(ticket)+'.started'),dict(started=True));return p

def dispatch(auth_path,output):
 p,registry=preflight(auth_path,output);root=Path(output).resolve();write(ROOT/'m3_5b2d/validation_attempts'/f'{HISTORICAL}.json',dict(output_root=str(root),authorization_sha256=digest(auth_path)));root.mkdir(parents=True,exist_ok=False)
 write(root/'authorization_binding.json',dict(authorization_sha256=digest(auth_path),binding_sha256=canonical(p['binding'])));write(root/'registry_before.json',registry);budget=Budget(root,p)
 # Standard-library original monitor, deferred until actual issued policy/scope.
 from m3_4b.monitoring import run,verify_trace
 try:
  for stage in ('replay','supplemental'):
   live_guard(auth_path,root);registry_check(p['binding']);budget.charge('workers');remaining=budget.remaining();check(remaining>0,'Time exhausted')
   ticket=root/'reservations'/f'{stage}.json';write(ticket,dict(authorization_sha256=digest(auth_path),binding_sha256=canonical(p['binding']),stage=stage,root=str(root),parent_pid=os.getpid()))
   try:
    s=run([sys.executable,'-m','m3_5b2d.worker',str(auth_path),str(root),stage,str(ticket)],root/'monitor',stage,HISTORICAL,{**LIMITS,'worker_timeout_seconds':min(900,remaining)},env=os.environ.copy());verify_trace(s);verify_resources(s)
   finally:
    summary=root/'monitor'/f'{stage}.json'
    if summary.exists():budget.charge('worker_seconds',amount=read(summary)['wall_seconds'])
  verify_binding(p['binding'])
  from .certificate import assemble
  result=assemble(root,p);parent_guard();write(root/'validation_candidate.json',result)
 except Exception as e:
  write(root/'quarantine.json',dict(reason=str(e),numerical_validation_accepted=False,release_eligible=False,inputs_untouched=True));raise
if __name__=='__main__':dispatch(sys.argv[1],sys.argv[2])
