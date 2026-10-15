"""New selected30 runner. All production imports are AFTER issued approval gates."""
import argparse,os,sys
from .common import *
from .authorization import issued_authorization
from .registry import Registry
from .checkpoint import verify_checkpoint
from .budget import Budget

def verify_owned_root(root,p,authorization_sha,ids):
 root=Path(root);owned=root/'authorization_binding.json';check(owned.is_file(),'Attempted output overwrite/unowned root');prior=read(owned)['authorization_sha256']
 if prior!=authorization_sha:
  for pid in ids:
   binding=p['resume_bindings'].get(pid);check(binding and binding['prior_authorization_sha256']==prior and digest(root/'work_ledger.json')==binding['work_ledger_sha256'],'Unauthorized root recovery/history binding')
   stage=root/'native'/('incomplete' if binding['purpose']=='resume' else 'samples')/pid
   check(digest(stage/'journal.json')==binding['journal_sha256'],'Approved recovery/adoption journal changed')
 return prior

def preflight(auth_path,*,kind,stage,ids,output_root):
 selection=Selection();p=issued_authorization(auth_path,selection,kind=kind,stage=stage,ids=ids,output_root=output_root)
 import shutil
 disk_parent=Path(output_root).resolve()
 while not disk_parent.exists():disk_parent=disk_parent.parent
 check(shutil.disk_usage(disk_parent).free>=p['disk_requirement_bytes'],'Disk preflight failed')
 registry=Registry(ROOT,selection);root=Path(output_root).resolve();owned=root/'authorization_binding.json'
 if root.exists():verify_owned_root(root,p,digest(auth_path),ids)
 for pid in ids:
  actions=p['experiment']['actions'];native=root/'native/samples'/pid;incomplete=root/'native/incomplete'/pid
  if 'generate' in actions and not incomplete.exists():check(not native.exists(),'Attempted output overwrite');registry.eligible(pid)
  elif incomplete.exists():
   check('resume' in actions and pid in p['resume_bindings'],'Partial recovery requires explicit authorization binding')
   proof=verify_checkpoint(incomplete,selection.specs[pid],p['production_sources'] and selection.manifest['source_binding']['original_execution_source_id']);check(proof['journal_sha256']==p['resume_bindings'][pid]['journal_sha256'],'Approved partial checkpoint changed');registry.eligible(pid,own_root=root,allow_owned=True)
  else:check(native.exists() and set(actions)=={'adopt'},'Completed artifacts require explicit read-only adoption; no repeated numeric work');registry.eligible(pid,own_root=root,allow_owned=True)
 return selection,p,registry

def worker_gate(auth_path,kind,stage,pid,root,task,ticket):
 s=Selection();a=issued_authorization(auth_path,s,kind=kind,stage=stage,ids=[pid],output_root=root)
 check(task in a['experiment']['actions'],'Unauthorized worker action');t=read(ticket);check(t==dict(authorization_sha256=digest(auth_path),problem_id=pid,task=task,output_root=str(Path(root).resolve())),'Worker reservation ticket binding')
 check(Path(ticket).resolve().is_relative_to(Path(root).resolve()/'reservations'),'Worker ticket path')
 started=Path(str(ticket)+'.started');write(started,dict(started=True,authorization_sha256=digest(auth_path)))
 return s,a

def dispatch(auth_path,*,kind,stage,ids,output_root):
 # No mock backend, permit object, trust-store argument or test override exists.
 s,a,registry=preflight(auth_path,kind=kind,stage=stage,ids=ids,output_root=output_root);root=Path(output_root).resolve();sha=digest(auth_path)
 root.mkdir(parents=True,exist_ok=True)
 binding=root/'authorization_binding.json'
 if binding.exists() and read(binding)['authorization_sha256']!=sha:
  prior=read(binding);write(root/'authorization_history'/f'{prior["authorization_sha256"]}.json',prior);atomic(binding,dict(authorization_sha256=sha,manifest_sha256=PLAN_SHA))
 elif not binding.exists():write(binding,dict(authorization_sha256=sha,manifest_sha256=PLAN_SHA))
 write(root/f'registry_before_{a["authorization_id"]}.json',registry.report())
 budget=Budget(root,a,sha)
 # This module is standard-library monitoring; still intentionally deferred.
 from m3_4b.monitoring import run
 for pid in ids:
  lease=ROOT/'m3_5b2/execution_locks'/f'{pid}.json'
  if not lease.exists():write(lease,dict(problem_id=pid,configuration_equivalence_sha256=equivalent(s.specs[pid]),output_root=str(root),authorization_sha256=sha))
  else:check(read(lease)['output_root']==str(root) and read(lease)['authorization_sha256'] in (sha,a['resume_bindings'].get(pid,{}).get('prior_authorization_sha256')),'Global ID lease conflict')
  final=root/'native/samples'/pid;incomplete=root/'native/incomplete'/pid
  try:
   tasks=[]
   if not final.exists():tasks.append('resume' if incomplete.exists() else 'generate')
   if not final.exists():tasks+=['replay','supplemental']
   for task in tasks:
    check(task in a['experiment']['actions'],'Task not authorized');identifier=f'{a["authorization_id"]}_{pid}_{task}';ticket=root/'reservations'/f'{identifier}.json'
    write(ticket,dict(authorization_sha256=sha,problem_id=pid,task=task,output_root=str(root)));budget.charge('workers');remaining=budget.remaining_seconds();check(remaining>0,'Total authorization time exhausted')
    cmd=[sys.executable,'-m','m3_5b2.worker',str(auth_path),kind,str(stage),pid,str(root),task,str(ticket)]
    try:monitor=run(cmd,root/'monitor',identifier,pid,{**LIMITS,'worker_timeout_seconds':min(900,remaining)},env=os.environ.copy())
    finally:
     summary=root/'monitor'/f'{identifier}.json'
     if summary.exists():budget.charge('worker_seconds',read(summary)['wall_seconds'])
    check(monitor['worker_sampled_hwm_mib']<=1024,'Worker HWM gate')
    if task in ('generate','resume') and not final.exists():
     write(root/f'diagnostic_incomplete_{pid}.json',dict(problem_id=pid,accepted_for_release=False,qualification_only=True,completed=False));return
   # Acceptance is another clean read-only process, never trusting a passed flag.
   budget.charge('workers');identifier=f'{a["authorization_id"]}_{pid}_accept'
   cmd=[sys.executable,'-m','m3_5b2.accept',str(auth_path),kind,str(stage),pid,str(root)]
   try:m=run(cmd,root/'monitor',identifier,pid,{**LIMITS,'worker_timeout_seconds':min(900,budget.remaining_seconds())},env=os.environ.copy())
   finally:
    summary=root/'monitor'/f'{identifier}.json'
    if summary.exists():budget.charge('worker_seconds',read(summary)['wall_seconds'])
  except Exception as exc:
   atomic(root/f'quarantine_{pid}.json',dict(problem_id=pid,folder=str(final if final.exists() else incomplete),error=str(exc),training_eligible=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--authorization',required=True);p.add_argument('--kind',choices=['production','resource_qualification'],required=True);p.add_argument('--stage',type=int,required=True);p.add_argument('--ids',nargs='+',required=True);p.add_argument('--output',required=True);v=p.parse_args();dispatch(v.authorization,kind=v.kind,stage=v.stage,ids=v.ids,output_root=v.output)
