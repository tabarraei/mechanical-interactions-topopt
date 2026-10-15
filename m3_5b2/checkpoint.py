"""Generic original-checkpoint byte/order/source validation, no solver import."""
from .common import *
def verify_checkpoint(stage,spec,source_id):
 stage=Path(stage);check(stage.is_dir() and read(stage/'problem.json')==spec,'Checkpoint specification');j=read(stage/'journal.json')
 check(j['schema_version']=='m3.3b.checkpoint.v1' and j['binding']==dict(problem_id=spec['problem_id'],spec_sha=canonical(spec),source_id=source_id),'Checkpoint source/config binding')
 states=j['states'];updates=j['updates'];check(len(updates)==max(len(states)-1,0) and len(states)<=spec['optimizer']['maxiter']+1,'Checkpoint state/update counts')
 committed=[]
 for group,ext in [('states','npz'),('updates','json')]:
  directory='frames' if group=='states' else 'updates'
  for t,e in enumerate(j[group]):
   check(set(e)=={'file','sha256'} and e['file']==f'{directory}/{t:05d}.{ext}','Checkpoint ordering/path');p=inside(stage,e['file']);check(p.is_file() and digest(p)==e['sha256'],'Checkpoint exact byte integrity');committed.append(e['file'])
 orphans=[str(p.relative_to(stage)) for d in ('frames','updates') for p in (stage/d).glob('*') if str(p.relative_to(stage)) not in committed]
 return dict(passed=True,journal_sha256=digest(stage/'journal.json'),committed_states=len(states),committed_updates=len(updates),orphans=orphans,binding=j['binding'])
