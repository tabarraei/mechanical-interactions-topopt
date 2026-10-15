"""Original generation with observation-only counters/fault hooks; no math changes."""
import json,sys,os,time
from pathlib import Path
from .common import job
from m3_4b.monitoring import atomic,memory,digest

def is_fault(point,path,count=None):
 p=Path(path)
 return ((point=='frame_before_journal' and p.name=='00003.npz') or
 (point=='transition_before_journal' and p.parent.name=='updates' and p.name=='00002.json') or
 (point=='committed_update2' and p.name=='journal.json' and count==3) or
 (point=='metadata_before_rename' and p.name=='metadata.json'))

def serializable_answer(answer):
 return {**answer,'folder':str(answer['folder'])}

def execute(out,index,root,attempt,fault=None,reuse=False):
 from topopt_utils.fea import dataset_storage as ds
 from topopt_utils.fea.production_dataset import OwnedSolver
 from m3_4br2.memory import load_trajectory
 out=Path(out);root=Path(root);log=out/'work_counts'/attempt;log.mkdir(parents=True,exist_ok=False)
 m,j,spec=job(index);pid=j['problem_id'];calls=dict(FEM=0,OC=0);solvers=[]
 def counter():atomic(log/'counts.json',dict(**calls,source_id=m['source_id'],problem_id=pid,phase='actual numerical entry',**memory()))
 counter()
 class RecordedSolver(OwnedSolver):
  def __init__(self,*a,**kw):super().__init__(*a,**kw);solvers.append(self)
  def solve(self,*a,**kw):
   if reuse:raise AssertionError('Completed reuse attempted FEM')
   calls['FEM']+=1;counter();return super().solve(*a,**kw)
 ds.FeaSolver=RecordedSolver
 original_update=ds.update
 def update(*a,**kw):
  if reuse:raise AssertionError('Completed reuse attempted OC')
  calls['OC']+=1;counter();return original_update(*a,**kw)
 ds.update=update
 original_frame=ds.atomic_frame;original_json=ds.atomic_json
 def checkpoint(path):
  stage=root/'incomplete'/pid;journal=stage/'journal.json';body=json.loads(journal.read_text()) if journal.exists() else None
  atomic(log/'fault_evidence.json',dict(point=fault,path=str(path),journal=body,journal_sha256=digest(journal) if journal.exists() else None,committed_hashes={e['file']:digest(stage/e['file']) for e in (body or {}).get('states',[])+(body or {}).get('updates',[])},orphan_files=[str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file() and str(p.relative_to(stage)) not in {e['file'] for e in (body or {}).get('states',[])+(body or {}).get('updates',[])}],calls=calls.copy(),source_id=m['source_id']))
  atomic(log/'fault_ready.json',dict(states=[0,1,2]))
  while True:time.sleep(.1)
 def frame(path,state):
  answer=original_frame(path,state)
  if is_fault(fault,path):checkpoint(path)
  return answer
 def aj(path,body):
  answer=original_json(path,body)
  count=len(body.get('states',[])) if isinstance(body,dict) else None
  if is_fault(fault,path,count):checkpoint(path)
  return answer
 ds.atomic_frame=frame;ds.atomic_json=aj
 started=time.monotonic()
 try:answer=ds.generate_sample(root,spec,m['source_id'])
 finally:
  for solver in solvers:solver.close()
 tr=load_trajectory(answer['folder']);counter()
 if reuse and (calls['FEM'] or calls['OC'] or answer['status']!='completed_reused'):raise ValueError('Reuse contract')
 atomic(log/'result.json',dict(**calls,**serializable_answer(answer),problem_id=pid,states=tr.metadata['state_count'],transitions=tr.metadata['completed_updates'],termination_reason=tr.metadata['termination_reason'],source_id=m['source_id'],wall_seconds=time.monotonic()-started,native_bytes=sum(p.stat().st_size for p in tr.folder.rglob('*') if p.is_file()),array_hashes={k:v['sha256'] for k,v in tr.metadata['arrays'].items()},**memory()))
if __name__=='__main__':
 execute(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4],None if sys.argv[5]=='none' else sys.argv[5],sys.argv[6]=='reuse')
