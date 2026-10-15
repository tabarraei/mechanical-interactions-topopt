"""Recover408 API evidence and continue original saved states409..500."""
import json,sys,os,shutil
from pathlib import Path
from m3_4br2.memory import digest
from m3_4br2.preflight import ROOT,OLD,FAILED,LIMITS
from m3_4b.monitoring import run,verify_trace

def launch(out):
 out=Path(out).resolve();r2=ROOT/'m3_4br2/generations/20261014T124054Z'
 if not json.loads((out/'regression_revision2/regression_isolated.json').read_text())['passed']:raise ValueError('Revised regression')
 if '26 passed' not in (out/'focused_revision2.log').read_text():raise ValueError('Revised tests')
 for k,h in json.loads((out/'protected_before.json').read_text()).items():
  if digest(ROOT/k)!=h:raise ValueError('Protected '+k)
 source=json.loads((OLD/'source_manifest.json').read_text())
 for k,v in source.items():
  if digest(ROOT/k)!=v['sha256'] or digest(OLD/v['archive'])!=v['sha256']:raise ValueError('Original source')
 manifests=out/'source_snapshots_revision3';manifests.mkdir();records={}
 for p in (ROOT/'m3_4br3').glob('*.py'):
  dest=manifests/(p.name+'.txt');shutil.copyfile(p,dest);records[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(out)))
 (out/'executing_manifest_revision3.json').write_text(json.dumps(records,indent=2))
 monitors=[json.loads((r2/'replay_monitor'/n).read_text()) for n in ('r2_single_replay.json','r2_remaining_states.json','r2_final_states.json')]+[json.loads((out/'monitor/r3_remaining_states.json').read_text())]
 for m in monitors:
  verify_trace(m)
  if m['resource_event']:raise ValueError('Resource')
 with (out/'recovery408_reservation.json').open('x') as f:json.dump(dict(missing_operand='actual displacement from second API solve',required_repeated_API_FEM=1,new_states='409..500',first_solve_408_repeated=False,reason='Verified source dependency: failed certificate paired second-solve gradient with first-solve displacement; actual second displacement not recorded',manifest_sha256=digest(out/'executing_manifest_revision3.json')),f,indent=2)
 limits=dict(LIMITS,worker_timeout_seconds=900-sum(m['wall_seconds'] for m in monitors))
 m=run([sys.executable,'-m','m3_4br3.recover408',str(out)],out/'monitor','r3_recover408',FAILED,limits,env=os.environ.copy());monitors.append(m)
 # Keep original failed fragment unchanged; create an immutable passing prefix.
 prefix=out/'recovered_prefix42_407';prefix.mkdir()
 prior=out/'independent_replay'
 for filename,index_key in [('transitions.jsonl','transition'),('states.jsonl','state')]:
  with (prefix/filename).open('x') as f:
   for line in (prior/filename).read_text().splitlines():
    row=json.loads(line)
    if row[index_key]<408:
     assert row['passed'];f.write(line+'\n')
 for t in range(42,408):
  for stem in ('operands','state_operands'):os.link(prior/f'{stem}_{t:05d}.npz',prefix/f'{stem}_{t:05d}.npz')
 folders=[r2/x for x in ('independent_replay','independent_replay_continuation','independent_replay_final')]+[out/'cached_state41',prefix,out/'recovered_state408']
 cmd=[sys.executable,'-m','m3_4br3.replay','--trajectory',str(OLD/'native/samples'/FAILED),'--output',str(out/'independent_replay_completion'),'--policy','m3.4b.transition_change_audit.v3']
 for folder in folders:cmd+=['--resume-evidence',str(folder)]
 limits=dict(LIMITS,worker_timeout_seconds=900-sum(m['wall_seconds'] for m in monitors))
 m=run(cmd,out/'monitor','r3_complete409_500',FAILED,limits,env=os.environ.copy())
 (out/'completion_execution.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':launch(sys.argv[1])
