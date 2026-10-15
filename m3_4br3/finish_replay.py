"""Resume cached validation after retaining original-versus-adapted gate evidence."""
import json,sys,os,shutil
from pathlib import Path
from m3_4br2.preflight import ROOT,OLD,FAILED,LIMITS
from m3_4br2.memory import digest
from m3_4b.monitoring import run,verify_trace

def run_finish(out):
 out=Path(out).resolve();r2=ROOT/'m3_4br2/generations/20261014T124054Z'
 if '28 passed' not in (out/'focused_revision3.log').read_text():raise ValueError('Current focused')
 reg=json.loads((out/'regression_revision2/regression_isolated.json').read_text());assert reg['passed']
 for k,h in json.loads((out/'protected_before.json').read_text()).items():
  if digest(ROOT/k)!=h:raise ValueError('Protected '+k)
 archive=out/'source_snapshots_revision4';archive.mkdir();manifest={}
 for p in (ROOT/'m3_4br3').glob('*.py'):
  dest=archive/(p.name+'.txt');shutil.copyfile(p,dest);manifest[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(out)))
 (out/'executing_manifest_revision4.json').write_text(json.dumps(manifest,indent=2))
 paths=[r2/'replay_monitor'/n for n in ('r2_single_replay.json','r2_remaining_states.json','r2_final_states.json')]+[out/'monitor'/n for n in ('r3_remaining_states.json','r3_recover408.json','r3_complete409_500.json')]
 history=[json.loads(p.read_text()) for p in paths]
 for m in history:
  verify_trace(m)
  if m['resource_event']:raise ValueError('Resources')
 reservation=out/'completion_retry_reservation.json'
 with reservation.open('x') as f:json.dump(dict(first_new_state=409,cached_states=409,additional_FEM_repeats=0,cache_gate_bug='Prior R3 adapted incoming gate was used instead of retained original incoming comparison. Revalidation now consumes original_primary.',failed_cached_attempt_OC_calls=43,manifest_sha256=digest(out/'executing_manifest_revision4.json')),f,indent=2)
 folders=[r2/x for x in ('independent_replay','independent_replay_continuation','independent_replay_final')]+[out/'cached_state41',out/'recovered_prefix42_407',out/'recovered_state408']
 cmd=[sys.executable,'-m','m3_4br3.replay','--trajectory',str(OLD/'native/samples'/FAILED),'--output',str(out/'independent_replay_completed'),'--policy','m3.4b.transition_change_audit.v3']
 for folder in folders:cmd+=['--resume-evidence',str(folder)]
 monitor=run(cmd,out/'monitor','r3_complete_retry',FAILED,dict(LIMITS,worker_timeout_seconds=900-sum(m['wall_seconds'] for m in history)),env=os.environ.copy())
 (out/'final_completion_execution.json').write_text(json.dumps(monitor,indent=2))
if __name__=='__main__':run_finish(sys.argv[1])
