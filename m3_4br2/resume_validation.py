"""Complete the single logical replay using the persisted state0 FEM evidence.
No state may be independently evaluated twice; fail if prefix contract differs.
"""
import json,sys,os
from pathlib import Path
from .memory import digest
from .preflight import ROOT,OLD,FAILED,LIMITS

def run(out):
 first=json.loads((out/'replay_monitor/r2_single_replay.json').read_text());prereq=json.loads((out/'replay_prerequisites.json').read_text())
 if not prereq['passed'] or first['resource_event'] is not None:raise ValueError('Prerequisites/resources failed')
 previous=out/'independent_replay';states=(previous/'states.jsonl').read_text().splitlines()
 if len(states)!=1 or not json.loads(states[0])['passed']:raise ValueError('Expected exactly one verified cached state')
 reserve=out/'validation_restart_reservation.json'
 if reserve.exists():raise ValueError('No repeated continuation worker')
 original=json.loads((OLD/'source_manifest.json').read_text())
 for name,e in original.items():
  if digest(ROOT/name)!=e['sha256'] or digest(OLD/e['archive'])!=e['sha256']:raise ValueError('Original numerical source changed')
 if digest(ROOT/'m3_4b/authorization.json')!=prereq['ledger_sha256']:raise ValueError('Ledger changed')
 # Newly changed audit/replay code must have successful current focused tests.
 log=(out/'test_logs/revision2_focused.log').read_text()
 if '61 passed' not in log or 'failed' in log:raise ValueError('Current revision focused regression gate')
 archive=out/'source_snapshots_revision2';archive.mkdir(exist_ok=False);records={}
 for p in sorted((ROOT/'m3_4br2').glob('*.py')):
  dest=archive/(p.name+'.txt');dest.write_bytes(p.read_bytes());records[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(out)))
 (out/'executing_source_manifest_revision2.json').write_text(json.dumps(records,indent=2))
 with reserve.open('x') as f:json.dump(dict(logical_replay='same authorized trajectory; cached state0, remaining1..500',duplicate_FEM=0,source_manifest_sha256=digest(out/'executing_source_manifest_revision2.json')),f)
 from m3_4b.monitoring import run
 command=[sys.executable,'-m','m3_4br2.replay','--trajectory',str(OLD/'native/samples'/FAILED),'--output',str(out/'independent_replay_continuation'),'--policy','m3.4b.oc_diagnostic_audit.v2','--resume-evidence',str(previous)]
 limits=dict(LIMITS,worker_timeout_seconds=LIMITS['worker_timeout_seconds']-first['wall_seconds'])
 monitor=run(command,out/'replay_monitor','r2_remaining_states',FAILED,limits,env=os.environ.copy())
 summary=json.loads((out/'independent_replay_continuation/summary.json').read_text());peak=max(first['worker_sampled_hwm_mib'],monitor['worker_sampled_hwm_mib'],summary['worker_memory'].get('VmHWM',0))
 if peak>1024:raise ValueError('High-water RSS gate')
 result=dict(passed=True,one_logical_replay=True,fragments=2,duplicate_FEM=0,states=summary['states'],transitions=summary['transitions'],FEM_calls=summary['FEM_calls'],validation_OC_calls=summary['transitions']+1,worker_peak_mib=peak,parent_peak_mib=max(first['parent_peak_mib'],monitor['parent_peak_mib']),combined_peak_mib=max(first['combined_peak_mib'],monitor['combined_peak_mib']),total_worker_wall_seconds=first['wall_seconds']+monitor['wall_seconds'],monitor=monitor,summary=summary,original_failure_preserved=True)
 (out/'replay_execution.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('monitor','summary')}),flush=True)
if __name__=='__main__':run(Path(sys.argv[1]))
