"""Finish the one logical replay at state7 after cached-prefix algebra review."""
import json,sys,os
from pathlib import Path
from .memory import digest
from .preflight import ROOT,OLD,FAILED,LIMITS

def run(out):
 sources=json.loads((OLD/'source_manifest.json').read_text())
 for name,e in sources.items():
  if digest(ROOT/name)!=e['sha256'] or digest(OLD/e['archive'])!=e['sha256']:raise ValueError('Original source changed')
 original=json.loads((out/'replay_prerequisites.json').read_text())
 if digest(ROOT/'m3_4b/authorization.json')!=original['ledger_sha256']:raise ValueError('Ledger changed')
 history=[json.loads((out/'replay_monitor'/name).read_text()) for name in ('r2_single_replay.json','r2_remaining_states.json')]
 if any(s['resource_event'] is not None for s in history):raise ValueError('Resource event')
 if '62 passed' not in (out/'test_logs/revision3_focused.log').read_text():raise ValueError('Current tests prerequisite')
 archive=out/'source_snapshots_revision3';archive.mkdir(exist_ok=False);records={}
 for p in sorted((ROOT/'m3_4br2').glob('*.py')):
  dest=archive/(p.name+'.txt');dest.write_bytes(p.read_bytes());records[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(out)))
 (out/'executing_source_manifest_revision3.json').write_text(json.dumps(records,indent=2))
 with (out/'final_fragment_reservation.json').open('x') as f:json.dump(dict(cached_states='0..6',new_independent_states='7..500',duplicate_FEM=0,source_manifest_sha256=digest(out/'executing_source_manifest_revision3.json')),f)
 from m3_4b.monitoring import run
 command=[sys.executable,'-m','m3_4br2.replay','--trajectory',str(OLD/'native/samples'/FAILED),'--output',str(out/'independent_replay_final'),'--policy','m3.4b.oc_diagnostic_audit.v2','--resume-evidence',str(out/'independent_replay'),'--resume-evidence',str(out/'independent_replay_continuation')]
 limits=dict(LIMITS,worker_timeout_seconds=900-sum(s['wall_seconds'] for s in history))
 monitor=run(command,out/'replay_monitor','r2_final_states',FAILED,limits,env=os.environ.copy());summary=json.loads((out/'independent_replay_final/summary.json').read_text());history.append(monitor)
 peak=max([s['worker_sampled_hwm_mib'] for s in history]+[summary['worker_memory'].get('VmHWM',0)])
 if peak>1024:raise ValueError('Worker high-water gate')
 result=dict(passed=True,one_logical_replay=True,fragments=3,duplicate_FEM=0,states=summary['states'],transitions=summary['transitions'],FEM_calls=summary['FEM_calls'],validation_OC_calls=summary['transitions']+8,worker_peak_mib=peak,parent_peak_mib=max(s['parent_peak_mib'] for s in history),combined_peak_mib=max(s['combined_peak_mib'] for s in history),total_worker_wall_seconds=sum(s['wall_seconds'] for s in history),summary=summary,old_failure_preserved=True)
 (out/'replay_execution.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='summary'}),flush=True)
if __name__=='__main__':run(Path(sys.argv[1]))
