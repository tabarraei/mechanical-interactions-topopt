"""One-shot R2 saved-state replay, gated; never production continuation."""
import json,sys,os
from pathlib import Path
from .preflight import check,LIMITS,OLD,FAILED
from .memory import digest

def launch(out):
 # Exclusive irreversible-in-this-review reservation prevents accidental second
 # real FEM attempt. A failed attempt is retained, not retried automatically.
 prereq=check(out)
 if not prereq['passed']:raise ValueError('R2 replay prerequisites failed')
 reservation=out/'single_replay_reservation.json'
 with reservation.open('x') as f:json.dump(dict(kind='one validation replay only',trajectory=FAILED,policy='m3.4b.oc_diagnostic_audit.v2',source_manifest_sha256=digest(out/'executing_source_manifest.json')),f)
 from m3_4b.monitoring import run
 command=[sys.executable,'-m','m3_4br2.replay','--trajectory',str(OLD/'native/samples'/FAILED),'--output',str(out/'independent_replay'),'--policy','m3.4b.oc_diagnostic_audit.v2']
 summary=run(command,out/'replay_monitor','r2_single_replay',FAILED,LIMITS,env=os.environ.copy())
 detail=json.loads((out/'independent_replay/summary.json').read_text())
 actual_peak=max(summary['worker_sampled_hwm_mib'],detail['worker_memory'].get('VmHWM',0))
 if actual_peak>LIMITS['worker_rss_mib']:raise ValueError('Worker historical high-water exceeded gate; no passing replay resource label')
 (out/'replay_execution.json').write_text(json.dumps(dict(monitor=summary,worker_actual_hwm_mib=actual_peak,summary=detail,resource_pass=True),indent=2));print('Replay completed',detail['states'],detail['transitions'],actual_peak,flush=True)
if __name__=='__main__':launch(Path(sys.argv[1]))
