"""One authorized validation continuation, no generation or ledger writes."""
import json,os,sys,shutil
from pathlib import Path
from m3_4br2.memory import digest,load_trajectory
from m3_4br2.preflight import ROOT,OLD,FAILED,LIMITS
from m3_4b.monitoring import run,verify_trace

def launch(out):
 out=Path(out).resolve();r2=ROOT/'m3_4br2/generations/20261014T124054Z'
 regression=json.loads((out/'regression/regression_isolated.json').read_text())
 if not regression['passed']:raise ValueError('Full regression')
 if '24 passed' not in (out/'focused.log').read_text():raise ValueError('Focused tests')
 expected=json.loads((out/'protected_before.json').read_text());changed=[k for k,h in expected.items() if not (ROOT/k).is_file() or digest(ROOT/k)!=h]
 if changed:raise ValueError('Protected changed '+str(changed))
 source=json.loads((OLD/'source_manifest.json').read_text())
 for k,v in source.items():
  if digest(ROOT/k)!=v['sha256'] or digest(OLD/v['archive'])!=v['sha256']:raise ValueError('Source/archive')
 if digest(ROOT/'m3_4b/authorization.json')!=json.loads((r2/'replay_prerequisites.json').read_text())['ledger_sha256']:raise ValueError('Ledger')
 if shutil.disk_usage(out).free<2**30:raise ValueError('Disk')
 if os.environ.get('OPENBLAS_NUM_THREADS')!='1':raise ValueError('Thread environment')
 snapshots=out/('source_snapshots_revision2' if (out/'source_snapshots').exists() else 'source_snapshots');snapshots.mkdir();manifest={}
 for p in list((ROOT/'m3_4br3').glob('*.py'))+list((ROOT/'m3_4br2').glob('*.py')):
  dest=snapshots/(p.parent.name+'_'+p.name+'.txt');shutil.copyfile(p,dest);manifest[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(out)))
 (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
 tr=load_trajectory(OLD/'native/samples'/FAILED)
 cache=out/'cached_state41';cache.mkdir(exist_ok=True)
 from topopt_utils.fea.robust_oc import update,trial
 from topopt_utils.fea.trajectory_validation import comparison
 from topopt_utils.fea.mechanical_problem import array_digest
 import numpy as np
 prior=r2/'independent_replay_final'
 row=json.loads((prior/'transitions.jsonl').read_text().splitlines()[-1]);identity=row['identity']
 state=json.loads((prior/'states.jsonl').read_text().splitlines()[-1]);assert state['state']==41 and state['passed']
 shutil.copyfile(prior/'state_operands_00041.npz',cache/'state_operands_00041.npz')
 with np.load(cache/'state_operands_00041.npz') as z:
  sa={k:z['saved_'+k].copy() for k in ('x','rho','g_rho','g_x')};re=dict(x=z['replay_x'].copy(),rho=z['replay_rho'].copy(),g_rho=z['api_g_rho'].copy(),g_x=z['api_g_x'].copy())
 action,d=update(re['x'],re['g_x'],tr.problem.volume_fraction,'feasible_upper_v1');sa['action']=tr.arrays['x'][42];re['action']=action
 assert array_digest(sa['action']-sa['x'])==tr.transitions[41]['delta_digest']
 primary=state['primary'];primary.update(OC=comparison(action,sa['action'],1e-8,enforce=False),incoming=comparison(np.max(abs(action-re['x'])),tr.arrays['previous_change'][42],1e-12,enforce=False),selected_trial=comparison(trial(sa['x'],sa['g_x'],tr.transitions[41]['selected_multiplier']),sa['action'],1e-12,enforce=False))
 op=cache/'operands_00041.npz'
 if op.exists():
  with np.load(op) as z:
   for side,values in [('saved',sa),('replay',re)]:
    for k,v in values.items():np.testing.assert_array_equal(z[side+'_'+k],v)
 else:
  with op.open('xb') as f:np.savez_compressed(f,**{side+'_'+k:v for side,values in [('saved',sa),('replay',re)] for k,v in values.items()})
 old=dict(transition=41,passed=False,reason='Independent primary evidence failed/missing',identity=identity,primary=primary,operand_file=op.name,operand_sha256=digest(op),recovered_from_independent_state41=True)
 (cache/'transitions.jsonl').write_text(json.dumps(old,default=lambda x:x.item())+'\n');(cache/'states.jsonl').write_text(json.dumps(state,default=lambda x:x.item())+'\n')
 history=[json.loads(p.read_text()) for p in (r2/'replay_monitor').glob('*.json') if p.name in ('r2_single_replay.json','r2_remaining_states.json','r2_final_states.json')]
 for h in history:
  verify_trace(h)
  if h['resource_event']:raise ValueError('Historical resource event')
 remaining=900-sum(h['wall_seconds'] for h in history)
 folders=[r2/x for x in ('independent_replay','independent_replay_continuation','independent_replay_final')]+[cache]
 with (out/'replay_reservation.json').open('x') as f:json.dump(dict(first_new_state=42,last_state=500,cached_states=42,new_states=459,duplicate_FEM=0,limits=LIMITS,history_seconds=900-remaining,source_manifest_sha256=digest(out/'source_manifest.json')),f,indent=2)
 command=[sys.executable,'-m','m3_4br3.replay','--trajectory',str(OLD/'native/samples'/FAILED),'--output',str(out/'independent_replay'),'--policy','m3.4b.transition_change_audit.v3']
 for folder in folders:command+=['--resume-evidence',str(folder)]
 limits=dict(LIMITS,worker_timeout_seconds=remaining)
 result=run(command,out/'monitor','r3_remaining_states',FAILED,limits,env=os.environ.copy())
 (out/'replay_execution.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':launch(sys.argv[1])
