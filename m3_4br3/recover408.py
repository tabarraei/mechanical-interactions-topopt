"""Recover missing API displacement with exactly one charged validation FEM call."""
import json,sys,os
from pathlib import Path
import numpy as np
from .operands import compliance_with_displacement
from .audit import audit_transition
from m3_4br2.memory import load_trajectory,digest
from m3_4br2.preflight import OLD,FAILED
from m3_4br2.gradient_certificate import certify
from topopt_utils.fea.production_dataset import OwnedSolver
from topopt_utils.fea.mechanical_problem import DensityMap
from topopt_utils.fea.robust_oc import update,trial
from topopt_utils.fea.trajectory_validation import comparison
from topopt_utils.fea.dataset_storage import verify_state
from topopt_utils.fea.consistent_trajectory import equilibrium_diagnostics

def run(out):
 out=Path(out);prior=out/'independent_replay';dest=out/'recovered_state408';dest.mkdir(exist_ok=False)
 tr=load_trajectory(OLD/'native/samples'/FAILED);p=tr.problem;t=408
 old=json.loads((prior/'transitions.jsonl').read_text().splitlines()[-1]);assert old['transition']==t and old['reason']=='Conditioned local gradient certificate'
 state=json.loads((prior/'states.jsonl').read_text().splitlines()[-1]);assert state['state']==t and state['passed']
 with np.load(prior/f'state_operands_{t:05d}.npz') as z:body={k:z[k].copy() for k in z.files}
 with OwnedSolver() as solver:
  (c,g),api_u=compliance_with_displacement(solver,tr.arrays['rho'][t],*p.solver_args,sensitivities=True)
  gx=DensityMap(p).transpose(g);diag=equilibrium_diagnostics(p,tr.arrays['rho'][t],api_u,solver)
  truth={k:body['replay_'+k] for k in ('x','rho','u','compliance','g_rho','g_x')};verify_state(p,{**truth,**diag,'u':api_u,'compliance':c,'g_rho':g,'g_x':gx,'solver_returned':True,'solver_finite':True})
  state['primary'].update(api_u=comparison(api_u,tr.arrays['u'][t],1e-9,enforce=False),api_C=comparison(c,tr.arrays['compliance'][t],1e-9,enforce=False),api_gradient=comparison(g,tr.arrays['g_rho'][t],1e-9,enforce=False),transpose=comparison(gx,tr.arrays['g_x'][t],1e-9,enforce=False))
  if not all(v['passed'] for v in state['primary'].values()):raise ValueError('Recovered API mechanics')
 body.update(api_u=api_u,api_g_rho=g,api_g_x=gx)
 with (dest/f'state_operands_{t:05d}.npz').open('xb') as f:np.savez_compressed(f,**body)
 sa={k:body['saved_'+k] for k in ('x','rho','g_rho','g_x')};re=dict(x=body['replay_x'],rho=body['replay_rho'],g_rho=g,g_x=gx)
 nextx,d=update(re['x'],gx,p.volume_fraction,'feasible_upper_v1');sa.update(action=tr.arrays['x'][t+1],diagnostic=tr.transitions[t]['diagnostic'],selected_multiplier=tr.transitions[t]['selected_multiplier']);re.update(action=nextx,diagnostic=d,selected_multiplier=d['multiplier_upper'])
 sa['gradient_certificate']=certify(p,sa['rho'],body['saved_u'],sa['g_rho'],sa['g_x']);re['gradient_certificate']=certify(p,re['rho'],api_u,g,gx)
 primary=state['primary'].copy();primary.update(OC=comparison(nextx,sa['action'],1e-8,enforce=False),incoming=comparison(float(np.max(abs(nextx-re['x']))),float(tr.arrays['previous_change'][t+1]),1e-12,enforce=False),selected_trial=comparison(trial(sa['x'],sa['g_x'],sa['selected_multiplier']),sa['action'],1e-12,enforce=False))
 result=audit_transition(sa,re,p.volume_fraction,primary,identity=old['identity'],change_evidence=dict(transition=t,indices=[t,t+1],saved_change=float(tr.arrays['previous_change'][t+1]),outgoing_change=float(tr.arrays['update_change'][t]),threshold=p.tolerance))
 op=dest/f'operands_{t:05d}.npz'
 with op.open('xb') as f:np.savez_compressed(f,**{side+'_'+k:values[k] for side,values in [('saved',sa),('replay',re)] for k in ('x','rho','g_rho','g_x','action')})
 result.update(transition=t,operand_file=op.name,operand_sha256=digest(op),recovery_FEM_calls=1,reason_for_repeated_API='Missing actual API displacement invalidated failed cross-solve certificate; one necessary API repeat, no repeat of first independent state evaluation')
 (dest/'transitions.jsonl').write_text(json.dumps(result,default=lambda x:x.item())+'\n');(dest/'states.jsonl').write_text(json.dumps(state,default=lambda x:x.item())+'\n')
 (dest/'recovery.json').write_text(json.dumps(dict(passed=result['passed'],FEM_calls=1,OC_calls=1,source_identity=old['identity'],api_u_comparison=primary['api_u'],gradient_certificate=re['gradient_certificate']),indent=2,default=lambda x:x.item()))
 if not result['passed']:raise ValueError('Recovery failed closed: '+result['reason'])
if __name__=='__main__':run(sys.argv[1])
