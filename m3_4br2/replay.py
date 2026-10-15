"""Validation-only independent saved trajectory replay; no generation call.
Select --policy historical or m3.4b.oc_diagnostic_audit.v2 explicitly.
Caller must satisfy preflight and use the unchanged monitored envelope.
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from .audit import audit_transition,POLICY
from .gradient_certificate import certify
from .memory import digest,load_trajectory

def json_value(value):
 if isinstance(value,np.generic):return value.item()
 raise TypeError(type(value).__name__)


def run(folder,out,policy,resume_evidence=None):
 from topopt_utils.fea.production_dataset import OwnedSolver
 from topopt_utils.fea.trajectory_dataset import Trajectory
 from topopt_utils.fea.dataset_storage import verify_state,atomic_json
 from topopt_utils.fea.mechanical_problem import DensityMap,array_digest
 from topopt_utils.fea.consistent_trajectory import evaluate_state,independent_stiffness,symmetric_csr,stiffness_probes
 from topopt_utils.fea.trajectory_validation import comparison
 from topopt_utils.fea.robust_oc import update,trial
 from m3_4b.monitoring import memory
 from scipy.sparse import diags
 out=Path(out);out.mkdir(exist_ok=False,parents=True)
 # Trajectory's original integrity checks remain; additionally stream hashes.
 tr=load_trajectory(folder);a=tr.arrays;p=tr.problem;T=tr.metadata['state_count'];calls=0
 hashes={k:digest(tr.folder/e['file']) for k,e in tr.metadata['arrays'].items()}
 comparison(a['x'][0],tr.initial,1e-12,'initial');np.testing.assert_array_equal(a['iteration'],np.arange(T))
 if bool(a['has_incoming_update'][0]) or not np.all(a['has_incoming_update'][1:]):raise ValueError('State timing mask')
 if np.any(a['update_change'][:-1]<p.tolerance):raise ValueError('Missed first stopping point')
 reason='change_tolerance' if a['update_change'][-1]<p.tolerance else 'maxiter'
 if reason!=tr.metadata['termination_reason'] or (reason=='maxiter' and T-1!=p.maxiter):raise ValueError('Terminal timing')
 identity=dict(native_source=tr.metadata['source_id'],policy=policy,problem_id=tr.spec['problem_id'],array_hashes=hashes,validation_sources={str(f.name):digest(f) for f in Path(__file__).parent.glob('*.py')},environment=dict(numpy=np.__version__,python=__import__('sys').version,OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS','solver default16')))
 phases=out/'phases.jsonl';records=out/'transitions.jsonl';states=out/'states.jsonl'
 def phase(t,label):
  with phases.open('a') as f:f.write(json.dumps(dict(state=t,phase=label,**memory()))+'\n')
 start=0
 if resume_evidence is not None:
  folders=resume_evidence if isinstance(resume_evidence,list) else [resume_evidence]
  cached_rows=[]
  for location in folders:
   prior=Path(location);oldrows=[json.loads(line) for line in (prior/'transitions.jsonl').read_text().splitlines()];oldstates={r['state']:r for r in map(json.loads,(prior/'states.jsonl').read_text().splitlines())}
   for old in oldrows:cached_rows.append((prior,old,oldstates[old['transition']]))
  cached_rows.sort(key=lambda item:item[1]['transition'])
  if [item[1]['transition'] for item in cached_rows]!=list(range(len(cached_rows))):raise ValueError('Cached replay must be a unique contiguous prefix')
  for prior,old,oldstate in cached_rows:
   t=old['transition']
   if not oldstate['passed'] or old['identity']['native_source']!=tr.metadata['source_id'] or old['identity']['problem_id']!=tr.spec['problem_id']:raise ValueError('Cached primary/identity mismatch')
   if not old['passed'] and old['reason'] not in ('Diagnostic keys','Componentwise g_rho','Componentwise g_x'):raise ValueError('Cached failure requires separate investigation')
   op=prior/old['operand_file']
   if digest(op)!=old['operand_sha256']:raise ValueError('Cached operand corruption')
   with np.load(op,allow_pickle=False) as cached:
    for key in ('x','rho','g_rho','g_x'):np.testing.assert_array_equal(cached['saved_'+key],a[key][t])
    nextx,d=update(cached['replay_x'],cached['replay_g_x'],p.volume_fraction,'feasible_upper_v1');np.testing.assert_array_equal(nextx,cached['replay_action'])
    sa={k:cached['saved_'+k].copy() for k in ('x','rho','g_rho','g_x','action')};re={k:cached['replay_'+k].copy() for k in ('x','rho','g_rho','g_x','action')}
   with np.load(prior/f'state_operands_{t:05d}.npz') as evidence:
    for side,body in [('saved',sa),('replay',re)]:
     body['gradient_certificate']=certify(p,body['rho'],evidence[side+'_u'],body['g_rho'],body['g_x'])
   sa.update(diagnostic=tr.transitions[t]['diagnostic'],selected_multiplier=tr.transitions[t]['selected_multiplier']);re.update(diagnostic=d,selected_multiplier=d['multiplier_upper'])
   corrected=audit_transition(sa,re,p.volume_fraction,old['primary'],policy,identity)
   corrected.update(transition=t,cached_independent_FEM=True,original_attempt=str(prior),operand_sha256=digest(op))
   with (out/f'cached_transition_{t:05d}.json').open('x') as f:json.dump(corrected,f,default=json_value,allow_nan=False)
   if not corrected['passed']:raise ValueError('Cached transition failed: '+corrected['reason'])
  start=len(cached_rows)
 with OwnedSolver() as solver, records.open('x') as rows,states.open('x') as sr:
  for t in range(start,T):
   phase(t,'start');state={k:v[t] for k,v in a.items() if k!='update_change'};verify_state(p,state)
   truth=evaluate_state(p,a['x'][t],solver);calls+=1;verify_state(p,{**truth,'solver_returned':truth['solver_returned'],'solver_finite':truth['solver_finite']})
   primary={k:comparison(truth[k],a[k][t],1e-9,k,enforce=False) for k in ('rho','u','compliance','g_rho','g_x','k_diagonal','k_probe_actions')}
   C,g=solver.compliance(a['rho'][t],*p.solver_args,sensitivities=True);calls+=1;gx=DensityMap(p).transpose(g)
   primary.update(api_C=comparison(C,a['compliance'][t],1e-9,enforce=False),api_gradient=comparison(g,a['g_rho'][t],1e-9,enforce=False),transpose=comparison(gx,a['g_x'][t],1e-9,enforce=False))
   with (out/f'state_operands_{t:05d}.npz').open('xb') as f:
    np.savez_compressed(f,**{'saved_'+k:a[k][t] for k in ('x','rho','u','compliance','g_rho','g_x')},**{'replay_'+k:truth[k] for k in ('x','rho','u','compliance','g_rho','g_x')},api_g_rho=g,api_g_x=gx)
   physical=independent_stiffness(p,a['rho'][t],solver);free=np.ones(p.ndof);free[list(p.fixed_dofs)]=0
   expected=diags(free)@physical@diags(free)+diags(1-free)
   conn=solver.get_connectivity(p.nelx,p.nely);pat=solver.get_mesh_pattern(p.nelx,p.nely)
   actual=symmetric_csr(solver.assemble_dirichlet(a['rho'][t],pat,solver.get_KE(p.dx,p.dy),conn,np.asarray(p.fixed_dofs)))
   primary['COO']=comparison(expected@stiffness_probes(p),actual@stiffness_probes(p),1e-12,enforce=False)
   sr.write(json.dumps(dict(state=t,passed=all(r['passed'] for r in primary.values()),primary=primary),default=json_value)+'\n');sr.flush();phase(t,'mechanics')
   if not all(r['passed'] for r in primary.values()):raise ValueError('Independent mechanics failed; operands retained')
   if t<T-1:
    saved=tr.transitions[t];nextx,d=update(a['x'][t],gx,p.volume_fraction,'feasible_upper_v1')
    primary['OC']=comparison(nextx,a['x'][t+1],1e-8,'OC',enforce=False);primary['incoming']=comparison(np.max(abs(nextx-a['x'][t])),a['previous_change'][t+1],1e-12,'incoming',enforce=False)
    if array_digest(a['x'][t+1]-a['x'][t])!=saved['delta_digest']:raise ValueError('Delta fingerprint')
    primary['selected_trial']=comparison(trial(a['x'][t],a['g_x'][t],saved['selected_multiplier']),a['x'][t+1],1e-12,'selected upper',enforce=False)
    # Independently reconstruct from saved x/rho, and persist both operands one
    # transition per atomic NPZ. No parent receives arrays.
    sa=dict(x=a['x'][t],rho=a['rho'][t],g_rho=a['g_rho'][t],g_x=a['g_x'][t],action=a['x'][t+1],diagnostic=saved['diagnostic'],selected_multiplier=saved['selected_multiplier'])
    re=dict(x=a['x'][t],rho=truth['rho'],g_rho=g,g_x=gx,action=nextx,diagnostic=d,selected_multiplier=d['multiplier_upper'])
    operand=out/f'operands_{t:05d}.npz';tmp=operand.with_suffix('.tmp')
    with tmp.open('xb') as f:
     np.savez_compressed(f,**{side+'_'+k:v for side,values in [('saved',sa),('replay',re)] for k,v in values.items() if k not in ('diagnostic','selected_multiplier')});f.flush();os.fsync(f.fileno())
    os.replace(tmp,operand)
    sa['gradient_certificate']=certify(p,sa['rho'],a['u'][t],sa['g_rho'],sa['g_x'])
    re['gradient_certificate']=certify(p,re['rho'],truth['u'],re['g_rho'],re['g_x'])
    r=audit_transition(sa,re,p.volume_fraction,primary,policy,identity=identity)
    r.update(transition=t,operand_file=operand.name,operand_sha256=digest(operand));rows.write(json.dumps(r,allow_nan=False,default=json_value)+'\n');rows.flush();os.fsync(rows.fileno());phase(t,'audit')
    if not r['passed']:raise ValueError('Versioned audit failed closed: '+r['reason'])
 atomic_json(out/'summary.json',dict(passed=True,states=T,transitions=T-1,FEM_calls_this_fragment=calls,FEM_calls=calls+2*start,cached_states=start,validation_OC_recomputed_from_cache=start,identity=identity,worker_memory=memory(),native_labels_changed=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--trajectory',required=True);ap.add_argument('--output',required=True);ap.add_argument('--policy',required=True,choices=[POLICY,'historical']);ap.add_argument('--resume-evidence',action='append');args=ap.parse_args();run(args.trajectory,args.output,args.policy,args.resume_evidence)
