"""M3.3B atomic trajectory checkpoints, immutable publication and replay."""
import json,os,shutil
from pathlib import Path
from time import perf_counter
import numpy as np
from .trajectory_dataset import load_problem,Trajectory,canonical_hash,file_hash
from .mechanical_problem import DensityMap,array_digest
from .consistent_trajectory import evaluate_state,equilibrium_diagnostics,independent_stiffness
from .oc_pilot import replay_selected
from .robust_oc import update,trial,OPTIONS
from .convergence_reliability import stationarity_diagnostics
from .trajectory_validation import comparison,TOLERANCES
from .solver import FeaSolver


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_frame(path,state):
    path=Path(path);tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**state);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def verify_state(p,state):
    for key,value in state.items():
        if not np.isfinite(value).all():raise ValueError('Nonfinite state: '+key)
    if not state['solver_finite'] or not state['solver_returned']:raise ValueError('Native solver unsuccessful')
    for key,tol in [('residual_relative_force',1e-7),('residual_backward_error',1e-12),('energy_compliance_relative',1e-8),('force_balance_relative',1e-7),('moment_balance_relative',1e-7)]:
        if state[key]>tol:raise ValueError('Mechanical invariant failed: '+key)
    if max(abs(state['u'][list(p.fixed_dofs)]))>100*np.finfo(float).eps*np.linalg.norm(state['u']):raise ValueError('Boundary condition failure')
    if min(state['x'])<.001 or max(state['x'])>1:raise ValueError('Design bound failure')
    comparison(DensityMap(p).forward(state['x']),state['rho'],1e-12,'density filter')
    comparison(p.forces@state['u'],state['compliance'],1e-12,'compliance')
    if state['x'].mean()>p.volume_fraction:raise ValueError('Design mean volume is infeasible')


def enrich_state(p,x,t,previous,solver):
    state=evaluate_state(p,x,solver);state['iteration']=t;state['previous_change']=previous if previous is not None else 0.;state['has_incoming_update']=previous is not None
    state['volume_excess']=float(x.mean()-p.volume_fraction);state['historical_sum_excess']=float(sum(x)-p.n*p.volume_fraction)
    diag=stationarity_diagnostics(x,state['g_x'],p.volume_fraction)
    for key in ('projected_gradient_inf','box_stationarity_inf','volume_complementarity_scaled'):state[key]=diag[key]
    verify_state(p,state);return state


def generate_sample(root,spec,source_id,max_new_updates=None):
    """Exactly-once committed state transitions; resume never changes past frames.

    Files written before a journal commit are uncommitted and quarantined during
    recovery. A frame and its preceding transition become visible together.
    """
    root=Path(root);pid=spec['problem_id'];final=root/'samples'/pid;stage=root/'incomplete'/pid
    if final.exists():
        loaded=Trajectory(final)
        if loaded.spec!=spec or loaded.metadata['source_id']!=source_id:raise ValueError('Completed sample belongs to another configuration/source')
        return dict(status='completed_reused',folder=final)
    p,x0=load_problem(spec);option=spec['optimizer']['option']
    if option!='feasible_upper_v1':raise ValueError('M3.3B strict-feasibility pipeline requires explicit feasible_upper_v1; baseline remains in M3.3A')
    source_binding=dict(problem_id=pid,spec_sha=canonical_hash(spec),source_id=source_id)
    if not stage.exists():
        (stage/'frames').mkdir(parents=True);(stage/'updates').mkdir();atomic_json(stage/'problem.json',spec)
        journal=dict(schema_version='m3.3b.checkpoint.v1',binding=source_binding,states=[],updates=[])
        atomic_json(stage/'journal.json',journal)
    else:
        journal=json.loads((stage/'journal.json').read_text())
        if journal['binding']!=source_binding or json.loads((stage/'problem.json').read_text())!=spec:raise ValueError('Resume source/configuration mismatch')
        for entry in journal['states']+journal['updates']:
            if file_hash(stage/entry['file'])!=entry['sha256']:raise ValueError('Checkpoint corruption')
        committed={e['file'] for e in journal['states']+journal['updates']}
        for name in ('frames','updates'):
            for file in (stage/name).iterdir():
                if str(file.relative_to(stage)) not in committed:
                    recovery=stage/'quarantined';recovery.mkdir(exist_ok=True)
                    destination=recovery/file.name
                    suffix=0
                    while destination.exists():suffix+=1;destination=recovery/(file.name+f'.{suffix}')
                    file.rename(destination)
    solver=FeaSolver();start=perf_counter();new_updates=0
    def commit_state(state,transition=None):
        t=int(state['iteration']);frame=stage/'frames'/f'{t:05d}.npz';atomic_frame(frame,state)
        entry=dict(file=str(frame.relative_to(stage)),sha256=file_hash(frame))
        if transition is not None:
            file=stage/'updates'/f'{t-1:05d}.json';atomic_json(file,transition)
            journal['updates'].append(dict(file=str(file.relative_to(stage)),sha256=file_hash(file)))
        journal['states'].append(entry);atomic_json(stage/'journal.json',journal)
    if not journal['states']:commit_state(enrich_state(p,x0,0,None,solver))
    while True:
        with np.load(stage/journal['states'][-1]['file'],allow_pickle=False) as data:state={k:data[k] for k in data.files}
        t=int(state['iteration']);incoming=float(state['previous_change']) if t else None
        if t and incoming<p.tolerance:reason='change_tolerance';break
        if t==p.maxiter:reason='maxiter';break
        if max_new_updates is not None and new_updates>=max_new_updates:return dict(status='incomplete',completed_updates=t,new_updates=new_updates,folder=stage)
        next_x,diagnostic=update(state['x'],state['g_x'],p.volume_fraction,option)
        change=float(np.max(abs(next_x-state['x'])));next_state=enrich_state(p,next_x,t+1,change,solver)
        transition=dict(iteration=t,next_iteration=t+1,delta_digest=array_digest(next_x-state['x']),maximum_change=change,valid=True,
            optimizer=spec['optimizer'],selected_multiplier=diagnostic['multiplier_upper'],diagnostic={k:v for k,v in diagnostic.items() if k!='seconds'},oc_seconds=diagnostic['seconds'])
        commit_state(next_state,transition);new_updates+=1
    count=len(journal['states']);array_dir=stage/'arrays';array_dir.mkdir(exist_ok=True)
    # Finalization is restartable: incomplete derived arrays can be replaced;
    # committed frames remain the authoritative immutable checkpoint data.
    with np.load(stage/journal['states'][0]['file'],allow_pickle=False) as first:templates={k:(first[k].shape,first[k].dtype) for k in first.files}
    arrays={}
    for key,(shape,dtype) in templates.items():
        file=array_dir/(key+'.npy');a=np.lib.format.open_memmap(file,mode='w+',dtype=dtype,shape=(count,*shape))
        for t,entry in enumerate(journal['states']):
            with np.load(stage/entry['file'],allow_pickle=False) as frame:a[t]=frame[key]
        a.flush();del a;arrays[key]=dict(file=str(file.relative_to(stage)),sha256=file_hash(file),shape=[count,*shape],dtype=str(dtype))
    transitions=[json.loads((stage/entry['file']).read_text()) for entry in journal['updates']]
    file=array_dir/'update_change.npy';np.save(file,np.asarray([d['maximum_change'] for d in transitions]));arrays['update_change']=dict(file=str(file.relative_to(stage)),sha256=file_hash(file),shape=[count-1],dtype='float64')
    atomic_json(stage/'transitions.json',transitions)
    metadata=dict(schema_version='m3.3b.trajectory.v1',problem_id=pid,mechanical_id=p.problem_id,optimizer_option=option,optimizer_version='m3.3a.oc.v1',tolerance=p.tolerance,
        state_count=count,completed_updates=count-1,equilibrium_solves=count,termination_reason=reason,source_id=source_id,arrays=arrays,
        transitions_sha256=file_hash(stage/'transitions.json'),problem_sha256=file_hash(stage/'problem.json'),last_session_seconds=perf_counter()-start,
        generation_status='completed',optimality_certified=False,reference_label_eligible=False,
        state_timing='state t evaluated after t completed updates; terminal state has no outgoing transition',
        checkpoint_note='Committed frames are not recomputed on resume; source/config must match; orphan files quarantined. Nominal solves=count; crashes before commit can incur extra uncommitted solves, not counted as committed states.')
    atomic_json(stage/'metadata.json',metadata)
    Trajectory(stage)
    final.parent.mkdir(exist_ok=True);os.rename(stage,final)
    # Delete only redundant working checkpoints of this newly published sample.
    # The mmap arrays retain every committed state; transition deltas are bound
    # by fingerprints and reconstruct from adjacent states without duplication.
    shutil.rmtree(final/'frames');shutil.rmtree(final/'updates')
    return dict(status='completed_new',folder=final,new_updates=new_updates)


def validate_sample(trajectory,independent=True):
    p=trajectory.problem;a=trajectory.arrays;meta=trajectory.metadata;T=meta['state_count'];rows=[]
    comparison(a['x'][0],trajectory.initial,1e-12,'specified initial x')
    np.testing.assert_array_equal(a['iteration'],np.arange(T))
    if not bool(a['has_incoming_update'][0])==False or not np.all(a['has_incoming_update'][1:]):raise ValueError('Incoming change mask misaligned')
    if a['previous_change'][0]!=0:raise ValueError('Initial incoming change must use documented zero sentinel')
    for t in range(T):
        state={k:v[t] for k,v in a.items() if k!='update_change'};verify_state(p,state)
        if t<T-1:
            transition=trajectory.transition(t)
            if not transition['valid'] or transition['iteration']!=t or transition['next_iteration']!=t+1 or transition['optimizer']!=trajectory.spec['optimizer']:raise ValueError('Transition metadata inconsistency')
            if array_digest(transition['delta'])!=transition['delta_digest']:raise ValueError('Actual update fingerprint mismatch')
            xnext,d=update(a['x'][t],a['g_x'][t],p.volume_fraction,meta['optimizer_option'])
            rows.append(dict(iteration=t,quantity='OC_saved_gradient',**comparison(xnext,a['x'][t+1],1e-8,'transition reconstruction')))
            comparison(np.max(abs(transition['delta'])),a['update_change'][t],1e-12,'max design change')
            comparison(a['update_change'][t],a['previous_change'][t+1],1e-12,'incoming design change')
            for key,value in d.items():
                if key=='seconds':continue
                saved=transition['diagnostic'][key]
                if isinstance(value,(str,bool)):
                    if value!=saved:raise ValueError('OC diagnostic inconsistency: '+key)
                else:comparison(value,saved,1e-12,'OC diagnostic '+key)
            comparison(trial(a['x'][t],a['g_x'][t],transition['selected_multiplier']),a['x'][t+1],1e-12,'selected multiplier')
        if independent:
            physical=independent_stiffness(p,a['rho'][t],FeaSolver());diag=equilibrium_diagnostics(p,a['rho'][t],a['u'][t],FeaSolver(),physical)
            if max(diag['residual_relative_force'],diag['force_balance_relative'],diag['moment_balance_relative'])>1e-7 or diag['energy_compliance_relative']>1e-8:raise ValueError('Independent equilibrium failure')
    if independent:
        # Existing M3.3A validator independently derives gradients using another
        # solver API and checks COO stiffness actions and transition reconstruction.
        rows+=replay_selected(p,trajectory.initial,a,meta,range(T))
    else:
        expected='change_tolerance' if a['previous_change'][-1]<p.tolerance and T>1 else 'maxiter'
        if meta['termination_reason']!=expected:raise ValueError('Incorrect terminal reason')
    return rows
