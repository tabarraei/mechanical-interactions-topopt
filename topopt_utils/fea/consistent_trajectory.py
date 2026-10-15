"""Aligned OC trajectory recording without modifying the verified optimizer/core."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse import coo_matrix,diags
from .mechanical_problem import MechanicalProblem,DensityMap,array_digest
from .solver import FeaSolver
from .q4 import __simp_stiffness__,__simp_stiffness_sensitivity__
from ..topopt_2d import __oc_update_design__

TRAJECTORY_SCHEMA='m3.1.trajectory.v1'


def symmetric_csr(upper):return upper+upper.T-diags(upper.diagonal())


def independent_stiffness(problem,rho,solver):
    """Independent COO element assembly, retaining support reactions."""
    dofs=problem.element_dofs;ke=solver.get_KE(problem.dx,problem.dy)
    values=(__simp_stiffness__(rho,problem.penal)[:,None]*ke.ravel()).ravel()
    rows=np.repeat(dofs,8,axis=1).ravel();cols=np.tile(dofs,(1,8)).ravel()
    return coo_matrix((values,(rows,cols)),shape=(problem.ndof,problem.ndof)).tocsr()


def stiffness_probes(problem):return np.random.default_rng(701).normal(size=(problem.ndof,2))


def equilibrium_diagnostics(problem,rho,u,solver,k=None):
    if k is None:k=independent_stiffness(problem,rho,solver)
    f=problem.forces;residual=k@u-f
    free=np.ones(problem.ndof,dtype=bool);free[list(problem.fixed_dofs)]=False
    scale=np.linalg.norm(f[free]);absolute=np.linalg.norm(residual[free])
    backward=absolute/(np.linalg.norm(k[free][:,free].data)*np.linalg.norm(u[free])+scale)
    reactions=np.zeros(problem.ndof);reactions[list(problem.fixed_dofs)]=residual[list(problem.fixed_dofs)]
    balance=(f+reactions).reshape(-1,2).sum(axis=0)
    total=(f+reactions).reshape(-1,2);xy=problem.nodes
    moment=np.sum(xy[:,0]*total[:,1]-xy[:,1]*total[:,0])
    energy=float(u@(k@u));c=float(f@u)
    return dict(residual_absolute=float(absolute),residual_relative_force=float(absolute/scale),residual_backward_error=float(backward),
        constrained_displacement_max=float(np.max(abs(u[list(problem.fixed_dofs)]))),energy_compliance_absolute=abs(energy-c),energy_compliance_relative=abs(energy-c)/abs(c),
        force_balance_relative=float(np.linalg.norm(balance)/np.linalg.norm(f)),moment_balance_relative=float(abs(moment)/(np.linalg.norm(f)*np.hypot(problem.width,problem.height))))


def evaluate_state(problem,x,solver=None):
    solver=solver or FeaSolver();mapping=DensityMap(problem);rho=mapping.forward(x)
    u=solver.solve(rho,*problem.solver_args,penal=problem.penal)
    ke=solver.get_KE(problem.dx,problem.dy);ue=u[problem.element_dofs]
    energy=np.einsum('ei,ij,ej->e',ue,ke,ue)
    g=-__simp_stiffness_sensitivity__(rho,problem.penal)*energy
    conn=solver.get_connectivity(problem.nelx,problem.nely);pattern=solver.get_mesh_pattern(problem.nelx,problem.nely,conn=conn)
    upper=solver.assemble_dirichlet(rho,pattern,ke,conn,np.asarray(problem.fixed_dofs),penal=problem.penal)
    k=symmetric_csr(upper)
    return dict(solver_returned=True,solver_finite=bool(np.isfinite(u).all() and np.isfinite(g).all()),x=np.asarray(x).copy(),rho=rho,u=u,compliance=float(problem.forces@u),g_rho=g,g_x=mapping.transpose(g),
        design_volume=float(np.mean(x)),physical_volume=float(np.mean(rho)),k_diagonal=k.diagonal(),k_probe_actions=k@stiffness_probes(problem),k_frobenius=float(np.linalg.norm(k.data)),
        g_rho_norm=float(np.linalg.norm(g)),g_x_norm=float(np.linalg.norm(mapping.transpose(g))),
        **equilibrium_diagnostics(problem,rho,u,solver))


def generate_trajectory(problem,solver=None):
    """Evaluate state before each OC update, plus the terminal updated state.

    Unchanged OC transitions; maximum number of updates is maxiter, and stopping
    tests max(abs(x_next-x)) < tolerance. Saved terminal evaluation adds no update.
    """
    solver=solver or FeaSolver();states=[];changes=[];x=np.full(problem.n,problem.volume_fraction)
    start=perf_counter();previous=np.inf
    for t in range(problem.maxiter+1):
        state=evaluate_state(problem,x,solver);state['iteration']=t;states.append(state)
        if t and previous<problem.tolerance:reason='change_tolerance';break
        if t==problem.maxiter:reason='maxiter';break
        next_x=__oc_update_design__(x,state['g_x'],problem.volume_fraction)
        previous=float(np.max(abs(next_x-x)));changes.append(previous);x=next_x
    arrays={key:np.stack([s[key] for s in states]) for key in states[0]}
    arrays['update_change']=np.asarray(changes)
    arrays['previous_change']=np.r_[np.nan,changes]
    metadata=dict(schema_version=TRAJECTORY_SCHEMA,problem_id=problem.problem_id,termination_reason=reason,completed_updates=len(changes),state_count=len(states),
        state_timing='state t evaluated before outgoing update; terminal x after final OC update is evaluated and has no outgoing update',
        continuation='OC is stateless beyond x; retain completed_updates, previous_change, maxiter and tolerance to preserve remaining budget/termination; cross-library bitwise equality not guaranteed',
        generation_seconds=perf_counter()-start,solver_status='native solve returned finite states; numerical residual/energy checks are evaluated independently during validation')
    return arrays,metadata


def save_trajectory(folder,problem,arrays,metadata,provenance=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    metadata=dict(metadata,provenance=provenance or {},array_shapes={k:list(v.shape) for k,v in arrays.items()},array_fingerprints={k:array_digest(v) for k,v in arrays.items()})
    (folder/'problem.json').write_text(json.dumps(problem.to_dict(),indent=2)+'\n')
    np.savez_compressed(folder/'trajectory.npz',**arrays)
    (folder/'trajectory.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    return folder


def load_trajectory(folder):
    folder=Path(folder);problem=MechanicalProblem.from_dict(json.loads((folder/'problem.json').read_text()))
    metadata=json.loads((folder/'trajectory.json').read_text())
    if metadata.get('schema_version')!=TRAJECTORY_SCHEMA or metadata.get('problem_id')!=problem.problem_id:
        raise ValueError('Trajectory problem identity/schema mismatch')
    with np.load(folder/'trajectory.npz',allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    for k,a in arrays.items():
        if metadata['array_fingerprints'].get(k)!=array_digest(a) or metadata['array_shapes'].get(k)!=list(a.shape):
            raise ValueError(f'Trajectory integrity mismatch: {k}')
    return problem,arrays,metadata


def representative_indices(count):
    if count<1:raise ValueError('No states')
    # Distinct initial, early, middle, late and terminal whenever five states exist.
    return sorted(set([0,min(1,count-1),count//2,max(0,count-2),count-1]))
