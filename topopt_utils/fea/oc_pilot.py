"""M3.3A controlled specifications, trajectories and independent replay."""
import hashlib,json
from time import perf_counter
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy import ndimage
from scipy.sparse import diags
from .mechanical_problem import MechanicalProblem,DensityMap,array_digest
from .consistent_trajectory import evaluate_state,independent_stiffness,symmetric_csr,stiffness_probes,save_trajectory,load_trajectory
from .convergence_reliability import stationarity_diagnostics
from .trajectory_validation import comparison,TOLERANCES
from .solver import FeaSolver
from .robust_oc import update,OPTIONS


def make_case(label,nx=48,ny=24,support='left_clamp',location=('right',.5),direction=(1.,0.),volume=.5,initial='uniform',width=None,height=None,seed=0,cap=500):
    node=lambda ix,iy:iy*(nx+1)+ix
    if support=='left_clamp':fixed=[2*node(0,y)+a for y in range(ny+1) for a in (0,1)]
    elif support=='right_clamp':fixed=[2*node(nx,y)+a for y in range(ny+1) for a in (0,1)]
    elif support=='bottom_clamp':fixed=[2*node(x,0)+a for x in range(nx+1) for a in (0,1)]
    elif support=='pin_roller':fixed=[0,1,2*nx+1]
    else:raise ValueError('Unsupported support')
    side,fraction=location
    pos={'right':(nx,round(fraction*ny)),'left':(0,round(fraction*ny)),'top':(round(fraction*nx),ny),'bottom':(round(fraction*nx),0)}[side]
    p=MechanicalProblem(nx,ny,float(width or nx),float(height or ny),tuple(sorted(fixed)),((node(*pos),*direction),),volume_fraction=volume,maxiter=cap,seed=seed)
    x=np.full(p.n,volume)
    if initial=='feasible_perturbation':
        rng=np.random.default_rng(seed);xx,yy=np.meshgrid(np.linspace(0,1,nx),np.linspace(0,1,ny))
        perturb=np.sin(2*np.pi*xx)*np.cos(np.pi*yy)+.2*rng.uniform(-1,1,(ny,nx));perturb-=perturb.mean();perturb*=.03/np.max(abs(perturb))
        # Deliberately slight slack initial budget, constructed not repaired later.
        x=(volume-1e-5+perturb).ravel()
    if min(x)<.001 or max(x)>1 or x.mean()>volume:raise ValueError('Infeasible initial design')
    spec=dict(schema_version='m3.3a.problem.v1',label=label,mechanical=p.to_dict(),support_kind=support,
        initial_design=dict(kind=initial,seed=seed,values=x.tolist(),array_digest=array_digest(x),mean=float(x.mean())),
        filter_physical=dict(metric='sqrt((delta_x/(radius*dx))**2+(delta_y/(radius*dy))**2)',semi_axes=[p.filter_radius*p.dx,p.filter_radius*p.dy],units='physical domain length',note='Existing index-space kernel unchanged; rectangular elements give elliptical physical support.'),
        optimizer_options={name:dict(version='m3.3a.oc.v1',**setting,move=.1,x_min=.001,x_max=1.,lambda_lower=0.,lambda_upper=1e5,volume_variable='x',maxiter=cap) for name,setting in OPTIONS.items()})
    spec['problem_id']='m3_3a_'+hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
    return p,x,spec


def suite(cap=500):
    configs=[
      dict(label='axial_small',nx=12,ny=6),dict(label='axial_medium',nx=24,ny=12),dict(label='axial_large'),
      dict(label='right_mid_vertical',direction=(0.,-1.)),
      dict(label='right_quarter_vertical',location=('right',.25),direction=(0.,-1.)),
      dict(label='right_upper_horizontal',location=('right',.75)),
      dict(label='right_corner_oblique',location=('right',1.),direction=(.6,-.8)),
      dict(label='top_mid_vertical',location=('top',.5),direction=(0.,-1.)),
      dict(label='bottom_outer_oblique',location=('bottom',.75),direction=(.6,.8)),
      dict(label='right_clamp_left_vertical',support='right_clamp',location=('left',.5),direction=(0.,-1.)),
      dict(label='bottom_clamp_top_horizontal',support='bottom_clamp',location=('top',.5)),
      dict(label='pin_roller_top_mid',support='pin_roller',location=('top',.5),direction=(0.,-1.)),
      dict(label='pin_roller_top_quarter',support='pin_roller',location=('top',.25),direction=(.6,-.8)),
      dict(label='vertical_volume_03',volume=.3,direction=(0.,-1.)),
      dict(label='oblique_volume_04',volume=.4,direction=(.6,-.8)),
      dict(label='vertical_volume_06',volume=.6,direction=(0.,-1.)),
      dict(label='long_domain_vertical',nx=48,ny=16,direction=(0.,-1.)),
      dict(label='square_domain_oblique',nx=24,ny=24,direction=(.6,-.8)),
      dict(label='rectangular_elements',nx=24,ny=12,width=36.,height=12.,direction=(0.,-1.)),
      dict(label='perturbed_initial_vertical',initial='feasible_perturbation',seed=371,direction=(0.,-1.)),
    ]
    return [make_case(**c,cap=cap) for c in configs]


def run(problem,x0,option,tolerance=.001):
    solver=FeaSolver();x=x0.copy();states=[];changes=[];search=[];eval_seconds=[];update_seconds=[];previous=np.inf;start=perf_counter()
    for t in range(problem.maxiter+1):
        tick=perf_counter();state=evaluate_state(problem,x,solver);eval_seconds.append(perf_counter()-tick);state['iteration']=t
        if not state['solver_finite'] or any(state[k]>TOLERANCES['balance_rtol'] for k in ('residual_relative_force','force_balance_relative','moment_balance_relative')) or state['energy_compliance_relative']>TOLERANCES['energy_rtol']:
            raise ValueError(f'Mechanical diagnostics failed at state {t}')
        states.append(state)
        if t and previous<tolerance:reason='change_tolerance';break
        if t==problem.maxiter:reason='maxiter';break
        tick=perf_counter();next_x,diag=update(x,state['g_x'],problem.volume_fraction,option);update_seconds.append(perf_counter()-tick);search.append(dict(iteration=t,**diag))
        previous=float(np.max(abs(next_x-x)));changes.append(previous);x=next_x
    arrays={k:np.stack([s[k] for s in states]) for k in states[0]}
    arrays.update(update_change=np.asarray(changes),previous_change=np.r_[np.nan,changes],evaluation_seconds=np.asarray(eval_seconds),update_seconds=np.asarray(update_seconds))
    meta=dict(schema_version='m3.1.trajectory.v1',problem_id=problem.problem_id,optimizer_option=option,optimizer_version='m3.3a.oc.v1',optimizer_controls=OPTIONS[option],
        tolerance=tolerance,termination_reason=reason,completed_updates=len(changes),state_count=len(states),equilibrium_solves=len(states),runtime_seconds=perf_counter()-start,
        state_timing='state t is after t OC updates; terminal state additionally evaluated, no outgoing update',warnings=['maxiter: not change converged'] if reason=='maxiter' else [])
    return arrays,meta,search


def load_case(path):
    spec=json.loads(Path(path).read_text());body={k:v for k,v in spec.items() if k!='problem_id'}
    expected='m3_3a_'+hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
    if expected!=spec['problem_id']:raise ValueError('Outer specification fingerprint mismatch')
    p=MechanicalProblem.from_dict(spec['mechanical']);x=np.asarray(spec['initial_design']['values'])
    if array_digest(x)!=spec['initial_design']['array_digest']:raise ValueError('Initial state fingerprint mismatch')
    return p,x,spec


def replay_selected(p,x0,a,meta,indices):
    solver=FeaSolver();rows=[]
    comparison(a['x'][0],x0,1e-12,'specified initial state')
    np.testing.assert_array_equal(a['iteration'],np.arange(len(a['x'])))
    expected_reason='change_tolerance' if a['update_change'][-1]<meta['tolerance'] else 'maxiter'
    if meta['termination_reason']!=expected_reason or (expected_reason=='maxiter' and len(a['x'])-1!=p.maxiter):raise ValueError('Stop metadata incorrect')
    if np.any(a['update_change'][:-1]<meta['tolerance']):raise ValueError('Passed first stopping point')
    for t in sorted(set(indices)):
        truth=evaluate_state(p,a['x'][t],solver)
        for key in ('rho','u','compliance','g_rho','g_x','k_diagonal','k_probe_actions'):
            rows.append(dict(iteration=t,quantity=key,**comparison(truth[key],a[key][t],1e-9,key)))
        # Another API derives sensitivities independently of recorder's formula.
        c,g=solver.compliance(a['rho'][t],*p.solver_args,penal=p.penal,sensitivities=True)
        for key,v,ref in [('independent_C',c,a['compliance'][t]),('independent_g_rho',g,a['g_rho'][t]),('independent_g_x',DensityMap(p).transpose(g),a['g_x'][t])]:
            rows.append(dict(iteration=t,quantity=key,**comparison(v,ref,1e-9,key)))
        physical=independent_stiffness(p,a['rho'][t],solver);free=np.ones(p.ndof);free[list(p.fixed_dofs)]=0;mask=diags(free)
        expected=mask@physical@mask+diags(1-free)
        conn=solver.get_connectivity(p.nelx,p.nely);pattern=solver.get_mesh_pattern(p.nelx,p.nely,conn=conn)
        actual=symmetric_csr(solver.assemble_dirichlet(a['rho'][t],pattern,solver.get_KE(p.dx,p.dy),conn,np.asarray(p.fixed_dofs),penal=p.penal))
        rows.append(dict(iteration=t,quantity='independent_K_actions',**comparison(expected@stiffness_probes(p),actual@stiffness_probes(p),1e-12,'stiffness assembly')))
        if t<len(a['x'])-1:
            next_x,d=update(a['x'][t],DensityMap(p).transpose(g),p.volume_fraction,meta['optimizer_option'])
            rows.append(dict(iteration=t,quantity='OC_transition',**comparison(next_x,a['x'][t+1],1e-8,'OC transition')))
            rows.append(dict(iteration=t,quantity='change_alignment',**comparison(np.max(abs(next_x-a['x'][t])),a['previous_change'][t+1],1e-12,'change')))
    return rows


def topology(p,rho,threshold):
    mask=rho.reshape(p.nely,p.nelx)>threshold;labels,count=ndimage.label(mask,ndimage.generate_binary_structure(2,1))
    supported_nodes={d//2 for d in p.fixed_dofs};support_elements=np.flatnonzero([bool(set(nodes)&supported_nodes) for nodes in p.connectivity])
    support_labels=set(labels.ravel()[support_elements])-{0};paths=[]
    for node,fx,fy in p.loads:
        if fx==0 and fy==0:continue
        elements=np.flatnonzero(np.any(p.connectivity==node,axis=1));load_labels=set(labels.ravel()[elements])-{0};paths.append(bool(support_labels&load_labels))
    sizes=np.bincount(labels.ravel())[1:]
    return dict(components=int(count),all_loads_supported=all(paths),loaded_paths=paths,largest_component_fraction=float(sizes.max()/mask.sum()) if mask.any() else 0.)
