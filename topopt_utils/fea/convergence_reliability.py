"""M3.2 constrained diagnostics and explicit OC experiments; baseline untouched."""
from dataclasses import replace
from time import perf_counter
import numpy as np
from scipy import ndimage
from .solver import FeaSolver
from .consistent_trajectory import evaluate_state
from ..topopt_2d import __oc_update_design__

TOLERANCES=(.001,.005,.01)


def first_stopping_update(changes,tolerance):
    a=np.asarray(changes,dtype=float)
    if tolerance<=0 or a.ndim!=1 or not np.isfinite(a).all():raise ValueError('Invalid stopping data')
    hit=np.flatnonzero(a<tolerance)
    return int(hit[0]+1) if hit.size else None


def project_box_volume(y,target,lower=.001,upper=1.):
    """Euclidean projection onto box and sum(z)<=N*target, not filtered volume."""
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or not np.isfinite(y).all() or not lower<=target<=upper:
        raise ValueError('Invalid projection')
    clipped=np.clip(y,lower,upper);budget=len(y)*target
    if clipped.sum()<=budget:return clipped
    lo=0.;hi=max(0.,float(np.max(y-lower)))
    for _ in range(80):
        mid=(lo+hi)/2
        if np.clip(y-mid,lower,upper).sum()>budget:lo=mid
        else:hi=mid
    return np.clip(y-hi,lower,upper)


def stationarity_diagnostics(x,gradient,target,bound_atol=1e-8):
    """Scaled KKT and projected-gradient diagnostics, including complementarity.

    Constraint is sum(x)-N*target<=0, multiplier lambda>=0. Choose lambda
    minimizing the squared box-stationarity residual. At lower bound keep only
    negative g+lambda; at upper keep only positive; interior retains both signs.
    Scaling is s=max|g|. PG=x-Proj(x-g/s), reported in density units. This is
    not an unconstrained gradient norm and does not include OC move constraints.
    """
    x=np.asarray(x,dtype=float);g=np.asarray(gradient,dtype=float)
    if x.shape!=g.shape or x.ndim!=1 or not np.isfinite(x).all() or not np.isfinite(g).all():raise ValueError('Invalid stationarity inputs')
    lower=x<=.001+bound_atol;upper=x>=1.-bound_atol;interior=~(lower|upper)
    scale=float(np.max(abs(g),initial=0.))
    def residual(lam):
        r=g+lam;r=r.copy();r[lower]=np.minimum(r[lower],0.);r[upper]=np.maximum(r[upper],0.)
        return r
    lo=0.;hi=scale
    if residual(0.).sum()<0:
        for _ in range(80):
            mid=(lo+hi)/2
            if residual(mid).sum()<0:lo=mid
            else:hi=mid
        lam=(lo+hi)/2
    else:lam=0.
    r=residual(lam);excess=float(x.mean()-target)
    pg=x-project_box_volume(x-g/scale if scale else x,target)
    complementarity=lam*abs(excess)/scale if scale else 0.
    bound_violation=float(max(0.,np.max(.001-x),np.max(x-1.)))
    return dict(gradient_scale=scale,volume_multiplier=lam,multiplier_over_gradient_scale=lam/scale if scale else 0.,
        box_stationarity_rms=float(np.linalg.norm(r)/np.sqrt(len(x))/scale) if scale else 0.,box_stationarity_inf=float(np.max(abs(r))/scale) if scale else 0.,
        projected_gradient_rms=float(np.linalg.norm(pg)/np.sqrt(len(x))),projected_gradient_inf=float(np.max(abs(pg))),
        volume_complementarity_scaled=float(complementarity),volume_excess=excess,volume_violation=max(0.,excess),bound_violation=bound_violation,
        dual_feasibility_violation=max(0.,-lam),strict_volume_feasible=excess<=0.,lower_bound_count=int(lower.sum()),upper_bound_count=int(upper.sum()),interior_count=int(interior.sum()),
        combined_kkt_residual=max(float(np.max(abs(r))/scale) if scale else 0.,complementarity,max(0.,excess),bound_violation))


def oc_trial(x,g,multiplier):
    with np.errstate(divide='ignore',invalid='ignore'):
        arg=np.maximum(-g/multiplier,0.)
        return np.fmax(.001,np.fmax(x-.1,np.fmin(1.,np.fmin(x+.1,x*np.sqrt(arg)))))


def oc_bisection_variant(x,g,target,width=1e-4,selection='last_midpoint'):
    """Explicit experimental variant; same baseline formula, bracket and move.

    last_midpoint replicates the baseline at width=1e-4. feasible_upper returns
    the feasible bracket endpoint; no posthoc density rescaling is applied.
    """
    x=np.asarray(x,dtype=float);g=np.asarray(g,dtype=float)
    if x.shape!=g.shape or width<=0 or not np.isfinite(width) or selection not in ('last_midpoint','feasible_upper'):
        raise ValueError('Invalid bisection variant')
    lo=0.;hi=1e5;count=0;start=perf_counter()
    if oc_trial(x,g,hi).mean()>target:raise ValueError('Fixed OC multiplier bracket does not contain a feasible point')
    while hi-lo>width:
        mid=(lo+hi)/2;candidate=oc_trial(x,g,mid);count+=1
        if candidate.sum()>target*len(x):lo=mid
        else:hi=mid
        if count>100:raise RuntimeError('Bisection precision exhausted')
    if selection=='feasible_upper':candidate=oc_trial(x,g,hi)
    return candidate,dict(bisection_iterations=count,multiplier_lower=lo,multiplier_upper=hi,mean_x=float(candidate.mean()),volume_excess=float(candidate.mean()-target),
        strict_volume_feasible=bool(candidate.mean()<=target),bisection_seconds=perf_counter()-start,selection=selection,bisection_width=width)


def run_trajectory(problem,*,start_x=None,start_iteration=0,previous_change=np.inf,stop=True,variant=None,solver=None):
    """M3.1 evaluation/order reused; continuation retains global update budget.

    One equilibrium solve per evaluated state. Timing includes aligned state
    diagnostics and update computation but excludes serialization/plots/I/O.
    """
    if not 0<=start_iteration<=problem.maxiter:raise ValueError('Invalid checkpoint iteration')
    if start_iteration and (start_x is None or not np.isfinite(previous_change)):raise ValueError('Checkpoint requires design and incoming change')
    solver=solver or FeaSolver();x=np.full(problem.n,problem.volume_fraction) if start_x is None else np.asarray(start_x).copy()
    states=[];changes=[];times=[];bis_counts=[];evaluation_times=[];update_times=[];incoming=previous_change;start=perf_counter()
    for iteration in range(start_iteration,problem.maxiter+1):
        tick=perf_counter();state=evaluate_state(problem,x,solver);evaluation_times.append(perf_counter()-tick);state['iteration']=iteration;states.append(state)
        if stop and iteration>0 and previous_change<problem.tolerance:reason='change_tolerance';times.append(perf_counter()-start);break
        if iteration==problem.maxiter:reason='maxiter';times.append(perf_counter()-start);break
        tick=perf_counter()
        if variant is None:
            next_x=__oc_update_design__(x,state['g_x'],problem.volume_fraction);bis_counts.append(30)  # exact fixed bracket/absolute-width baseline
        else:
            next_x,diagnostic=oc_bisection_variant(x,state['g_x'],problem.volume_fraction,**variant);bis_counts.append(diagnostic['bisection_iterations'])
        update_times.append(perf_counter()-tick)
        previous_change=float(np.max(abs(next_x-x)));changes.append(previous_change);x=next_x
        times.append(perf_counter()-start)
    arrays={key:np.stack([s[key] for s in states]) for key in states[0]}
    arrays['update_change']=np.asarray(changes);arrays['previous_change']=np.r_[np.nan if start_iteration==0 else incoming,changes]
    arrays['elapsed_seconds']=np.asarray(times);arrays['evaluation_seconds']=np.asarray(evaluation_times);arrays['update_seconds']=np.asarray(update_times);arrays['bisection_iterations']=np.asarray(bis_counts,dtype=int)
    return arrays,dict(termination_reason=reason,completed_updates=int(arrays['iteration'][-1]),state_count=len(states),start_iteration=start_iteration,
        generation_seconds=perf_counter()-start,equilibrium_solve_count=len(states),variant=variant or 'unchanged_baseline',problem_id=problem.problem_id,schema_version='m3.1.trajectory.v1',
        state_timing='state t before outgoing update; terminal updated x additionally evaluated',continuation='x, absolute iteration, incoming change, saved spec and optional variant; same-platform numerical replay, not cross-BLAS bitwise guarantee')


def final_design_difference(relaxed,reference,shape):
    result={}
    c=float(relaxed['compliance']);refc=float(reference['compliance'])
    result.update(relative_compliance_difference=(c-refc)/refc,relative_absolute_compliance_difference=abs(c-refc)/abs(refc))
    for space in ('x','rho'):
        a=np.asarray(relaxed[space]);b=np.asarray(reference[space]);delta=a-b
        result.update({space+'_mean_absolute_difference':float(np.mean(abs(delta))),space+'_max_absolute_difference':float(np.max(abs(delta))),
            space+'_relative_l2_difference':float(np.linalg.norm(delta)/np.linalg.norm(b)),space+'_binary_disagreement_fraction':float(np.mean((a>.5)!=(b>.5)))})
    ta=topology_diagnostics(relaxed['rho'],shape);tb=topology_diagnostics(reference['rho'],shape)
    result.update(relaxed_components=ta['component_count'],reference_components=tb['component_count'],relaxed_loaded_path=ta['loaded_path'],reference_loaded_path=tb['loaded_path'],
        relaxed_largest_component_fraction=ta['largest_component_fraction'],reference_largest_component_fraction=tb['largest_component_fraction'])
    return result


def topology_diagnostics(rho,shape,threshold=.5):
    """Four-neighbor binary components and connection from loaded element to left.

    The nodal load touches two right-edge elements on the even-row M3.1 family.
    At least one adjacent solid element must share a component with the left edge.
    This is thresholded geometry, not an Ersatz-mechanics disconnection test.
    """
    mask=np.asarray(rho).reshape(shape)>threshold;labels,count=ndimage.label(mask,ndimage.generate_binary_structure(2,1))
    sizes=np.bincount(labels.ravel())[1:];left=set(labels[:,0])-{0};ny,nx=shape
    adjacent={int(labels[max(0,ny//2-1),-1]),int(labels[min(ny-1,ny//2),-1])}-{0}
    return dict(component_count=int(count),largest_component_fraction=float(sizes.max()/mask.sum()) if mask.any() else 0.,
        loaded_path=bool(left&adjacent),solid_fraction=float(mask.mean()))
