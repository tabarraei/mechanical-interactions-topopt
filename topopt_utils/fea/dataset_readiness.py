"""M3.3C completion analysis and training-only sampling; numerical core unchanged."""
from collections import defaultdict,Counter
from time import perf_counter
import numpy as np
from .mechanical_problem import DensityMap
from .convergence_reliability import stationarity_diagnostics
from .robust_oc import update

PROJECTION_VERSION='m3.3c.prediction_projection.v2'

POLICY=dict(version='m3.3c.quality.v1',pg=.01,box_kkt=.01,complementarity=1e-5,relative_compliance_span=1e-3,binary_disagreement=.005,stability_updates=20,change=.001,
            meaning='Provisional approximate-reference screen; not certified optimality. Native metadata stays unmodified; sidecar eligibility requires independent dynamics validation.')

def project_prediction(y,target):
    """Versioned Euclidean box/sum projection, with explicit conservative roundoff.

    No FEM. Bisection80; retain feasible high endpoint. Final density-scale guard
    requires both reported mean and historical Python scalar sum feasibility.
    """
    y=np.asarray(y,float)
    if y.ndim!=1 or not y.size or not np.isfinite(y).all() or not .001<=target<=1:raise ValueError('Invalid prediction projection')
    z=np.clip(y,.001,1);budget=len(y)*target
    if z.mean()<=target and sum(z)<=budget:return z
    lo=0.;hi=max(float(np.max(y-.001)),0.)
    for _ in range(80):
        mid=lo+(hi-lo)/2
        if mid==lo or mid==hi:break
        trial=np.clip(y-mid,.001,1)
        if trial.mean()>target or trial.sum()>budget:lo=mid
        else:hi=mid
    roundoff_step=np.finfo(float).eps*max(1.,float(np.max(abs(y))))*len(y)
    for _ in range(128):
        z=np.clip(y-hi,.001,1)
        if z.mean()<=target and sum(z)<=budget:return z
        # A multiplier ULP can be too small to change any density. Use a
        # documented density-scale shift instead, without any FEM or OC change.
        hi+=roundoff_step;roundoff_step*=2
    raise ValueError('Projection roundoff feasibility guard exhausted')

def predict(x,history,h,method,target):
    if not isinstance(h,int) or h<1:raise ValueError('Positive horizon required')
    if method.endswith('_projected'):
        return project_prediction(predict(x,history,h,method[:-10],target),target)
    if method=='no_change':return np.asarray(x).copy()
    if method=='linear':
        if len(history)<2:raise ValueError('Linear extrapolation needs past state')
        return x+h*(history[-1]-history[-2])
    if method=='two_update':
        if len(history)<3:raise ValueError('Two-update extrapolation needs three states')
        return x+h*(history[-1]-history[-3])/2
    raise ValueError('Unknown prediction method')

def balanced_training_indices(windows,count,seed):
    """IID problem-uniform then window-uniform; only train rows are eligible."""
    if not isinstance(count,int) or count<0:raise ValueError('Nonnegative count required')
    groups=defaultdict(list)
    for i,row in enumerate(windows):
        if row['partition']=='train':groups[row['problem_id']].append(i)
    ids=sorted(groups)
    if not ids:raise ValueError('No training windows')
    rng=np.random.default_rng(seed);chosen=[]
    for _ in range(count):
        pid=ids[int(rng.integers(len(ids)))];chosen.append(groups[pid][int(rng.integers(len(groups[pid])))])
    return chosen

def stage(iteration,cap=500):
    return 'early' if iteration<.1*cap else 'middle' if iteration<.6*cap else 'late'

def metrics(p,y,truth):
    delta=y-truth;rho=DensityMap(p).forward(y);actual=DensityMap(p).forward(truth)
    return dict(x_mae=float(abs(delta).mean()),x_max=float(abs(delta).max()),x_relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(truth),1e-12)),rho_mae=float(abs(rho-actual).mean()),rho_relative_l2=float(np.linalg.norm(rho-actual)/max(np.linalg.norm(actual),1e-12)),binary_rho_disagreement=float(np.mean((rho>.5)!=(actual>.5))),volume_excess=float(y.mean()-p.volume_fraction),volume_feasible=bool(y.mean()<=p.volume_fraction),historical_sum_feasible=bool(sum(y)<=len(y)*p.volume_fraction),box_feasible=bool(np.min(y)>=.001 and np.max(y)<=1))

def quality_flags(summary,policy=POLICY,dynamics_valid=True):
    stationarity=summary['projected_gradient_inf']<=policy['pg'] and summary['box_stationarity_inf']<=policy['box_kkt'] and summary['volume_complementarity_scaled']<=policy['complementarity']
    eligible=dynamics_valid and summary['change_converged'] and summary['volume_feasible'] and stationarity and summary['relative_compliance_span']<=policy['relative_compliance_span'] and summary['binary_disagreement_tail']<=policy['binary_disagreement'] and summary['tail_updates']>=policy['stability_updates']
    return dict(dynamics_valid=bool(dynamics_valid),change_converged=bool(summary['change_converged']),maxiter_capped=bool(summary['maxiter_capped']),stationarity_pass=bool(stationarity),volume_feasible=bool(summary['volume_feasible']),reference_label_eligible=bool(eligible),optimality_certified=False)

def endpoint_analysis(tr):
    p=tr.problem;a=tr.arrays;T=len(a['x']);last=T-1;tail=min(20,last);start=last-tail;C=np.asarray(a['compliance']);x=a['x'][last];g=a['g_x'][last]
    diag=stationarity_diagnostics(x,g,p.volume_fraction);d=np.diff(np.asarray(a['x'][max(0,last-50):]),axis=0)
    norms=np.linalg.norm(d,axis=1);cos=[]
    for j in range(1,len(d)):
        denom=norms[j]*norms[j-1]
        if denom>1e-15:cos.append(float(d[j]@d[j-1]/denom))
    fractions=np.mean(abs(d)>.001,axis=1) if len(d) else np.array([0.])
    summary=dict(label=tr.spec['label'],problem_id=tr.spec['problem_id'],mesh=[p.nelx,p.nely],states=T,updates=last,stop=tr.metadata['termination_reason'],change_converged=tr.metadata['termination_reason']=='change_tolerance',maxiter_capped=tr.metadata['termination_reason']=='maxiter',volume_feasible=bool(np.all(a['volume_excess']<=0) and all(t['diagnostic']['strict_feasible'] for t in tr.transitions)),tail_updates=tail,relative_compliance_span=float(np.ptp(C[start:])/max(abs(C[-1]),1e-12)),binary_disagreement_tail=float(np.mean((a['rho'][start]>.5)!=(a['rho'][last]>.5))),terminal_change=float(a['previous_change'][last]),terminal_compliance=float(C[-1]),initial_compliance=float(C[0]),tail_compliance_progress=float((C[max(0,last-50)]-C[-1])/abs(C[-1])),tail_negative_update_cosine_fraction=float(np.mean(np.asarray(cos)<0)) if cos else 0.,tail_median_update_cosine=float(np.median(cos)) if cos else 0.,tail_mean_changed_cell_fraction=float(np.mean(fractions)),tail_move_limit_update_fraction=float(np.mean(np.max(abs(d),axis=1)>=.099999)) if len(d) else 0.,lower_bound_fraction=float(np.mean(x<=.001+1e-8)),upper_bound_fraction=float(np.mean(x>=1-1e-8)),**diag)
    # Evidence can coexist; do not assert one causal explanation from descriptive flags.
    summary['capped_evidence']=dict(continued_compliance_progress=summary['tail_compliance_progress']>1e-4,opposed_updates=summary['tail_negative_update_cosine_fraction']>.5,localized_changes=summary['tail_mean_changed_cell_fraction']<.1,stable_compliance_poor_stationarity=summary['relative_compliance_span']<1e-3 and diag['projected_gradient_inf']>.01)
    summary.update(quality_flags(summary));return summary

def aggregate(rows,fields):
    result=dict(count=len(rows))
    for field in fields:
        a=np.asarray([r[field] for r in rows],float)
        result[field]=dict(mean=float(np.mean(a)),median=float(np.median(a)),p90=float(np.percentile(a,90)),max=float(np.max(a)))
    return result
