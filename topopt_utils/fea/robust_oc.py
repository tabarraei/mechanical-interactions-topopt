"""M3.3A versioned OC options; the historical numerical core is untouched."""
from time import perf_counter
import numpy as np
from ..topopt_2d import __oc_update_design__

OPTIONS={
 'baseline_v1':dict(width=1e-4,selection='historical_last_midpoint'),
 'tight_midpoint_v1':dict(width=1e-8,selection='last_midpoint'),
 'feasible_upper_v1':dict(width=1e-8,selection='feasible_upper'),
}

class OCFailure(ValueError):
    def __init__(self,message,**diagnostic):
        super().__init__(message);self.diagnostic=dict(reason=message,**diagnostic)


def checked_inputs(x,g,target):
    x=np.asarray(x,dtype=float);g=np.asarray(g,dtype=float)
    if x.ndim!=1 or not x.size or x.shape!=g.shape or not np.isfinite(x).all() or not np.isfinite(g).all():
        raise OCFailure('OC requires nonempty matching finite vectors')
    if not np.isfinite(target) or not .001<=target<=1:
        raise OCFailure('Target outside design bounds',target=float(target))
    if np.any(x<.001) or np.any(x>1):raise OCFailure('Initial design outside box')
    return x,g


def trial(x,g,multiplier):
    """Same clipped square-root treatment, with an explicit right limit at zero."""
    if multiplier<0 or not np.isfinite(multiplier):raise OCFailure('Invalid multiplier')
    if multiplier==0:
        # zero/positive sensitivities get the same zero square-root factor.
        return np.where(g<0,np.minimum(1.,x+.1),np.maximum(.001,x-.1))
    with np.errstate(over='ignore',under='ignore',divide='ignore',invalid='raise'):
        factor=np.sqrt(np.maximum(-g/multiplier,0.))
        result=np.fmax(.001,np.fmax(x-.1,np.fmin(1.,np.fmin(x+.1,x*factor))))
    if not np.isfinite(result).all():raise OCFailure('Nonfinite OC trial')
    return result


def update(x,g,target,option='feasible_upper_v1'):
    """Fixed historical bracket; fail if move/box or endpoint makes it invalid.

    No adaptive bracket, gradient rescaling, density repair or default mutation.
    Feasibility uses BOTH the historical scalar sum and NumPy mean, so a final
    endpoint cannot pass on one reduction while violating the reported other.
    The small conservative difference is explicitly recorded.
    """
    start=perf_counter();x,g=checked_inputs(x,g,target)
    if option not in OPTIONS:raise OCFailure('Unknown optimizer option',option=option)
    config=OPTIONS[option];lo=0.;hi=1e5;budget=target*len(x)
    def feasible(z):return bool(sum(z)<=budget and z.mean()<=target)
    minimum=np.maximum(.001,x-.1)
    if not feasible(minimum):raise OCFailure('Volume unattainable within move bounds',minimum_mean=float(minimum.mean()),target=target)
    high=trial(x,g,hi);low=trial(x,g,lo)
    if not feasible(high):raise OCFailure('Fixed upper multiplier is infeasible',upper=hi,upper_mean=float(high.mean()),target=target)
    if feasible(low):
        # No active root: all reachable OC trial volumes already satisfy budget.
        if option=='baseline_v1':
            result=__oc_update_design__(x,g,target);count=30;lo=0.;hi=1e5/(2**30)
        else:result=low;count=0;hi=0.
        status='slack_volume_no_root'
    else:
        count=0;status='active_root'
        while hi-lo>config['width']:
            mid=lo+(hi-lo)/2
            if mid==lo or mid==hi:raise OCFailure('Absolute precision exhausted',lower=lo,upper=hi,width=hi-lo)
            result=trial(x,g,mid);count+=1
            # Preserve historical reduction for both midpoint comparators.
            is_feasible=feasible(result) if option=='feasible_upper_v1' else sum(result)<=budget
            if is_feasible:hi=mid
            else:lo=mid
            if count>100:raise OCFailure('Multiplier iteration guard exhausted')
        if option=='baseline_v1':
            truth=__oc_update_design__(x,g,target)
            if not np.array_equal(result,truth):raise OCFailure('Baseline reconstruction disagrees')
            result=truth
        elif option=='feasible_upper_v1':result=trial(x,g,hi)
    high=trial(x,g,hi);low=trial(x,g,lo)
    if option=='feasible_upper_v1' and not feasible(result):raise OCFailure('Final feasible endpoint invariant failed')
    signed=float(result.mean()-target)
    return result,dict(option=option,version='m3.3a.oc.v1',width=config['width'],selection=config['selection'],
        multiplier_lower=float(lo),multiplier_upper=float(hi),bracket_width=float(hi-lo),iterations=count,
        lower_mean=float(low.mean()),upper_mean=float(high.mean()),upper_feasible=feasible(high),lower_above_budget=bool(not feasible(low)),
        root_status=status,volume_excess=signed,strict_feasible=feasible(result),historical_sum_excess=float(sum(result)-budget),
        minimum_reachable_mean=float(minimum.mean()),negative_gradient_count=int(np.sum(g<0)),zero_gradient_count=int(np.sum(g==0)),positive_gradient_count=int(np.sum(g>0)),
        seconds=perf_counter()-start)
