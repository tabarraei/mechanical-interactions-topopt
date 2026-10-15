"""Independent numerical reconstruction and derivative checks for M3.1 states."""
from time import perf_counter
import numpy as np
from scipy.sparse import diags
from scipy.linalg import eigvalsh
from .solver import FeaSolver
from .mechanical_problem import DensityMap
from .consistent_trajectory import (evaluate_state,independent_stiffness,symmetric_csr,stiffness_probes,equilibrium_diagnostics,representative_indices)
from ..topopt_2d import __oc_update_design__

TOLERANCES=dict(reconstruction_rtol=1e-9,filter_rtol=1e-12,stiffness_rtol=1e-12,trajectory_update_rtol=1e-8,
    equilibrium_force_rtol=1e-7,equilibrium_backward_tolerance=1e-12,energy_rtol=1e-8,balance_rtol=1e-7,
    symmetry_rtol=1e-10,psd_factor=100.,hvp_identity_rtol=1e-8,gradient_fd_rtol=3e-5,hvp_fd_rtol=2e-3,
    finite_difference_roundoff_factor=100.,required_acceptable_step_sizes=2,interior_direction_margin=1e-3)


def comparison(actual,reference,rtol,label='',enforce=True):
    a=np.asarray(actual);r=np.asarray(reference)
    if a.shape!=r.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(r)):
        raise ValueError(f'{label}: shape/nonfinite inconsistency')
    absolute=float(np.linalg.norm(a-r));norm=float(np.linalg.norm(r));scale=max(norm,float(np.linalg.norm(a)))
    tolerance=rtol*norm+100*max(a.size,1)*np.finfo(float).eps*scale
    result=dict(absolute_error=absolute,relative_error=absolute/norm if norm else (0. if not absolute else np.nan),reference_norm=norm,absolute_tolerance=tolerance,passed=absolute<=tolerance)
    if enforce and not result['passed']:raise ValueError(f'{label}: numerical reconstruction failed: {result}')
    return result


def oc_budget_diagnostic(x,gradient,target):
    """Inspect the unchanged OC multiplier bracket without altering its update."""
    def trial(multiplier):
        with np.errstate(divide='ignore',invalid='ignore'):
            factor=np.sqrt(np.maximum(-gradient/multiplier,0.))
        return np.fmax(.001,np.fmax(x-.1,np.fmin(1.,np.fmin(x+.1,x*factor))))
    lower=0.;upper=1e5
    while upper-lower>1e-4:
        midpoint=(lower+upper)/2
        if trial(midpoint).sum()>target*len(x):lower=midpoint
        else:upper=midpoint
    volume_high=float(trial(lower).mean());volume_low=float(trial(upper).mean())
    return dict(oc_lambda_lower=lower,oc_lambda_upper=upper,oc_volume_lower=volume_low,oc_volume_upper=volume_high,
        oc_volume_uncertainty=volume_high-volume_low,oc_bracket_contains_target=volume_low<=target<=volume_high)


def validate_trajectory(problem,arrays,metadata,solver=None):
    solver=solver or FeaSolver();mapping=DensityMap(problem);rows=[];history=[]
    count=len(arrays['x'])
    if count<2:raise ValueError('A complete OC trajectory requires an initial and terminal state')
    if metadata.get('problem_id')!=problem.problem_id:raise ValueError('Problem identity mismatch')
    if not np.isnan(arrays['previous_change'][0]) or not np.isfinite(arrays['previous_change'][1:]).all():raise ValueError('Invalid incoming transition metrics')
    shapes=dict(x=(count,problem.n),rho=(count,problem.n),u=(count,problem.ndof),g_rho=(count,problem.n),g_x=(count,problem.n),
        solver_returned=(count,),solver_finite=(count,),k_diagonal=(count,problem.ndof),k_probe_actions=(count,problem.ndof,2),compliance=(count,),iteration=(count,),update_change=(count-1,),previous_change=(count,))
    for key,shape in shapes.items():
        if arrays[key].shape!=shape:raise ValueError(f'{key}: invalid trajectory shape')
        if key!='previous_change' and not np.all(np.isfinite(arrays[key])):raise ValueError(f'{key}: nonfinite trajectory')
    np.testing.assert_array_equal(arrays['iteration'],np.arange(count))
    if not arrays['solver_returned'].all() or not arrays['solver_finite'].all():raise ValueError('Saved native solver status not successful/finite')
    comparison(arrays['x'][0],np.full(problem.n,problem.volume_fraction),TOLERANCES['filter_rtol'],'initial uniform state')
    for t in range(count):
        x=arrays['x'][t];rho=arrays['rho'][t]
        if np.min(x)<.001-64*np.finfo(float).eps or np.max(x)>1+64*np.finfo(float).eps:
            raise ValueError('Design bound violation')
        recovered=evaluate_state(problem,x,solver)
        def record(quantity,a,b,rtol):
            row=comparison(a,b,rtol,f'iteration {t} {quantity}')
            rows.append(dict(iteration=t,quantity=quantity,**row))
        record('rho_mapping',mapping.forward(x),rho,TOLERANCES['filter_rtol'])
        for key in ('u','compliance','g_rho','g_x','k_diagonal','k_probe_actions','k_frobenius','design_volume','physical_volume','g_rho_norm','g_x_norm'):
            record(key,recovered[key],arrays[key][t],TOLERANCES['reconstruction_rtol'])
        c,g=solver.compliance(rho,*problem.solver_args,penal=problem.penal,sensitivities=True)
        record('solver_compliance',c,arrays['compliance'][t],TOLERANCES['reconstruction_rtol'])
        record('solver_gradient',g,arrays['g_rho'][t],TOLERANCES['reconstruction_rtol'])
        # Independent unconstrained COO assembly -> Dirichlet restriction.
        physical=independent_stiffness(problem,rho,solver)
        free=np.ones(problem.ndof);free[list(problem.fixed_dofs)]=0
        expected=diags(free)@physical@diags(free)+diags(1-free)
        conn=solver.get_connectivity(problem.nelx,problem.nely);pattern=solver.get_mesh_pattern(problem.nelx,problem.nely,conn=conn)
        upper=solver.assemble_dirichlet(rho,pattern,solver.get_KE(problem.dx,problem.dy),conn,np.asarray(problem.fixed_dofs),penal=problem.penal)
        actual=symmetric_csr(upper)
        delta=actual-expected;knorm=np.linalg.norm(expected.data)
        error=np.linalg.norm(delta.data)/knorm
        if error>TOLERANCES['stiffness_rtol']:raise ValueError('Independent stiffness assembly failed')
        rows.append(dict(iteration=t,quantity='independent_stiffness_entries',absolute_error=float(np.linalg.norm(delta.data)),relative_error=float(error),reference_norm=float(knorm),passed=True))
        diagnostic=equilibrium_diagnostics(problem,rho,arrays['u'][t],solver,physical)
        for key,tol in [('residual_relative_force',TOLERANCES['equilibrium_force_rtol']),('residual_backward_error',TOLERANCES['equilibrium_backward_tolerance']),('energy_compliance_relative',TOLERANCES['energy_rtol']),('force_balance_relative',TOLERANCES['balance_rtol']),('moment_balance_relative',TOLERANCES['balance_rtol'])]:
            if diagnostic[key]>tol:raise ValueError(f'Equilibrium/energy failure at {t}: {key}={diagnostic[key]}')
        for key,value in diagnostic.items():
            check=comparison(value,arrays[key][t],0.,enforce=False)
            if key=='residual_backward_error':allowance=TOLERANCES['equilibrium_backward_tolerance']
            elif key=='energy_compliance_absolute':allowance=TOLERANCES['energy_rtol']*abs(c)
            elif key=='energy_compliance_relative':allowance=TOLERANCES['energy_rtol']
            elif key=='constrained_displacement_max':allowance=100*np.finfo(float).eps*np.linalg.norm(arrays['u'][t])
            elif key=='residual_absolute':allowance=TOLERANCES['equilibrium_force_rtol']*np.linalg.norm(problem.forces)
            else:allowance=TOLERANCES['equilibrium_force_rtol']
            check.update(absolute_tolerance=allowance,passed=check['absolute_error']<=allowance)
            if not check['passed']:raise ValueError(f'Saved solver diagnostic inconsistency: {key}')
            rows.append(dict(iteration=t,quantity='saved_'+key,**check))
        history.append(dict(iteration=t,compliance=float(c),design_volume=float(x.mean()),physical_volume=float(rho.mean()),volume_excess=float(x.mean()-problem.volume_fraction),
            previous_change=float(arrays['previous_change'][t]),g_rho_norm=float(np.linalg.norm(g)),g_x_norm=float(np.linalg.norm(mapping.transpose(g))),**diagnostic))
        if t<count-1:
            budget=oc_budget_diagnostic(x,mapping.transpose(g),problem.volume_fraction)
            history[-1].update(budget)
            if not budget['oc_bracket_contains_target']:raise ValueError('OC multiplier bracket does not contain the design volume target')
            if not budget['oc_volume_lower']-1e-12<=float(arrays['x'][t+1].mean())<=budget['oc_volume_upper']+1e-12:raise ValueError('Saved volume outside OC bisection bracket')
            next_x=__oc_update_design__(x,mapping.transpose(g),problem.volume_fraction)
            record('oc_update_independent_gradient',next_x,arrays['x'][t+1],TOLERANCES['trajectory_update_rtol'])
            change=float(np.max(abs(arrays['x'][t+1]-x)))
            record('outgoing_update_change',change,arrays['update_change'][t],TOLERANCES['filter_rtol'])
            record('incoming_update_change',change,arrays['previous_change'][t+1],TOLERANCES['filter_rtol'])
            if t<count-2 and change<problem.tolerance:raise ValueError('Trajectory continued beyond stopping criterion')
    if metadata['state_count']!=count or metadata['completed_updates']!=count-1:raise ValueError('Count/iteration alignment mismatch')
    converged=arrays['update_change'][-1]<problem.tolerance
    reason='change_tolerance' if converged else 'maxiter'
    if metadata['termination_reason']!=reason or (not converged and count-1!=problem.maxiter):
        raise ValueError('Incorrect optimizer termination')
    warnings=[]
    if reason=='maxiter':warnings.append('Optimizer exhausted maxiter; change convergence not achieved')
    if arrays['design_volume'][-1]>problem.volume_fraction:
        warnings.append('Final design volume is above target; exact excess reported, unchanged OC finite-bisection behavior')
    return rows,history,dict(passed=True,termination_reason=reason,completed_updates=count-1,warnings=warnings,tolerances=TOLERANCES)


def reconstruct_interaction(problem,arrays,index,solver=None):
    solver=solver or FeaSolver();rho=arrays['rho'][index];start=perf_counter()
    p,d,h,info=solver.compliance_interaction_matrices(rho,*problem.solver_args,penal=problem.penal,return_info=True)
    comparison(info['u'],arrays['u'][index],TOLERANCES['reconstruction_rtol'],'interaction displacement')
    c_check=comparison(problem.forces@info['u'],arrays['compliance'][index],TOLERANCES['reconstruction_rtol'],'interaction compliance')
    from .q4 import __simp_stiffness_sensitivity__
    ue=info['u'][problem.element_dofs]
    energy=np.einsum('ei,ij,ej->e',ue,solver.get_KE(problem.dx,problem.dy),ue)
    gradient=-__simp_stiffness_sensitivity__(rho,problem.penal)*energy
    g_check=comparison(gradient,arrays['g_rho'][index],TOLERANCES['reconstruction_rtol'],'interaction gradient rho')
    gx_check=comparison(DensityMap(problem).transpose(gradient),arrays['g_x'][index],TOLERANCES['reconstruction_rtol'],'interaction gradient x')
    comparison(2*p-np.diag(d),h,1e-12,'H identity')
    norm=np.linalg.norm(p,'fro');asym=np.linalg.norm(p-p.T,'fro')/norm
    if asym>TOLERANCES['symmetry_rtol']:raise ValueError('Interaction symmetry failed')
    # No eigenvalue clipping or PSD correction; average only accepted tiny asymmetry for eigh.
    values=eigvalsh((p+p.T)/2,driver='evd');tol=TOLERANCES['psd_factor']*problem.n*np.finfo(float).eps*np.max(np.abs(values))
    if values[0]<-tol:raise ValueError('Interaction PSD failed')
    conn=solver.get_connectivity(problem.nelx,problem.nely);pattern=solver.get_mesh_pattern(problem.nelx,problem.nely,conn=conn)
    expected=solver.assemble_dirichlet(rho,pattern,solver.get_KE(problem.dx,problem.dy),conn,np.asarray(problem.fixed_dofs),penal=problem.penal)
    k_check=comparison(info['K'].data,expected.data,1e-12,'interaction K')
    np.testing.assert_array_equal(conn.element_dofs,problem.element_dofs)
    comparison(symmetric_csr(info['K'])@stiffness_probes(problem),arrays['k_probe_actions'][index],TOLERANCES['reconstruction_rtol'],'interaction saved stiffness actions')
    a=info['A'];z=info['Z'];k=symmetric_csr(info['K']);rhs_error=np.linalg.norm(k@z-a)/np.linalg.norm(a)
    if rhs_error>TOLERANCES['equilibrium_force_rtol']:raise ValueError('Interaction RHS equilibrium failed')
    comparison(a.T@z,p,1e-12,'Gram construction')
    return p,d,h,dict(iteration=index,dimension=problem.n,symmetry_relative_error=float(asym),raw_minimum_eigenvalue=float(values[0]),negative_count=int(np.sum(values<-tol)),psd_tolerance=float(tol),
        rhs_equilibrium_relative_error=float(rhs_error),construction_seconds=perf_counter()-start,dense_guard_estimated_bytes=int(8*(2*problem.n**2+2*problem.ndof*problem.n+problem.n)+8*1024**2),
        factorization_count=info['factorizations'],solve_count=info['solves'],
        **{'compliance_'+key:value for key,value in c_check.items()},
        **{'gradient_rho_'+key:value for key,value in g_check.items()},
        **{'gradient_x_'+key:value for key,value in gx_check.items()},
        **{'stiffness_'+key:value for key,value in k_check.items()},
        **{'u_'+key:value for key,value in comparison(info['u'],arrays['u'][index],TOLERANCES['reconstruction_rtol']).items()})


def perturbation_families(problem,arrays,index):
    from generate_large_mesh_scaling import _vector_batch_random,_vector_batch_smooth,_vector_batch_localized
    result=dict(random=_vector_batch_random(problem.n,1,10)[0],smooth=_vector_batch_smooth(problem.nelx,problem.nely,1,14)[0],localized=_vector_batch_localized(problem.nelx,problem.nely,1,16)[0])
    count=len(arrays['x'])
    other=index+1 if index+1<count else index-1
    if other>=0:result['trajectory_change']=arrays['x'][other]-arrays['x'][index]
    return result


def derivative_checks(problem,arrays,index,p,d,solver=None):
    """Both variable spaces, signed feasible directions, four centered FD steps.

    At boundary states directions are explicitly restricted to strictly interior
    entries; initial uniform/early states provide unrestricted full-space checks.
    The trajectory family uses physical differences for rho and design differences
    for x. No optimization or mechanics is rerun to fabricate an interior state.
    """
    solver=solver or FeaSolver();mapping=DensityMap(problem);gradient_rows=[];hvp_rows=[];skipped=[]
    rho=arrays['rho'][index];families=perturbation_families(problem,arrays,index)
    for space in ('rho','x'):
        base=arrays[space][index];margin=np.minimum(base-.001,1-base);interior=margin>TOLERANCES['interior_direction_margin']
        for family,original in families.items():
            original=np.asarray(original).copy()
            if family=='trajectory_change' and space=='rho':
                other=index+1 if index+1<len(arrays['rho']) else index-1
                original=arrays['rho'][other]-rho
            direction=original*interior
            if np.linalg.norm(direction)<1e-14:
                skipped.append(dict(iteration=index,space=space,family=family,reason='No nonzero strictly interior feasible direction',interior_elements=int(interior.sum())));continue
            direction/=np.max(np.abs(direction));nonzero=abs(direction)>0
            safe=float(np.min(margin[nonzero]/abs(direction[nonzero]))*.2)
            largest=min(.01,safe);steps=largest*np.array([1.,.1,.01,.001])
            physical_direction=direction if space=='rho' else mapping.forward(direction)
            analytic=solver.hessian_vector_product(rho,physical_direction,*problem.solver_args,penal=problem.penal)
            modal=2*p@physical_direction-d*physical_direction
            if space=='x':analytic=mapping.transpose(analytic);modal=mapping.transpose(modal)
            identity=comparison(analytic,modal,TOLERANCES['hvp_identity_rtol'],'P/D HVP identity')
            g=arrays['g_rho' if space=='rho' else 'g_x'][index];derivative=float(g@direction)
            accepted_g=accepted_h=0
            for step in steps:
                plus=base+step*direction;minus=base-step*direction
                bound_roundoff=64*np.finfo(float).eps
                if np.min(minus)<.001-bound_roundoff or np.max(minus)>1+bound_roundoff or np.min(plus)<.001-bound_roundoff or np.max(plus)>1+bound_roundoff:
                    raise ValueError('FD direction violates density bounds')
                rp=plus if space=='rho' else mapping.forward(plus);rm=minus if space=='rho' else mapping.forward(minus)
                cp,gp=solver.compliance(rp,*problem.solver_args,penal=problem.penal,sensitivities=True)
                cm,gm=solver.compliance(rm,*problem.solver_args,penal=problem.penal,sensitivities=True)
                if space=='x':gp=mapping.transpose(gp);gm=mapping.transpose(gm)
                fd=(cp-cm)/(2*step);hfd=(gp-gm)/(2*step)
                ga=abs(fd-derivative);roundoff=TOLERANCES['finite_difference_roundoff_factor']*np.finfo(float).eps*max(abs(cp),abs(cm))/step
                gt=TOLERANCES['gradient_fd_rtol']*abs(derivative)+roundoff
                hr=comparison(hfd,analytic,TOLERANCES['hvp_fd_rtol'],enforce=False)
                hroundoff=100*np.finfo(float).eps*max(np.linalg.norm(gp),np.linalg.norm(gm))/step
                hpass=hr['absolute_error']<=hr['absolute_tolerance']+hroundoff
                accepted_g+=int(ga<=gt);accepted_h+=int(hpass)
                base_row=dict(iteration=index,space=space,family=family,step=float(step),interior_elements=int(interior.sum()),restricted_elements=int((~interior).sum()),direction_norm=float(np.linalg.norm(direction)))
                gradient_rows.append(dict(base_row,analytic_directional_derivative=derivative,finite_difference=fd,absolute_error=ga,relative_error=ga/abs(derivative) if abs(derivative)>roundoff else np.nan,
                    near_zero_directional_derivative=abs(derivative)<=roundoff,roundoff_allowance=roundoff,absolute_tolerance=gt,passed=ga<=gt))
                hvp_rows.append({**base_row,**hr,'passed':hpass,'roundoff_allowance':float(hroundoff),'identity_absolute_error':identity['absolute_error'],'identity_relative_error':identity['relative_error']})
            if min(accepted_g,accepted_h)<TOLERANCES['required_acceptable_step_sizes']:
                error=ValueError(f'Multiple-step derivative validation failed: state {index}, {space}, {family}, gradient {accepted_g}/4 HVP {accepted_h}/4')
                error.gradient_rows=gradient_rows;error.hvp_rows=hvp_rows;error.skipped=skipped
                raise error
    return gradient_rows,hvp_rows,skipped
