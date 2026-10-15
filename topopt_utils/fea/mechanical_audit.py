"""Controlled, same-density audit of the documented legacy fallback mismatch.

This comparison is not a regeneration/reinterpretation of historical trajectories.
It intentionally holds the initial uniform density fixed while changing one problem
component at a time; each case has its own complete specification and identity.
"""
from .mechanical_problem import MechanicalProblem
from .solver import FeaSolver


def fallback_mismatch_comparison(problem,rho,solver=None):
    solver=solver or FeaSolver()
    nx,ny=problem.nelx,problem.nely
    row=ny//2
    expected=((row*(nx+1)+nx,1.,0.),)
    if problem.loads!=expected:raise ValueError('Fallback audit requires the canonical unit axial M1 load')
    nodes=(row*(nx+1)+nx-1,row*(nx+1)+nx,(row+1)*(nx+1)+nx-1,(row+1)*(nx+1)+nx)
    distributed=tuple((node,.25,0.) for node in nodes)
    cases=[('consistent_M1',problem.width,problem.height,problem.loads),
        ('legacy_geometry_only',1.,2.,problem.loads),
        ('legacy_element_load_only',problem.width,problem.height,distributed),
        ('legacy_generated_fallback',1.,2.,distributed)]
    rows=[];reference=None
    for name,width,height,loads in cases:
        case=MechanicalProblem(nx,ny,width,height,problem.fixed_dofs,loads,volume_fraction=problem.volume_fraction,penal=problem.penal,filter_radius=problem.filter_radius,maxiter=problem.maxiter,tolerance=problem.tolerance,seed=problem.seed)
        c,g=solver.compliance(rho,*case.solver_args,penal=case.penal,sensitivities=True)
        if reference is None:reference=c
        rows.append(dict(case=name,problem_id=case.problem_id,width=width,height=height,dx=case.dx,dy=case.dy,
            load_representation='single nodal force' if loads==problem.loads else 'equal quarters on four element nodes',compliance=float(c),
            compliance_ratio_to_consistent=float(c/reference),relative_compliance_change=float((c-reference)/reference)))
    return rows
