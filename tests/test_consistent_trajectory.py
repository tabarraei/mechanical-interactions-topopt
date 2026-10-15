import copy
import numpy as np
import pytest
from topopt_utils.fea.mechanical_problem import MechanicalProblem,cantilever_problem,DensityMap
from topopt_utils.fea.consistent_trajectory import (generate_trajectory,save_trajectory,load_trajectory,evaluate_state,representative_indices,independent_stiffness)
from topopt_utils.fea.trajectory_validation import validate_trajectory,reconstruct_interaction,derivative_checks
from topopt_utils.fea.solver import FeaSolver
from topopt_utils.topopt_2d import optimize
from topopt_utils.topology_data import TopOptSample

@pytest.fixture(scope='module')
def trajectory():
    p=cantilever_problem(6,3,maxiter=4);a,m=generate_trajectory(p)
    return p,a,m


def test_spec_roundtrip_and_geometry():
    p=cantilever_problem(4,2)
    q=MechanicalProblem.from_dict(p.to_dict())
    assert q.problem_id==p.problem_id
    np.testing.assert_array_equal(p.connectivity[0],[0,1,6,5])
    np.testing.assert_array_equal(p.connectivity[4],[5,6,11,10])
    np.testing.assert_allclose(p.centers[4],[.5,1.5])
    assert p.forces.sum()==1 and len(p.fixed_dofs)==6
    assert p.forces[2*(1*5+4)]==1


@pytest.mark.parametrize('field',['force_vector','connectivity','ordering','material'])
def test_inconsistent_problem_fixture_fails(field):
    d=cantilever_problem(4,2).to_dict()
    if field=='force_vector':d[field][-1]=1.
    elif field=='connectivity':d[field][0],d[field][1]=d[field][1],d[field][0]
    elif field=='ordering':d[field]='Fortran'
    else:d[field]['E0']=2.
    with pytest.raises(ValueError):MechanicalProblem.from_dict(d)


def test_support_and_force_validation():
    with pytest.raises(ValueError):MechanicalProblem(4,2,4.,2.,(0,1),((4,1.,0.),))
    with pytest.raises(ValueError):MechanicalProblem(4,2,4.,2.,(0,1,10,11,20,21),((0,1.,0.),))


def test_filter_jacobian_adjoint_and_boundary():
    p=cantilever_problem(6,3);f=DensityMap(p);rng=np.random.default_rng(4)
    np.testing.assert_allclose(f.forward(np.ones(p.n)),1.)
    x=rng.normal(size=p.n);g=rng.normal(size=p.n)
    assert g@f.forward(x)==pytest.approx(x@f.transpose(g),rel=1e-13)
    jac=np.column_stack([f.forward(v) for v in np.eye(p.n)])
    np.testing.assert_allclose(f.transpose(g),jac.T@g,rtol=1e-13,atol=1e-13)
    assert not np.allclose(jac,jac.T)


def test_saved_state_reconstruction_and_equilibrium(trajectory):
    p,a,m=trajectory;rows,history,summary=validate_trajectory(p,a,m)
    assert summary['passed']
    assert max(r['residual_relative_force'] for r in history)<1e-10
    assert max(r['energy_compliance_relative'] for r in history)<1e-10
    assert all(r['passed'] for r in rows)


def test_numerical_wrong_load_fixture_fails_not_just_hash(trajectory):
    p,a,m=trajectory;d=p.to_dict();node=p.loads[0][0]
    wrong=MechanicalProblem(p.nelx,p.nely,p.width,p.height,p.fixed_dofs,((node,2.,0.),),maxiter=p.maxiter)
    # Direct validation deliberately bypasses integrity hashes. Same shape/order,
    # changed mechanics must fail through reconstructed displacement/compliance.
    with pytest.raises(ValueError,match='numerical reconstruction failed'):
        validate_trajectory(wrong,a,dict(m,problem_id=wrong.problem_id))


def test_iteration_alignment_failure(trajectory):
    p,a,m=trajectory;bad={k:v.copy() for k,v in a.items()};bad['u'][1]=bad['u'][0]
    with pytest.raises(ValueError):validate_trajectory(p,bad,m)


def test_serialization_integrity(tmp_path,trajectory):
    p,a,m=trajectory;folder=save_trajectory(tmp_path/'run',p,a,m)
    q,b,meta=load_trajectory(folder)
    assert q.problem_id==p.problem_id
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    b['rho'][0,0]+=.01;np.savez_compressed(folder/'trajectory.npz',**b)
    with pytest.raises(ValueError,match='integrity'):load_trajectory(folder)


def test_exact_interactions_hvp_and_multi_step_fd(trajectory):
    p,a,m=trajectory;matrix,diag,hessian,info=reconstruct_interaction(p,a,0)
    np.testing.assert_allclose(hessian,2*matrix-np.diag(diag),rtol=0,atol=1e-13)
    gs,hs,skips=derivative_checks(p,a,0,matrix,diag)
    assert set(r['space'] for r in gs)=={'rho','x'}
    assert set(r['family'] for r in hs)=={'random','smooth','localized','trajectory_change'}
    assert len(gs)==len(hs)==32
    assert not skips
    for space in ('rho','x'):
        for family in ('random','smooth','localized','trajectory_change'):
            assert sum(r['passed'] for r in gs if r['space']==space and r['family']==family)>=2
            assert sum(r['passed'] for r in hs if r['space']==space and r['family']==family)>=2


def test_deterministic_reproducibility(trajectory):
    p,a,m=trajectory;b,n=generate_trajectory(p)
    for key in a:np.testing.assert_allclose(a[key],b[key],rtol=1e-11,atol=1e-12,equal_nan=True)
    assert m['termination_reason']==n['termination_reason']


def test_legacy_oc_equivalence_for_same_distributed_mechanics():
    nx,ny=6,3;bc=np.zeros((ny+1,nx+1,2));bc[:,0,:]=1
    sample=TopOptSample(mesh=dict(nelx=nx,nely=ny,width=float(nx),height=float(ny)),load=dict(x=[nx-1],y=[ny//2],fx=[1.],fy=[0.]),boundary_conditions=bc,volume_fraction=.5)
    nodes=((ny//2)*(nx+1)+nx-1,(ny//2)*(nx+1)+nx,(ny//2+1)*(nx+1)+nx-1,(ny//2+1)*(nx+1)+nx)
    p=cantilever_problem(nx,ny,maxiter=4)
    p=MechanicalProblem(nx,ny,float(nx),float(ny),p.fixed_dofs,tuple((n,.25,0.) for n in nodes),maxiter=4)
    a,m=generate_trajectory(p)
    optimize(sample,maxiter=4,tolerance=.001,store_trajectory=True,quiet=True)
    np.testing.assert_allclose(a['rho'][:-1].reshape(-1,ny,nx),sample.density_trajectory,rtol=1e-11,atol=1e-12)


def test_short_selection_and_terminal_alignment(trajectory):
    p,a,m=trajectory
    assert representative_indices(2)==[0,1]
    assert len(a['x'])==m['completed_updates']+1
    assert len(a['update_change'])==len(a['x'])-1
    np.testing.assert_allclose(a['rho'][-1],DensityMap(p).forward(a['x'][-1]))


def test_legacy_mismatch_audit_matches_original_fallback(trajectory):
    from topopt_utils.fea.mechanical_audit import fallback_mismatch_comparison
    from generate_m1_interactions import _make_topopt_sample
    from topopt_utils.fea.q4 import load_to_forces
    p,a,m=trajectory;rows=fallback_mismatch_comparison(p,a['rho'][0])
    assert len({r['problem_id'] for r in rows})==4
    assert rows[0]['compliance']==pytest.approx(a['compliance'][0],rel=1e-12)
    old=_make_topopt_sample(p.nelx,p.nely)
    force=load_to_forces(old.load,p.nelx,p.nely)
    value=FeaSolver().compliance(a['rho'][0],force,old.get_boundary_condition_dofs(),p.nelx,p.nely,old.mesh['width']/p.nelx,old.mesh['height']/p.nely,penal=3.)
    assert rows[-1]['compliance']==pytest.approx(value,rel=1e-12)
