import numpy as np
import pytest
from dataclasses import replace
from topopt_utils.fea.mechanical_problem import cantilever_problem
from topopt_utils.fea.consistent_trajectory import save_trajectory,load_trajectory,evaluate_state
from topopt_utils.fea.convergence_reliability import (first_stopping_update,project_box_volume,stationarity_diagnostics,oc_bisection_variant,run_trajectory,final_design_difference,topology_diagnostics)
from topopt_utils.topopt_2d import __oc_update_design__


def test_strict_threshold_and_iteration_number():
    assert first_stopping_update([.1,.01,.009],.01)==3
    assert first_stopping_update([.01,.01],.01) is None
    assert first_stopping_update([.0005,.1],.001)==1


def test_projection_box_volume_known_solution():
    np.testing.assert_allclose(project_box_volume([2.,2.],.5),[.5,.5],atol=1e-14)
    np.testing.assert_allclose(project_box_volume([-.5,.2],.5),[.001,.2])
    p=project_box_volume([2.,-.2,.9],.4)
    assert p.min()>=.001 and p.max()<=1 and p.sum()<=1.2+1e-14


def test_kkt_volume_active_interior():
    d=stationarity_diagnostics([.5,.5],[-2.,-2.],.5)
    assert d['volume_multiplier']==pytest.approx(2.)
    assert d['combined_kkt_residual']<1e-14
    assert d['projected_gradient_inf']<1e-14


def test_kkt_box_signs_and_wrong_stationarity():
    d=stationarity_diagnostics([.001,1.],[2.,-2.],.6)
    assert d['projected_gradient_inf']<1e-14
    wrong=stationarity_diagnostics([.001,1.],[-2.,2.],.6)
    assert wrong['projected_gradient_inf']>.1


def test_kkt_complementarity_and_infeasibility():
    slack=stationarity_diagnostics([.3,.3],[-1.,-1.],.5)
    assert slack['volume_complementarity_scaled']==pytest.approx(.2)
    excess=stationarity_diagnostics([.8,.8],[-1.,-1.],.5)
    assert not excess['strict_volume_feasible'] and excess['volume_violation']==pytest.approx(.3)


def test_stationarity_force_scale_invariant():
    a=stationarity_diagnostics([.2,.6,.9],[-1.,-.2,-2.],.5)
    b=stationarity_diagnostics([.2,.6,.9],np.array([-1.,-.2,-2.])*1e7,.5)
    for key in ('projected_gradient_inf','box_stationarity_rms','combined_kkt_residual'):
        assert a[key]==pytest.approx(b[key],abs=1e-14)


@pytest.mark.parametrize('width',[1e-4,1e-6,1e-8])
def test_bisection_variants(width):
    x=np.array([.3,.5,.7]);g=np.array([-2.,-1.,-.4])
    result,diag=oc_bisection_variant(x,g,.5,width)
    if width==1e-4:np.testing.assert_array_equal(result,__oc_update_design__(x,g,.5))
    feasible,fd=oc_bisection_variant(x,g,.5,width,'feasible_upper')
    assert fd['strict_volume_feasible']
    assert fd['multiplier_upper']-fd['multiplier_lower']<=width
    assert np.max(abs(feasible-x))<=.1+1e-14


@pytest.fixture(scope='module')
def run():
    p=cantilever_problem(4,2,maxiter=8,tolerance=.001);a,m=run_trajectory(p)
    return p,a,m


def test_terminal_evaluation_and_stopping(run):
    p,a,m=run
    np.testing.assert_allclose(a['rho'][-1],evaluate_state(p,a['x'][-1])['rho'])
    assert m['equilibrium_solve_count']==m['completed_updates']+1
    relaxed=replace(p,tolerance=.2)
    b,n=run_trajectory(relaxed)
    assert n['completed_updates']==1 and n['termination_reason']=='change_tolerance'
    np.testing.assert_allclose(b['x'][-1],a['x'][1])


def test_checkpoint_continuation(run):
    p,a,m=run;k=3
    b,n=run_trajectory(p,start_x=a['x'][k],start_iteration=k,previous_change=a['previous_change'][k])
    for name in ('x','rho','compliance','g_rho','g_x'):
        np.testing.assert_allclose(b[name],a[name][k:],rtol=1e-10,atol=1e-12)
    assert n['termination_reason']==m['termination_reason']
    assert b['previous_change'][0]==a['previous_change'][k]


def test_serialization_reconstruction(tmp_path,run):
    p,a,m=run
    folder=save_trajectory(tmp_path/'run',p,a,m)
    q,b,n=load_trajectory(folder)
    assert q.problem_id==p.problem_id
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_allclose(evaluate_state(q,b['x'][-1])['compliance'],b['compliance'][-1])


def test_compliance_and_density_comparison():
    a=dict(compliance=11.,x=np.array([.5,.5]),rho=np.array([.5,.5]))
    b=dict(compliance=10.,x=np.array([.4,.6]),rho=np.array([.4,.6]))
    d=final_design_difference(a,b,(1,2))
    assert d['relative_compliance_difference']==pytest.approx(.1)
    assert d['x_mean_absolute_difference']==pytest.approx(.1)


def test_connectivity_four_neighbors():
    connected=topology_diagnostics([1,1,1,1,0,1],(2,3))
    disconnected=topology_diagnostics([1,0,1,0,0,1],(2,3))
    assert connected['loaded_path'] and connected['component_count']==1
    assert not disconnected['loaded_path'] and disconnected['component_count']==2
