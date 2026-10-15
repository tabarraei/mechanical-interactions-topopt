import json
from dataclasses import replace
import numpy as np
import pytest
from topopt_utils.fea.robust_oc import update,trial,OCFailure,OPTIONS
from topopt_utils.fea.oc_pilot import make_case,suite,run,replay_selected,load_case,topology
from topopt_utils.fea.consistent_trajectory import save_trajectory,load_trajectory
from topopt_utils.topopt_2d import __oc_update_design__

@pytest.mark.parametrize('target',[.3,.4,.5,.6])
def test_nonuniform_bracket_and_feasible_endpoint(target):
    x=np.array([target-.08,target+.06,target+.02]);g=np.array([-3.,-.2,-1.])
    y,d=update(x,g,target)
    assert d['upper_feasible'] and d['strict_feasible'] and y.mean()<=target
    assert d['lower_above_budget'] and d['bracket_width']<=1e-8
    assert np.max(abs(y-x))<=.1+1e-14
    assert np.all((y>=.001)&(y<=1))


def test_monotone_multiplier_with_zero_positive_and_bound_gradients():
    x=np.array([.001,.2,.5,.9,1.]);g=np.array([-1.,0.,1.,-1e-300,-1e300])
    volumes=[trial(x,g,l).sum() for l in np.r_[0.,np.logspace(-300,300,80)]]
    assert np.all(np.diff(volumes)<=1e-14)

@pytest.mark.parametrize('g',[np.zeros(3),np.array([-1e-300,0.,1e-300])])
def test_zero_nearzero_sensitivities(g):
    y,d=update(np.array([.2,.5,.8]),g,.5)
    assert np.isfinite(y).all() and d['strict_feasible']
    if not g.any():assert d['root_status']=='slack_volume_no_root'

@pytest.mark.parametrize('option',list(OPTIONS))
def test_options_metadata_and_baseline_exact(option):
    x=np.array([.35,.5,.65]);g=np.array([-2.,-.2,-3.]);y,d=update(x,g,.5,option)
    assert d['option']==option and d['version']=='m3.3a.oc.v1'
    if option=='baseline_v1':np.testing.assert_array_equal(y,__oc_update_design__(x,g,.5))

@pytest.mark.parametrize('bad',[np.array([np.nan,-1.]),np.array([np.inf,-1.])])
def test_nonfinite_gradients_fail(bad):
    with pytest.raises(OCFailure,match='finite'):update([.5,.5],bad,.5)


def test_unattainable_move_constraint_fails():
    with pytest.raises(OCFailure,match='unattainable'):update([.9,.9],[-1.,-1.],.3)


def test_fixed_upper_not_unconditionally_feasible():
    with pytest.raises(OCFailure,match='upper multiplier'):update([.5,.5],[-1e300,-1e300],.5)


def test_box_and_bad_target_rejected():
    with pytest.raises(OCFailure):update([0.,.5],[-1.,-1.],.5)
    with pytest.raises(OCFailure):update([.5,.5],[-1.,-1.],0.)


def test_feasible_upper_strict_both_reductions():
    rng=np.random.default_rng(871)
    for n in (3,72,1152):
        x=rng.uniform(.35,.65,n);g=-rng.uniform(.01,20,n);y,d=update(x,g,.5)
        assert sum(y)<=.5*n and y.mean()<=.5


def test_pilot_specs_cover_supported_diversity():
    cases=suite(cap=2)
    assert len(cases)==20 and len({s['problem_id'] for p,x,s in cases})==20
    assert {p.volume_fraction for p,x,s in cases}>={.3,.4,.5,.6}
    assert {s['support_kind'] for p,x,s in cases}=={'left_clamp','right_clamp','bottom_clamp','pin_roller'}
    for p,x,s in cases:assert x.mean()<=p.volume_fraction


def test_nonuniform_spec_serialization_and_replay(tmp_path):
    p,x,s=make_case('small',nx=6,ny=3,initial='feasible_perturbation',seed=4,cap=4)
    spec=tmp_path/'spec.json';spec.write_text(json.dumps(s));q,z,t=load_case(spec)
    np.testing.assert_array_equal(x,z)
    a,m,d=run(p,x,'feasible_upper_v1');folder=save_trajectory(tmp_path/'run',p,a,m)
    q,b,n=load_trajectory(folder);rows=replay_selected(q,z,b,n,[0,1,len(b['x'])-1])
    assert all(r['passed'] for r in rows)
    assert m['equilibrium_solves']==m['completed_updates']+1
    assert m['optimizer_option']=='feasible_upper_v1'
    assert len(d)==len(a['update_change'])


def test_wrong_spec_replay_detected():
    p,x,s=make_case('small',nx=4,ny=2,cap=2);a,m,d=run(p,x,'baseline_v1')
    q=replace(p,loads=((p.loads[0][0],0.,-1.),))
    with pytest.raises(ValueError):replay_selected(q,x,a,m,[0])


def test_support_load_path_not_cantilever_specific():
    p,x,s=make_case('bridge',nx=4,ny=2,support='pin_roller',location=('top',.5),direction=(0.,-1.),cap=2)
    assert topology(p,np.ones(p.n),.5)['all_loads_supported']
    empty=np.zeros(p.n);assert not topology(p,empty,.5)['all_loads_supported']
