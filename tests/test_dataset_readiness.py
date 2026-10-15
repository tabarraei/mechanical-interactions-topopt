import numpy as np
import pytest
from topopt_utils.fea.dataset_readiness import project_prediction,predict,balanced_training_indices,quality_flags,POLICY,stage
from topopt_utils.fea.trajectory_dataset import sample_problem,split_problems,validate_splits
from run_m3_3c_completion import configuration

def test_projection_strict_constraints_and_shift_kkt():
    y=np.array([-.3,.4,1.4,1.2]);z=project_prediction(y,.4)
    assert min(z)>=.001 and max(z)<=1 and z.mean()<=.4 and sum(z)<=4*.4
    interior=(z>.001)&(z<1);np.testing.assert_allclose((y-z)[interior],(y-z)[interior][0],atol=1e-14)
    np.testing.assert_array_equal(project_prediction(z,.4),z)

@pytest.mark.parametrize('bad',[np.array([np.nan]),np.empty(0)])
def test_projection_rejects_invalid(bad):
    with pytest.raises(ValueError):project_prediction(bad,.5)

def test_predictions_only_use_available_history():
    history=np.array([[.2,.4],[.3,.4],[.4,.4]])
    np.testing.assert_allclose(predict(history[-1],history,5,'linear',.5),[.9,.4])
    np.testing.assert_allclose(predict(history[-1],history,10,'two_update',.5),[1.4,.4])
    with pytest.raises(ValueError):predict(history[-1],history[-1:],1,'linear',.5)
    assert predict(history[-1],history,10,'two_update_projected',.5).mean()<=.5

def test_balanced_sampler_train_only_deterministic_and_problem_uniform():
    rows=[dict(partition='train',problem_id='a')]*100+[dict(partition='train',problem_id='b')]*2+[dict(partition='test_id',problem_id='c')]*200
    indices=balanced_training_indices(rows,10000,23);assert indices==balanced_training_indices(rows,10000,23)
    assert all(i<102 for i in indices)
    assert .47<np.mean([rows[i]['problem_id']=='a' for i in indices])<.53
    with pytest.raises(ValueError):balanced_training_indices(rows[102:],1,1)

def test_quality_separates_caps_from_reference_and_never_certifies():
    s=dict(projected_gradient_inf=.001,box_stationarity_inf=.001,volume_complementarity_scaled=1e-8,change_converged=False,maxiter_capped=True,volume_feasible=True,relative_compliance_span=1e-5,binary_disagreement_tail=0,tail_updates=20)
    assert quality_flags(s)['dynamics_valid'] and not quality_flags(s)['reference_label_eligible']
    s.update(change_converged=True,maxiter_capped=False);assert quality_flags(s)['reference_label_eligible']
    assert not quality_flags(s,dynamics_valid=False)['reference_label_eligible'];assert not quality_flags(s)['optimality_certified']

def test_stage_uses_configured_cap_not_future_length():
    assert [stage(t) for t in (0,49,50,299,300,500)]==['early','early','middle','middle','late','late']

def test_ten_problem_configuration_and_force_groups():
    c=configuration();specs=[sample_problem(x,c['seed']+i,c['controls'])[2] for i,x in enumerate(c['problems']+c['force_variants'])]
    assert len(c['problems'])==10 and len(specs)==12
    splits=split_problems(specs,577,['bottom_clamp']);validate_splits(specs,splits)
    assignment=splits['assignments'];assert assignment[specs[0]['problem_id']]==assignment[specs[-1]['problem_id']]==assignment[specs[-2]['problem_id']]
    assert specs[2]['related_group']==specs[9]['related_group']
    assert all(assignment[s['problem_id']]['partition']=='test_ood' for s in specs if s['support_pattern']=='bottom_clamp')


def test_projection_near_feasible_scalar_reduction_roundoff():
    y=np.array([0.40391734503082355, 0.48039618647914206, 0.38449218090016624, 0.3814940006701598, 0.37466836191646946, 0.3546070740363606, 0.3171423043707408, 0.2575407522366288, 0.21989965532138014, 0.19999999, 0.19999999, 0.19999999, 0.4801475263462369, 0.6198568755917608, 0.5391451352471492, 0.5429538818474039, 0.5415335858704837, 0.5311738111134113, 0.5148827208920571, 0.5037679509960663, 0.5297584383758589, 0.6134184234838522, 0.79999999, 0.79999999, 0.3801225666442886, 0.5295961408228971, 0.49333301967316795, 0.5104166172556874, 0.519853284824574, 0.5281912173872947, 0.5477515565742095, 0.6064422776790134, 0.6934973415579844, 0.79999999, 0.79999999, 0.79999999, 0.38012256664429256, 0.5295961408229024, 0.4933330196731733, 0.5104166172556921, 0.5198532848245786, 0.5281912173872987, 0.5477515565742128, 0.6064422776790157, 0.6934973415579846, 0.79999999, 0.79999999, 0.79999999, 0.4801475263462529, 0.6198568755917792, 0.539145135247164, 0.5429538818474166, 0.5415335858704946, 0.5311738111134204, 0.5148827208920638, 0.5037679509960713, 0.5297584383758643, 0.6134184234838542, 0.79999999, 0.79999999, 0.4039173450308437, 0.48039618647916205, 0.3844921809001808, 0.3814940006701717, 0.37466836191647857, 0.3546070740363672, 0.3171423043707451, 0.25754075223663253, 0.21989965532139114, 0.19999999, 0.19999999, 0.19999999])
    z=project_prediction(y,0.5)
    assert z.mean()<=0.5 and sum(z)<=len(z)*0.5
    assert np.max(abs(z-y))<1e-7


def test_completion_initialization_preserves_existing_output(tmp_path):
    from run_m3_3c_completion import initialize
    prior=tmp_path/'prior_result.json';prior.write_text('preserve these bytes')
    with pytest.raises(ValueError,match='Output must be new'):initialize(tmp_path)
    assert prior.read_text()=='preserve these bytes' and list(tmp_path.iterdir())==[prior]
