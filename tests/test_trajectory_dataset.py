import copy,json
from pathlib import Path
import numpy as np
import pytest
from topopt_utils.fea.trajectory_dataset import (sample_problem,load_problem,features,feature_schema,split_problems,validate_splits,Trajectory,window_index,WindowDataset,evaluate_prediction,batch_indices,file_hash)
from topopt_utils.fea.dataset_storage import generate_sample,validate_sample,atomic_json
from topopt_utils.fea.consistent_trajectory import evaluate_state

CONTROLS=dict(option='feasible_upper_v1',maxiter=12,tolerance=.001)

def cfg(label='case',support='left_clamp',direction=(1.,0.),magnitude=1.):
    return dict(label=label,mesh=[6,4],domain=[6.,4.],support=support,volume_fraction=.5,related_group=label,
        loads=[dict(position=[3.,4.] if support in ('pin_roller','bottom_clamp') else [6.,2.],direction=list(direction),magnitude=magnitude)],initial=dict(kind='uniform'))

@pytest.fixture(scope='module')
def complete(tmp_path_factory):
    root=tmp_path_factory.mktemp('dataset');p,x,s=sample_problem(cfg(),31,CONTROLS)
    result=generate_sample(root,s,'test-source');return root,s,Trajectory(result['folder'])


def test_sampler_deterministic_full_identity_and_initial():
    c=cfg();c['initial']={'kind':'perturbed','amplitude':.02};p,x,s=sample_problem(c,123,CONTROLS);q,y,t=sample_problem(c,123,CONTROLS)
    assert s==t;np.testing.assert_array_equal(x,y)
    assert x.mean()<=.5 and np.max(abs(x-.5))>.01
    c['loads'][0]['magnitude']=2.;_,_,other=sample_problem(c,123,CONTROLS);assert other['problem_id']!=s['problem_id']


def test_physical_load_not_silently_snapped():
    c=cfg();c['loads'][0]['position']=[6.,2.3]
    with pytest.raises(ValueError,match='not on'):sample_problem(c,1,CONTROLS)

@pytest.mark.parametrize('edit',[('magnitude',0.),('direction',[0.,0.])])
def test_invalid_load_rejected(edit):
    c=cfg();c['loads'][0][edit[0]]=edit[1]
    with pytest.raises(ValueError):sample_problem(c,1,CONTROLS)


def test_load_on_fixed_node_and_unstable_support_rejected():
    c=cfg();c['loads'][0]['position']=[0.,2.]
    with pytest.raises(ValueError):sample_problem(c,1,CONTROLS)
    c=cfg();c['support']='custom';c['fixed_dofs']=[0,1]
    with pytest.raises(ValueError):sample_problem(c,1,CONTROLS)


def test_spec_tamper_detected():
    _,_,s=sample_problem(cfg(),1,CONTROLS);s['mechanical']['force_vector'][-1]=99
    with pytest.raises(ValueError):load_problem(s)


def test_resume_and_completed_sample_not_recomputed(tmp_path):
    _,_,s=sample_problem(cfg(),1,CONTROLS)
    first=generate_sample(tmp_path,s,'source',max_new_updates=2);assert first['status']=='incomplete' and first['completed_updates']==2
    stage=first['folder'];old=file_hash(stage/'frames/00000.npz')
    (stage/'frames/99999.npz').write_bytes(b'uncommitted orphan')
    done=generate_sample(tmp_path,s,'source');assert done['status']=='completed_new'
    assert (done['folder']/'quarantined/99999.npz').exists()
    reused=generate_sample(tmp_path,s,'source');assert reused['status']=='completed_reused'
    with pytest.raises(ValueError,match='another'):generate_sample(tmp_path,s,'changed-source')
    _,_,other=sample_problem(cfg('continuous'),1,CONTROLS);otherroot=tmp_path/'continuous';r=generate_sample(otherroot,other,'source')
    a=Trajectory(done['folder']);b=Trajectory(r['folder'])
    for key in ('x','rho','u','compliance','g_x','g_rho'):np.testing.assert_allclose(a.arrays[key],b.arrays[key],rtol=1e-10,atol=1e-12)


def test_checkpoint_corruption_fails(tmp_path):
    _,_,s=sample_problem(cfg(),1,CONTROLS);r=generate_sample(tmp_path,s,'source',max_new_updates=1)
    (r['folder']/'frames/00000.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corruption'):generate_sample(tmp_path,s,'source')


def test_mmap_storage_terminal_and_transition_alignment(complete):
    root,s,t=complete;assert isinstance(t.arrays['x'],np.memmap)
    assert len(t.arrays['x'])==t.metadata['completed_updates']+1
    r=t.transition(2);np.testing.assert_array_equal(r['delta'],t.arrays['x'][3]-t.arrays['x'][2])
    assert r['valid'] and r['optimizer']['option']=='feasible_upper_v1'
    truth=evaluate_state(t.problem,t.arrays['x'][-1]);np.testing.assert_allclose(truth['compliance'],t.arrays['compliance'][-1])
    assert all(r['passed'] for r in validate_sample(t,independent=True))


def test_force_support_channels_native_grid_exact(complete):
    root,s,t=complete;p=t.problem;f=features(p,t.arrays['x'][0],0,'no_fem_v1')
    assert f['node'].shape==(8,p.nely+1,p.nelx+1)
    node=f['node'];scale=s['force_scale'];np.testing.assert_array_equal(node[0].ravel()*scale,p.forces[0::2])
    np.testing.assert_array_equal(node[1].ravel()*scale,p.forces[1::2])
    fixed=np.zeros(p.ndof);fixed[list(p.fixed_dofs)]=1;np.testing.assert_array_equal(node[2].ravel(),fixed[0::2])
    assert np.count_nonzero(node[0])==1
    np.testing.assert_array_equal(f['element'][2].ravel(),p.centers[:,0])


def test_no_fem_and_fresh_modes_explicit(complete):
    _,_,t=complete;p=t.problem;x=t.arrays['x'][0]
    f=features(p,x,0,'no_fem_v1');assert not f['fresh_fem_required'] and all('g_x' not in n for n in f['element_channel_names'])
    with pytest.raises(ValueError,match='requires'):features(p,x,0,'fresh_gradient_v1')
    f=features(p,x,0,'fresh_full_v1',t.state(0));assert f['fresh_fem_required'] and f['element'].shape[0]==12
    wrong=t.state(0);wrong['mechanical_id']='different'
    with pytest.raises(ValueError):features(p,x,0,'fresh_full_v1',wrong)


def test_known_load_normalization_not_future_statistics():
    p,x,s=sample_problem(cfg(magnitude=1),1,CONTROLS);q,y,r=sample_problem(cfg(magnitude=2),1,CONTROLS)
    a=evaluate_state(p,x);a['mechanical_id']=p.problem_id;b=evaluate_state(q,y);b['mechanical_id']=q.problem_id
    fa=features(p,x,0,'fresh_full_v1',a);fb=features(q,y,0,'fresh_full_v1',b)
    np.testing.assert_allclose(fa['element'][9:],fb['element'][9:],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(fa['node'],fb['node'],rtol=1e-11,atol=1e-12)
    assert fa['element'][8,0,0]==1 and fb['element'][8,0,0]==2


def split_specs():
    configs=[cfg('axial'),cfg('scaled',magnitude=2),cfg('vertical',direction=(0.,-1.)),cfg('oblique',direction=(.6,-.8)),cfg('pin','pin_roller',direction=(0.,-1.)),cfg('ood','bottom_clamp')]
    configs[1]['related_group']='axial'
    return [sample_problem(c,i+10,CONTROLS)[2] for i,c in enumerate(configs)]


def test_group_split_deterministic_and_ood_no_leakage():
    specs=split_specs();a=split_problems(specs,712,['bottom_clamp']);b=split_problems(specs,712,['bottom_clamp']);assert a==b and validate_splits(specs,a)
    assert a['assignments'][specs[0]['problem_id']]==a['assignments'][specs[1]['problem_id']]
    assert a['assignments'][specs[-1]['problem_id']]['partition']=='test_ood'
    partitions={x['partition'] for x in a['assignments'].values()};assert len(partitions)==4


def test_split_tampering_and_duplicate_id_rejected():
    specs=split_specs();s=split_problems(specs,712,['bottom_clamp']);s['assignments'][specs[0]['problem_id']]['partition']='test_id'
    with pytest.raises(ValueError):validate_splits(specs,s)
    with pytest.raises(ValueError):split_problems(specs+[specs[0]],1,['bottom_clamp'])


def test_window_index_targets_no_future_input_and_short_history():
    specs=split_specs();s=split_problems(specs,712,['bottom_clamp']);e=[dict(problem_id=specs[0]['problem_id'],state_count=12)]
    rows=window_index(e,s,(1,3),(1,5,10));assert {r['horizon'] for r in rows}=={1,5,10}
    for r in rows:assert r['target']==r['anchor']+r['horizon'] and r['anchor']>=r['history_length']-1 and r['target']<12
    assert window_index([dict(problem_id=e[0]['problem_id'],state_count=2)],s,(3,),(10,))==[]


def test_window_loader_and_torch_batch(complete,tmp_path):
    root,s,t=complete;specs=split_specs();splits=split_problems(specs,712,['bottom_clamp'])
    # The fixture's specimen gets a self-contained assignment for loading tests.
    assignment=dict(partition='train',group_id='one');splits['assignments']={s['problem_id']:assignment};splits['groups']=[]
    entry=dict(problem_id=s['problem_id'],state_count=t.metadata['state_count'],folder=str(t.folder),metadata_sha256=file_hash(t.folder/'metadata.json'),mesh=[t.problem.nelx,t.problem.nely])
    rows=window_index([entry],splits,(1,3),(1,5,10));atomic_json(tmp_path/'windows.json',rows);atomic_json(tmp_path/'splits.json',splits);atomic_json(tmp_path/'feature_schema.json',feature_schema())
    atomic_json(tmp_path/'manifest.json',dict(schema_version='m3.3b.dataset.v1',samples=[entry],files={n:file_hash(tmp_path/n) for n in ('windows.json','splits.json','feature_schema.json')}))
    ds=WindowDataset(tmp_path,'train','no_fem_v1',history_length=3,horizon=5);sample=ds[0];r=sample['index']
    np.testing.assert_array_equal(sample['target_x'].ravel(),t.arrays['x'][r['target']]);np.testing.assert_array_equal(sample['element'][0,0].ravel(),t.arrays['x'][r['anchor']-2])
    import torch
    batch=next(iter(torch.utils.data.DataLoader(ds,batch_sampler=batch_indices(ds,2))));assert batch['element'].shape[0]<=2


def test_periodic_fem_evaluation_and_correction(complete):
    _,_,t=complete;result=evaluate_prediction(t.problem,t.arrays['x'][1],correct=True)
    assert result['equilibrium_solves']==1 and result['oc_diagnostics']['strict_feasible']
    np.testing.assert_allclose(result['corrected_x'],t.arrays['x'][2],rtol=1e-10,atol=1e-12)


def test_physical_energy_channel_sums_to_half_compliance(complete):
    _,s,t=complete;p=t.problem;state=t.state(2)
    f=features(p,state['x'],2,'fresh_full_v1',state)
    assert np.sum(f['element'][-1])*s['force_scale']**2==pytest.approx(state['compliance']/2,rel=1e-10)


def test_auto_grouping_force_reversal_and_scale():
    specs=split_specs();c=cfg('reverse',direction=(-1.,0.),magnitude=3.)
    specs.append(sample_problem(c,31,CONTROLS)[2]);s=split_problems(specs,712,['bottom_clamp'])
    assert s['assignments'][specs[0]['problem_id']]['group_id']==s['assignments'][specs[-1]['problem_id']]['group_id']
