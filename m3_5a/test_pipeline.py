"""Real-release integration and deliberate leakage/corruption attempts; no physics."""
import copy,importlib,json
from pathlib import Path
import numpy as np
import pytest
from .loader import Release,TrajectoryDataset
from .normalization import Normalization
from .features import history_indices,collate,crop_predictions,schema
from .io import RELEASE,array_digest
@pytest.fixture(scope='session')
def release():
 with Release() as r:yield r
@pytest.fixture(scope='session')
def norm(release):return Normalization.fit(release)
def dataset(release,norm,mode='A',k=3):return TrajectoryDataset(release,mode,k,norm)
@pytest.mark.parametrize('index',range(12))
@pytest.mark.parametrize('mode',['A','B'])
def test_every_mesh_family_and_endpoint(release,norm,index,mode):
 e=release.entries[index];d=dataset(release,norm,mode);pid=e['problem_id'];p=release.specs[pid]['mechanical'];H,W=p['nely'],p['nelx']
 for t in (0,e['transitions']//2,e['transitions']-1):
  s=d.sample(pid,t);assert d.validate_sample(s);assert s['target'].shape==(1,H,W);assert s['inputs']['static_node'].shape==(4,H+1,W+1)
  assert np.array_equal(s['target'][0],release.target(pid,t).astype(np.float32));assert np.all(s['inputs']['history_indices']<=t)
  assert 'termination_reason' not in s['inputs'];assert s['proof']['next_state']==t+1
@pytest.mark.parametrize('k',[1,3,8])
def test_padding_and_causal_masks(release,norm,k):
 d=dataset(release,norm,k=k);s=d[0];x=s['inputs'];assert x['history_indices'].tolist()==[-1]*(k-1)+[0];assert not x['history_valid'][:-1].any();assert not x['element_history'][:-1].any();assert x['causal_attention'].sum()==1
 s=d.sample(d.ids[0],4);x=s['inputs'];assert not np.triu(x['causal_attention'],1).any();assert np.all(x['history_indices'][x['history_valid']]<=4)
def test_all_global_indices_unique(release,norm):
 d=dataset(release,norm);keys=[d.key(i) for i in range(len(d))];assert len(keys)==len(set(keys))==3697
 assert all(t<release.by_id[pid]['transitions'] for pid,t in keys)
def test_terminal_and_capped_not_reference(release,norm):
 d=dataset(release,norm)
 for e in release.entries:
  with pytest.raises(ValueError,match='Terminal'):d.sample(e['problem_id'],e['states']-1)
  if e['termination_reason']=='maxiter':assert e['states']==501 and d.sample(e['problem_id'],499)['proof']['next_state']==500 and not e['reference_label_eligible']
@pytest.mark.parametrize('mode',['A','B'])
def test_mixed_mesh_batch_crop(release,norm,mode):
 d=dataset(release,norm,mode);samples=[d.sample(release.entries[i]['problem_id'],0) for i in (0,1,6)];b=collate(samples);pred=b['target'].copy();rest=crop_predictions(pred,b)
 for p,s in zip(rest,samples):np.testing.assert_array_equal(p,s['target'])
 assert b['target'].shape==(3,1,48,96) and b['element_valid'][0].sum()==24*48 and b['node_valid'][0].sum()==25*49
 assert not b['element_history'][0,...,24:,:].any()
@pytest.mark.parametrize('attack',['future_x','future_sensitivity','target','transition','cross_trajectory','padding','history_mask','causal_mask','mesh','nodal_elemental','load','support','duplicate_future_channel','normalization_id'])
def test_adversarial_sample_injections_detected(release,norm,attack):
 d=dataset(release,norm,'B');s=copy.deepcopy(d[0]);x=s['inputs'];pid=s['proof']['problem_id'];a=release.open(pid)
 if attack=='future_x':x['element_history'][-1,0]=a['x'][1].reshape(24,48)
 elif attack=='future_sensitivity':x['element_history'][-1,4]=a['g_rho'][1].reshape(24,48)
 elif attack=='target':s['target']*=0
 elif attack=='transition':s['proof']['transition']=1
 elif attack=='cross_trajectory':s['proof']['problem_id']=release.entries[1]['problem_id']
 elif attack=='padding':x['element_history'][0,0]=1
 elif attack=='history_mask':x['history_valid'][0]=True
 elif attack=='causal_mask':x['causal_attention'][0,-1]=True
 elif attack=='mesh':x['element_history']=x['element_history'].swapaxes(-1,-2)
 elif attack=='nodal_elemental':x['static_node']=x['static_element']
 elif attack=='load':x['static_node'][0,0,0]+=1
 elif attack=='support':x['static_node'][2,0,0]=1-x['static_node'][2,0,0]
 elif attack=='duplicate_future_channel':x['future_g_x']=np.asarray(a['g_x'][1])
 elif attack=='normalization_id':x['normalization_id']='forbidden future fit'
 with pytest.raises(ValueError):d.validate_sample(s)
def test_wrong_saved_transition_index_detected(release,norm,monkeypatch):
 pid=release.entries[0]['problem_id'];release.open(pid)
 with monkeypatch.context() as m:
  m.setitem(release._native_transitions[0],'next_iteration',2)
  with pytest.raises(ValueError,match='indexing'):release.target(pid,0)
@pytest.mark.parametrize('attack',['future_prefix','unknown_problem','duplicate','nontraining_partition','wrong_release','zero_scale'])
def test_forbidden_normalization_detected(release,norm,attack,monkeypatch):
 pid=release.entries[0]['problem_id']
 if attack=='future_prefix':
  with pytest.raises(ValueError,match='forbidden'):Normalization.fit(release,[(pid,1)])
 elif attack=='unknown_problem':
  with pytest.raises(ValueError,match='forbidden'):Normalization.fit(release,[('forbidden_held_out',0)])
 elif attack=='duplicate':
  with pytest.raises(ValueError,match='Duplicate'):Normalization.fit(release,[(pid,0),(pid,0)])
 elif attack=='nontraining_partition':
  with monkeypatch.context() as m:
   m.setitem(release.by_id[pid],'partition','development_only')
   with pytest.raises(ValueError,match='forbidden'):Normalization.fit(release,[(pid,0)])
 else:
  r=copy.deepcopy(norm.record)
  if attack=='wrong_release':r['release_sha256']='bad'
  else:r['scales']['u']=0
  with pytest.raises(ValueError):Normalization(r).validate(release)
def test_zero_variance_finite_and_vector_scaling(release,norm):
 assert all(np.isfinite(v) and v>0 for v in norm.scales.values());d=dataset(release,norm,'B');s=d[0];u=release.open(s['proof']['problem_id'])['u'][0].reshape(-1,2).T.reshape(2,25,49)
 np.testing.assert_allclose(s['inputs']['node_history'][-1],u/release.specs[s['proof']['problem_id']]['force_scale']/norm.scales['u'],rtol=1e-6,atol=1e-7)
def test_duplicate_dataset_problems_rejected(release,norm):
 pid=release.entries[0]['problem_id']
 with pytest.raises(ValueError,match='Duplicate'):TrajectoryDataset(release,problem_ids=[pid,pid])
def test_readonly_mapping(release):
 a=release.open(release.entries[0]['problem_id'])['x'];assert not a.flags.writeable
 with pytest.raises(ValueError):a[0,0]=0
def test_release_index_tamper_detected(tmp_path):
 v=json.loads((RELEASE/'accepted_dataset_index.json').read_text());v['configurations'][0]['states']+=1;p=tmp_path/'accepted_dataset_index.json';p.write_text(json.dumps(v))
 with pytest.raises(ValueError,match='SHA256'):Release(p,'not_the_hash')
def test_forbidden_physics_import():
 with pytest.raises(ImportError,match='prohibits'):importlib.import_module('topopt_utils.fea.solver')
def test_modes_have_separate_schema_and_no_A_mechanics():
 assert schema('A')['version']!=schema('B')['version'];assert not schema('A')['fresh_FEM_at_inference'];assert schema('B')['fresh_FEM_at_inference'];assert schema('A')['node_history']==[]
 assert not {'rho','g_rho','g_x','compliance','u'}&set(schema('A')['element_history'])
def test_filter_density_not_design(release):
 e=release.entries[0];a=release.open(e['problem_id']);assert np.max(abs(a['x'][10]-a['rho'][10]))>1e-4

def test_duplicate_batch_examples_rejected(release,norm):
 d=dataset(release,norm);s=d[0]
 with pytest.raises(ValueError,match='Duplicate batch'):collate([s,s])
@pytest.mark.parametrize('mode',['A','B'])
def test_feature_extractor_cannot_read_future(release,norm,monkeypatch,mode):
 from .features import extract
 pid=release.entries[0]['problem_id'];a=release.open(pid);t=4
 class PastOnly:
  def __init__(self,array,forbidden=False):self.array=array;self.forbidden=forbidden
  def __getitem__(self,key):
   assert not self.forbidden,'ModeA accessed mechanics'
   assert isinstance(key,(int,np.integer)) and 0<=key<=t,'Future array access'
   return self.array[key]
 guarded={k:PastOnly(v,mode=='A' and k in ('u','g_rho','g_x','compliance','rho')) for k,v in a.items()}
 with monkeypatch.context() as m:
  m.setattr(release,'open',lambda key:guarded)
  result=extract(release,pid,t,mode,3,norm)
  assert result['history_indices'].tolist()==[2,3,4]
def test_zero_mechanical_variance_fallback():
 class Tiny:
  index_sha256='synthetic';by_id={'train':{'partition':'train','array_hashes':{}}};specs={'train':{'force_scale':1.}}
  def open(self,pid):return {k:np.zeros((1,4)) for k in ('u','g_rho','g_x','compliance')}
 n=Normalization.fit(Tiny());assert all(v==1 for v in n.scales.values());assert all(n.record['nearly_zero_fallback'].values())
def test_hash_mismatch_and_missing_artifact_fail_closed(tmp_path):
 r=Release.__new__(Release);p=tmp_path/'validation.json';p.write_text('{"passed":false}')
 r.manifests={tmp_path:{'validation.json':'bad_hash'}};r.verified_files={}
 with pytest.raises(ValueError,match='hash'):r.verify_file(p)
 p.unlink()
 with pytest.raises(FileNotFoundError):r.verify_file(p)
def test_model_encoder_excludes_targets_and_proof(release,norm):
 from .model import encode
 d=dataset(release,norm);b=collate([d[0]]);x=encode(b).numpy().copy();b['target'][:]=999;b['proof'][0]['next_state']=999
 np.testing.assert_array_equal(encode(b).numpy(),x)
