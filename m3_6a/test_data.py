"""Read-only original accepted data; any fabricated arrays explicitly fixtures."""
import copy
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from m3_6a.data import AcceptedData,collate,mechanics_key,NORMALIZATION
from m3_6a.model import MultiscaleTemporalOperator

@pytest.fixture(scope='module')
def data():
 d=AcceptedData();yield d;d.close()

def test_exact_original_release_no_historical_alias(data):
 assert len(data.entries)==12 and sum(e['states'] for e in data.entries)==3709 and sum(e['transitions'] for e in data.entries)==3697
 assert all(e['partition']=='train' for e in data.entries)
 for bad in ('m3_3b_d88cb10eb219fb432141045c','m3_3b_84d8d729bc9abf70d17abec2','unknown'):
  with pytest.raises(ValueError):data.example(bad,0)
 assert len({mechanics_key(data.release.specs[e['problem_id']]) for e in data.entries})==12

@pytest.mark.parametrize('h',[1,5,10])
def test_real_multistep_target_and_alignment(data,h):
 pid=data.entries[0]['problem_id'];s=data.example(pid,3,k=3,h=h);a=data.release.open(pid)
 np.testing.assert_array_equal(s['target'].reshape(-1),np.asarray(a['x'][3+h]-a['x'][3],np.float32))
 assert s['proof']['history']==[1,2,3]
 np.testing.assert_array_equal(s['inputs']['element_history'][-1,0].reshape(-1),np.asarray(a['x'][3],np.float32))
 f=s['inputs']['node_fields'][:2].transpose(1,2,0).reshape(-1)
 np.testing.assert_allclose(f,data.release.specs[pid]['mechanical']['force_vector'],atol=1e-7)
 assert s['inputs']['element_coords'].shape==(2,24,48) and s['inputs']['node_coords'].shape==(2,25,49)
 with pytest.raises(ValueError):data.example(pid,data.entries[0]['states']-h,h=h)

@pytest.mark.parametrize('i',[0,1,6])
def test_one_model_same_parameters_on_accepted_mesh(data,i):
 # The example script additionally executes all3 meshes with one actual instance.
 pid=data.entries[i]['problem_id'];s=data.example(pid,2,k=3,h=10);b=collate([s]);model=MultiscaleTemporalOperator().eval()
 with torch.no_grad():p=model(b)
 assert p.shape==b['target'].shape and torch.isfinite(p).all()


def test_mode_B_has_past_mechanics_and_reuses_frozen_normalization(data):
 pid=data.entries[0]['problem_id'];s=data.example(pid,1,k=2,h=5,mode='B')
 assert s['inputs']['element_history'].shape==(2,5,24,48) and s['inputs']['node_history'].shape==(2,2,25,49)
 assert all(r['state']==0 for r in data.normalization.record['fit_samples'])
 assert data.normalization.record['targets_used'] is False and len(data.normalization.record['fit_samples'])==12


def test_actual_future_mutation_only_in_synthetic_fixture(data):
 # Copy accepted specification only. No actual dataset arrays are writable.
 original=data.entries[0]['problem_id'];spec=copy.deepcopy(data.release.specs[original]);pid='SYNTHETIC_HISTORY_NOT_ACCEPTED';spec['problem_id']=pid
 N=spec['mechanical']['nelx']*spec['mechanical']['nely'];T=20
 x=np.broadcast_to(.2+np.arange(T)[:,None]*.001,(T,N)).copy();arrays={'x':x}
 source=SimpleNamespace(specs={pid:spec},open=lambda p:arrays)
 d=object.__new__(AcceptedData);d.by_id={pid:dict(partition='train',states=T,source_id='SYNTHETIC_ONLY')};d.release=source
 a=d.example(pid,3,k=3,h=5);arrays['x'][4:]=.9;b=d.example(pid,3,k=3,h=5)
 for key in a['inputs']:
  if isinstance(a['inputs'][key],np.ndarray):np.testing.assert_array_equal(a['inputs'][key],b['inputs'][key])
 assert not np.array_equal(a['target'],b['target'])


def test_no_cross_mode_batches(data):
 pid=data.entries[0]['problem_id']
 with pytest.raises(ValueError):collate([data.example(pid,1,mode='A'),data.example(pid,2,mode='B')])
