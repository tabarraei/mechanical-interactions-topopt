"""Real original release plus EXPLICITLY synthetic new-entry schema fixtures.
The fixture subclass exists ONLY here; production has no stub-verifier flag.
"""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from .common import *
import m3_5b2.release as mod
from .release import ExpandedRelease,TrainingNormalization,TrajectoryDataset
from .test_controls import pid_at
from .validation import independent,exact_final_checkpoint,verify_acceptance
@pytest.fixture(scope='session')
def selected():return Selection()
@pytest.fixture(scope='session')
def original():
 from m3_5a.loader import Release
 with Release() as r:yield r
@pytest.fixture
def cached_original(monkeypatch,original):
 monkeypatch.setattr(mod,'OriginalRelease',lambda *a,**kw:original)
 return original
class FixtureRelease(ExpandedRelease):
 """NOT exported by the implementation; synthetic schema acceptance only."""
 def _verify_new(self,e,spec,selection):
  check(e['authorization']=={'fixture_domain':'synthetic-NOT-executable'},'Fixture segregation')
  c=read(e['validation']['acceptance_path']);check(c['evidence_kind']=='synthetic-record-fixture-NOT-mechanical' and c['passed'] and c['spec_sha256']==canonical(spec),'Invalid synthetic validation certificate')
  meta=read(Path(e['folder'])/'metadata.json')
  for key,v in meta['arrays'].items():check(digest(Path(e['folder'])/v['file'])==v['sha256']==e['array_hashes'][key],'Fixture native hash')

def fixture_index(tmp_path,selection):
 entries=copy.deepcopy(selection.original)
 for i in (0,72,80,138):
  pid=pid_at(selection,i);spec=selection.specs[pid];p=spec['mechanical'];N=p['nelx']*p['nely'];nd=2*(p['nelx']+1)*(p['nely']+1);folder=tmp_path/'synthetic_native'/pid;folder.mkdir(parents=True)
  x=np.asarray(spec['initial_x']);arrays=dict(x=np.stack([x,x]),rho=np.stack([x,x]),g_x=np.zeros((2,N)),g_rho=np.zeros((2,N)),u=np.zeros((2,nd)),compliance=np.ones(2),iteration=np.arange(2),update_change=np.array([0.]),previous_change=np.array([0.,0.]),has_incoming_update=np.array([False,True]));meta_arrays={}
  for k,v in arrays.items():
   np.save(folder/(k+'.npy'),v);meta_arrays[k]=dict(file=k+'.npy',sha256=digest(folder/(k+'.npy')),shape=list(v.shape),dtype=str(v.dtype))
  from m3_5a.io import array_digest
  transitions=[dict(iteration=0,next_iteration=1,valid=True,optimizer=spec['optimizer'],delta_digest=array_digest(x-x),maximum_change=0.)];write(folder/'transitions.json',transitions);write(folder/'problem.json',spec)
  meta=dict(state_count=2,completed_updates=1,termination_reason='change_tolerance',reference_label_eligible=False,optimality_certified=False,source_id=selection.manifest['source_binding']['original_execution_source_id'],arrays=meta_arrays)
  write(folder/'metadata.json',meta);cert=folder/'synthetic_certificate.json';write(cert,dict(evidence_kind='synthetic-record-fixture-NOT-mechanical',passed=True,spec_sha256=canonical(spec)))
  entries.append(dict(problem_id=pid,family=selection.jobs[pid]['family'],partition=selection.jobs[pid]['partition'],folder=str(folder),states=2,transitions=1,termination_reason='change_tolerance',source_id=meta['source_id'],metadata_sha256=digest(folder/'metadata.json'),array_hashes={k:v['sha256'] for k,v in meta_arrays.items()},validation=dict(acceptance_path=str(cert),evidence=str(folder),execution=dict(authorization_sha256='synthetic',resumed=False,restart_control=None),acceptance_sha256=digest(cert)),authorization={'fixture_domain':'synthetic-NOT-executable'},mechanically_validated=True,reference_label_eligible=False,validation_status='complete'))
 doc=dict(schema_version='m3.5b2.combined_release.v1',manifest_sha256=PLAN_SHA,original_index=dict(path=str(ORIGINAL),sha256=ORIGINAL_SHA),entries=entries);path=tmp_path/'synthetic_combined_index.json';write(path,doc);return path,doc

def test_original_only_real_adapter(original,tmp_path):
 p=tmp_path/'original_only_combined.json';write(p,dict(schema_version='m3.5b2.combined_release.v1',manifest_sha256=PLAN_SHA,original_index=dict(path=str(ORIGINAL),sha256=ORIGINAL_SHA),entries=original.entries))
 with ExpandedRelease(p,digest(p)) as r:
  assert len(r.entries)==12 and sum(e['transitions'] for e in r.entries)==3697
  d=TrajectoryDataset(r,'A',3);assert d.validate_sample(d[0])
def test_real_evidence_checkpoint_no_FEM(original):
 e=original.entries[0];spec=original.specs[e['problem_id']];side=independent(e['folder'],e['validation']['evidence'],spec);proof=exact_final_checkpoint(side);assert proof['status']=='verified_exact' and proof['committed_states']==192
 result=verify_acceptance(e['folder'],e['validation']['evidence'],spec,e['source_id'],dict(authorization_sha256='read-only-existing-evidence-not-new-grant',resumed=False,restart_control=None));assert result['passed'] and result['transitions']==191
@pytest.mark.parametrize('attack',['unvalidated','incomplete','duplicate','partition','family','counts','mesh','source','reference','certificate','missing_provenance'])
def test_synthetic_corrupt_release_rejected(selected,cached_original,tmp_path,attack):
 p,doc=fixture_index(tmp_path,selected);e=doc['entries'][-1]
 if attack=='unvalidated':e['mechanically_validated']=False
 elif attack=='incomplete':e['validation_status']='incomplete'
 elif attack=='duplicate':doc['entries'].append(copy.deepcopy(e))
 elif attack=='partition':e['partition']='train'
 elif attack=='family':e['family']='wrong-family'
 elif attack=='counts':e['states']=501
 elif attack=='mesh':
  f=Path(e['folder'])/'metadata.json';m=read(f);m['arrays']['u']['shape']=[2,2];atomic(f,m);e['metadata_sha256']=digest(f)
 elif attack=='source':e['source_id']='missing-source'
 elif attack=='reference':e['reference_label_eligible']=True;e['termination_reason']='maxiter'
 elif attack=='certificate':
  f=Path(e['validation']['acceptance_path']);c=read(f);c['passed']=False;atomic(f,c)
 elif attack=='missing_provenance':e.pop('authorization')
 atomic(p,doc)
 with pytest.raises((ValueError,KeyError)):FixtureRelease(p,digest(p))
def test_four_partitions_layout_causal_features_and_default_rejects_fixtures(selected,cached_original,tmp_path):
 p,doc=fixture_index(tmp_path,selected)
 with FixtureRelease(p,digest(p)) as r:
  assert {e['partition'] for e in r.entries}==PARTITIONS and len(r.entries)==16
  for mode in ('A','B'):
   d=TrajectoryDataset(r,mode,3)
   for e in r.entries:
    sample=d.sample(e['problem_id'],0);assert d.validate_sample(sample)
    assert sample['inputs']['node_history'].shape[-2:]==(r.specs[e['problem_id']]['mechanical']['nely']+1,r.specs[e['problem_id']]['mechanical']['nelx']+1)
   sample=d.sample(doc['entries'][-1]['problem_id'],0);sample['inputs']['history_indices'][-1]=1
   with pytest.raises(ValueError):d.validate_sample(sample)
 with pytest.raises(ValueError,match='authorization provenance'):ExpandedRelease(p,digest(p))
def test_heldout_changes_cannot_change_training_normalization(selected,cached_original,tmp_path):
 p,doc=fixture_index(tmp_path,selected)
 with FixtureRelease(p,digest(p)) as r:
  a=TrainingNormalization.fit(r);a.validate(r);train_identity=r.training_release_identity
  with pytest.raises(ValueError):TrainingNormalization.fit(r,[(doc['entries'][-1]['problem_id'],0)])
 held=doc['entries'][-1];folder=Path(held['folder']);array=folder/'compliance.npy';np.save(array,np.array([1000000.,2000000.]));meta=read(folder/'metadata.json');meta['arrays']['compliance']['sha256']=digest(array);atomic(folder/'metadata.json',meta);held['array_hashes']['compliance']=digest(array);held['metadata_sha256']=digest(folder/'metadata.json');atomic(p,doc)
 with FixtureRelease(p,digest(p)) as r:
  b=TrainingNormalization.fit(r);assert train_identity==r.training_release_identity and a.record==b.record;a.validate(r)
  assert len(b.record['training_ids'])==13 and all(row['state']==0 and r.by_id[row['problem_id']]['partition']=='train' for row in b.record['fit_samples'])

def test_default_raw_independent_certificate_cannot_be_replaced_by_failed_summary(original,tmp_path):
 from m3_4br4r1.policy import EvidenceError
 e=original.entries[0];summary=read(Path(e['validation']['evidence'])/'summary.json');summary['passed']=False;write(tmp_path/'summary.json',summary)
 with pytest.raises(EvidenceError):independent(e['folder'],tmp_path,original.specs[e['problem_id']])
def test_raw_checkpoint_corruption_rejects_even_with_good_independent_evidence(original,tmp_path):
 e=original.entries[0];side=independent(e['folder'],e['validation']['evidence'],original.specs[e['problem_id']]);j=read(Path(e['folder'])/'journal.json');j['states'][0]['sha256']='bad';write(tmp_path/'journal.json',j);side['folder']=str(tmp_path)
 with pytest.raises(ValueError,match='NPZ bytes'):exact_final_checkpoint(side)

def test_forged_positive_normalization_statistics_are_rejected(selected,cached_original,tmp_path):
 p,_=fixture_index(tmp_path,selected)
 with FixtureRelease(p,digest(p)) as r:
  n=TrainingNormalization.fit(r);record=copy.deepcopy(n.record);record['scales']['compliance']*=2
  file=tmp_path/'forged_statistics.json';write(file,record)
  with pytest.raises(ValueError,match='reproduce'):TrainingNormalization.load(file,r)


def test_normalizer_validation_does_not_invalidate_borrowed_trajectory_maps(selected,cached_original,tmp_path):
 p,_=fixture_index(tmp_path,selected)
 with FixtureRelease(p,digest(p)) as r:
  norm=TrainingNormalization.fit(r);pid=r.entries[0]['problem_id'];a=r.open(pid);norm.validate(r);assert r._opened==pid and np.isfinite(a['x'][0]).all()
  d=TrajectoryDataset(r,'B',3,norm);assert d.validate_sample(d.sample(pid,0))
