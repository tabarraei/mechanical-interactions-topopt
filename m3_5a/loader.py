"""Verified release, one mapped trajectory at a time, causal supervised examples."""
import time
from pathlib import Path
import numpy as np
from .io import ROOT,RELEASE,R4,R3,R2,OLD,read,digest,array_digest,canonical,check,bound_path
POLICY='m3.4b.transition_change_audit.v3'
class Release:
 def __init__(self,index=RELEASE/'accepted_dataset_index.json',expected_sha256=None,expected_manifest_sha256=None):
  self.index=Path(index).resolve();self.index_sha256=digest(self.index)
  if expected_sha256:check(self.index_sha256==expected_sha256,'Release index SHA256')
  self.entries=read(self.index);check(self.entries.get('passed') is True and self.entries.get('original_only') is True,'Release not passed/original')
  self.entries=self.entries['configurations'];check(len(self.entries)==12,'Original12 release count')
  check([e['index'] for e in self.entries]==list(range(12)),'Release ordering')
  self.by_id={e['problem_id']:e for e in self.entries};check(len(self.by_id)==12,'Duplicate trajectory ID')
  check(sum(e['states'] for e in self.entries)==3709 and sum(e['transitions'] for e in self.entries)==3697,'Release totals')
  # Hash roots from the inspected release; no source modules are imported.
  self.manifests={};self.verified_files={};self.bindings={}
  for base in (self.index.parent,R4,R3):
   path=base/'artifact_manifest.json';self.manifests[base]=read(path);self.bindings[str(path.relative_to(ROOT))]=digest(path)
  if expected_manifest_sha256:check(digest(self.index.parent/'artifact_manifest.json')==expected_manifest_sha256,'Release artifact manifest SHA256')
  self.verify_file(self.index)
  self.verify_file(self.index.parent/'protected_before.json');release_protected=read(self.index.parent/'protected_before.json')
  for base in (R4,R3):check(digest(base/'artifact_manifest.json')==release_protected[str((base/'artifact_manifest.json').relative_to(ROOT))],'Retained artifact manifest binding')
  check(read(self.index.parent/'effective_production_preflight.json')['passed'],'Approved preflight')
  source=read(OLD/'source_manifest.json')
  for name,e in source.items():
   check(digest(ROOT/name)==e['sha256'] and digest(OLD/e['archive'])==e['sha256'],'Original source/archive identity '+name)
   self.verified_files[str(ROOT/name)]=e['sha256'];self.verified_files[str(OLD/e['archive'])]=e['sha256']
  # R2 artifacts used by the complete R3 logical replay are bound by the
  # protected-before manifest, itself covered by the frozen R3 manifest.
  self.verify_file(R3/'protected_before.json');self.protected={**read(R3/'protected_before.json'),**release_protected}
  self.verify_file(OLD/'source_manifest.json');self.verify_file(OLD/'execution_manifest.json')
  self.verify_file(self.index.parent/'effective_production_preflight.json')
  self.verify_file(R3/'complete_validation.json');check(read(R3/'complete_validation.json')['passed'],'R3 complete replay')
  self._opened=None;self._arrays={};self.records={};self.specs={};started=time.perf_counter()
  original=read(OLD/'execution_manifest.json')['jobs'];source_id=read(OLD/'execution_manifest.json')['source_id']
  for e,j in zip(self.entries,original):
   check(e['problem_id']==j['problem_id'] and e['partition']==j['partition']=='train' and e['source_id']==source_id,'Original ID/source/split')
   check(e['mechanically_validated'] and e['validation']['passed'] and not e['reference_label_eligible'],'Release acceptance')
   check(e['transitions']==e['states']-1,'Trajectory lengths')
   folder=Path(e['folder']);check(digest(folder/'metadata.json')==e['metadata_sha256'],'Native metadata hash')
   self.verified_files[str(folder/'metadata.json')]=e['metadata_sha256'];meta=read(folder/'metadata.json');spec=read(folder/'problem.json')
   check(canonical(spec)==j['spec_sha256'],'Canonical original specification')
   check(spec==read(OLD/j['spec']),'Original specification unchanged')
   for key,value in [('problem_id',e['problem_id']),('source_id',e['source_id']),('state_count',e['states']),('completed_updates',e['transitions']),('termination_reason',e['termination_reason'])]:check(meta[key]==value,'Native metadata identity '+key)
   check(not meta['reference_label_eligible'] and not meta['optimality_certified'],'No reference label')
   check(meta['optimizer_option']=='feasible_upper_v1' and meta['optimizer_version']=='m3.3a.oc.v1','Original OC semantics')
   for file,key in [('problem.json','problem_sha256'),('transitions.json','transitions_sha256')]:
    p=folder/file;check(digest(p)==meta[key],'Native '+file+' hash');self.verified_files[str(p)]=meta[key]
   check(e['array_hashes']=={k:v['sha256'] for k,v in meta['arrays'].items()},'Release array hashes')
   for key,v in meta['arrays'].items():
    p=bound_path(folder,v['file']);check(digest(p)==v['sha256'],'Native array hash '+key);self.verified_files[str(p)]=v['sha256']
    a=np.load(p,mmap_mode='r',allow_pickle=False)
    check(list(a.shape)==v['shape'] and str(a.dtype)==v['dtype'],'Native array shape/dtype '+key)
    close_array(a)
   self.specs[e['problem_id']]=spec
   self.records[e['problem_id']]=self._validate_evidence(e)
  self.close();self.verification_seconds=time.perf_counter()-started
 def verify_file(self,path):
  p=Path(path).resolve();expected=None
  for base,manifest in self.manifests.items():
   if p.is_relative_to(base):expected=manifest.get(str(p.relative_to(base)));break
  if expected is None and hasattr(self,'protected'):expected=self.protected.get(str(p.relative_to(ROOT)))
  check(expected is not None,'Unbound validation artifact '+str(p));check(digest(p)==expected,'Validation artifact hash '+str(p));self.verified_files[str(p)]=expected
 def _validate_evidence(self,e):
  pid=e['problem_id'];T=e['states'];v=e['validation'];native_hashes=e['array_hashes'];states=[];paths=[]
  if e['index']==9:
   fragments=[R2/n for n in ('independent_replay','independent_replay_continuation','independent_replay_final')]+[R3/'independent_replay',R3/'independent_replay_completed']
   for f in fragments:
    self.verify_file(f/'states.jsonl');states.extend(read_lines(f/'states.jsonl'))
   self.verify_file(R3/'recovered_state408/states.jsonl');states[408]=read_lines(R3/'recovered_state408/states.jsonl')[0]
   ev=R3/'independent_replay_completed'
   for t in range(409):
    p=ev/f'cached_transition_{t:05d}.json';self.verify_file(p);paths.append(p)
   self.verify_file(ev/'transitions.jsonl');rows=[read(p) for p in paths]+read_lines(ev/'transitions.jsonl');refs=[str(p) for p in paths]+[str(ev/'transitions.jsonl')]*(T-1-409)
  else:
   ev=Path(v['evidence']);self.verify_file(ev/'summary.json');s=read(ev/'summary.json')
   check(s['passed'] and s['states']==T and s['transitions']==T-1,'Complete independent summary')
   check(s['identity']['array_hashes']==native_hashes,'Replay native identity')
   for n,h in s['identity']['validation_sources'].items():check(digest(ROOT/'m3_4br3'/n)==h,'Replay source identity')
   for p in (ev/'states.jsonl',ev/'transitions.jsonl'):self.verify_file(p)
   states=read_lines(ev/'states.jsonl');rows=read_lines(ev/'transitions.jsonl');refs=[str(ev/'transitions.jsonl')]*(T-1)
  check([r['state'] for r in states]==list(range(T)) and all(r['passed'] for r in states),'Independent state coverage')
  for r in states:check(all(c['passed'] for c in r['primary'].values()),'Primary mechanical comparison')
  check([r['transition'] for r in rows]==list(range(T-1)),'Independent transition ordering')
  for t,r in enumerate(rows):
   check(r['passed'] and r['policy']==POLICY and r['change_audit']['passed'],'Independent transition audit')
   ident=r['identity'];check(ident['problem_id']==pid and ident['native_source']==e['source_id'] and ident['array_hashes']==native_hashes,'Transition source/native binding')
   check(all(r['strict_feasibility'].values()) and r['checks']['selected_multiplier']['passed'],'Independent feasibility/multiplier')
  # Store only compact proof references, not thousands of JSON record objects.
  return refs
 def open(self,pid):
  check(pid in self.by_id,'Unknown trajectory')
  if self._opened!=pid:
   self.close();e=self.by_id[pid];folder=Path(e['folder']);meta=read(folder/'metadata.json')
   self._arrays={k:np.load(folder/v['file'],mmap_mode='r',allow_pickle=False) for k,v in meta['arrays'].items()};self._opened=pid
   p=self.specs[pid]['mechanical'];H,W=p['nely'],p['nelx'];T=e['states'];a=self._arrays
   check(a['x'].shape==a['rho'].shape==a['g_x'].shape==a['g_rho'].shape==(T,H*W),'Element layout')
   check(a['u'].shape==(T,2*(H+1)*(W+1)),'Nodal layout')
   check(np.array_equal(a['iteration'],np.arange(T)),'Iteration state timing')
   reason='change_tolerance' if a['update_change'][-1]<self.specs[pid]['optimizer']['tolerance'] else 'maxiter'
   check(reason==e['termination_reason'],'Capped/stopping classification')
   check(reason!='maxiter' or T-1==self.specs[pid]['optimizer']['maxiter'],'Capped length')
   self._native_transitions=read(folder/'transitions.json')
  return self._arrays
 def target(self,pid,t):
  e=self.by_id.get(pid);check(e is not None and isinstance(t,(int,np.integer)) and 0<=t<e['transitions'],'Terminal/invalid transition')
  a=self.open(pid);tr=self._native_transitions[t];delta=a['x'][t+1]-a['x'][t]
  check(tr['iteration']==t and tr['next_iteration']==t+1 and tr['valid'] and tr['optimizer']==self.specs[pid]['optimizer'],'Native OC action indexing')
  check(array_digest(delta)==tr['delta_digest'],'Target delta OC fingerprint')
  check(float(np.max(abs(delta)))==tr['maximum_change']==a['update_change'][t],'Target change bookkeeping')
  return delta.reshape(self.specs[pid]['mechanical']['nely'],self.specs[pid]['mechanical']['nelx']).copy()
 def close(self):
  for a in self._arrays.values():close_array(a)
  self._arrays={};self._opened=None
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
def close_array(a):
 if hasattr(a,'_mmap') and a._mmap is not None:a._mmap.close()
def read_lines(path):return [__import__('json').loads(l) for l in Path(path).read_text().splitlines()]

class TrajectoryDataset:
 def __init__(self,release,mode='A',history=3,normalization=None,problem_ids=None):
  from .features import schema
  schema(mode);check(isinstance(history,int) and history>=1,'Positive history length');self.release=release;self.mode=mode;self.history=history;self.normalization=normalization
  self.ids=list(problem_ids) if problem_ids is not None else list(release.by_id)
  check(len(set(self.ids))==len(self.ids) and all(p in release.by_id for p in self.ids),'Duplicate/unknown problems')
  self.offsets=np.cumsum([0]+[release.by_id[p]['transitions'] for p in self.ids])
 def __len__(self):return int(self.offsets[-1])
 def key(self,i):
  check(isinstance(i,(int,np.integer)) and 0<=i<len(self),'Example index');j=int(np.searchsorted(self.offsets,i,side='right')-1);return self.ids[j],int(i-self.offsets[j])
 def __getitem__(self,i):return self.sample(*self.key(i))
 def sample(self,pid,t):
  from .features import extract
  check(pid in self.ids,'Cross-dataset trajectory');target=self.release.target(pid,t)
  inputs=extract(self.release,pid,t,self.mode,self.history,self.normalization)
  return dict(inputs=inputs,target=target[None].astype(np.float32),proof=dict(problem_id=pid,transition=t,next_state=t+1,delta_digest=array_digest(target.ravel()),validation_record=self.release.records[pid][t],release_sha256=self.release.index_sha256))
 def validate_sample(self,sample):
  check(set(sample)=={'inputs','target','proof'},'Unexpected sample fields')
  p=sample['proof'];expected=self.sample(p['problem_id'],p['transition']);check(p==expected['proof'],'Sample proof/index/source')
  check(sample['inputs'].keys()==expected['inputs'].keys(),'Input schema fields')
  for k,v in expected['inputs'].items():
   actual=sample['inputs'][k]
   if isinstance(v,np.ndarray):check(isinstance(actual,np.ndarray) and actual.shape==v.shape and actual.dtype==v.dtype and np.array_equal(actual,v),'Causal input mismatch '+k)
   else:check(actual==v,'Input metadata '+k)
  check(np.array_equal(sample['target'],expected['target']),'Wrong target/action alignment')
  return True
