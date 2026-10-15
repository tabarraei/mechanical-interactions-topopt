"""Verified combined-release adapter; unchanged M3.5A tensors/dataset reused."""
from pathlib import Path
import numpy as np
from .common import *
# This module is read-only. Numerical workers must use a separate process.
from m3_5a.loader import Release as OriginalRelease,TrajectoryDataset,close_array
from m3_5a.normalization import Normalization
from m3_5a.features import schema
from .validation import verify_acceptance
class ExpandedRelease(OriginalRelease):
 def __init__(self,index,expected_sha256):
  index=Path(index).resolve();check(digest(index)==expected_sha256,'Combined release index hash');doc=read(index)
  check(set(doc)=={'schema_version','manifest_sha256','original_index','entries'} and doc['schema_version']=='m3.5b2.combined_release.v1' and doc['manifest_sha256']==PLAN_SHA,'Combined release schema/provenance')
  check(doc['original_index']==dict(path=str(ORIGINAL),sha256=ORIGINAL_SHA),'Original release reference')
  selection=Selection();base=OriginalRelease(ORIGINAL,ORIGINAL_SHA);self.original_verification=base;self.index=index;self.index_sha256=expected_sha256;self.entries=[];self.by_id={};self.specs={};self.records={};self._arrays={};self._opened=None
  ids=[e['problem_id'] for e in doc['entries']];check(len(ids)==len(set(ids)) and selection.original_ids.issubset(ids),'Duplicate IDs/missing original12');check(set(ids).issubset(selection.original_ids|set(selection.jobs)),'Unknown expansion release ID');families={};equivalences=set()
  for entry in doc['entries']:
   pid=entry['problem_id']
   if pid in selection.original_ids:
    check(entry==base.by_id[pid],'Original released entry changed');spec=base.specs[pid];refs=base.records[pid]
   else:
    job=selection.jobs[pid];spec=selection.specs[pid]
    check(set(entry)=={'problem_id','partition','family','folder','states','transitions','termination_reason','source_id','metadata_sha256','array_hashes','validation','authorization','mechanically_validated','reference_label_eligible','validation_status'},'Missing/unknown new release provenance')
    check(entry['source_id']==selection.manifest['source_binding']['original_execution_source_id'],'Missing/wrong source provenance')
    check(entry['partition']==job['partition'] and entry['family']==job['family'],'Cross-partition/family reassignment')
    check(entry.get('mechanically_validated') is True and entry.get('reference_label_eligible') is False and entry.get('validation_status')=='complete','Unvalidated/incomplete/reference release')
    self._verify_new(entry,spec,selection);refs=[str(Path(entry['validation']['evidence'])/'transitions.jsonl')]*entry['transitions']
   family=spec['related_group'];families.setdefault(family,set()).add(entry['partition']);check(len(families[family])==1,'Cross-partition family conflict');eq=equivalent(spec);check(eq not in equivalences,'Configuration-equivalent duplicate');equivalences.add(eq)
   folder=Path(entry['folder']);meta=read(folder/'metadata.json');check(digest(folder/'metadata.json')==entry['metadata_sha256'],'Release metadata hash');check(meta['state_count']==entry['states'] and meta['completed_updates']==entry['transitions']==entry['states']-1,'State/transition count')
   p=spec['mechanical'];N=p['nelx']*p['nely'];nd=2*(p['nelx']+1)*(p['nely']+1)
   check(meta['arrays']['x']['shape']==[entry['states'],N] and meta['arrays']['rho']['shape']==[entry['states'],N] and meta['arrays']['u']['shape']==[entry['states'],nd],'Incompatible element/nodal mesh layout')
   check(meta['termination_reason']==entry['termination_reason'] and not meta['reference_label_eligible'] and not meta['optimality_certified'],'Termination/reference provenance')
   self.entries.append(entry);self.by_id[pid]=entry;self.specs[pid]=spec;self.records[pid]=refs
  base.close();self.training_release_identity=canonical([dict(problem_id=e['problem_id'],metadata_sha256=e['metadata_sha256'],array_hashes=e['array_hashes']) for e in self.entries if e['partition']=='train'])
 def _verify_new(self,entry,spec,selection):
  from .authorization import historical_authorization
  a=entry['authorization'];check(set(a)=={'path','sha256'},'Missing issued authorization provenance');payload=read(a['path']);check(digest(a['path'])==a['sha256'],'Release authorization provenance')
  check(set(entry['validation']['execution'])=={'authorization_sha256','resumed','restart_control','executed_at'},'Missing execution-time provenance')
  historical_authorization(a['path'],selection,kind='production',stage=payload['stage'],ids=[entry['problem_id']],output_root=payload['output_root'],executed_at=entry['validation']['execution']['executed_at'])
  check(Path(entry['folder']).resolve()==Path(payload['output_root'])/'native/samples'/entry['problem_id'],'Release folder outside approved root')
  check(payload['experiment']['release_allowed'] and 'adopt' in payload['experiment']['actions'],'Explicit release adoption required')
  from .registry import Registry
  Registry(ROOT,selection).eligible(entry['problem_id'],own_root=payload['output_root'],allow_owned=True)
  v=entry['validation'];check(v['execution']['authorization_sha256']==a['sha256'] and (payload['resume_bindings'].get(entry['problem_id'],{}).get('purpose')!='resume' or v['execution']['resumed'] is True),'Execution/resume authorization binding');check(set(v)=={'evidence','acceptance_path','acceptance_sha256','execution'},'Validation provenance fields')
  check(digest(v['acceptance_path'])==v['acceptance_sha256'],'Acceptance file hash');actual=verify_acceptance(entry['folder'],v['evidence'],spec,entry['source_id'],v['execution']);check(read(v['acceptance_path'])==actual,'Independent certificate inconsistent')
  check(actual['native_hashes']==entry['array_hashes'] and actual['states']==entry['states'] and actual['transitions']==entry['transitions'] and actual['termination_reason']==entry['termination_reason'],'Release validation binding')

class TrainingNormalization(Normalization):
 @classmethod
 def fit(cls,release,fit_samples=None):
  train=sorted(pid for pid,e in release.by_id.items() if e['partition']=='train');check(bool(train),'No accepted training data')
  requested=fit_samples if fit_samples is not None else [(pid,0) for pid in train]
  check(sorted(requested)==[(pid,0) for pid in train],'Normalization must fit exactly accepted train state0; held-out/future data forbidden')
  # Reuse the original numerical RMS implementation, with a train-only view.
  class View:
   by_id={pid:release.by_id[pid] for pid in train};specs=release.specs;index_sha256=release.training_release_identity
   def open(self,pid):check(pid in self.by_id,'Held-out normalization access');return release.open(pid)
  original=Normalization.fit(View(),requested)
  record=dict(original.record,adapter_version='m3.5b2.training_only_normalization.v1',training_release_identity=release.training_release_identity,training_ids=train,feature_schemas=[schema('A')['version'],schema('B')['version']])
  obj=cls(record);obj._verified_training_identity=release.training_release_identity;return obj
 @classmethod
 def load(cls,path,release):
  record=read(path);expected=cls.fit(release);check(record==expected.record,'Normalization statistics do not reproduce accepted train state0');return expected
 def validate(self,release):
  check(canonical(self.record)==self.identifier,'Normalization record mutated');check(self.record.get('adapter_version')=='m3.5b2.training_only_normalization.v1' and self.record['training_release_identity']==release.training_release_identity,'Training release identity/version')
  train=sorted(pid for pid,e in release.by_id.items() if e['partition']=='train');check(self.record['training_ids']==train and self.record['feature_schemas']==[schema('A')['version'],schema('B')['version']],'Normalizer train IDs/schema')
  class View:
   by_id={pid:release.by_id[pid] for pid in train};index_sha256=release.training_release_identity
  check(sorted(x['problem_id'] for x in self.record['fit_samples'])==train,'Normalization must bind all accepted training prefixes')
  super().validate(View())
  check(getattr(self,'_verified_training_identity',None)==release.training_release_identity,'Normalizer must be verified by fit/load before use')
  return True
