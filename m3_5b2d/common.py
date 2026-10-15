"""Standard-library-only target and immutable evidence binding."""
from pathlib import Path
from m3_5b2.common import ROOT,Selection,read,write,atomic,digest,canonical,equivalent,environment,check,LIMITS
from m3_5b2s.alias_schema import HISTORICAL,SELECTED,MANIFEST,EQUIVALENCE
HROOT=ROOT/'m3_3c/completion_generations/20261013T203017221805Z'
INPUT=HROOT/'samples'/HISTORICAL
REPORT=ROOT/'m3_5b2d/reports/20261015_candidate_v1'
AUTHORITY=Path('/users/atabarra/.codex/approvals/m3_5b2d_historical_admission.json')
PROTOCOL_BINDING_SHA='6a3d2554f109205e6313e2052dcd404dba0437c62926e3c0de07d7ada5e4bb74'
WORK=dict(FEM=1005,OC=500,selected_trials=500,bisection_trials=52500,workers=2,worker_seconds=1800)
STAGES={'replay':dict(FEM=1002,OC=500,selected_trials=500,bisection_trials=52500),'supplemental':dict(FEM=3,OC=0,selected_trials=0,bisection_trials=0)}
GATES={'identity','primary_mechanics','boundary','displacement','compliance','sensitivities','COO','OC_action','selected_multiplier','strict_feasibility','stopping','v2','v3','raw_operands','supplemental','final_checkpoint','resources'}
def exact(d,fields,label):check(isinstance(d,dict) and set(d)==set(fields),label+' fields')
def target_binding():
 s=Selection();s.verify_sources();check(digest(ROOT/'m3_5b2s/reports/20261015_protocol_v1/alias_bindings.json')==PROTOCOL_BINDING_SHA,'Frozen protocol binding');b=read(ROOT/'m3_5b2s/reports/20261015_protocol_v1/alias_bindings.json');a=read(INPUT/'problem.json');meta=read(INPUT/'metadata.json')
 check(digest(HROOT/'source_manifest.json')==b['historical']['source_manifest_sha256'],'Historical source manifest bytes')
 check(a==b['historical']['spec'] and meta['source_id']==b['historical']['source_id'],'Historical target/source identity')
 check(b['selected']['source_id']==s.manifest['source_binding']['original_execution_source_id'],'Selected source binding')
 for name,h in b['historical']['artifact_hashes'].items():check(digest(ROOT/name)==h,'Originally pinned historical input hash')
 check(equivalent(a)==equivalent(s.specs[SELECTED])==EQUIVALENCE,'Target equivalence')
 hashes={str(INPUT/p):digest(INPUT/p) for p in ('problem.json','metadata.json','journal.json','transitions.json')}
 for v in meta['arrays'].values():
  p=(INPUT/v['file']).resolve();check(p.is_relative_to(INPUT),'Native path traversal');check(digest(p)==v['sha256'],'Historical input data hash');hashes[str(p)]=v['sha256']
 check(meta['state_count']==501 and meta['completed_updates']==500 and meta['termination_reason']=='maxiter' and not meta['reference_label_eligible'] and not meta['optimality_certified'],'Target completion/reference')
 sources={**s.production_sources,**s.validation_sources}
 old=read(HROOT/'source_manifest.json');check(canonical(old)==meta['source_id'],'Original67 source identity')
 for n,e in old.items():check(digest(HROOT/e['archive'])==e['sha256'],'Original67 archive')
 sources.update({str((HROOT/e['archive']).relative_to(ROOT)):e['sha256'] for e in old.values()})
 sources.update({str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).parent.glob('*.py')})
 sources['m3_5b2s/reports/20261015_protocol_v1/alias_bindings.json']=PROTOCOL_BINDING_SHA
 sources['m3_4b/monitoring.py']=digest(ROOT/'m3_4b/monitoring.py')
 return dict(historical_id=HISTORICAL,selected_id=SELECTED,manifest_sha256=MANIFEST,equivalence_sha256=EQUIVALENCE,input_root=str(INPUT),historical_source_id=meta['source_id'],selected_source_id=b['selected']['source_id'],input_hashes=hashes,sources=sources,physical_history_key=b['physical_history_key'],historical_spec_sha256=canonical(a),selected_spec_sha256=canonical(s.specs[SELECTED]))
def verify_binding(b):
 check(b==target_binding(),'Exact target/source/data binding changed');return b
def registry_check(binding):
 from m3_5b2.registry import Registry
 r=Registry(ROOT,Selection());found=[]
 for pid,records in r.records.items():
  for x in records:
   if pid in (HISTORICAL,SELECTED) or x['equivalence_sha256']==EQUIVALENCE:
    check(pid==HISTORICAL and Path(x['path']).resolve().is_relative_to(INPUT) and x['status']=='previously_attempted','Unknown duplicate/quarantine/released equivalent')
    found.append(x)
 check(len(found)>=3,'Global target registry coverage missing')
 return dict(coverage_complete=True,coverage_sha256=r.coverage_sha256,physical_history_key=binding['physical_history_key'],target_remains_ineligible_for_generation=True)
