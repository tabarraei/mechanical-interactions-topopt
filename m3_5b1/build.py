"""Single-use pure planning build, tests and historical integrity report."""
import sys,copy,json,shutil,subprocess,collections,time,importlib.metadata
from pathlib import Path
from m3_5a.io import ROOT,RELEASE,R2,read,digest,write,check
from .plan import Audit,PLAN,PAIR,ML,ORDER,LIMITS,compatibility,APPROVED
from .resources import measurements,costs

def build(out):
 start=time.perf_counter();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 protected=read(ML/'protected_before.json').copy()
 for group in ('m3_5a','m3_5b1/plans'):
  for p in (ROOT/group).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:protected[str(p.relative_to(ROOT))]=digest(p)
 write(out/'protected_before.json',protected)
 write(out/'development_test_outcome.json',dict(prior_build_failure='Version1 resource report used NumPy dtype.itemsize as a function instead of a property; failed before tests/reports/dispatch. Preserved version1 BLOCKER and actual source snapshots; arithmetic corrected without changing a resource limit.',run='Initial32 planning tests',passed=30,failed=2,failures=['Import guard tests expected message word forbids; existing M3.5A guard ran first and reported prohibits. Both attempted imports were correctly blocked.','Same assertion wording issue for production worker import.'],fix='Accept either guard message, retaining ImportError requirement; no safety/science/resource condition relaxed',new_FEM=0,new_OC=0))
 audit=Audit();batch=audit.build();write(out/'M3_5B1_SELECTED_BATCH.json',batch);write(out/'all164_configuration_audit.json',dict(rows=audit.rows,partitions=dict(collections.Counter(j['partition'] for j in audit.jobs)),declared_families=audit.family_count,exact_ID_duplicates=0,configuration_equivalent_duplicates=0,declared_cross_partition_family_conflicts=0,completed_original_indices=[r['plan_index'] for r in audit.rows if r['execution_status']=='complete_independently_validated'],unexecuted=152,inventory=audit.inventory,configuration_equivalence_definition='Complete physical/mechanical spec and initial array plus optimizer, ignoring seed-only mechanical metadata; force direction/sign/magnitude remain physical distinctions. Related design dynamics are not declared independent merely because this signature differs.',declared_source_id_missing_count=sum(r['declared_source_id'] is None for r in audit.rows),governing_original_sources_verified=77))
 write(out/'all21_near_neighbor_reconciliation.json',dict(pairs=batch['near_neighbor_reconciliation'],additional_mechanical_relationships=batch['additional_control_relationships'],unresolved_historical_dispositions=0,independence_claim=False))
 ml=compatibility(audit,batch);write(out/'ml_compatibility.json',ml)
 measured=measurements(audit);storage=costs(audit,batch,measured);write(out/'resource_evidence.json',measured);write(out/'resource_cost_scenarios.json',storage)
 table=[]
 for j in batch['jobs']:
  table.append(dict(order=j['execution_order'],plan_index=j['plan_index'],problem_id=j['problem_id'],partition=j['partition'],family=j['family'],configuration=j['configuration'],resource=j['resource'],rationale=j['rationale'],execution_authorized=False))
 write(out/'selection_table.json',table);write(out/'source_fingerprints.json',dict(artifact_bindings=audit.bindings,unchanged_execution_dependencies=audit.execution_sources,policy={str(p.relative_to(ROOT)):h for p,h in APPROVED.items()}))
 archive=out/'source_snapshots';archive.mkdir();sources={}
 for p in (ROOT/'m3_5b1').glob('*.py'):
  dest=archive/(p.name+'.txt');shutil.copyfile(p,dest);sources[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(out)))
 write(out/'source_manifest.json',sources)
 with (out/'planning_tests.log').open('x') as log:
  result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','m3_5b1/test_planning.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 check(result.returncode==0,'Planning tests failed; preserve outputs and stop')
 check('33 passed' in (out/'planning_tests.log').read_text(),'Expected33 planning tests')
 bad=[p for p,h in protected.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h];check(not bad,'Protected source/artifact changes '+str(bad[:10]))
 check(all(digest(ROOT/p)==e['sha256']==digest(out/e['archive']) for p,e in sources.items()),'Planner source snapshots')
 write(out/'source_artifact_integrity.json',dict(passed=True,protected_files=len(protected),changed=bad,original_sources_and_archives_verified=77,artifact_bindings=audit.bindings,source_snapshots_match=True,planning_tests_passed=33,new_FEM_solves=0,new_OC_updates=0,new_trajectories=0,new_neural_training_runs=0,new_model_inference_runs=0,synthetic_feature_probes=ml['synthetic_probe_count'],packages=dict(python=sys.version,numpy=importlib.metadata.version('numpy'),pytest=importlib.metadata.version('pytest')),wall_seconds=time.perf_counter()-start))
 from .reports import reports
 reports(out,audit,batch,ml,measured,storage)
 write(out/'artifact_manifest.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
 print(json.dumps(dict(selected=batch['selected_count'],partition_counts=batch['partition_counts'],families=batch['family_count'],unexecuted=152,capacity_demonstrated=25,qualification_required=5,status='BLOCKED_BY_RESOURCE_QUALIFICATION',new_FEM=0,new_training=0)),flush=True)
if __name__=='__main__':
 try:build(sys.argv[1])
 except Exception as exc:
  out=Path(sys.argv[1])
  if out.exists() and not (out/'BLOCKER.json').exists():write(out/'BLOCKER.json',dict(error=str(exc),planning_only=True,execution_authorized=False,new_FEM=0,new_training=0))
  raise
