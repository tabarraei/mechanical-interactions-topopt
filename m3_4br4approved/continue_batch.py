"""Single-use sequential continuation; historical numerical code is unchanged."""
import json,os,sys,shutil,platform,importlib.metadata
from pathlib import Path
from datetime import datetime,timezone
from m3_4br4.common import ROOT,OLD,LIMITS,job,read
from m3_4br4.pipeline import Pipeline
from m3_4b.monitoring import atomic,digest,reduce_summaries,verify_trace
R4=ROOT/'m3_4br4/results/20261014T174028Z'
PROPOSAL=ROOT/'m3_4br4r1/results/20261015T020242Z'
APPROVED={ROOT/'m3_4br4r1/policy.py':'c74a81c5477622aaccce4c0866ebbda190dfe4d01fa74f985cd38d15cd31b120',ROOT/'m3_4br4r1/evidence.py':'fd30fdb7eb9392ce77d6a7203d5ee162f62b8d7c1d66416b31863e7be2349761',PROPOSAL/'R4_R1_ACCEPTANCE_MATRIX.json':'417686f60b4e1462cf76cb84264d14ad07640c5bb259422f524b83827e1a4881'}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def approved():
 for p,h in APPROVED.items():require(digest(p)==h,'Approved hash mismatch: '+str(p))
def preserve(out):
 before=read(out/'protected_before.json');bad=[p for p,h in before.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
 require(not bad,'Historical/source changes '+str(bad[:10]));return dict(passed=True,files=len(before),changed=bad)
def run(out):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 for n in ('reservations','monitor','checks','validation','preflight_analysis'): (out/n).mkdir()
 approved()
 atomic(out/'approval_receipt.json',dict(approved=True,user_response='yes',context='User approving the exact active R4_R1_POLICY_APPROVAL_REQUEST.md; category-C replacement only',recorded_utc=datetime.now(timezone.utc).isoformat(),policy='m3.4b.restart_acceptance.v2',bindings={str(p.relative_to(ROOT)):h for p,h in APPROVED.items()},scope='Category C restart/control equivalence only; A exact checkpoint and B full independent correctness remain mandatory',production_scope='Existing R4 original twelve-ID authorization; only unexecuted original indices10/11',excluded='164 planning manifest, training, SLURM, commit/push, new workload IDs'))
 protected=dict(read(PROPOSAL/'protected_before.json'))
 for group in ('m3_4br4r1',):
  for p in (ROOT/group).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:protected[str(p.relative_to(ROOT))]=digest(p)
 atomic(out/'protected_before.json',protected)
 sources={};archive=out/'source_snapshots';archive.mkdir()
 for group in ('m3_4br4approved','m3_4br4','m3_4br3','m3_4br2','m3_4br4r1','m3_4b'):
  for p in (ROOT/group).glob('*.py'):
   dest=archive/(group+'_'+p.name+'.txt');shutil.copyfile(p,dest);sources[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(out)))
 atomic(out/'executing_source_manifest.json',sources)
 pipeline=Pipeline(out)
 # Pure comparator process rechecks all frozen original sources/artifacts, exact
 # checkpoint content and complete retained correctness before the approved C.
 pipeline.execute('retained_preflight','retained-four-fixtures',[sys.executable,'-m','m3_4br4r1.analyze',str(out/'preflight_analysis')])
 matrix=read(out/'preflight_analysis/R4_R1_ACCEPTANCE_MATRIX.json')
 require(len(matrix)==4 and all(r['proposed']=='Accepted' and r['checkpoint']=='Accepted' and r['independent_mechanical_OC']=='Accepted' for r in matrix),'Retained A/B/C comparison')
 pipeline.execute('approved_policy_tests','comparator-only',[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','m3_4br4r1/test_policy.py','m3_4br4r1/test_retained.py'])
 require('55 passed' in (out/'monitor/approved_policy_tests.log').read_text(),'Comparator tests')
 oldgate=read(R4/'production_preflight.json');checks=oldgate['checks'].copy();require(all(v for k,v in checks.items() if k!='real_restart_control_numerical_equivalence'),'Other mandatory retained gates')
 # Bind current worker code to the already tested production observation wrapper.
 for path,e in read(R4/'executing_source_manifest.json').items():require(digest(ROOT/path)==e['sha256'],'Tested source changed '+path)
 manifest=read(OLD/'execution_manifest.json');require(len(manifest['jobs'])==12,'Original manifest count')
 jobs=[job(i)[1] for i in range(12)];require(len({j['problem_id'] for j in jobs})==12,'Duplicate IDs')
 require(all(job(i)[1]['partition']=='train' for i in range(12)),'Fixed partitions')
 for i in (10,11):
  pid=jobs[i]['problem_id']
  require(not (OLD/'native/samples'/pid).exists() and not (OLD/'native/incomplete'/pid).exists(),'Original job already present/incomplete')
 require(os.environ.get('OPENBLAS_NUM_THREADS')=='1','Original OPENBLAS environment')
 require(os.environ.get('MKL_NUM_THREADS') in (None,'16') and os.environ.get('OMP_NUM_THREADS') in (None,'16'),'Preserve solver default16 environment')
 env=dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','pypardiso')},OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS','solver default16'),OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS','solver default16'))
 retained=read(R4/'validation/native96_full_replay/summary.json')['identity']['environment']
 require(env['python']==retained['python'] and env['packages']['numpy']==retained['numpy'] and env['MKL_NUM_THREADS']==retained['MKL_NUM_THREADS'] and env['OPENBLAS_NUM_THREADS']==retained['OPENBLAS_NUM_THREADS'],'Retained environment changed')
 free=shutil.disk_usage(out).free;require(free>4*1024**3,'Require4GiB free for bounded two-job continuation')
 prior=read(R4/'execution_ledger.json');pipeline.production=prior['production'].copy();pipeline.validation=prior['validation'].copy();pipeline.save()
 # Read-only verified reuse; any numerical call raises before it occurs.
 for i in range(10):
  reused=pipeline.generate(i,OLD/'native',f'original_reuse_{i}',reuse=True)
  require(reused['result']['FEM']==reused['result']['OC']==0,'Completed reuse work')
  pipeline.production[str(i)]['approved_continuation_reuse']=reused;pipeline.save()
 integrity=preserve(out);approved();checks['real_restart_control_numerical_equivalence']=True
 preflight=dict(passed=True,checks=checks,category_C_policy='m3.4b.restart_acceptance.v2',approval_receipt_sha256=digest(out/'approval_receipt.json'),approved_bindings={str(p.relative_to(ROOT)):h for p,h in APPROVED.items()},category_A='Exact checkpoint verified',category_B='Complete mechanical/OC correctness verified',historical_failed_preflight=str(R4/'production_preflight.json'),retained_matrix=str(out/'preflight_analysis/R4_R1_ACCEPTANCE_MATRIX.json'),environment=env,free_disk_bytes=free,limits=LIMITS,integrity=integrity,original_indices_to_execute=[10,11],retained_existing_full_replays=10)
 atomic(out/'effective_production_preflight.json',preflight);print('APPROVED EFFECTIVE PREFLIGHT PASS',flush=True)
 for i in (10,11):
  approved();require(read(out/'effective_production_preflight.json')['passed'],'Effective preflight missing')
  for path,e in sources.items():require(digest(ROOT/path)==e['sha256'],'Dispatch source changed '+path)
  folder=out/'native/samples'/jobs[i]['problem_id'];require(not folder.exists() and not (out/'native/incomplete'/jobs[i]['problem_id']).exists(),'No duplicate/restart dispatch')
  try:
   generated=pipeline.generate(i,out/'native',f'production_generate_{i}')
   pipeline.production[str(i)]=dict(status='completed_new_approved_R4',folder=str(folder),generation=generated);pipeline.save()
   pipeline.validation[str(i)]=pipeline.replay(i,folder,f'production_replay_{i}');pipeline.save()
   pipeline.validation[str(i)]['supplemental_checks']=pipeline.checks(i,folder,f'features_FD_{i}');pipeline.save()
  except Exception as exc:
   atomic(out/f'quarantine_{i}.json',dict(folder=str(folder),error=str(exc),training_eligible=False));raise
 finalize(out,pipeline)
def finalize(out,pipeline):
 entries=[]
 for i in range(12):
  row=pipeline.production[str(i)];folder=Path(row['folder']);meta=read(folder/'metadata.json');spec=read(folder/'problem.json');val=pipeline.validation[str(i)]
  require(val['passed'] and val['supplemental_checks']['passed'],'All12 validation/feature gates')
  for name,e in meta['arrays'].items():require(digest(folder/e['file'])==e['sha256'],'Released native hash')
  require(spec==job(i)[2] and meta['source_id']==job(i)[0]['source_id'],'Release identity')
  require(not meta.get('reference_label_eligible',False),'Reference flag')
  entries.append(dict(index=i,problem_id=spec['problem_id'],partition=job(i)[1]['partition'],folder=str(folder),states=meta['state_count'],transitions=meta['completed_updates'],termination_reason=meta['termination_reason'],source_id=meta['source_id'],metadata_sha256=digest(folder/'metadata.json'),array_hashes={k:e['sha256'] for k,e in meta['arrays'].items()},validation=val,reference_label_eligible=False,mechanically_validated=True))
 resources=reduce_summaries(sorted((out/'monitor').glob('*.json')));atomic(out/'resource_measurements.json',resources)
 integrity=preserve(out);approved()
 atomic(out/'source_artifact_integrity.json',integrity)
 atomic(out/'accepted_dataset_index.json',dict(passed=True,policy='m3.4b.restart_acceptance.v2',original_only=True,configurations=entries))
 counts=dict(new_generation_FEM=sum(pipeline.production[str(i)]['generation']['result']['FEM'] for i in (10,11)),new_generation_OC=sum(pipeline.production[str(i)]['generation']['result']['OC'] for i in (10,11)),new_independent_validation_FEM=sum(pipeline.validation[str(i)]['FEM'] for i in (10,11)),new_independent_validation_OC=sum(pipeline.validation[str(i)]['OC'] for i in (10,11)),new_supplemental_FEM=6,new_supplemental_OC=0,completed_reuse_FEM=0,completed_reuse_OC=0,retained_prior_accounting=str(R4/'execution_ledger.json'))
 summary=dict(passed=True,complete=12,previously_generated=10,newly_generated=2,independently_validated=12,states=sum(e['states'] for e in entries),transitions=sum(e['transitions'] for e in entries),termination_reasons={k:sum(e['termination_reason']==k for e in entries) for k in {e['termination_reason'] for e in entries}},quarantined=0,volume_feasibility_violations=0,reference_labels=0,calls=counts,resources=resources,scaled_execution_authorized=False)
 atomic(out/'final_readiness.json',summary)
 atomic(out/'production_ledger.json',pipeline.production);atomic(out/'validation_ledger.json',pipeline.validation)
 print(json.dumps({k:v for k,v in summary.items() if k not in ('resources',)}),flush=True)
if __name__=='__main__':
 out=Path(sys.argv[1])
 try:run(out)
 except Exception as exc:
  if out.exists():atomic(out/'BLOCKER.json',dict(passed=False,error=str(exc),production_started=(out/'native').exists(),accepted_dataset_released=False))
  raise
