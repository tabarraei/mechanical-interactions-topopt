#!/usr/bin/env python
"""M3.4B sequential, streamed, manifest-bound twelve-problem controlled batch."""
import argparse,json,os,sys,time,shutil,fcntl,subprocess,platform,math
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
from m3_4b.monitoring import atomic,digest,memory,run,reduce_summaries,trace_rows,verify_trace
ROOT=Path(__file__).resolve().parent
SCHEMA='m3.4b.production.v1'
MAX_AUTHORIZED=16
LIMITS=dict(worker_rss_mib=1024,parent_rss_mib=768,combined_rss_mib=1792,worker_timeout_seconds=900,poll_seconds=.05,sequential_only=True)
def stamp():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def read(p):return json.loads(Path(p).read_text())
def protected(out):
 old=read(out/'protected_before.json');bad=[n for n,h in old.items() if not (ROOT/n).is_file() or digest(ROOT/n)!=h]
 if bad:raise ValueError('Protected files changed: '+str(bad))
 return len(old)
def archive(out):
 files=list(ROOT.glob('*.py'))
 for folder in ('topopt_utils','topo_transformer','inr_render','tests','m3_4b'):files.extend((ROOT/folder).rglob('*.py'))
 records={}
 for file in sorted(set(files)):
  if 'generations' in file.parts:continue
  name=file.relative_to(ROOT);target=out/'source_snapshots'/name
  if name.parts[0]=='tests':target=target.with_suffix('.py.txt')
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target);records[str(name)]=dict(sha256=digest(file),archive=str(target.relative_to(out)))
 atomic(out/'source_manifest.json',records);return records

def compatibility():
 import importlib,importlib.metadata
 result={}
 for name in ('numpy','scipy','torch','pypardiso'):
  module=importlib.import_module(name);result[name]=dict(version=importlib.metadata.version(name),path=module.__file__)
 # Check native package linkage without solving a state; real calls remain worker-owned.
 from pypardiso import PyPardisoSolver
 context=PyPardisoSolver();context.free_memory(everything=True)
 return dict(packages=result,python=sys.version,executable=sys.executable,threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},solver_default16_preserved=True)

def binding(out):
 from topopt_utils.fea.trajectory_dataset import canonical_hash,load_problem
 m=read(out/'execution_manifest.json');b=read(out/'manifest_binding.json');sources=read(out/'source_manifest.json')
 if canonical_hash(m)!=b['manifest_sha256'] or canonical_hash(sources)!=m['source_id']:raise ValueError('Immutable manifest/source identity changed')
 if not 12<=len(m['jobs'])<=MAX_AUTHORIZED or len({j['problem_id'] for j in m['jobs']})!=len(m['jobs']):raise ValueError('Authorized count/unique ID guard')
 if m['limits']!=LIMITS or m['schema_version']!=SCHEMA:raise ValueError('Resource/schema gate changed')
 for n,r in sources.items():
  if digest(ROOT/n)!=r['sha256'] or digest(out/r['archive'])!=r['sha256']:raise ValueError('Executing/archive source incompatibility: '+n)
 for n,h in b['files'].items():
  if digest(out/n)!=h:raise ValueError('Immutable configuration changed: '+n)
 for j in m['jobs']:
  spec=read(out/j['spec']);p,x=load_problem(spec)
  if spec['problem_id']!=j['problem_id'] or canonical_hash(spec)!=j['spec_sha256'] or p.n>4608 or p.nelx>96 or p.nely>48 or p.maxiter!=500 or p.tolerance!=.001 or spec['optimizer']['option']!='feasible_upper_v1':raise ValueError('Unsupported execution specification')
 return m

def reserve_authorization(out,m,ledger_path=None):
 ledger_path=Path(ledger_path) if ledger_path else ROOT/'m3_4b/authorization.json';ledger_path.parent.mkdir(parents=True,exist_ok=True)
 with ledger_path.with_suffix('.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);ids=[j['problem_id'] for j in m['jobs']]
  if len(ids)>MAX_AUTHORIZED or len(set(ids))!=len(ids):raise ValueError('Batch exceeds authorized total')
  record=dict(generation=str(out.resolve()),problem_ids=sorted(ids),source_id=m['source_id'],maximum_authorized=MAX_AUTHORIZED)
  if ledger_path.exists():
   if read(ledger_path)!=record:raise ValueError('One batch only: commands/new generations cannot bypass authorization ledger')
  else:atomic(ledger_path,record)
 return record

def check_authorization(out,m):
 expected=dict(generation=str(out.resolve()),problem_ids=sorted(j['problem_id'] for j in m['jobs']),source_id=m['source_id'],maximum_authorized=MAX_AUTHORIZED)
 if read(ROOT/'m3_4b/authorization.json')!=expected:raise ValueError('Worker has no matching persistent authorization')

def plan(out):
 from topopt_utils.fea.trajectory_dataset import canonical_hash
 from m3_4b.design import configurations,specification,entry,batch_indices,audit
 if out.exists() and set(p.name for p in out.iterdir())-{'M3_4B_IMPLEMENTATION_PLAN.md','protected_before.json','development_source_snapshots','development_test_logs'}:raise ValueError('New generation required')
 out.mkdir(parents=True,exist_ok=True)
 if not (out/'protected_before.json').exists():
  old={}
  for p in ROOT.rglob('*'):
   if p.is_file() and not any(n in p.parts for n in ('.git','__pycache__','.pytest_cache','m3_4b')) and p.name not in ('run_m3_4b.py','test_scalable_production.py'):old[str(p.relative_to(ROOT))]=digest(p)
  atomic(out/'protected_before.json',old)
 if not (out/'M3_4B_IMPLEMENTATION_PLAN.md').exists():shutil.copyfile(sorted((ROOT/'m3_4b/generations').glob('*/M3_4B_IMPLEMENTATION_PLAN.md'))[0],out/'M3_4B_IMPLEMENTATION_PLAN.md')
 for folder in ('specs','summaries','attempts','test_logs'): (out/folder).mkdir(exist_ok=True)
 sources=archive(out);entries=[]
 for i,(cfg,part,axis) in enumerate(configurations()):
  spec=specification(**cfg);again=specification(**cfg)
  if spec!=again:raise ValueError('Nondeterministic configuration')
  path=f'specs/{i:03d}.json';atomic(out/path,spec);entries.append(entry(spec,path,part,axis))
 proposal=dict(schema_version=SCHEMA,status='planning_only_not_authorized',jobs=entries,future192='Planning envelope only, no executable192 specification')
 atomic(out/'proposed_manifest.json',proposal);a=audit(entries);atomic(out/'diversity_audit.json',a);atomic(out/'split_assignments.json',{e['problem_id']:dict(partition=e['partition'],family=e['family'],ood_axis=e['ood_axis']) for e in entries})
 jobs=[entries[i] for i in batch_indices(entries)];m=dict(schema_version=SCHEMA,generation=out.name,source_id=canonical_hash(sources),jobs=jobs,limits=LIMITS,authorized_maximum=MAX_AUTHORIZED,proposal_execution=False)
 atomic(out/'execution_manifest.json',m);files={e['spec']:digest(out/e['spec']) for e in entries};files.update({n:digest(out/n) for n in ('proposed_manifest.json','split_assignments.json','diversity_audit.json')});atomic(out/'manifest_binding.json',dict(manifest_sha256=canonical_hash(m),files=files));atomic(out/'progress.json',dict(status='planned',generation=[],validation=[],failures=[]));atomic(out/'environment.json',compatibility())
 old=ROOT/'m3_4/generations/20261014T000500998078Z';prior=read(old/'final_supplement_validation.json');old_jobs=read(old/'execution_manifest.json')['jobs']
 for n,r in read(old/'source_manifest.json').items():
  if digest(old/r['archive'])!=r['sha256'] or digest(ROOT/n)!=r['sha256']:raise ValueError('M3.4 archived/live source changed')
 native=[read(old/'pilot/samples'/j['problem_id']/'metadata.json') for j in old_jobs]
 if len(native)!=7 or sum(r['state_count'] for r in native)!=1619 or sum(r['completed_updates'] for r in native)!=1612 or len(read(old/'recovery_tests.json')['cases'])!=4:raise ValueError('Prior evidence mismatch')
 prior_a=read(old/'diversity_audit.json');prior_counts=Counter(e['family'] for e in read(old/'proposed_manifest.json')['jobs'] if e['partition']=='train')
 if len(prior_counts)!=18 or sum(prior_counts.values())!=92 or max(prior_counts.values())!=20:raise ValueError('Prior diversity mismatch')
 atomic(out/'prior_evidence_audit.json',dict(verified=True,native_states=1619,native_updates=1612,worker_peak_mib=prior['worker_peak_mib'],parent_peak_mib=prior['pilot_parent_peak_mib'],old_tests=prior['tests'],old_families=prior_a['mechanical_families'],old_train_configurations=92,old_largest_train_family=20,old_maximum8_unchanged=True,archived_sources_checked=True,ownership_calls=[x['result']['FEM_evaluations'] for x in read(old/'lifecycle_benchmarks.json')]))
 atomic(out/'before_after_design.json',dict(original=dict(configurations=164,families=36,train_configurations=92,train_families=18,maximum_family=20,train_aspects=[2]),revised=a,quota_rationale='Four volume levels per family plus distributed controlled variants. Max6=4volumes+at most2controls; retains164 and36 families, includes train geometry and held96. No category erased to meet quota.'))
 preflight(out);print(out,flush=True)

def disk_gate(required,available,safety=2.):
 if required*safety>available:raise ValueError('Disk preflight failed: estimated native/checkpoint/replay/archive/temp plus safety exceeds free bytes')
 return True

def preflight(out):
 m=binding(out);current=compatibility();recorded=read(out/'environment.json')
 if current!=recorded:raise ValueError('Source/package/environment mismatch')
 native=0;work=[]
 for j in m['jobs']:
  n=math.prod(j['configuration']['mesh']);ratio=n/1152;gen=.45*501*ratio**.75;val=.1*501*ratio**.75;rss=700+60*ratio;size=int(501*.12*2**20*ratio);native+=size
  if max(gen,val)>=LIMITS['worker_timeout_seconds'] or rss>=LIMITS['worker_rss_mib']:raise ValueError('Estimated resource gate failed; do not launch')
  work.append(dict(problem_id=j['problem_id'],generation_seconds=gen,replay_seconds=val,worker_peak_estimate_mib=rss,native_bytes=size,partition=j['partition']))
 # Concurrent working compressed checkpoints + final arrays +4 recovery copies + source/temp/export allowance.
 sources=sum((out/r['archive']).stat().st_size for r in read(out/'source_manifest.json').values());required=3*native+4*work[0]['native_bytes']+sources+512*2**20;free=shutil.disk_usage(out).free;temp_free=shutil.disk_usage('/tmp').free
 disk_gate(required,free);disk_gate(512*2**20,temp_free)
 answer=dict(passed=True,optimization_executed=False,workload=work,job_count=len(m['jobs']),proposal_count=164,split_counts=read(out/'diversity_audit.json')['partitions'],limits=LIMITS,estimated_native_bytes=native,estimated_checkpoint_replay_recovery_source_temp_bytes=required,free_bytes=free,temp_free_bytes=temp_free,disk_safety_factor=2.,source_archive_bytes=sources,exact_packages_verified=True,solver_available=True,count_guard='12reserved IDs from one batch, hard16 across resume/new generations; no flag override')
 atomic(out/('preflight_'+stamp()+'.json'),answer);return answer

def launch(out,index,task,work=None,fault=None,monitor_fault=False,expected=0):
 m=binding(out);check_authorization(out,m);j=m['jobs'][index];attempt=task+'_'+str(index)+'_'+stamp();result=out/'attempts'/(attempt+'_result.json');directory=out/'attempts/monitor';args=[sys.executable,str(ROOT/'run_m3_4b.py'),'_worker','--output',str(out.resolve()),'--index',str(index),'--task',task,'--result',str(result.resolve())]
 if work:args+=['--work',str(Path(work).resolve())]
 if fault:args+=['--fault',fault]
 interrupt=str(Path(work)/'incomplete'/j['problem_id']/'journal.json') if monitor_fault else False
 summary=run(args,directory,attempt,j['problem_id'],m['limits'],expected=expected,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),interrupt_monitor=interrupt)
 if result.exists():
  data=read(result);summary['worker_reported_hwm_mib']=data.get('worker_self_hwm_mib');summary['result_path']=str(result);summary['result_sha256']=digest(result)
  atomic(directory/(attempt+'.json'),summary)
 else:data={}
 with (out/'worker_index.jsonl').open('a') as f:f.write(json.dumps(dict(summary_path=str(directory/(attempt+'.json')),job_id=attempt,problem_id=j['problem_id'],task=task,index=index))+'\n');f.flush();os.fsync(f.fileno())
 return data,summary

def stress(out):
 if (out/'monitor_stress.json').exists():return read(out/'monitor_stress.json')
 baseline=memory().get('VmRSS',0);rows=[]
 for i in range(100):
  s=run([sys.executable,'-c','import time; a=bytearray(1048576); time.sleep(.025)'],out/'synthetic',f'synthetic_{i:04d}','synthetic_no_FEM',LIMITS)
  rows.append(dict(job=i,parent_after_mib=s['parent_after_mib'],parent_peak_mib=s['parent_peak_mib'],trace_bytes=s['trace_bytes'],samples=s['sample_count']))
 reduced=reduce_summaries(sorted((out/'synthetic').glob('*.json')));after=memory().get('VmRSS',0);late=[r['parent_after_mib'] for r in rows[20:]]
 if max(late)-min(late)>16:raise RuntimeError('Sustained synthetic parent growth; stop before production')
 comparison=[]
 # Same controlled raw workload in separate inexpensive processes; no FEM.
 for mode in ('retain','stream'):
  result=out/(f'retention_{mode}.json');cmd=[sys.executable,str(ROOT/'run_m3_4b.py'),'_retention','--output',str(out),'--mode',mode,'--result',str(result)]
  s=run(cmd,out/'synthetic_compare',mode,'synthetic_retention_comparison',LIMITS);comparison.append(dict(mode=mode,result=read(result),monitor=s))
 answer=dict(passed=True,jobs=100,real_FEM=0,parent_before_mib=baseline,parent_after_jobs_mib=after,parent_after_reducer_mib=memory().get('VmRSS',0),rows=rows,reduced=reduced,late_range_mib=max(late)-min(late),comparison=comparison,qualification='Observed100 short jobs; CPU/write wall and trace bytes measured; not a constant-memory theorem. Raw traces streamed, row summaries are compact O(jobs).')
 atomic(out/'monitor_stress.json',answer);return answer

def batch(out):
 m=binding(out);progress=read(out/'progress.json')
 if progress['status']=='completed':raise ValueError('Completed generation cannot be overwritten or resumed')
 preflight(out);stress(out);reserve_authorization(out,m)
 for i,j in enumerate(m['jobs']):
  print('M3.4B job',i+1,'/',len(m['jobs']),j['configuration']['mesh'],j['configuration']['direction'],flush=True)
  try:
   if i not in progress['generation']:
    data,s=launch(out,i,'generate');atomic(out/'summaries'/f'gen_{i}.json',dict(result=data,monitor=s));progress['generation'].append(i);atomic(out/'progress.json',progress)
   if i not in progress['validation']:
    data,s=launch(out,i,'validate');atomic(out/'summaries'/f'val_{i}.json',dict(result=data,monitor=s));progress['validation'].append(i);atomic(out/'progress.json',progress)
  except Exception as e:
   progress['status']='failed_closed';progress['failures'].append(dict(index=i,error=str(e),utc=stamp()));atomic(out/'progress.json',progress);raise
 recovery(out);finish(out)

def recovery(out):
 if (out/'recovery_validation.json').exists():return
 import numpy as np
 from topopt_utils.fea.trajectory_dataset import Trajectory
 from run_m3_4 import checkpoint_snapshot
 m=binding(out);pid=m['jobs'][0]['problem_id'];original=Trajectory(out/'native/samples'/pid);records=[]
 for label,fault in [('generation','before_update'),('checkpoint','recording'),('monitor',None),('finalization','finalization')]:
  work=out/'recovery'/label;work.mkdir(parents=True,exist_ok=True)
  _,failed=launch(out,0,'generate',work=work,fault=fault,monitor_fault=label=='monitor',expected=77 if fault else 0)
  snapshot=checkpoint_snapshot(work,pid);stage=work/'incomplete'/pid;spec_sha=digest(stage/'problem.json');committed=read(stage/'journal.json');prefix={}
  for e in committed['states']:
   with np.load(stage/e['file']) as frame:prefix[int(frame['iteration'])]={k:np.array(frame[k]) for k in ('x','rho','u','g_x','g_rho','compliance')}
  result,resumed=launch(out,0,'generate',work=work);tr=Trajectory(work/'samples'/pid);journal=read(tr.folder/'journal.json');finalhash={e['file']:e['sha256'] for e in journal['states']+journal['updates']}
  if digest(tr.folder/'problem.json')!=spec_sha or any(finalhash[k]!=h for k,h in snapshot.items()):raise ValueError('Committed/spec bytes changed on recovery')
  for k in original.arrays:np.testing.assert_allclose(tr.arrays[k],original.arrays[k],rtol=1e-10,atol=1e-12)
  for t,fields in prefix.items():
   for k,value in fields.items():np.testing.assert_array_equal(tr.arrays[k][t],value)
  verify_trace(failed);verify_trace(resumed);records.append(dict(stage=label,committed_prefix_unchanged=True,spec_unchanged=True,complete_arrays_match=True,trace_attribution_verified=True,failed_monitor=failed,resumed_monitor=resumed,nominal_committed_FEM=tr.metadata['state_count'],uncommitted_work_may_repeat=True))
 # Completed worker result is not recorded in progress: another dispatch must reuse native output, zero FEM.
 result,s=launch(out,0,'generate')
 if result['status']!='completed_reused' or result['FEM_evaluations_this_attempt']!=0:raise ValueError('Completed worker restart duplicated FEM')
 atomic(out/'recovery_validation.json',dict(passed=True,cases=records,completed_worker_restart_zero_FEM=True,exactly_once_FEM_claim=False))

def finish(out):
 from m3_4b.reporting import build_report
 tests={}
 for name,extra in [('focused',['tests/test_scalable_production.py']),('full',[])]:
  r=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*extra],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'test_logs'/f'{name}.log').write_text(r.stdout+f'\nExit code {r.returncode}\n');tests[name]=dict(exit_code=r.returncode,stdout=r.stdout);print(r.stdout,flush=True)
  if r.returncode:raise RuntimeError('Regression gate failed')
 atomic(out/'test_results.json',tests);binding(out);count=protected(out);progress=read(out/'progress.json');progress['status']='completed';atomic(out/'progress.json',progress);atomic(out/'final_validation.json',dict(passed=True,protected_files_unchanged=count,source_configuration_unchanged=True,batch_count=len(binding(out)['jobs']),tests=tests,full_dataset_not_executed=True,training_not_executed=True,no192_FEM=True));print(build_report(out),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['plan','dry-run','batch','resume','validate','report','audit','stress','_worker','_retention']);p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--task');p.add_argument('--result',type=Path);p.add_argument('--work',type=Path);p.add_argument('--fault');p.add_argument('--mode');a=p.parse_args();out=a.output.resolve()
 if a.command=='plan':plan(out)
 elif a.command=='dry-run':print(json.dumps(preflight(out),indent=2))
 elif a.command in ('batch','resume'):batch(out)
 elif a.command=='stress':binding(out);print(stress(out)['passed'])
 elif a.command=='_worker':
  from m3_4b.jobs import execute
  execute(out,a.index,a.task,a.result,a.work,a.fault)
 elif a.command=='_retention':
  # 200jobs x1000records; same descriptors and record count in both modes.
  baseline=memory();raw=[];count=0
  for i in range(200000):
   row=dict(job_id=str(i//1000),seconds=i*.05,worker_rss_mib=600.,parent_rss_mib=baseline['VmRSS'],problem_id='synthetic')
   if a.mode=='retain':raw.append(row)
   count+=1
  atomic(a.result,dict(mode=a.mode,records=count,before=baseline,after=memory(),retained_records=len(raw),qualification='Controlled materialized-record comparison, not real FEM or200 real jobs'))
 elif a.command=='validate':
  m=binding(out);check_authorization(out,m)
  for i in range(len(m['jobs'])):data,s=launch(out,i,'validate');print(i,data['passed'],flush=True)
 elif a.command=='report':
  from m3_4b.reporting import build_report
  print(build_report(out))
 elif a.command=='audit':binding(out);print(json.dumps(dict(protected_files_unchanged=protected(out),source_config_unchanged=True)))
