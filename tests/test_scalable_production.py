"""M3.4B non-neural orchestration and deterministic config invariants."""
import json,sys
from pathlib import Path
import pytest
from m3_4b.monitoring import run,atomic,digest,verify_trace,trace_rows,reduce_summaries,memory
from run_m3_4b import LIMITS,MAX_AUTHORIZED,reserve_authorization,disk_gate,binding
from m3_4b.design import configurations,specification,entry,audit,batch_indices

@pytest.fixture(scope='module')
def proposed():
 return [entry(specification(**c),f'specs/{i}.json',p,a) for i,(c,p,a) in enumerate(configurations())]

def test_streamed_monitor_compact_integrity_cleanup(tmp_path):
 s=run([sys.executable,'-c','import time; a=bytearray(100000); time.sleep(.06)'],tmp_path,'stable_attempt','stable_problem',LIMITS)
 assert 'samples' not in s and s['worker_exited'] and s['sample_count']>0 and s['trace_bytes']>0
 assert verify_trace(s)==s['sample_count']
 assert all(r['problem_id']=='stable_problem' for r in trace_rows(s['trace_path'],'stable_attempt'))
 assert reduce_summaries([tmp_path/'stable_attempt.json'])['jobs']==1

def test_trace_tampering_and_attribution_fail(tmp_path):
 s=run([sys.executable,'-c','import time;time.sleep(.02)'],tmp_path,'a','p',LIMITS)
 with pytest.raises(ValueError):list(trace_rows(s['trace_path'],'wrong'))
 with Path(s['trace_path']).open('a') as f:f.write('{}\n')
 with pytest.raises(ValueError,match='checksum'):verify_trace(s)

def test_partial_monitor_survives_controlled_interruption(tmp_path):
 s=run([sys.executable,'-c','import time;time.sleep(10)'],tmp_path,'interrupt','p',LIMITS,interrupt_monitor=True)
 assert s['controlled_interruption'] and s['worker_exited'] and verify_trace(s)==3

def test_completed_monitor_cannot_overwrite(tmp_path):
 run([sys.executable,'-c','pass'],tmp_path,'a','p',LIMITS)
 with pytest.raises(ValueError):run([sys.executable,'-c','pass'],tmp_path,'a','p',LIMITS)

def test_limit_failure_is_closed_and_retained(tmp_path):
 limits=dict(LIMITS,worker_timeout_seconds=.05)
 with pytest.raises(RuntimeError,match='closed'):run([sys.executable,'-c','import time;time.sleep(3)'],tmp_path,'a','p',limits)
 s=json.loads((tmp_path/'a.json').read_text());assert s['resource_event']=='Time gate exceeded' and s['worker_exited'];verify_trace(s)

@pytest.mark.parametrize('n',[17,164])
def test_no_count_flag_or_split_command_bypass(tmp_path,n):
 m=dict(source_id='source',jobs=[dict(problem_id=str(i)) for i in range(n)])
 with pytest.raises(ValueError):reserve_authorization(tmp_path,m,tmp_path/'ledger.json')

def test_ledger_counts_completed_resume_and_new_generation(tmp_path):
 m=dict(source_id='s',jobs=[dict(problem_id=str(i)) for i in range(12)]);ledger=tmp_path/'ledger.json';out=tmp_path/'first'
 assert reserve_authorization(out,m,ledger)==reserve_authorization(out,m,ledger)
 with pytest.raises(ValueError):reserve_authorization(tmp_path/'second',m,ledger)
 with pytest.raises(ValueError):reserve_authorization(out,dict(m,jobs=m['jobs']+[dict(problem_id='new')]),ledger)

@pytest.mark.parametrize('free,ok',[(199,False),(200,True),(1000,True)])
def test_disk_safety_preflight(free,ok):
 if ok:assert disk_gate(100,free)
 else:
  with pytest.raises(ValueError):disk_gate(100,free)

def test_revision_quotas_and_geometry_resolution(proposed):
 a=audit(proposed);assert len(proposed)==164 and a['families']==36 and a['max_per_family']<=6
 assert set(a['coverage']['train']['aspect'])=={'1.5','2.0'} and '[96, 48]' in a['coverage']['train']['mesh']
 assert len(batch_indices(proposed))==12 and len(set(batch_indices(proposed)))==12
 assert any(e['partition']=='test_id' and e['configuration']['mesh']==[96,48] for e in proposed)
 assert all(max(e['configuration']['mesh'])<=96 for e in proposed)

def test_group_split_leakage_is_rejected(proposed):
 e=dict(proposed[0],problem_id='different',partition='test_id')
 with pytest.raises(ValueError,match='family leakage'):audit(proposed+[e])

def test_near_neighbor_review_not_automatic_merge(proposed):
 a=audit(proposed);assert a['cross_partition_candidates']>0
 assert all(n['status']=='review_candidate_not_automatic_union' for n in a['near_neighbor_candidates'])
 assert any(e['ood_axis']=='force_direction_only' for e in proposed)

def test_global_sign_mirror_and_start_relatives():
 s=specification(direction='horizontal');t=specification(direction='horizontal',force_sign=-1);u=specification(direction='horizontal',support='right_clamp');v=specification(direction='horizontal',initial='perturbed',seed=3)
 assert len({s['related_group'],t['related_group'],u['related_group'],v['related_group']})==1
 assert len({s['problem_id'],t['problem_id'],u['problem_id'],v['problem_id']})==4

def test_refinement_native_physical_metadata():
 s=specification(mesh=(96,48),regime='refinement');assert s['filter_physical_semi_axes']==[2,2] and s['configuration']['dx']==.5
 g=specification(mesh=(96,48),domain=(96.,48.),regime='growth');assert g['configuration']['dx']==1 and g['configuration']['domain']==[96.,48.]
 with pytest.raises(ValueError):specification(mesh=(192,96))
 with pytest.raises(ValueError):specification(location=.123)

def test_spec_determinism_and_initial_strict_feasibility():
 s=specification(support='bottom_clamp',direction='horizontal',initial='perturbed',seed=1101)
 assert s==specification(support='bottom_clamp',direction='horizontal',initial='perturbed',seed=1101)
 from topopt_utils.fea.trajectory_dataset import load_problem
 p,x=load_problem(s);assert x.mean()<p.volume_fraction and x.sum()<p.n*p.volume_fraction and x.min()>=.001 and x.max()<=1

def test_manifest_immutability_failure_precedes_execution(tmp_path):
 atomic(tmp_path/'execution_manifest.json',dict(schema_version='changed'));atomic(tmp_path/'manifest_binding.json',dict(manifest_sha256='wrong'));atomic(tmp_path/'source_manifest.json',{})
 with pytest.raises(ValueError,match='manifest'):binding(tmp_path)

def test_streaming_reducer_many_summaries_no_raw_retention(tmp_path):
 paths=[]
 for i in range(100):
  trace=tmp_path/f'{i}.jsonl';trace.write_text(json.dumps(dict(job_id=str(i),problem_id='p'))+'\n')
  s=dict(job_id=str(i),trace_path=str(trace),trace_sha256=digest(trace),sample_count=1,trace_bytes=trace.stat().st_size,parent_peak_mib=10,worker_peak_rss_mib=20,worker_sampled_hwm_mib=21,combined_peak_mib=30,monitor_parent_cpu_seconds=.01,monitor_write_seconds=.001)
  p=tmp_path/f'{i}.json';atomic(p,s);paths.append(p)
 r=reduce_summaries(iter(paths));assert r['jobs']==100 and r['samples']==100 and 'samples_list' not in r and r['worker_peak_mib']==21

def test_package_compatibility_refuses_before_worker(tmp_path,monkeypatch):
 import run_m3_4b as cli
 monkeypatch.setattr(cli,'binding',lambda out:{'jobs':[]})
 monkeypatch.setattr(cli,'compatibility',lambda:{'changed_package':True})
 atomic(tmp_path/'environment.json',{'original_package':True})
 with pytest.raises(ValueError,match='package'):cli.preflight(tmp_path)
