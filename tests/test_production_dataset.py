import json
from pathlib import Path
import numpy as np
import pytest
from topopt_utils.fea.production_dataset import specification,proposed_cases,audit_families,window_ranges,draw_training,OwnedSolver,stream_replay,ProductionWindowLoader,digest
from topopt_utils.fea.trajectory_dataset import load_problem,canonical_hash,Trajectory
from topopt_utils.fea.dataset_storage import generate_sample
from run_m3_4 import entry,binding,fault_hooks,report,proc_memory

@pytest.fixture(scope='module')
def small(tmp_path_factory):
    root=tmp_path_factory.mktemp('m34');spec=specification(mesh=(6,4),domain=(6.,4.),direction='horizontal',cap=4)
    result=generate_sample(root,spec,'test-source');return root,spec,Trajectory(result['folder'])

def test_owned_context_repeated_numeric_equivalence_and_cleanup():
    s=specification(mesh=(6,4),domain=(6.,4.),cap=4);p,x=load_problem(s)
    with OwnedSolver() as solver:
        first=solver.compliance(x,*p.solver_args)
        for _ in range(10):assert abs(solver.compliance(x,*p.solver_args)-first)<1e-12
        assert len(solver._symbolic_cache)==1
    assert not solver._symbolic_cache

def test_owned_context_cleanup_on_exception():
    s=specification(mesh=(6,4),domain=(6.,4.),cap=4);p,x=load_problem(s)
    with pytest.raises(RuntimeError):
        with OwnedSolver() as solver:
            solver.compliance(x,*p.solver_args);raise RuntimeError('test')
    assert not solver._symbolic_cache

def test_replay_all_states_strict_volume_terminal(small):
    root,s,tr=small;a=stream_replay(tr)
    assert a['passed'] and a['states']==5 and a['updates']==4 and a['independent_FEM_evaluations']==10
    assert np.all(tr.arrays['volume_excess']<=0) and len(tr.transitions)==4

def test_native_window_shapes_index_and_cache_bound(small):
    root,s,tr=small;e=dict(problem_id=s['problem_id'],partition='train');loader=ProductionWindowLoader(root,[e],cache_size=1)
    a=dict(problem_id=s['problem_id'],partition='train',anchor=2,k=3,h=1);f=loader.load(a)
    assert f['element'].shape==(3,9,4,6) and f['node'].shape==(3,8,5,7)
    np.testing.assert_array_equal(f['target_x'].ravel(),tr.arrays['x'][3]);assert len(loader.cache)==1
    with pytest.raises(ValueError):loader.load(dict(a,partition='test_id'))

def test_native_checkpoint_prefix_resume_equivalence(tmp_path,small):
    root,s,tr=small;r=generate_sample(tmp_path,s,'test-source',max_new_updates=2);stage=r['folder'];before={p.name:digest(p) for p in (stage/'frames').iterdir()};result=generate_sample(tmp_path,s,'test-source');again=Trajectory(result['folder'])
    for k in tr.arrays:np.testing.assert_allclose(tr.arrays[k],again.arrays[k],rtol=1e-10,atol=1e-12)
    journal=json.loads((again.folder/'journal.json').read_text());assert all(e['sha256']==before[Path(e['file']).name] for e in journal['states'][:3])

def test_orphan_quarantine_and_corrupt_committed_frame_rejection(tmp_path,small):
    _,s,_=small;result=generate_sample(tmp_path,s,'test-source',max_new_updates=1);stage=result['folder'];orphan=stage/'frames/uncommitted.npz';orphan.write_bytes(b'orphan')
    generate_sample(tmp_path,s,'test-source',max_new_updates=1);assert (stage/'quarantined/uncommitted.npz').read_bytes()==b'orphan'
    committed=stage/'frames/00000.npz';committed.write_bytes(committed.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='corruption'):generate_sample(tmp_path,s,'test-source')

@pytest.mark.parametrize('mode',['uniform_windows','uniform_problems','uniform_families','family_first'])
def test_sampler_train_only_deterministic_probabilities(mode):
    es=[dict(problem_id='a',family='one',partition='train',states=501),dict(problem_id='b',family='one',partition='train',states=50),dict(problem_id='c',family='two',partition='train',states=100),dict(problem_id='d',family='held',partition='test_id',states=500)]
    ranges=list(window_ranges(es));a=list(draw_training(ranges,1000,7,mode));assert a==list(draw_training(ranges,1000,7,mode));assert all(r['problem_id']!='d' and r['target']<next(e['states'] for e in es if e['problem_id']==r['problem_id']) and r['probability']>0 for r in a)
    if mode in ('uniform_families','family_first'):assert .45<np.mean([r['family']=='one' for r in a])<.55

def test_compact_stream_index_storage_and_stage_balance():
    entries=[dict(problem_id=str(i),family=str(i),partition='train',states=501) for i in range(200)]
    r=list(window_ranges(entries));assert len(r)==1200
    rows=list(draw_training(r,1000,3,'family_first',True));counts=[sum(x['stage']==s for x in rows) for s in ('early','middle','late')];assert all(250<n<400 for n in counts)

def test_physical_refinement_and_forbidden_large_mesh():
    s=specification(mesh=(96,48),regime='refinement');p,x=load_problem(s);assert p.width==48 and p.height==24 and p.dx==.5 and p.filter_radius==4
    assert s['filter_physical_semi_axes']==[2,2]
    with pytest.raises(ValueError):specification(mesh=(192,96))

def test_deterministic_specs_and_family_mirrors():
    s=specification(direction='horizontal',location=.25);t=specification(direction='horizontal',location=.25,support='right_clamp',magnitude=10)
    assert s==specification(direction='horizontal',location=.25) and s['related_group']==t['related_group'] and s['problem_id']!=t['problem_id']
    with pytest.raises(ValueError):specification(location=.123)

def test_proposal_diversity_and_group_leakage():
    # Config-only audit of the complete deterministic plan is executed in plan;
    # unit audit validates categories and rejects crossing related families.
    es=[]
    for i,d in enumerate(('horizontal','vertical','oblique')):
        for v in (.3,.4,.5,.6):es.append(dict(problem_id=f'{i}_{v}',family=str(i),partition='train',direction=d,volume=v,support=('left','bottom','pin')[i],regime='baseline',ood_axis='ID'))
    assert audit_families(es)['mechanical_families']==3
    with pytest.raises(ValueError):audit_families(es+[dict(es[0],problem_id='new',partition='test_id')])
    assert 100<=len(proposed_cases())<=200

def test_manifest_tampering_is_rejected(tmp_path):
    (tmp_path/'execution_manifest.json').write_text(json.dumps(dict(status='changed')));(tmp_path/'manifest_binding.json').write_text(json.dumps(dict(sha256='wrong')))
    with pytest.raises(ValueError,match='manifest changed'):binding(tmp_path)

def test_report_does_not_execute_fem(tmp_path,monkeypatch):
    import run_m3_4 as cli
    def forbidden(*a,**kw):raise AssertionError('FEM in report')
    monkeypatch.setattr(cli,'evaluate_state',forbidden);monkeypatch.setattr(cli,'binding',forbidden)
    files={'streaming_summary.json':dict(samples=[]),'generation_results.json':[],'lifecycle_benchmarks.json':[],'test_results.json':dict(focused=dict(stdout='passed'),full=dict(stdout='passed')),'diversity_audit.json':dict(unique_configurations=160,mechanical_families=30),'execution_manifest.json':dict(limits={})}
    for n,v in files.items():(tmp_path/n).write_text(json.dumps(v))
    (tmp_path/'worker_results').mkdir();(tmp_path/'worker_results/a_monitor.json').write_text(json.dumps(dict(worker_peak_mib=1,parent_peak_mib=1)))
    first=report(tmp_path);second=report(tmp_path);assert first!=second and first.exists() and second.exists()

def test_monitor_missing_process_and_stream_hash(tmp_path):
    f=tmp_path/'data';f.write_bytes(b'x'*3000000)
    import hashlib
    assert digest(f)==hashlib.sha256(f.read_bytes()).hexdigest() and proc_memory(999999999)=={}
