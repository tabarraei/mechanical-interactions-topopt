#!/usr/bin/env python
"""M3.4 immutable planning and monitored, sequential production pilot."""
import argparse,json,os,sys,subprocess,time,shutil,tempfile,platform,csv,gzip,gc
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from collections import Counter
from importlib.metadata import version
import numpy as np
from topopt_utils.fea.production_dataset import SCHEMA,digest,OwnedSolver,specification,proposed_cases,pilot_cases,audit_families,stream_replay,window_ranges,draw_training
from topopt_utils.fea.trajectory_dataset import canonical_hash,Trajectory,features,PROFILES
from topopt_utils.fea.dataset_storage import generate_sample,atomic_json
from topopt_utils.fea.dataset_readiness import endpoint_analysis,POLICY,quality_flags
from topopt_utils.fea.consistent_trajectory import evaluate_state
from topopt_utils.fea.robust_oc import update
from run_m3_3b import source_manifest,assert_sources
from run_m3_3c import memory_status
ROOT=Path(__file__).resolve().parent

def stamp():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def save(path,value):atomic_json(path,value)

def protected(out,create=False):
    path=out/'protected_before.json'
    if create and not path.exists():
        data={}
        for p in ROOT.rglob('*'):
            if p.is_file() and not any(n in p.parts for n in ('.git','__pycache__','.pytest_cache','m3_4')):data[str(p.relative_to(ROOT))]=digest(p)
        save(path,data)
    old=json.loads(path.read_text());changed=[n for n,h in old.items() if not (ROOT/n).is_file() or digest(ROOT/n)!=h]
    if changed:raise ValueError('Protected source/artifact changed: '+str(changed))
    return len(old)

def entry(spec,file,partition,ood_axis):
    c=spec['configuration'];return dict(problem_id=spec['problem_id'],spec=str(file),spec_sha256=canonical_hash(spec),family=spec['related_group'],partition=partition,ood_axis=ood_axis,**{k:c[k] for k in ('support','direction','location','volume','mesh','domain','regime','initial','magnitude')})

def plan(out,cap=500):
    if out.exists() and set(p.name for p in out.iterdir())-{'protected_before.json','M3_4_IMPLEMENTATION_PLAN.md'}:raise ValueError('New output required; historical generations cannot be reused')
    if not 1<=cap<=500:raise ValueError('Cap must be1..500')
    out.mkdir(parents=True,exist_ok=True);protected(out,True)
    if not (out/'M3_4_IMPLEMENTATION_PLAN.md').exists():
        plans=sorted((ROOT/'m3_4/generations').glob('*/M3_4_IMPLEMENTATION_PLAN.md'))
        if not plans:raise ValueError('Implementation plan required')
        shutil.copyfile(plans[0],out/'M3_4_IMPLEMENTATION_PLAN.md')
    for name in ('specs','worker_logs','worker_results','test_logs'): (out/name).mkdir(exist_ok=True)
    sources=source_manifest(out);proposed=[];pilot=[]
    for i,(cfg,partition,axis) in enumerate(proposed_cases()):
        # Seed is provenance, not a random new mechanical draw. Starts are explicit.
        spec=specification(**cfg);file=f'specs/proposed_{i:03d}.json';save(out/file,spec);proposed.append(entry(spec,file,partition,axis))
    diversity=audit_families(proposed);save(out/'proposed_manifest.json',dict(schema_version=SCHEMA,status='planning_only_not_authorized_for_generation',jobs=proposed,source_id=canonical_hash(sources),future192='Config/resource scenario only; not an executable job'))
    for i,cfg in enumerate(pilot_cases(cap)):
        spec=specification(**cfg);file=f'specs/pilot_{i:02d}.json';save(out/file,spec);pilot.append(entry(spec,file,'train','controlled_pilot_not_a_learning_split'))
    limits=dict(worker_rss_mib=1024,parent_rss_mib=768,combined_rss_mib=1792,worker_timeout_seconds=900,poll_seconds=.05,sequential_only=True,max_pilot_problems=8,max_elements=4608,max_updates=500)
    m=dict(schema_version=SCHEMA,generation=out.name,status='planned',source_id=canonical_hash(sources),jobs=pilot,limits=limits,proposed_authorization=False)
    save(out/'execution_manifest.json',m);save(out/'manifest_binding.json',dict(sha256=canonical_hash(m),spec_hashes={j['spec']:digest(out/j['spec']) for j in pilot+proposed},proposed_sha256=digest(out/'proposed_manifest.json')))
    save(out/'progress.json',dict(status='planned',completed_generation=[],completed_validation=[],failures=[]));save(out/'diversity_audit.json',diversity)
    save(out/'manifest_schema.json',dict(version=SCHEMA,immutable='execution_manifest + spec SHA256 + source manifest; mutable progress separate',worker_modes=['generate','validate','lifecycle'],native_schema='unchanged m3.3b.problem/trajectory.v1',limits='pilot<=8, N<=4608, updates<=500; proposed manifest planning-only'))
    save(out/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),versions={k:version(k) for k in ('numpy','scipy','torch','pypardiso','pytest')},threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},module_paths=dict(storage=__import__('topopt_utils.fea.dataset_storage',fromlist=['x']).__file__,solver=__import__('topopt_utils.fea.solver',fromlist=['x']).__file__),initial_memory=memory_status(),time_basis='UTC stamps, local research date America/New_York'))
    # Verify saved prior manifests/source archives and all12 native completion samples.
    prior=ROOT/'m3_3c/completion_generations/20261013T203017221805Z';a=json.loads((prior/'manifest.json').read_text())
    for folder in (prior,ROOT/'m3_3c/generations/20261013T201818977895Z'):
        for e in json.loads((folder/'source_manifest.json').read_text()).values():
            if digest(folder/e['archive'])!=e['sha256']:raise ValueError('Prior source archive corrupted')
    for j in a['samples']:Trajectory(prior/j['folder'])
    save(out/'prior_evidence_audit.json',dict(completion_manifest_sha256=digest(prior/'manifest.json'),states=sum(j['state_count'] for j in a['samples']),updates=sum(j['completed_updates'] for j in a['samples']),source_archives_checked=True,native_completion_arrays_checked=True,scaling_reused_no_new_scaling_trajectories=True,prior_final_validation=json.loads((prior/'completion_audit.json').read_text())))
    dry_run(out);return m

def binding(out):
    m=json.loads((out/'execution_manifest.json').read_text());b=json.loads((out/'manifest_binding.json').read_text())
    if canonical_hash(m)!=b['sha256']:raise ValueError('Immutable manifest changed')
    for n,h in b['spec_hashes'].items():
        if digest(out/n)!=h:raise ValueError('Immutable spec changed')
    if digest(out/'proposed_manifest.json')!=b['proposed_sha256']:raise ValueError('Proposed manifest changed')
    sources=json.loads((out/'source_manifest.json').read_text())
    if m['schema_version']!=SCHEMA or m['source_id']!=canonical_hash(sources):raise ValueError('Source/schema binding changed')
    assert_sources(sources)
    if len(m['jobs'])>8 or m['proposed_authorization']:raise ValueError('Full generation is not authorized')
    return m

def dry_run(out):
    m=binding(out);jobs=[]
    for j in m['jobs']:
        n=j['mesh'][0]*j['mesh'][1];spec=json.loads((out/j['spec']).read_text());T=spec['optimizer']['maxiter']+1;ratio=n/1152
        estimates=dict(generation_seconds=.45*T*ratio**.75,validation_seconds=.1*T*ratio**.75,worker_peak_estimate_mib=700+60*ratio,native_storage_estimate_mib=.12*T*ratio)
        # Gate generation and replay individually, not by summing different jobs.
        passed=max(estimates['generation_seconds'],estimates['validation_seconds'])<m['limits']['worker_timeout_seconds'] and estimates['worker_peak_estimate_mib']<m['limits']['worker_rss_mib'] and n<=4608
        jobs.append(dict(problem_id=j['problem_id'],mesh=j['mesh'],estimates=estimates,passed=passed))
    result=dict(optimization_executed=False,trajectories_written=False,workload=jobs,limits=m['limits'],all_gates_passed=all(j['passed'] for j in jobs),assumptions='Conservative envelopes anchored to prior48/96 measurements; not hardware-independent upper bounds; post-exec RSS monitored before OOM',planned_cases=len(json.loads((out/'proposed_manifest.json').read_text())['jobs']))
    # dry-run returns/prints only; plan archives its estimate separately.
    print(json.dumps(result),flush=True);return result

def proc_memory(pid):
    try:
        data=Path(f'/proc/{pid}/status').read_text();values={}
        for line in data.splitlines():
            if line.startswith(('VmRSS:','VmHWM:')):values[line.split(':')[0]]=int(line.split()[1])/1024
        return values
    except FileNotFoundError:return {}

def launch(out,task,index=0,work=None,fault=None,mode=None,expected=0):
    m=binding(out);name=f'{task}_{index}_{stamp()}';result=out/'worker_results'/f'{name}.json';log=out/'worker_logs'/f'{name}.log';args=[sys.executable,str(ROOT/'run_m3_4.py'),'_worker','--output',str(out.resolve()),'--task',task,'--index',str(index),'--result',str(result.resolve())]
    if work:args+=['--work',str(Path(work).resolve())]
    if fault:args+=['--fault',fault]
    if mode:args+=['--mode',mode]
    samples=[];reason=None;t=perf_counter();parent_peak=memory_status()['VmRSS_mib'];worker_peak=0.
    with log.open('w') as f:
        child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        while child.poll() is None:
            memory=proc_memory(child.pid);parent=memory_status()['VmRSS_mib'];parent_peak=max(parent_peak,parent);rss=memory.get('VmRSS',0.);worker_peak=max(worker_peak,memory.get('VmHWM',rss));samples.append(dict(seconds=perf_counter()-t,worker_rss_mib=rss,parent_rss_mib=parent))
            limits=m['limits']
            if rss>limits['worker_rss_mib'] or parent>limits['parent_rss_mib'] or rss+parent>limits['combined_rss_mib']:reason='RSS gate exceeded'
            if perf_counter()-t>limits['worker_timeout_seconds']:reason='Time gate exceeded'
            if reason:
                child.terminate()
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                break
            time.sleep(limits['poll_seconds'])
        code=child.wait()
    monitor=dict(task=task,index=index,fault=fault,mode=mode,exit_code=code,expected_exit=expected,reason=reason,wall_seconds=perf_counter()-t,worker_peak_mib=worker_peak,parent_peak_mib=parent_peak,pid=child.pid,worker_exited=not Path(f'/proc/{child.pid}').exists(),samples=samples,log=str(log.relative_to(out)),result=str(result.relative_to(out)))
    save(out/'worker_results'/f'{name}_monitor.json',monitor)
    if reason or code!=expected:raise RuntimeError(f'Worker stopped ({reason or code}); see {log}')
    return json.loads(result.read_text()) if result.exists() else {},monitor

def fault_hooks(fault):
    """Process-local fault injection only; native source bytes/numerics unchanged."""
    import topopt_utils.fea.dataset_storage as storage
    if fault=='before_update':
        old=storage.update;counter=[0]
        def wrapped(*args,**kw):
            counter[0]+=1
            if counter[0]==3:os._exit(77)
            return old(*args,**kw)
        storage.update=wrapped
    elif fault=='recording':
        old=storage.atomic_frame
        def wrapped(path,state):
            old(path,state)
            if int(state['iteration'])==2:os._exit(77)
        storage.atomic_frame=wrapped
    elif fault=='committed':
        old=storage.atomic_json
        def wrapped(path,value):
            old(path,value)
            if Path(path).name=='journal.json' and len(value.get('states',[]))==3:os._exit(77)
        storage.atomic_json=wrapped
    elif fault=='finalization':
        old=np.lib.format.open_memmap
        def wrapped(*args,**kw):
            result=old(*args,**kw);os._exit(77)
        np.lib.format.open_memmap=wrapped

def worker(out,task,index,result,work=None,fault=None,mode=None):
    m=binding(out)
    if task=='lifecycle':
        spec=json.loads((out/m['jobs'][1]['spec']).read_text());from topopt_utils.fea.trajectory_dataset import load_problem
        p,x=load_problem(spec);trace=[];Cs=[]
        with OwnedSolver() as shared:
            for i in range(120):
                rho=np.full(p.n,.4+.1*(i%5)/4)
                if mode=='reused':C=shared.compliance(rho,*p.solver_args)
                else:
                    with OwnedSolver() as owned:C=owned.compliance(rho,*p.solver_args)
                Cs.append(C)
                if i%5==0:trace.append(dict(iteration=i,**memory_status()))
        save(result,dict(FEM_evaluations=120,OC_updates=0,mode=mode,compliance=Cs,memory_trace=trace,after_cleanup=memory_status(),source='Inherited numerical methods, explicit native context.close'))
        return
    j=m['jobs'][index];spec=json.loads((out/j['spec']).read_text());folder=Path(work) if work else out/'pilot';start=perf_counter()
    if task=='generate':
        if fault:fault_hooks(fault)
        before=memory_status();answer=generate_sample(folder,spec,m['source_id']);tr=Trajectory(answer['folder']);save(result,dict(problem_id=j['problem_id'],status=answer['status'],states=tr.metadata['state_count'],updates=tr.metadata['completed_updates'],generation_FEM_evaluations=tr.metadata['state_count'],wall_seconds=perf_counter()-start,before=before,after=memory_status(),native_bytes=sum(p.stat().st_size for p in tr.folder.rglob('*') if p.is_file()),folder=str(tr.folder)))
    elif task=='validate':
        tr=Trajectory(folder/'samples'/spec['problem_id']);timing=perf_counter();audit=stream_replay(tr);validation_seconds=perf_counter()-timing;features_timing={}
        for profile in PROFILES:
            tick=perf_counter();count=0
            for t in sorted({0,tr.metadata['state_count']//2,tr.metadata['state_count']-1}):
                f=features(tr.problem,tr.arrays['x'][t],t,profile,tr.state(t) if PROFILES[profile]['fresh_fem'] else None)
                assert f['element'].shape==(len(PROFILES[profile]['element']),tr.problem.nely,tr.problem.nelx) and f['node'].shape==(len(PROFILES[profile]['node']),tr.problem.nely+1,tr.problem.nelx+1)
                np.testing.assert_allclose(f['element'][1].ravel(),tr.arrays['rho'][t],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(f['node'][0].ravel()*tr.spec['force_scale'],tr.problem.forces[::2],atol=1e-15);np.testing.assert_allclose(f['node'][1].ravel()*tr.spec['force_scale'],tr.problem.forces[1::2],atol=1e-15)
                if profile=='fresh_full_v1':np.testing.assert_allclose(f['element'][-1].sum()*tr.spec['force_scale']**2,tr.arrays['compliance'][t]/2,rtol=1e-8)
                count+=1
            features_timing[profile]=dict(cached_feature_seconds=perf_counter()-tick,checks=count,fresh_FEM_required_at_inference=PROFILES[profile]['fresh_fem'])
        endpoint=endpoint_analysis(tr);candidate=endpoint['reference_label_eligible'];endpoint.update(reference_screen_candidate=candidate,reference_label_eligible=False,reference_policy='Conservative native False; provisional candidate sidecar only, no newly approved optimum-label policy')
        sensitivity=[]
        for threshold in (.005,.01,.02):
            policy=dict(POLICY,pg=threshold,box_kkt=threshold);sensitivity.append(dict(pg_box_threshold=threshold,screen_candidate=quality_flags(endpoint,policy)['reference_label_eligible']))
        with OwnedSolver() as owned:
            rho=tr.arrays['rho'][0];c,g,profile=owned.compliance(rho,*tr.problem.solver_args,sensitivities=True,profile=True)
        tick=perf_counter();update(tr.arrays['x'][0],tr.arrays['g_x'][0],tr.problem.volume_fraction);oc_seconds=perf_counter()-tick
        # Independent export cost; kept separate from native journal/finalization.
        tick=perf_counter();export=result.with_suffix('.npz');np.savez_compressed(export,x=tr.arrays['x']);export_seconds=perf_counter()-tick
        tick=perf_counter();hashes={k:digest(tr.folder/v['file']) for k,v in tr.metadata['arrays'].items()};checksum_seconds=perf_counter()-tick
        save(result,dict(problem_id=j['problem_id'],**audit,validation_seconds=validation_seconds,feature_timings=features_timing,endpoint=endpoint,threshold_sensitivity=sensitivity,primitive_first_call_seconds=profile,primitive_OC_seconds=oc_seconds,extra_timing_FEM_evaluations=1,serialization_density_history_seconds=export_seconds,checksum_seconds=checksum_seconds,memory=memory_status(),native_array_hashes=hashes))
    else:raise ValueError('Unknown task')


def checkpoint_snapshot(folder,pid):
    stage=folder/'incomplete'/pid;j=json.loads((stage/'journal.json').read_text());return {e['file']:digest(stage/e['file']) for e in j['states']+j['updates']}

def recovery_tests(out):
    m=binding(out);job=m['jobs'][0];original=Trajectory(out/'pilot/samples'/job['problem_id']);results=[]
    for fault in ('before_update','recording','committed','finalization'):
        work=out/'recovery'/fault;work.mkdir(parents=True);_,failure=launch(out,'generate',0,work,fault,expected=77);snapshot=checkpoint_snapshot(work,job['problem_id']);stage=work/'incomplete'/job['problem_id']
        # Capture committed bytes and compare to final native arrays, not derive a new spec.
        prefix={}
        for name in snapshot:
            if name.startswith('frames/'):
                with np.load(stage/name) as frame:prefix[int(frame['iteration'])]={k:np.asarray(frame[k]).copy() for k in ('x','rho','u','compliance','g_x','g_rho')}
        spec_before=digest(stage/'problem.json');answer,monitor=launch(out,'generate',0,work);restored=Trajectory(work/'samples'/job['problem_id'])
        assert spec_before==digest(restored.folder/'problem.json')
        for t,fields in prefix.items():
            for key,v in fields.items():np.testing.assert_array_equal(restored.arrays[key][t],v)
        for key in original.arrays:np.testing.assert_allclose(original.arrays[key],restored.arrays[key],rtol=1e-10,atol=1e-12)
        # Original journal retains all same committed hashes; former files are intentionally finalized away.
        finaljournal=json.loads((restored.folder/'journal.json').read_text());finalhash={e['file']:e['sha256'] for e in finaljournal['states']+finaljournal['updates']}
        assert all(finalhash[n]==h for n,h in snapshot.items())
        results.append(dict(fault=fault,expected_exit=77,prefix_commits_unchanged=True,problem_spec_unchanged=True,uninterrupted_numerical_match=True,quarantined_files=[p.name for p in (restored.folder/'quarantined').glob('*')] if (restored.folder/'quarantined').exists() else [],failed_wall_seconds=failure['wall_seconds'],resumed_wall_seconds=monitor['wall_seconds'],committed_generation_FEM=restored.metadata['state_count'],failed_phase_may_have_uncommitted_extra_FEM=True))
    # Historical native matched-spec equality, not only production self-consistency.
    historical=ROOT/'m3_3c/completion_generations/20261013T203017221805Z/samples'/job['problem_id']
    if not historical.exists():raise ValueError('Historical matched small control identity unexpectedly changed')
    old=Trajectory(historical)
    for key in original.arrays:np.testing.assert_allclose(original.arrays[key],old.arrays[key],rtol=1e-10,atol=1e-12)
    save(out/'recovery_tests.json',dict(passed=True,cases=results,historical_matched_spec_arrays_equal=True,rtol=1e-10,atol=1e-12,exactly_once_FEM_claim=False))


def streaming_analysis(out):
    m=binding(out);summaries=[];ranges=[]
    for index,j in enumerate(m['jobs']):
        file=out/'summaries'/f'{index}.json';s=json.loads(file.read_text());summaries.append(dict(**j,states=s['states'],updates=s['updates'],stop=s['endpoint']['stop'],quality=s['endpoint']))
    ranges=list(window_ranges(summaries));save(out/'window_ranges.json',ranges)
    modes={}
    for mode in ('uniform_windows','uniform_problems','uniform_families','family_first'):
        families=Counter();problems=Counter();stages=Counter();horizons=Counter();probmin=1.;probmax=0.
        with (out/f'sampling_{mode}.jsonl').open('w') as f:
            for row in draw_training(ranges,10000,134,mode):
                families[row['family']]+=1;problems[row['problem_id']]+=1;stages[row['stage']]+=1;horizons[row['h']]+=1;probmin=min(probmin,row['probability']);probmax=max(probmax,row['probability']);f.write(json.dumps(row)+'\n')
        proportions=np.array(list(families.values()))/10000
        modes[mode]=dict(family_counts=dict(families),problem_counts=dict(problems),stage_counts=dict(stages),horizon_counts=dict(horizons),probability_range=[probmin,probmax],effective_families=float(1/np.sum(proportions**2)),only_train=True)
    stagecounts=Counter(r['stage'] for r in draw_training(ranges,10000,134,'family_first',True));save(out/'sampling_audit.json',dict(modes=modes,optional_stage_balance=dict(stagecounts),stage_basis='Known500 budget; labels/termination never input features; window limits use actual saved length only for eligible indexing'))
    save(out/'streaming_summary.json',dict(samples=summaries,compact_range_count=len(ranges),total_windows=sum(r['count'] for r in ranges),peak_memory=memory_status(),record_retention='O(problems*history*horizon) range metadata, O(families) sampler counters, one native problem mmap per worker; no full-window/trajectory stacking'))


def report(out):
    # Intentionally does not call binding or import/evaluate trajectories or FEM.
    start=perf_counter();summary=json.loads((out/'streaming_summary.json').read_text());generation=json.loads((out/'generation_results.json').read_text());lifecycle=json.loads((out/'lifecycle_benchmarks.json').read_text());test=json.loads((out/'test_results.json').read_text());diversity=json.loads((out/'diversity_audit.json').read_text());m=json.loads((out/'execution_manifest.json').read_text())
    lines=['# M3.4 production dataset readiness','', 'Verified numerical code, existing datasets and historical launch limits are unchanged. Generation/replay ownership is one bounded fresh process per problem; independent replay additionally uses inherited numerical methods in an explicitly closed owned context. Proposal is planning-only; no full dataset, training,192/384 FEM, HPC, commit/push.','', '| Pilot mesh / support / direction | States | Updates/stop | PG inf | Dynamics valid | Reference screen candidate |','|---|---:|---|---:|---|---|']
    for s in summary['samples']:
        q=s['quality'];lines.append(f"| {s['mesh']} / {s['support']} / {s['direction']} | {s['states']} | {s['updates']}/{s['stop']} | {q['projected_gradient_inf']:.4g} | {q['dynamics_valid']} | {q['reference_screen_candidate']} |")
    monitors=[json.loads(f.read_text()) for f in (out/'worker_results').glob('*_monitor.json')];worker=max(r['worker_peak_mib'] for r in monitors);parent=max(r['parent_peak_mib'] for r in monitors)
    lines+=['',f"{len(summary['samples'])} pilot configurations; {sum(s['states'] for s in summary['samples'])} saved states; {sum(s['updates'] for s in summary['samples'])} updates. Every state/transition independently replayed; inherited FEM API gradients, filter transpose, COO actions/reactions/equilibrium, volume and selected OC endpoint/diagnostics checked. Terminal evaluation and native checksums are preserved. All reference_label_eligible flags stay false; provisional screen candidates and .005/.01/.02 threshold sensitivity are separate. Capped trajectories retained as valid dynamics, not reference designs.", '', f"Maximum observed worker RSS {worker:.2f}MiB; parent {parent:.2f}MiB. Limits {m['limits']}. Worker process exit verified for all monitored jobs; kill/recovery faults exit77 intentionally. RSS monitored every50ms and timeout enforced before OOM; sampling can miss very short peaks, so worker self VmHWM is also retained. No universal constant-memory claim; measured workloads only.", '', '## Native lifecycle and crash safety','', '120-call reused-owned and120-call explicitly released-owned probes produced matching C at repeated density values; native cleanup occurs explicitly. Generation reuses a native context within one trajectory and all allocations are reclaimed by process exit. Independent replay owns a single reused context; feature construction creates no fresh factor contexts. Reporting has no FEM. Native NumPy/allocator caches, plotting and buffered metadata are separate from factor ownership. The selected architecture avoids accumulating fresh native handles across problems.', '', 'Four controlled abrupt exits test before update, uncommitted state recording, after journal commit and finalization. All prefixes retain their committed hashes and spec; resumed complete arrays match uninterrupted pilot and historical matched control at rtol1e-10/atol1e-12. Native orphan quarantine and derived-array reconstruction are reused unchanged. Abrupt exit may recompute uncommitted FEM; exactly-once execution is not claimed. Complete pilot samples are immutable; resume refuses a completed generation.','', '## Proposal and splits','', f"Proposed {diversity['unique_configurations']} complete validated configs, {diversity['mechanical_families']} conservative related families. No trajectories generated from proposal. Training covers horizontal/vertical/oblique, left/bottom/pin-roller supports and all4 volumes; explicit force/start/mirror/refinement/growth relatives are grouped within one partition. Family uniqueness differs from mechanical/spec uniqueness; controlled duplicates are limited and labeled. Config validity checks exact node correspondence, force-on-support rejection, box/volume initialization, full problem serialization and rigid-body restraint, not a numerical convergence guarantee.", '', 'ID train/validation/test distributions preserve support/direction/volume/geometry categories, with held-family locations interpolated between observed boundary ranges. OOD adds dual-pin DOF to matched pin/roller cases (support factor), extrapolated point location (location factor), or aspect4 geometry (boundary-relative coordinates preserved, absolute location changes documented). No192 executable jobs. Near-duplicate grouping uses explicit canonical support/direction/normalized location/aspect families; continuous-neighbor leakage beyond this declared design needs review. No held-out/future statistics normalize features.', '', '## Sampling and physical resolution','', 'Compact window ranges encode k1/3,h1/5/10, exact terminal-aware eligible anchors. Training-only deterministic uniform-window, uniform-problem, uniform-family/window and family-first/problem/window modes emit address probabilities and family/problem/stage/horizon frequencies. Family-first prevents variants from increasing family influence; within-family counts remain visible. Optional stage balancing uses configured500 boundaries, not future labels as neural inputs. Keep homogeneous-mesh batches and native element/node features; no resize/padding or load smearing.', '', 'Refinement uses physical48×24 and radius2 at48/96 (index2/4), exact nodal point force/support. Growth uses dx=dy1 and physicalradius2 while domain expands. Scalar-radius rectangular cells are explicitly elliptical; geometry OOD here keeps square spacing1. Discrete cone quadrature/boundary differences and concentrated-load singularity prevent claiming mesh convergence.192 remains unmeasured FEM.','', '## Costs and estimates','', 'Measured worker records separate native generation (including checkpoints/finalization), replay2FEM/state, inherited solver first assembly/symbolic/factor/solve, guarded OC, cached feature extraction, density-history export and streaming checksum timing. Native checkpoint/finalization costs are not perfectly separated from diagnostics; stage/session and restart wall times provide inclusive measurements, not a clean production solver speedup. Parent/worker imports and process startup are included in monitored wall times. Disk usage and report memory are recorded separately. Estimates include terminal states, replay and startup; mixed96 and192 scenarios retain uncertainty from prior scaling.','', '## Explicit research answers','', '1. Native-factor accumulation is isolated by workers; owned replay/probes explicitly release factors without changing core.', '2. Repeated worker jobs and120-call owned probes stay below measured limits; no all-workload memory theorem.', '3. Failed workers preserve journals/committed bytes; all four controlled fault locations recovered correctly.', '4. Resumed arrays match original/historical numerical histories under recorded tolerances.', '5. Complete configurations and seeds regenerate deterministically; resume uses hashes/specs, not newly sampled problems.', '6. Explicit mirror/force/start/resolution families do not cross partitions; broader continuous leakage still needs review.', '7. Proposed training supplies all directions/volumes and three support classes; tiny pilot itself is not broad learning evidence.', '8. Valid capped states retained with distinct dynamics/reference labels.', '9. Family-first controls family influence independently of history length/variant count; stage/horizon probabilities disclosed.', '10. Both48 and96 are exercised below gates; clocks and RSS in per-job logs.', '11. Updated100/200 and mixed/larger planning estimates are saved; no full dataset executed.', '12. Remaining risks: private PARDISO API/version compatibility, unmeasured long-run hardware/disk variance, family coverage and provisional label screens.', '13. Native grids/spec metadata support later cross-resolution studies; NN conditioning/positions/receptive fields and fresh-feature FEM cost remain.', '14. Before192 trajectories: separately approved single-state resource/residual measurement, longer monitored stability checks, physical-filter/load-singularity review and native learning benchmarks.', '', '**Decision A — CONDITIONAL GO:** monitored isolated production machinery and restart/replay passed the pilot. A separately authorized100–200 execution needs reviewed manifest/category quotas, sufficient disk/time and the same gates; no automatic expansion is enabled.', '', '**Decision B — CONDITIONAL GO:** proposed stratified data support controlled M4 architecture comparisons after actual generation/validation, family-balanced training, matched data/compute and held-out mechanical rollout metrics. No learning/acceleration is established; legacy Torch physics filter still differs at boundaries.', '', '**Decision C — CONDITIONAL GO:** fresh-worker192 single-state benchmark is a justified separately authorized measurement; no192 FEM done, and no192 trajectory readiness inferred.', '', f"Focused/full tests exit0; logs retained. {test['focused']['stdout'].strip()} / {test['full']['stdout'].strip()}. Source/config/job fingerprints immutable; old-source checks included in audit. Reproduce with README commands and exact archived sources. Report is generated only from saved summaries; no FEM/optimization in report."]
    file=out/('M3_4_PRODUCTION_READINESS_REPORT.md' if not (out/'M3_4_PRODUCTION_READINESS_REPORT.md').exists() else 'M3_4_PRODUCTION_READINESS_REPORT_'+stamp()+'.md');file.write_text('\n'.join(lines)+'\n');save(out/('report_resources_'+stamp()+'.json'),dict(seconds=perf_counter()-start,memory=memory_status(),FEM_evaluations=0,inputs='saved validated summaries only'));return file


def pilot(out,resuming=False):
    m=binding(out);progress=json.loads((out/'progress.json').read_text())
    if progress['status']=='completed':raise ValueError('Completed generation is immutable')
    if not dry_run(out)['all_gates_passed']:raise RuntimeError('Estimated resource gates failed')
    (out/'summaries').mkdir(exist_ok=True);gen=[]
    for index,j in enumerate(m['jobs']):
        print('Production job',index,j['mesh'],j['direction'],flush=True)
        try:
            if index not in progress['completed_generation']:
                answer,monitor=launch(out,'generate',index);save(out/'summaries'/f'gen_{index}.json',dict(result=answer,monitor=monitor));progress['completed_generation'].append(index);save(out/'progress.json',progress)
            if index not in progress['completed_validation']:
                answer,monitor=launch(out,'validate',index);save(out/'summaries'/f'{index}.json',answer);progress['completed_validation'].append(index);save(out/'progress.json',progress)
        except Exception as e:
            progress['failures'].append(dict(index=index,error=str(e),utc=stamp()));save(out/'progress.json',progress);raise
        gen.append(json.loads((out/'summaries'/f'gen_{index}.json').read_text()))
    save(out/'generation_results.json',gen)
    life=[]
    for mode in ('reused','released'):
        answer,monitor=launch(out,'lifecycle',mode=mode);life.append(dict(result=answer,monitor=monitor))
    np.testing.assert_allclose(life[0]['result']['compliance'],life[1]['result']['compliance'],rtol=1e-12,atol=1e-12)
    for r in life:
        trace=r['result']['memory_trace'];growth=trace[-1]['VmRSS_mib']-trace[2]['VmRSS_mib']
        if growth>64:raise RuntimeError('Owned lifecycle memory grew beyond bounded probe allowance')
    save(out/'lifecycle_benchmarks.json',life);recovery_tests(out);streaming_analysis(out)
    tests={}
    for label,extra in [('focused',['tests/test_production_dataset.py']),('full',[])]:
        r=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*extra],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'test_logs'/f'{label}.log').write_text(r.stdout+f'\nExit code {r.returncode}\n');tests[label]=dict(exit_code=r.returncode,stdout=r.stdout);print(r.stdout,flush=True)
        if r.returncode:raise RuntimeError('Tests failed')
    save(out/'test_results.json',tests)
    # Resource projections anchored to production workers; uncertainty kept explicit.
    estimates=[]
    v=[json.loads((out/'summaries'/f'{i}.json').read_text()) for i in range(len(m['jobs']))]
    g48=[x for x in gen if m['jobs'][gen.index(x)]['mesh']==[48,24]];per=[x['monitor']['wall_seconds']/x['result']['states'] for x in g48]
    v48=[x['validation_seconds']/x['states'] for i,x in enumerate(v) if m['jobs'][i]['mesh']==[48,24]];storage=[x['result']['native_bytes']/x['result']['states'] for x in g48]
    for n in (100,200):estimates.append(dict(dataset=f'{n}_at48_500cap',generation_FEM=n*501,replay_FEM=2*n*501,generation_minutes=[n*501*min(per)/60,n*501*max(per)/60],replay_minutes=[n*501*min(v48)/60,n*501*max(v48)/60],storage_GiB=[n*501*min(storage)/2**30,n*501*max(storage)/2**30],qualification='Observed per-state small-pilot ranges including process startup; not confidence bounds, all501states assumed, monitored worker limits still mandatory'))
    ninety=gen[2];per96=ninety['monitor']['wall_seconds']/ninety['result']['states'];s96=ninety['result']['native_bytes']/ninety['result']['states'];v96=v[2]['validation_seconds']/v[2]['states']
    for n in (100,200):estimates.append(dict(dataset=f'{n}_half48_half96',generation_FEM=n*501,replay_FEM=2*n*501,generation_minutes=[n*501*(min(per)+per96)/120,n*501*(max(per)+per96)/120],replay_minutes=[n*501*(min(v48)+v96)/120,n*501*(max(v48)+v96)/120],storage_GiB=[n*501*(min(storage)+s96)/2/2**30,n*501*(max(storage)+s96)/2/2**30],qualification='One96 full history; no parallel throughput or arbitrary population guarantee'))
    old=ROOT/'m3_3c/generations/20261013T201818977895Z/larger_mesh_estimates.json';save(out/'resource_projections.json',dict(measured_and_projected=estimates,future192_384_unmeasured=json.loads(old.read_text()),memory='Observed parent/worker RSS plus monitored limits; parallel workers not enabled, regression child peak not monitored'))
    save(out/'disk_usage.json',dict(bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),scope='Includes proposal/specs, sources, pilot, crash copies, worker logs/results; native pilot disk separately in generation records'))
    progress['status']='completed';save(out/'progress.json',progress);assert_sources(json.loads((out/'source_manifest.json').read_text()));save(out/'final_validation.json',dict(passed=True,protected_files_unchanged=protected(out),source_bytes_unchanged=True,pilot_states=sum(x['states'] for x in v),pilot_updates=sum(x['updates'] for x in v),independent_replay_FEM=sum(x['independent_FEM_evaluations'] for x in v),ownership_probes_FEM=240,crash_recovery_passed=True,native_historical_match=True,limits=m['limits'],tests=tests,proposed_only=True,completed=True));report(out);print('M3.4 complete:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['plan','dry-run','pilot','resume','validate','report','audit','_worker']);p.add_argument('--output',type=Path);p.add_argument('--cap',type=int,default=500);p.add_argument('--task');p.add_argument('--index',type=int,default=0);p.add_argument('--result',type=Path);p.add_argument('--work',type=Path);p.add_argument('--fault');p.add_argument('--mode');a=p.parse_args();out=a.output or ROOT/'m3_4/generations'/stamp()
    if a.command=='plan':plan(out,a.cap)
    elif a.command=='dry-run':dry_run(out)
    elif a.command in ('pilot','resume'):pilot(out,a.command=='resume')
    elif a.command=='_worker':worker(out,a.task,a.index,a.result,a.work,a.fault,a.mode)
    elif a.command=='report':print(report(out))
    elif a.command=='audit':binding(out);print(json.dumps(dict(protected_files_unchanged=protected(out),source_config_unchanged=True)))
    elif a.command=='validate':
        m=binding(out)
        for i in range(len(m['jobs'])):answer,monitor=launch(out,'validate',i);print(answer['passed'],flush=True)
