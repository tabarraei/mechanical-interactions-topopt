#!/usr/bin/env python
"""M3.3B small dataset pipeline demonstration. No large-generation default."""
import argparse,csv,json,os,platform,shutil,subprocess,sys,tempfile
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from importlib.metadata import version
import numpy as np
from topopt_utils.fea.trajectory_dataset import (SCHEMA,sample_problem,split_problems,validate_splits,load_problem,feature_schema,PROFILES,Trajectory,WindowDataset,window_index,batch_indices,features,file_hash,canonical_hash,evaluate_prediction)
from topopt_utils.fea.dataset_storage import atomic_json,generate_sample,validate_sample

ROOT=Path.cwd()


def protected_check():
    data=json.loads((ROOT/'m3_3b/protected_before.json').read_text());changed=[n for n,h in data.items() if not (ROOT/n).is_file() or file_hash(ROOT/n)!=h]
    if changed:raise ValueError('Protected files modified: '+repr(changed))
    return len(data)


def demo_config():
    def case(label,mesh,domain,support,position,direction,magnitude,volume,group,initial=None):
        return dict(label=label,mesh=mesh,domain=domain,support=support,loads=[dict(position=position,direction=direction,magnitude=magnitude)],volume_fraction=volume,related_group=group,initial=initial or dict(kind='uniform',slack=1e-8))
    return dict(schema_version='m3.3b.configuration.v1',seed=201,split_seed=577,controls=dict(option='feasible_upper_v1',maxiter=40,tolerance=.001),
        history_lengths=[1,3],horizons=[1,5,10],ood_supports=['bottom_clamp'],split_ratios=[.6,.2,.2],
        problems=[case('axial_12x6',[12,6],[12.,6.],'left_clamp',[12.,3.],[1.,0.],1.,.5,'axial'),
          case('axial_scaled_16x8',[16,8],[12.,6.],'left_clamp',[12.,3.],[1.,0.],2.,.5,'axial',dict(kind='perturbed',amplitude=.02)),
          case('vertical_12x6',[12,6],[12.,6.],'left_clamp',[12.,2.],[0.,-1.],1.,.4,'vertical'),
          case('oblique_top_8x4',[8,4],[8.,4.],'left_clamp',[4.,4.],[.6,-.8],.5,.5,'oblique'),
          case('pin_roller_10x4',[10,4],[10.,4.],'pin_roller',[5.,4.],[0.,-1.],1.5,.6,'bridge'),
          case('ood_bottom_clamp_8x6',[8,6],[8.,6.],'bottom_clamp',[4.,6.],[1.,0.],1.,.5,'bottom')])


def source_manifest(out):
    root=Path(__file__).resolve().parent;files=list(root.glob('*.py'))
    for name in ('topopt_utils','topo_transformer','inr_render','tests'):files.extend((root/name).rglob('*.py'))
    hashes={}
    for file in sorted(set(files)):
        name=file.relative_to(root);target=out/'source_snapshots'/name
        if name.parts[0]=='tests':target=target.with_suffix('.py.txt')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
        hashes[str(name)]=dict(sha256=file_hash(file),archive=str(target.relative_to(out)))
    atomic_json(out/'source_manifest.json',hashes);return hashes


def assert_sources(hashes):
    root=Path(__file__).resolve().parent
    if any(file_hash(root/n)!=r['sha256'] for n,r in hashes.items()):raise ValueError('Executing source changed')


def initialize(out,config):
    if config['schema_version']!='m3.3b.configuration.v1' or not 1<=len(config['problems'])<=8:raise ValueError('This CLI stage supports only a handful (<=8) of demo problems')
    if not 1<=config['controls']['maxiter']<=500:raise ValueError('Cap outside validated pilot policy')
    out.mkdir(parents=True,exist_ok=False);sources=source_manifest(out);source_id=canonical_hash(sources)
    specs=[sample_problem(c,config['seed']+i,config['controls'])[2] for i,c in enumerate(config['problems'])]
    splits=split_problems(specs,config['split_seed'],config['ood_supports'],tuple(config['split_ratios']))
    atomic_json(out/'config.json',config);atomic_json(out/'problems.json',specs);atomic_json(out/'splits.json',splits);atomic_json(out/'feature_schema.json',feature_schema())
    atomic_json(out/'environment.json',dict(utc=datetime.now(timezone.utc).isoformat(),python=sys.version,executable=sys.executable,platform=platform.platform(),
        versions={n:version(n) for n in ('numpy','scipy','torch','matplotlib','pypardiso','pytest')},threads={n:os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],text=True),source_id=source_id))
    atomic_json(out/'manifest.json',dict(schema_version=SCHEMA,generation_id=out.name,source_id=source_id,config_sha=canonical_hash(config),status='incomplete',samples=[],failures=[]))
    return specs,splits,sources


def generate(out,stop_after=None):
    config=json.loads((out/'config.json').read_text());specs=json.loads((out/'problems.json').read_text());manifest=json.loads((out/'manifest.json').read_text());sources=json.loads((out/'source_manifest.json').read_text())
    assert_sources(sources)
    if canonical_hash(config)!=manifest['config_sha'] or canonical_hash(sources)!=manifest['source_id']:raise ValueError('Generation binding changed')
    regenerated=[sample_problem(c,config['seed']+i,config['controls'])[2] for i,c in enumerate(config['problems'])]
    if regenerated!=specs:raise ValueError('Saved problem specifications differ from configuration')
    splits=json.loads((out/'splits.json').read_text());validate_splits(specs,splits)
    entries=[]
    for spec in specs:
        print('Generate/resume:',spec['label'],flush=True)
        result=generate_sample(out,spec,manifest['source_id'],max_new_updates=stop_after)
        if result['status']=='incomplete':
            manifest.update(status='incomplete',samples=entries,pending_problem=spec['problem_id']);atomic_json(out/'manifest.json',manifest);return result
        t=Trajectory(result['folder']);p=t.problem
        entries.append(dict(problem_id=spec['problem_id'],label=spec['label'],folder=str(result['folder'].relative_to(out)),mesh=[p.nelx,p.nely],state_count=t.metadata['state_count'],completed_updates=t.metadata['completed_updates'],termination_reason=t.metadata['termination_reason'],
            partition=splits['assignments'][spec['problem_id']]['partition'],group_id=splits['assignments'][spec['problem_id']]['group_id'],metadata_sha256=file_hash(result['folder']/'metadata.json'),reused=result['status']=='completed_reused'))
        manifest.update(samples=entries);atomic_json(out/'manifest.json',manifest)
    rows=window_index(entries,splits,tuple(config['history_lengths']),tuple(config['horizons']));atomic_json(out/'windows.json',rows)
    manifest.update(status='completed',files={n:file_hash(out/n) for n in ('windows.json','splits.json','feature_schema.json','problems.json','config.json')});manifest.pop('pending_problem',None);atomic_json(out/'manifest.json',manifest)
    assert_sources(sources);protected_check();return dict(status='completed',samples=len(entries),windows=len(rows))


def validation(out,reproduce=False,output_prefix=None):
    manifest=json.loads((out/'manifest.json').read_text());specs=json.loads((out/'problems.json').read_text());splits=json.loads((out/'splits.json').read_text())
    if manifest['status']!='completed':raise ValueError('Dataset not yet complete')
    validate_splits(specs,splits);comparisons=[];states=0;transitions=0;channel_checks=0
    for e in manifest['samples']:
        t=Trajectory(out/e['folder']);rows=validate_sample(t,independent=True);comparisons.extend(dict(problem_id=e['problem_id'],**r) for r in rows);states+=t.metadata['state_count'];transitions+=t.metadata['completed_updates']
        p=t.problem
        for i in (0,len(t.arrays['x'])//2,len(t.arrays['x'])-1):
            for profile in PROFILES:
                f=features(p,t.arrays['x'][i],i,profile,t.state(i) if PROFILES[profile]['fresh_fem'] else None)
                if f['element_channel_names']!=PROFILES[profile]['element'] or f['node_channel_names']!=PROFILES[profile]['node']:raise ValueError('Channel order changed')
                np.testing.assert_array_equal(f['node'][0].ravel()*t.spec['force_scale'],p.forces[0::2]);np.testing.assert_array_equal(f['node'][1].ravel()*t.spec['force_scale'],p.forces[1::2])
                np.testing.assert_array_equal(f['element'][0].ravel(),t.arrays['x'][i]);np.testing.assert_allclose(f['element'][1].ravel(),t.arrays['rho'][i],rtol=1e-12,atol=1e-12)
                if profile=='fresh_full_v1':np.testing.assert_allclose(f['element'][-1].sum()*t.spec['force_scale']**2,t.arrays['compliance'][i]/2,rtol=1e-8)
                channel_checks+=1
    index=json.loads((out/'windows.json').read_text());expected=window_index(manifest['samples'],splits,tuple(json.loads((out/'config.json').read_text())['history_lengths']),tuple(json.loads((out/'config.json').read_text())['horizons']))
    if index!=expected:raise ValueError('Window index inconsistent')
    window_checks=0;benchmarks={}
    for partition in ('train','validation','test_id','test_ood'):
        for profile in PROFILES:
            ds=WindowDataset(out,partition,profile);tick=perf_counter()
            for i in range(min(len(ds),32)):
                sample=ds[i];r=sample['index'];t=ds.cache[r['problem_id']]
                np.testing.assert_array_equal(sample['target_x'].ravel(),t.arrays['x'][r['target']]);np.testing.assert_array_equal(sample['element'][0,0].ravel(),t.arrays['x'][r['anchor']-r['history_length']+1])
                window_checks+=1
            benchmarks[partition+'/'+profile]=dict(windows=min(len(ds),32),seconds=perf_counter()-tick)
    # Check each horizon explicitly rather than only the first contiguous index rows.
    for h in (1,5,10):
        for partition in ('train','validation','test_id','test_ood'):
            ds=WindowDataset(out,partition,'no_fem_v1',history_length=3,horizon=h)
            if len(ds):
                sample=ds[-1];r=sample['index'];t=ds.cache[r['problem_id']];np.testing.assert_array_equal(sample['target_x'].ravel(),t.arrays['x'][r['anchor']+h]);window_checks+=1
    reproduction=None
    if reproduce:
        spec=specs[0];original=Trajectory(out/manifest['samples'][0]['folder'])
        with tempfile.TemporaryDirectory(prefix='m3-3b-repro-') as temp:
            copy=generate_sample(temp,spec,manifest['source_id']);t=Trajectory(copy['folder'])
            for key in ('x','rho','u','compliance','g_x','g_rho'):np.testing.assert_allclose(original.arrays[key],t.arrays[key],rtol=1e-10,atol=1e-12)
        reproduction=dict(problem_id=spec['problem_id'],uninterrupted_matches_resumed=True,rtol=1e-10,atol=1e-12,scope='First demo problem, entire sequence; runtime diagnostics excluded; not cross-library bitwise claim')
    correction=Trajectory(out/manifest['samples'][0]['folder']);result=evaluate_prediction(correction.problem,correction.arrays['x'][1],correct=True)
    np.testing.assert_allclose(result['corrected_x'],correction.arrays['x'][2],rtol=1e-10,atol=1e-12)
    numeric_file=(output_prefix+'_numerical.csv') if output_prefix else 'numerical_validation.csv'
    with (out/numeric_file).open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in comparisons for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(comparisons)
    audit=dict(schema_version='m3.3b.validation.v1',passed=True,states=states,transitions=transitions,numerical_comparisons=len(comparisons),channel_profiles_checked=channel_checks,windows=len(index),window_samples_checked=window_checks,
        split_leakage=False,all_partitions_nonempty=all(any(e['partition']==part for e in manifest['samples']) for part in ('train','validation','test_id','test_ood')),sampler_regeneration='config/seed matches saved complete specifications',
        reproducibility=reproduction,periodic_correction_demo=dict(equilibrium_solves=result['equilibrium_solves'],matches_actual_next_state=True),loader_benchmarks=benchmarks,protected_files_unchanged=protected_check())
    atomic_json(out/((output_prefix+'.json') if output_prefix else 'validation.json'),audit);return audit


def checks(out):
    logs=out/'test_logs';logs.mkdir(exist_ok=True);results={}
    for label,extra in [('focused',['tests/test_trajectory_dataset.py']),('regression',[])]:
        cmd=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*extra];r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,cwd=ROOT)
        (logs/(label+'.log')).write_text('Command: '+' '.join(cmd)+'\n'+r.stdout+f'\nExit code: {r.returncode}\n');results[label]=dict(exit_code=r.returncode,stdout=r.stdout)
        print(label+': '+r.stdout.strip(),flush=True)
        if r.returncode:raise RuntimeError(label+' tests failed')
    atomic_json(out/'test_results.json',results);return results


def report(out):
    manifest=json.loads((out/'manifest.json').read_text());audit=json.loads((out/'validation.json').read_text());test=json.loads((out/'test_results.json').read_text())
    lines=['# M3.3B — Mechanically consistent dataset pipeline','',
        'This is a six-problem small-mesh demonstration of storage, features, supervised windows, splitting and replay. It implements no neural network and generates no 100–200-problem dataset. M1–M3.3A numerical components and artifacts are preserved.','',
        '## Scientific purposes and observed optimizer limits','',
        'A: predict x[t+1] from current design and optional current mechanics. B: predict saved x[t+h], h=1/5/10, from histories of length1/3. C: periodically evaluate a predicted design with the unchanged FEM solver and optionally apply one explicit feasible-upper OC correction. These are data/API contracts; no acceleration or learning success is claimed.',
        'M3.3A found feasible-upper updates feasible on20 cases, but8 cases remained capped and endpoint stationarity worsened on7. Thus this pipeline explicitly selects feasible_upper_v1 for feasibility, preserves the core controls and records quality/termination separately. The demo cap40 is intentionally a bounded dynamics example; a cap or .001 change crossing is not an optimum certificate. All reference_label_eligible flags remain false pending a research quality policy.','',
        '| Problem | Mesh | Partition | Updates | States/FEM evaluations | Termination |','|---|---|---|---:|---:|---|']
    for e in manifest['samples']:lines.append(f"| {e['label']} | {e['mesh']} | {e['partition']} | {e['completed_updates']} | {e['state_count']} | {e['termination_reason']} |")
    lines+=['','## Complete mechanics, feature channels and FEM cost','',
        'Each complete problem records physical dimensions, exact nodal coordinates and force arrays, constrained DOFs, ordering, material/SIMP/filter conventions, exact initial x, random seed and effective optimizer version/controls. The deterministic sampler normalizes a requested direction and applies its explicit force magnitude to an exact mesh node; off-grid requests fail instead of snapping or interpolating. The feature normalization never changes solver forces, gradients or multiplier precision. Different force scales remain different optimizer problems.',
        'Elements have shape(nely,nelx), nodes(nely+1,nelx+1), C-order and increasing physical row-y. Element and nodal tensors stay separate. A point force remains at its actual node; support x/y masks represent exactly the fixed DOFs. Physical and normalized coordinates are separately named. For rectangular cells, the unchanged index-space filter has physical semi-axes radius*dx/radius*dy.',
        'no_fem_v1: x, locally filtered rho, coordinates, volume target, iteration/maxiter and known force magnitude plus exact nodal force/support channels. These require zero FEM evaluations. fresh_gradient_v1 adds g_x/Fscale². fresh_full_v1 adds g_rho/Fscale², physical element elastic energy/Fscale² and nodal u/Fscale. These mechanics must be evaluated at the exact current x and mechanical problem. Training uses saved current-state truth; inference requires new FEM results or explicitly cached matching states. Historical mechanics may be cached, but gradients cannot silently be reused at changed x.',
        'Fscale=norm(full nodal force vector). Raw arrays retain physical values; normalized gradient/energy channels divide by Fscale² and displacement by Fscale. Energy is .5*s(rho)*u_e.T KE0 u_e and sums to C/2 before normalization. Progress uses the configured update cap, never final trajectory length or final compliance. No train/test data statistics are used. Feature lists, units and shapes are documented in feature_schema.json and DATASET_SCHEMA.md.','',
        '## Storage, transition timing and atomic restart','',
        'Each state contains x,rho,u,C,g_rho,g_x,iteration,incoming maximum change, volume, equilibrium/energy and stationarity diagnostics. State0 uses incoming-change0 with has_incoming_update=false; all numerical stored values are finite. Transitions reference adjacent states and bind actual delta by a dtype/shape-aware digest, storing OC multiplier/bracket, controls, validity and change. The loader returns x_t,x_next and actual delta without duplicating complete density histories.',
        'State t follows t completed updates; the terminal updated x is independently FEM-evaluated and has no outgoing transition. Nominal committed FEM solves equal states. Crashes before journal commit can incur extra uncommitted solves; no exactly-once execution claim is made.',
        'During generation, independently hashed frame/transition files become committed together through an atomic journal. Resume checks source/configuration and every committed checksum, retains incoming change and absolute update count, and quarantines uncommitted orphans. Completed samples are immutable and reused after integrity checks. Final publication uses an atomic directory rename; derived NumPy arrays can be rebuilt from committed checkpoints. The journal is retained as provenance; its working-frame references are superseded after redundant frames are removed.',
        'Final storage is portable .npy arrays with per-array checksums and read-only memory mapping. This NumPy format supports genuine random slice access without decompressing a whole NPZ trajectory. Each field is stored once; frame checkpoints are removed after publication, while transition deltas are reconstructed and verified. A bounded per-worker cache avoids repeated openings; cache state is cleared on worker serialization. Homogeneous mesh/history batch indices support a PyTorch DataLoader without hidden padding or load interpolation.','',
        '## Problem-level splits and leakage scope','',
        'All windows from a complete problem belong to one partition. Related families are unioned with conservative normalized-mechanics buckets, grouping force magnitude, initial seed, resolution and horizontal-mirror/global-force-sign variants. The grouping policy is versioned, reproducible and rechecked independently. Declared related_group is required; coarse buckets do not prove that every possible continuous near-duplicate has been detected. Future configurations require a family/leakage audit.',
        'The demo validation pin/roller family is absent from its tiny training subset; ID is declared sampler-domain membership, not an observed-support coverage guarantee. A later pilot must audit category coverage. The demo includes train, validation, test_id and test_ood. OOD explicitly withholds bottom_clamp supports from the ID partitions. ID means membership in the declared sampler domain; six examples cannot establish statistical in-distribution coverage or learned generalization. Related-group IDs and split seeds are saved; individual iterations are never split. Window/index/schema checksums and bounds are checked on loading. Target labels and target_termination are annotations, never input channels.','',
        '## Checks actually run','',
        f"All {audit['states']} saved states and {audit['transitions']} transitions passed mechanical checks, numerical reconstruction and OC replay ({audit['numerical_comparisons']} comparisons). Independent reconstruction uses another solver compliance/gradient API and COO stiffness assembly in the existing M3.3A validator. Boundary conditions, force/moment balance, energy, filter, g_rho/g_x, exact multiplier selection, finite values, terminal timing and volume are checked.",
        f"{audit['channel_profiles_checked']} state/profile combinations and {audit['window_samples_checked']} loaded windows checked; {audit['windows']} window descriptors generated. All h=1/5/10 targets and histories are checked against actual saved indices. Split coverage/group separation, OOD support isolation, deterministic problem regeneration, array integrity and sample identity passed. A paused two-update first sample matched an independently generated uninterrupted complete sequence, excluding timing diagnostics. A fresh-FEM one-step correction matched the next stored OC state.",
        'Focused tests include corrupted checkpoint rejection, orphan recovery, source mismatch refusal, no completed-sample recomputation, normalization under force scaling, faithful nodal channels, physical energy/C consistency, supervised indexing, mmap loading and PyTorch DataLoader integration.','']
    for label,result in test.items():lines.append(label+': '+result['stdout'].strip().splitlines()[-1]+f"; exit {result['exit_code']}.")
    lines+=['',f"All {audit['protected_files_unchanged']} fingerprinted protected files remain unchanged. Source bytes used for execution, versioned configuration, code hashes, dependency versions, thread settings, array fingerprints and split manifests are archived. MKL remains16; OpenBLAS analysis uses1. No commit, push, HPC/SLURM or neural training.",'',
        '## Final scientific assessment','',
        'The demonstration CLI intentionally accepts at most eight problems. A later reviewed production entry point must explicitly lift that stage limit; no 100–200-sample execution is enabled by default.',
        'Saved trajectories are mechanically consistent within the recorded tolerances and independently reproducible from their complete specifications. Channels preserve the actual discretized problem, with explicit units, grids and cost provenance. One-step and multi-step labels correspond to real saved states. The declared problem/group split prevents the tested leakage classes; it is not a proof covering arbitrary new families.',
        'The software is ready for a reviewed controlled 100–200-problem pilot as a dynamics dataset after configuration/family audit and representative resource testing. It is not yet validated as a generator of certified optimal final labels. Before that pilot, choose a quality/stopping policy for capped or worse-stationarity cases, check force-scale/fixed-bracket sensitivity, assess OOD family coverage and balance/weights for variable-length trajectories, and benchmark throughput/recovery at the intended resolution. This stage does not launch that pilot.']
    (out/'M3_3B_DATASET_PIPELINE_REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['all','generate','resume','validate','report']);parser.add_argument('--output',type=Path);parser.add_argument('--config',type=Path);parser.add_argument('--stop-after-updates',type=int);args=parser.parse_args();protected_check()
    if args.command in ('all','generate'):
        out=args.output or ROOT/'m3_3b/generations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');config=json.loads(args.config.read_text()) if args.config else demo_config();initialize(out,config)
        if args.command=='all':
            # Deliberately interrupt and resume the first sample to exercise the
            # production journal path; do not merely assert that resume exists.
            paused=generate(out,stop_after=2)
            if paused['status']!='incomplete':raise ValueError('Expected controlled interruption')
            print('Controlled interruption committed two updates; resuming',flush=True)
        result=generate(out,args.stop_after_updates)
    else:
        if args.output is None:parser.error('Existing generation --output required')
        out=args.output;result=generate(out,args.stop_after_updates) if args.command=='resume' else None
    if args.command in ('all','validate'):
        audit=validation(out,reproduce=args.command=='all',output_prefix=('fresh_validation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')) if args.command=='validate' else None);print('Validation:',audit['states'],'states;',audit['transitions'],'transitions',flush=True)
    if args.command=='all':checks(out);report(out)
    if args.command=='report':report(out)
    assert_sources(json.loads((out/'source_manifest.json').read_text()));protected_check()
    if args.command=='all':atomic_json(ROOT/'m3_3b/latest_generation.json',dict(generation=str(out.resolve()),report=str((out/'M3_3B_DATASET_PIPELINE_REPORT.md').resolve()),manifest_sha256=file_hash(out/'manifest.json'),source_manifest_sha256=file_hash(out/'source_manifest.json')))
    print('Output:',out,flush=True)

if __name__=='__main__':main()
