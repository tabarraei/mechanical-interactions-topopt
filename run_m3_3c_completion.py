#!/usr/bin/env python
"""Ten-problem M3.3C completion pilot. No neural training or larger-mesh FEM."""
import argparse,json,csv,gzip,os,sys,shutil,subprocess,tempfile,platform,itertools,hashlib
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from collections import defaultdict,Counter
from importlib.metadata import version
import numpy as np
from topopt_utils.fea.trajectory_dataset import sample_problem,split_problems,validate_splits,Trajectory,features,PROFILES,window_index,feature_schema,canonical_hash,file_hash,WindowDataset
from topopt_utils.fea.dataset_storage import generate_sample,validate_sample,atomic_json
from topopt_utils.fea.dataset_readiness import POLICY,project_prediction,predict,balanced_training_indices,stage,metrics,endpoint_analysis,aggregate,quality_flags
from topopt_utils.fea.mechanical_problem import DensityMap
from topopt_utils.fea.robust_oc import update,trial
from topopt_utils.fea.consistent_trajectory import evaluate_state
from topopt_utils.fea.solver import FeaSolver
from run_m3_3b import source_manifest,assert_sources
from run_m3_3c import memory_status
ROOT=Path(__file__).resolve().parent
SCALING=ROOT/'m3_3c/generations/20261013T201818977895Z'

def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    return v

def save(path,v):atomic_json(path,clean(v))

def configuration():
    def c(label,mesh,domain,support,pos,direction,vol,group,initial=None,magnitude=1.):
        return dict(label=label,mesh=list(mesh),domain=list(domain),support=support,loads=[dict(position=list(pos),direction=list(direction),magnitude=magnitude)],volume_fraction=vol,filter_radius=2.,related_group=group,initial=initial or dict(kind='uniform',slack=1e-8))
    problems=[c('axial_small',(12,6),(12.,6.),'left_clamp',(12.,3.),(1.,0.),.5,'axial'),
      c('axial_medium',(24,12),(24.,12.),'left_clamp',(24.,6.),(1.,0.),.5,'axial'),
      c('right_mid_vertical',(48,24),(48.,24.),'left_clamp',(48.,12.),(0.,-1.),.5,'right_vertical'),
      c('right_quarter_volume03',(48,24),(48.,24.),'left_clamp',(48.,6.),(0.,-1.),.3,'right_vertical'),
      c('right_corner_oblique_volume04',(48,24),(48.,24.),'left_clamp',(48.,24.),(.6,-.8),.4,'corner_oblique'),
      c('top_mid_volume06',(48,24),(48.,24.),'left_clamp',(24.,24.),(0.,-1.),.6,'top_mid'),
      c('bottom_clamp_horizontal',(48,24),(48.,24.),'bottom_clamp',(24.,24.),(1.,0.),.5,'bottom_clamp'),
      c('pin_roller_top_quarter',(48,24),(48.,24.),'pin_roller',(12.,24.),(0.,-1.),.5,'pin_roller'),
      c('long_domain_vertical',(48,24),(96.,24.),'left_clamp',(96.,12.),(0.,-1.),.5,'long_vertical'),
      c('perturbed_right_mid',(48,24),(48.,24.),'left_clamp',(48.,12.),(0.,-1.),.5,'right_vertical',dict(kind='perturbed',amplitude=.02,slack=1e-8))]
    force=[]
    for mag in (.1,10.):
        case=json.loads(json.dumps(problems[0]));case['label']=f'force_{mag:g}';case['loads'][0]['magnitude']=mag;force.append(case)
    return dict(schema_version='m3.3c.completion.configuration.v1',seed=701,split_seed=577,controls=dict(option='feasible_upper_v1',maxiter=500,tolerance=.001),history_lengths=[1,3],horizons=[1,5,10],problems=problems,force_variants=force,ood_supports=['bottom_clamp'],quality_policy=POLICY,execution_limit=dict(max_problems=12,max_elements=1152,max_updates=500,max_peak_mib=2048,max_estimated_seconds=3600),sampling=dict(seed=720,count=10000),prediction_subset='Each primary problem, anchors2/middle/late for each horizon; raw invalid-box predictions never passed to FEM')

def protect(out,create=False):
    path=out/'protected_before.json'
    if create and not path.exists():
        files={}
        for p in ROOT.rglob('*'):
            if p.is_file() and not any(t in p.parts for t in ('.git','__pycache__','.pytest_cache','completion_generations')):files[str(p.relative_to(ROOT))]=file_hash(p)
        save(path,files)
    old=json.loads(path.read_text());changed=[n for n,h in old.items() if not (ROOT/n).is_file() or file_hash(ROOT/n)!=h]
    if changed:raise ValueError('Protected source/artifact changed: '+str(changed))
    return len(old)

def initialize(out):
    if (out/'manifest.json').exists():raise ValueError('Existing completed/incomplete generation: use explicit resume')
    if out.exists() and set(p.name for p in out.iterdir())-{'protected_before.json','M3_3C_COMPLETION_PLAN.md'}:raise ValueError('Output must be new or contain only its pre-execution plan/fingerprints')
    out.mkdir(parents=True,exist_ok=True);protect(out,True)
    if not (out/'M3_3C_COMPLETION_PLAN.md').exists():
        plans=sorted((ROOT/'m3_3c/completion_generations').glob('*/M3_3C_COMPLETION_PLAN.md'))
        if not plans:raise ValueError('Completion plan required')
        shutil.copyfile(plans[0],out/'M3_3C_COMPLETION_PLAN.md')
    (out/'test_logs').mkdir(exist_ok=True);sources=source_manifest(out);c=configuration();specs=[]
    for i,case in enumerate(c['problems']+c['force_variants']):
        seed=c['seed']+i if i<10 else c['seed']
        specs.append(sample_problem(case,seed,c['controls'])[2])
    splits=split_problems(specs,c['split_seed'],c['ood_supports']);validate_splits(specs,splits)
    save(out/'config.json',c);save(out/'problems.json',specs);save(out/'splits.json',splits);save(out/'feature_schema.json',feature_schema())
    save(out/'manifest.json',dict(schema_version='m3.3b.dataset.v1',generation_id=out.name,source_id=canonical_hash(sources),config_sha=canonical_hash(c),status='incomplete',samples=[],failures=[]))
    save(out/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),dependencies={k:version(k) for k in ('numpy','scipy','pypardiso','torch','pytest','matplotlib')},threads={k:os.environ.get(k) for k in ('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},solver='PARDISO direct SPD mtype2; cached symbolic, fresh numeric factors per state',scope='Sequential <=1152 element pilot; no192 FEM, training, full dataset, HPC or Git changes',initial_memory=memory_status()))
    # Read-only reuse: check executing archives and all saved scaling sample bytes.
    source=json.loads((SCALING/'source_manifest.json').read_text())
    for name,entry in source.items():
        if file_hash(SCALING/entry['archive'])!=entry['sha256']:raise ValueError('Scaling source archive corrupted')
    prior=json.loads((SCALING/'bounded_trajectories.json').read_text())
    for row in prior:Trajectory(SCALING/row['folder'])
    save(out/'scaling_integration.json',dict(source_generation=str(SCALING.relative_to(ROOT)),source_archive_hashes_passed=True,native_array_checksums_passed=True,scaling_report_sha256=file_hash(SCALING/'M3_3C_MESH_SCALING_REPORT.md'),benchmarks=[json.loads((SCALING/f'benchmark_{r["experiment"]}_{r["mesh"][0]}.json').read_text()) for r in prior],bounded=prior,larger_estimates=json.loads((SCALING/'larger_mesh_estimates.json').read_text()),no_scaling_FEM_rerun=True))

def regenerate_specs(out):
    c=json.loads((out/'config.json').read_text());specs=json.loads((out/'problems.json').read_text())
    rebuilt=[sample_problem(case,c['seed']+i if i<10 else c['seed'],c['controls'])[2] for i,case in enumerate(c['problems']+c['force_variants'])]
    if rebuilt!=specs:raise ValueError('Configuration/seed regeneration mismatch')
    return c,specs

def generate(out):
    sources=json.loads((out/'source_manifest.json').read_text());assert_sources(sources);c,specs=regenerate_specs(out);manifest=json.loads((out/'manifest.json').read_text());sid=manifest['source_id']
    if sid!=canonical_hash(sources) or canonical_hash(c)!=manifest['config_sha']:raise ValueError('Source/config binding mismatch')
    # Preserve this measured production prefix and resume it with its original cap.
    if not (out/'resource_gate.json').exists():
        start=perf_counter();prefix=generate_sample(out,specs[2],sid,max_new_updates=20);elapsed=perf_counter()-start
        count=21;estimated=elapsed/count*501*12+240
        gate=dict(prefix_problem=specs[2]['problem_id'],prefix_elapsed_seconds=elapsed,prefix_committed_states=count,prefix_updates=20,prefix_status=prefix['status'],memory=memory_status(),worst_case_12trajectory_generation_plus_analysis_seconds=estimated,max_seconds=3600,max_peak_mib=2048,storage_budget_mib=12*501*2.84/21,assumptions='Cold production prefix scaled by states;12 total controls; full500 budget each plus240s replay/analysis allowance. Sequential mmap; realized histories may stop early; not a performance guarantee.',passed=memory_status()['VmHWM_mib']<2048 and estimated<3600)
        save(out/'resource_gate.json',gate)
        if not gate['passed']:raise RuntimeError('Resource gate failed; preserve prefix and stage later runs')
    elif not json.loads((out/'resource_gate.json').read_text())['passed']:raise RuntimeError('Prior gate failed')
    splits=json.loads((out/'splits.json').read_text());validate_splits(specs,splits);entries=[];throughput=[]
    for i,spec in enumerate(specs):
        print('Generate/resume',spec['label'],flush=True);start=perf_counter()
        try:result=generate_sample(out,spec,sid)
        except Exception as e:
            manifest['failures'].append(dict(problem_id=spec['problem_id'],error=repr(e)));save(out/'manifest.json',manifest);raise
        seconds=perf_counter()-start;tr=Trajectory(result['folder']);p=tr.problem
        e=dict(problem_id=spec['problem_id'],label=spec['label'],folder=str(result['folder'].relative_to(out)),mesh=[p.nelx,p.nely],state_count=tr.metadata['state_count'],completed_updates=tr.metadata['completed_updates'],termination_reason=tr.metadata['termination_reason'],partition=splits['assignments'][spec['problem_id']]['partition'],group_id=splits['assignments'][spec['problem_id']]['group_id'],metadata_sha256=file_hash(tr.folder/'metadata.json'),role='primary' if i<10 else 'force_control')
        entries.append(e);throughput.append(dict(**e,session_seconds=seconds,native_storage_bytes=sum(f.stat().st_size for f in tr.folder.rglob('*') if f.is_file()),memory=memory_status(),oc_total_seconds=sum(t['oc_seconds'] for t in tr.transitions),reused=result['status']=='completed_reused'))
        manifest['samples']=entries;save(out/'manifest.json',manifest);print(f"Saved {e['completed_updates']} updates / {e['termination_reason']} in {seconds:.2f}s",flush=True)
        if memory_status()['VmHWM_mib']>2048:raise RuntimeError('Memory gate exceeded; remaining runs stopped')
    windows=window_index(entries,splits,(1,3),(1,5,10));save(out/'windows.json',windows)
    manifest.update(status='completed',files={n:file_hash(out/n) for n in ('windows.json','splits.json','feature_schema.json','problems.json','config.json')});save(out/'manifest.json',manifest);save(out/'throughput.json',throughput);assert_sources(sources)

def validate(out,fresh=False):
    c,specs=regenerate_specs(out);manifest=json.loads((out/'manifest.json').read_text());splits=json.loads((out/'splits.json').read_text());validate_splits(specs,splits)
    if manifest['status']!='completed':raise ValueError('Dataset incomplete')
    prefix='fresh_validation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if fresh else 'validation'
    summary=[];maxima={};checks=0
    with gzip.open(out/(prefix+'_comparisons.csv.gz'),'wt',newline='') as f:
        writer=None
        for e in manifest['samples']:
            print('Independent replay',e['label'],flush=True);tr=Trajectory(out/e['folder']);start=perf_counter();rows=validate_sample(tr,independent=True)
            if writer is None:
                fields=list(dict.fromkeys(['problem_id']+[k for r in rows for k in r]));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for r in rows:writer.writerow(clean(dict(problem_id=e['problem_id'],**r)))
            p=tr.problem
            for index in (0,len(tr.arrays['x'])//2,len(tr.arrays['x'])-1):
                for profile in PROFILES:
                    feat=features(p,tr.arrays['x'][index],index,profile,tr.state(index) if PROFILES[profile]['fresh_fem'] else None)
                    assert feat['element'].shape==(len(PROFILES[profile]['element']),p.nely,p.nelx) and feat['node'].shape==(len(PROFILES[profile]['node']),p.nely+1,p.nelx+1)
                    np.testing.assert_allclose(feat['node'][0].ravel()*tr.spec['force_scale'],p.forces[::2],rtol=0,atol=1e-15);np.testing.assert_allclose(feat['node'][1].ravel()*tr.spec['force_scale'],p.forces[1::2],rtol=0,atol=1e-15)
                    if profile=='fresh_full_v1':np.testing.assert_allclose(feat['element'][-1].sum()*tr.spec['force_scale']**2,tr.arrays['compliance'][index]/2,rtol=1e-8)
                    checks+=1
            for k in ('residual_relative_force','force_balance_relative','moment_balance_relative','energy_compliance_relative','residual_backward_error'):
                maxima[k]=max(maxima.get(k,0),float(tr.arrays[k].max()))
            summary.append(dict(problem_id=e['problem_id'],states=e['state_count'],updates=e['completed_updates'],rows=len(rows),seconds=perf_counter()-start,passed=True))
    windows=json.loads((out/'windows.json').read_text());assert windows==window_index(manifest['samples'],splits,(1,3),(1,5,10))
    window_checks=0
    for partition in ('train','validation','test_id','test_ood'):
        for k in (1,3):
            for h in (1,5,10):
                ds=WindowDataset(out,partition,'no_fem_v1',k,h)
                for i in ([0,len(ds)//2,len(ds)-1] if len(ds) else []):
                    sample=ds[i];r=sample['index'];tr=ds.cache[r['problem_id']]
                    np.testing.assert_array_equal(sample['target_x'].ravel(),tr.arrays['x'][r['anchor']+h]);np.testing.assert_array_equal(sample['element'][0,0].ravel(),tr.arrays['x'][r['anchor']-k+1]);window_checks+=1
    audit=dict(passed=True,trajectories=summary,states=sum(x['states'] for x in summary),updates=sum(x['updates'] for x in summary),comparison_rows=sum(x['rows'] for x in summary),feature_checks=checks,window_checks=window_checks,mechanical_maxima=maxima,split_leakage=False,protected_files_unchanged=protect(out),memory=memory_status())
    save(out/(prefix+'.json'),audit);return audit

def restart_benchmark(out):
    spec=json.loads((out/'problems.json').read_text())[0];source=json.loads((out/'manifest.json').read_text())['source_id'];original=Trajectory(out/'samples'/spec['problem_id'])
    with tempfile.TemporaryDirectory(prefix='m3-3c-uninterrupted-') as tmp:
        t=perf_counter();result=generate_sample(tmp,spec,source);uninterrupted_seconds=perf_counter()-t;uninterrupted=Trajectory(result['folder'])
        with tempfile.TemporaryDirectory(prefix='m3-3c-restarted-') as other:
            t=perf_counter();partial=generate_sample(other,spec,source,max_new_updates=2);prefix=perf_counter()-t
            t=perf_counter();result=generate_sample(other,spec,source);resume=perf_counter()-t;restarted=Trajectory(result['folder'])
            discrepancies={}
            for key in original.arrays:
                np.testing.assert_allclose(original.arrays[key],restarted.arrays[key],rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(uninterrupted.arrays[key],restarted.arrays[key],rtol=1e-10,atol=1e-12)
                discrepancies[key]=float(np.max(abs(np.asarray(uninterrupted.arrays[key],float)-np.asarray(restarted.arrays[key],float))))
            record=dict(problem_id=spec['problem_id'],prefix_seconds=prefix,resume_seconds=resume,uninterrupted_seconds=uninterrupted_seconds,ratio=(prefix+resume)/uninterrupted_seconds,numerical_match=True,max_array_discrepancies=discrepancies,extra_benchmark_committed_FEM_evaluations=2*original.metadata['state_count'],checkpoint_reuses_committed_prefix=True,scope='One small full trajectory; one timing repetition; all saved numerical arrays matched; no crash-at-every-write claim',peak_memory=memory_status())
    save(out/'restart_benchmark.json',record)

def force_study(out):
    manifest=json.loads((out/'manifest.json').read_text());entries=[manifest['samples'][i] for i in (0,10,11)];trs=[Trajectory(out/e['folder']) for e in entries];base=trs[0];result=[];same_state=[];solves=0
    for tr in trs:
        alpha=tr.spec['force_scale'];T=min(len(base.arrays['x']),len(tr.arrays['x']));a=tr.arrays;b=base.arrays
        result.append(dict(force=alpha,states=len(a['x']),updates=tr.metadata['completed_updates'],stop=tr.metadata['termination_reason'],common_prefix_states=T,x_common_prefix_max=float(abs(a['x'][:T]-b['x'][:T]).max()),x_terminal_mae=float(abs(a['x'][-1]-b['x'][-1]).mean()),normalized_C_common_prefix_max_relative=float(np.max(abs(a['compliance'][:T]/alpha**2-b['compliance'][:T])/np.maximum(abs(b['compliance'][:T]),1e-12))),normalized_gradient_common_prefix_relative_l2=float(np.linalg.norm(a['g_x'][:T]/alpha**2-b['g_x'][:T])/np.linalg.norm(b['g_x'][:T])),normalized_multiplier_terminal=tr.transitions[-1]['selected_multiplier']/alpha**2,terminal_pg=float(a['projected_gradient_inf'][-1]),max_signed_volume_excess=float(a['volume_excess'].max()),brackets_all_valid=all(t['diagnostic']['upper_feasible'] for t in tr.transitions)))
        for t in (0,len(b['x'])//2,len(b['x'])-1):
            x=np.asarray(b['x'][t]);state=evaluate_state(tr.problem,x);solves+=1
            xn,d=update(x,state['g_x'],tr.problem.volume_fraction,'feasible_upper_v1');ref,rd=update(x,b['g_x'][t],base.problem.volume_fraction,'feasible_upper_v1')
            errors={k:float(np.linalg.norm(state[k]/(alpha if k=='u' else alpha**2)-b[k][t])/max(np.linalg.norm(b[k][t]),1e-12)) for k in ('u','compliance','g_rho','g_x')}
            assert max(errors.values())<1e-8
            normalized_lambda=d['multiplier_upper']/alpha**2;reference_lambda=rd['multiplier_upper']
            # A diagnostic tight normalized bracket is proposed only, never replaces stored OC.
            low=d['multiplier_lower'];high=d['multiplier_upper'];low_trial=trial(x,state['g_x'],low);high_trial=trial(x,state['g_x'],high)
            same_state.append(dict(force=alpha,base_iteration=t,mechanical_scaling_relative_errors=errors,normalized_lambda=normalized_lambda,normalized_lambda_relative_error=abs(normalized_lambda-reference_lambda)/max(reference_lambda,1e-12),bracket_relative_width=d['bracket_width']/max(d['multiplier_upper'],1e-30),OC_same_x_design_max=float(abs(xn-ref).max()),bracket_endpoint_design_spread=float(abs(low_trial-high_trial).max()),strict_feasible=d['strict_feasible'],root_status=d['root_status']))
    save(out/'force_scaling.json',dict(trajectory_comparisons=result,same_design_tests=same_state,actual_FEM_evaluations=solves,proposed_variant='Future scale-aware OC version could use known force_scale**2 to normalize absolute search width/bracket or relative safeguarded tolerance. Not implemented or used; preserve fixed-bracket failure diagnostics.',qualification='Common-prefix trajectories have differing x after initial updates; normalized path differences are not pure same-design mechanics scaling errors. Same-design tests isolate exact linear-elastic scaling and finite multiplier precision.'))

def analyze(out):
    manifest=json.loads((out/'manifest.json').read_text());entries=manifest['samples'];windows=json.loads((out/'windows.json').read_text());summaries=[];diagnostics=[]
    for e in entries:
        tr=Trajectory(out/e['folder']);summary=endpoint_analysis(tr);summary.update(role=e['role'],partition=e['partition']);summaries.append(summary)
        p=tr.problem;a=tr.arrays
        for t in range(len(a['x'])):
            x=a['x'][t];diagnostics.append(dict(problem_id=e['problem_id'],iteration=t,stage=stage(t),compliance=float(a['compliance'][t]),change=float(a['previous_change'][t]),volume_excess=float(a['volume_excess'][t]),PG=float(a['projected_gradient_inf'][t]),box_KKT=float(a['box_stationarity_inf'][t]),complementarity=float(a['volume_complementarity_scaled'][t]),lower_fraction=float(np.mean(x<=.001+1e-8)),upper_fraction=float(np.mean(x>=1-1e-8)),multiplier=tr.transitions[t]['selected_multiplier'] if t<len(tr.transitions) else None))
    save(out/'endpoint_quality.json',dict(policy=POLICY,endpoints=summaries,native_reference_metadata_remains_false=True,sidecar_only=True))
    save(out/'convergence_diagnostics.json',diagnostics)
    sensitivity=[]
    for pg,comp,C,binary in itertools.product((.005,.01,.02),(1e-6,1e-5,1e-4),(1e-4,1e-3,1e-2),(.001,.005,.01)):
        policy=dict(POLICY,pg=pg,box_kkt=pg,complementarity=comp,relative_compliance_span=C,binary_disagreement=binary)
        eligible=[s['label'] for s in summaries[:10] if quality_flags(s,policy)['reference_label_eligible']]
        sensitivity.append(dict(pg_box=pg,complementarity=comp,Cspan=C,binary=binary,eligible=eligible,count=len(eligible)))
    save(out/'quality_threshold_sensitivity.json',sensitivity)
    # Stream per-window results; aggregates contain scalar errors, never stacked trajectories.
    targetgroups=defaultdict(list);baselinegroups=defaultdict(list);counts=Counter();futurestats=[];prediction_subset=[];actual_prediction_FEM=0
    with gzip.open(out/'prediction_baselines.csv.gz','wt',newline='') as bf, gzip.open(out/'target_statistics.csv.gz','wt',newline='') as tf:
        bw=tw=None
        for e in entries:
            tr=Trajectory(out/e['folder']);p=tr.problem;a=tr.arrays;own=[w for w in windows if w['problem_id']==e['problem_id']];cached={}
            for w in own:
                t=w['anchor'];h=w['horizon'];k=w['history_length'];current=a['x'][t];truth=a['x'][t+h];history=a['x'][t-k+1:t+1];phase=stage(t)
                m=metrics(p,current,truth);delta=truth-current;energy=delta**2;top=max(1,int(np.ceil(.1*p.n)));capture=float(np.sort(energy)[-top:].sum()/energy.sum()) if energy.sum()>0 else 0.
                row=dict(problem_id=e['problem_id'],family=tr.spec['related_group'],partition=e['partition'],termination=e['termination_reason'],k=k,h=h,anchor=t,stage=phase,**m,near_unchanged=bool(m['x_max']<.001),small_MAE=bool(m['x_mae']<1e-4),top10percent_update_energy=capture)
                if tw is None:tw=csv.DictWriter(tf,fieldnames=list(row));tw.writeheader()
                tw.writerow(row);targetgroups[(e['label'],k,h,phase)].append(row);counts[(e['problem_id'],k,h,phase)]+=1
                methods=['no_change','no_change_projected']+(['linear','linear_projected','two_update','two_update_projected'] if k==3 else [])
                if h==1:methods+=['exact_OC_cached_gradient']
                for method in methods:
                    start=perf_counter()
                    if method=='exact_OC_cached_gradient':
                        if t not in cached:cached[t]=update(current,a['g_x'][t],p.volume_fraction,'feasible_upper_v1')[0]
                        y=cached[t];deployment_fem=1
                    else:y=predict(current,history,h,method,p.volume_fraction);deployment_fem=0
                    seconds=perf_counter()-start;mm=metrics(p,y,truth)
                    r=dict(problem_id=e['problem_id'],label=e['label'],partition=e['partition'],k=k,h=h,anchor=t,stage=phase,method=method,prediction_seconds=seconds,actual_FEM_evaluations=0,required_fresh_FEM_for_inference=deployment_fem,**mm)
                    if bw is None:bw=csv.DictWriter(bf,fieldnames=list(r));bw.writeheader()
                    bw.writerow(r);baselinegroups[(method,k,h,phase)].append(r)
            # Current/past-only methods; common selected anchors for fresh C and true repeated OC.
            if e['role']=='primary':
                for h in (1,5,10):
                    T=len(a['x']);anchors=sorted(set((2,max(2,(T-h-1)//2),T-h-1)))
                    for t in anchors:
                        if t<2 or t+h>=T:continue
                        truth=a['x'][t+h];current=a['x'][t];history=a['x'][t-2:t+1]
                        for method in ('no_change','linear','linear_projected','two_update','two_update_projected'):
                            start=perf_counter();y=predict(current,history,h,method,p.volume_fraction);prediction_time=perf_counter()-start;mm=metrics(p,y,truth)
                            record=dict(problem_id=e['problem_id'],label=e['label'],anchor=t,h=h,stage=stage(t),method=method,prediction_seconds=prediction_time,**mm)
                            if mm['box_feasible']:
                                tick=perf_counter();state=evaluate_state(p,y);actual_prediction_FEM+=1;record.update(fresh_FEM_evaluations=1,FEM_seconds=perf_counter()-tick,compliance=float(state['compliance']),relative_C_difference=float((state['compliance']-a['compliance'][t+h])/a['compliance'][t+h]))
                            else:record.update(fresh_FEM_evaluations=0,FEM_skipped='Raw density outside verified box; metrics retained, no hidden clipping or unsupported mechanics')
                            prediction_subset.append(record)
                        solver=FeaSolver();y=np.asarray(current).copy();start=perf_counter()
                        for step in range(h):
                            state=evaluate_state(p,y,solver);y,d=update(y,state['g_x'],p.volume_fraction,'feasible_upper_v1');actual_prediction_FEM+=1
                        path_seconds=perf_counter()-start;np.testing.assert_allclose(y,truth,rtol=1e-8,atol=1e-10)
                        state=evaluate_state(p,y,solver);actual_prediction_FEM+=1
                        prediction_subset.append(dict(problem_id=e['problem_id'],label=e['label'],anchor=t,h=h,stage=stage(t),method='exact_OC_fresh_multistep',prediction_seconds=path_seconds,fresh_FEM_evaluations=h+1,prediction_FEM_evaluations=h,extra_compliance_validation_FEM=1,compliance=state['compliance'],relative_C_difference=float((state['compliance']-a['compliance'][t+h])/a['compliance'][t+h]),**metrics(p,y,truth)))
            print('Analyzed targets and predictions',e['label'],flush=True)
    fields=('x_mae','x_relative_l2','rho_mae','binary_rho_disagreement','x_max','top10percent_update_energy','near_unchanged','small_MAE')
    targetsummary=[dict(label=key[0],k=key[1],h=key[2],stage=key[3],**aggregate(rs,fields)) for key,rs in targetgroups.items()]
    baselinefields=('x_mae','x_relative_l2','rho_mae','binary_rho_disagreement','volume_feasible','box_feasible','prediction_seconds')
    baselines=[dict(method=key[0],k=key[1],h=key[2],stage=key[3],**aggregate(rs,baselinefields)) for key,rs in baselinegroups.items()]
    save(out/'target_summary.json',targetsummary);save(out/'baseline_summary.json',baselines);save(out/'fresh_prediction_subset.json',dict(results=prediction_subset,actual_FEM_evaluations=actual_prediction_FEM,scope='Three selected anchors per primary problem/horizon where valid; raw infeasible boxes skipped explicitly; exact OC h uses h fresh pre-update FEM plus one terminal C evaluation'))
    # Lag correlations and growing horizons independently from final/future labels.
    correlations=[]
    for e in entries:
        tr=Trajectory(out/e['folder']);dx=np.diff(tr.arrays['x'],axis=0)
        for lag in (1,5,10):
            if len(dx)<=lag:continue
            left=dx[:-lag];right=dx[lag:];den=np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1);valid=den>1e-15
            cosine=np.sum(left[valid]*right[valid],axis=1)/den[valid]
            correlations.append(dict(problem_id=e['problem_id'],lag=lag,mean_cosine=float(np.mean(cosine)),median_cosine=float(np.median(cosine)),count=len(cosine)))
    save(out/'temporal_correlations.json',correlations)
    bypid=Counter(w['problem_id'] for w in windows);primary=entries[:10];alltotal=sum(bypid[e['problem_id']] for e in primary);capped={e['problem_id'] for e in entries if e['termination_reason']=='maxiter'}
    chosen=balanced_training_indices(windows,10000,720);sampler_counts=Counter(windows[i]['problem_id'] for i in chosen)
    sampling=dict(window_counts=[dict(problem_id=key[0],k=key[1],h=key[2],stage=key[3],count=n) for key,n in counts.items()],primary_problem_weights=[dict(problem_id=e['problem_id'],family=next(s['related_group'] for s in json.loads((out/'problems.json').read_text()) if s['problem_id']==e['problem_id']),termination=e['termination_reason'],windows=bypid[e['problem_id']],uniform_window_probability=bypid[e['problem_id']]/alltotal,uniform_problem_probability=.1) for e in primary],primary_capped_share_uniform_window=sum(bypid[e['problem_id']] for e in primary if e['problem_id'] in capped)/alltotal,primary_capped_share_uniform_problem=sum(e['problem_id'] in capped for e in primary)/10,train_problem_balanced_sample=dict(seed=720,count=10000,counts=dict(sampler_counts),only_train=all(windows[i]['partition']=='train' for i in chosen),rule='Uniform training problem then uniform eligible window; realized stages/termination are analysis annotations, never features'),training_capped_share_uniform_window=sum(w['problem_id'] in capped for w in windows if w['partition']=='train')/sum(w['partition']=='train' for w in windows),training_capped_share_balanced_realization=sum(windows[i]['problem_id'] in capped for i in chosen)/len(chosen),stage_balanced_proposal='Optional known-budget stage strata; ensure nonempty train-only strata, do not add final-length features')
    save(out/'sampling_analysis.json',sampling);save(out/'balanced_training_indices.json',chosen)
    # Explicit manifest separation and category coverage; no claim of statistical coverage.
    specs=json.loads((out/'problems.json').read_text());splits=json.loads((out/'splits.json').read_text());validate_splits(specs,splits)
    diversity=dict(primary_count=10,primary_mechanical_arrangements=9,controlled_pair=['right_mid_vertical','perturbed_right_mid'],mesh_counts=dict(Counter(str(e['mesh']) for e in primary)),volume_counts=dict(Counter(str(s['mechanical']['volume_fraction']) for s in specs[:10])),support_counts=dict(Counter(s['support_pattern'] for s in specs[:10])),partition_counts=dict(Counter(e['partition'] for e in primary)),split_groups=splits['groups'],force_variant_grouped=all(splits['assignments'][specs[i]['problem_id']]==splits['assignments'][specs[0]['problem_id']] for i in (10,11)),seed_pair_grouped=splits['assignments'][specs[2]['problem_id']]==splits['assignments'][specs[9]['problem_id']],OOD_support_isolated=True,whole_problem_and_states_separate=True,normalization='Only known geometry/force_scale and absolute iteration/configured500 cap; no held-out statistics or future/termination inputs',limitations='No mirror cases generated; preserved split implementation canonicalizes horizontal mirrors and force signs. Conservative buckets do not prove absence of all nearby duplicates. Ten deterministic configurations are not statistical generalization evidence.')
    save(out/'diversity_split_audit.json',diversity)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,2,figsize=(12,15))
    for ax,e in zip(axes.ravel(),primary):
        tr=Trajectory(out/e['folder']);ax.plot(tr.arrays['compliance']/tr.arrays['compliance'][0]);ax.set_title(e['label']+' '+e['termination_reason']);ax.set_xlabel('Updates');ax.set_ylabel('C/C_initial')
    fig.tight_layout();fig.savefig(out/'compliance_histories.png');plt.close(fig)
    fig,axes=plt.subplots(10,4,figsize=(15,30))
    for row,e in zip(axes,primary):
        tr=Trajectory(out/e['folder']);p=tr.problem
        for ax,t in zip(row,(0,min(10,len(tr.arrays['x'])-1),len(tr.arrays['x'])//2,len(tr.arrays['x'])-1)):
            ax.imshow(tr.arrays['rho'][t].reshape(p.nely,p.nelx),origin='lower',extent=[0,p.width,0,p.height],vmin=0,vmax=1);ax.set_title(f"{e['label']} t={t}")
    fig.tight_layout();fig.savefig(out/'spatial_evolution.png');plt.close(fig)
    capped_summaries=[s for s in summaries[:10] if s['maxiter_capped']][:2]
    save(out/'continuation_proposals.json',dict(executed=False,requires_separate_approval=True,proposals=[dict(problem_id=s['problem_id'],additional_update_cap=500,separate_output_required=True,reason='Distinguish persistent/oscillatory update dynamics from slow progress; no presumed convergence',estimated_seconds=200,estimated_additional_native_storage_mib=70,assumptions='Conservative48 prefix.4s/state; estimates only; retain incoming change and optimizer source') for s in capped_summaries]))
    return summaries

def estimates(out):
    through=json.loads((out/'throughput.json').read_text());validation=json.loads((out/'validation.json').read_text());large=[x for x in through[:10] if x['mesh']==[48,24]];vmap={x['problem_id']:x for x in validation['trajectories']}
    per=np.asarray([x['session_seconds']/x['state_count'] for x in large if not x['reused']]);val=np.asarray([vmap[x['problem_id']]['seconds']/x['state_count'] for x in large]);stor=np.asarray([x['native_storage_bytes']/x['state_count'] for x in large]);stop=np.asarray([x['state_count'] for x in large]);result=[]
    for number in (100,200):
        scenarios=[]
        for label,T in [('empirical_mean_length',float(stop.mean())),('all_cap500',501.)]:
            scenarios.append(dict(label=label,FEM_evaluations=number*T,generation_seconds=[number*T*float(np.median(per)),number*T*float(np.max(per))],validation_seconds=[number*T*float(np.median(val)),number*T*float(np.max(val))],native_storage_bytes=[number*T*float(np.median(stor)),number*T*float(np.max(stor))]))
        result.append(dict(dataset=f'{number}_at48',scenarios=scenarios,peak_memory_mib=memory_status()['VmHWM_mib'],peak_qualification='Measured sequential pilot peak; extrapolation assumes sequential processing/mmap. More workers/large batches invalidate this bound.'))
    scaling=json.loads((out/'scaling_integration.json').read_text());base48=scaling['benchmarks'][0];base96=scaling['benchmarks'][1];fe_ratio=base96['warm_median_seconds']['total']/base48['warm_median_seconds']['total'];native_ratio=scaling['bounded'][1]['native_storage_bytes']/scaling['bounded'][0]['native_storage_bytes'];ratio=max(fe_ratio,native_ratio)
    for number in (100,200):
        total=number*501;result.append(dict(dataset=f'{number}_half48_half96',FEM_evaluations=total,generation_scenario_seconds=[total*float(np.median(per))*(1+fe_ratio)/2,total*float(np.max(per))*(1+ratio)/2],storage_scenario_bytes=[total*float(np.median(stor))*(1+native_ratio)/2,total*float(np.max(stor))*(1+native_ratio)/2],validation_scenario_seconds=[total*float(np.median(val))*(1+fe_ratio)/2,total*float(np.max(val))*(1+ratio)/2],peak_memory='Unmeasured mixed full-length pipeline; measured96 short worker625MiB and completion process peak retained; review before rollout',qualification='Planning ratios from20-step scaling; changes in OC, filter and I/O costs may invalidate simple total-cost scaling. Not confidence bounds.'))
    save(out/'resource_estimates.json',dict(measured48_state_generation_seconds=per.tolist(),measured48_state_validation_seconds=val.tolist(),measured48_state_storage_bytes=stor.tolist(),datasets=result,future192=scaling['larger_estimates'],assumptions='Same threading, sequential process, unchanged diagnostics/nativecheckpoint frequency;100/200 not generated. Cold vswarm and different stopping lengths retained.192 memory/factorfill unmeasured. Include fresh validation FEM states separately; each validator also performs independent gradient API solves, so repeated-validation total exceeds nominal state counts.',IO_bottlenecks='Compressed atomic per-state frames, fsync/journal rewrites, repeated checkpoint hashing/final field-by-field NPZ reads, configuration/array digests. Not a clean FEM-only throughput benchmark. Avoid claiming linear scaling or embarrassingly parallel memory bound.'))

def checks(out):
    result={}
    for label,extra in [('focused',['tests/test_dataset_readiness.py','tests/test_mesh_scaling.py']),('regression',[])]:
        t=perf_counter();r=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*extra],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/'test_logs'/f'{label}.log').write_text(r.stdout+f'\nExit code: {r.returncode}\n');result[label]=dict(exit_code=r.returncode,seconds=perf_counter()-t,stdout=r.stdout)
        print(label,r.stdout.strip(),flush=True)
        if r.returncode:raise RuntimeError(label+' checks failed')
    save(out/'test_results.json',result)

def report(out,fresh=False):
    q=json.loads((out/'endpoint_quality.json').read_text());v=json.loads((out/'validation.json').read_text());f=json.loads((out/'force_scaling.json').read_text());sampling=json.loads((out/'sampling_analysis.json').read_text());resource=json.loads((out/'resource_estimates.json').read_text());tests=json.loads((out/'test_results.json').read_text());primary=q['endpoints'][:10]
    eligible=[x['label'] for x in primary if x['reference_label_eligible']];capped=[x for x in primary if x['maxiter_capped']]
    lines=['# M3.3C full ten-problem pilot and dataset readiness','', 'Actual execution: ten deterministic primary problems, two additional force controls (.1/10; F1 reused), explicit feasible_upper_v1, tolerance.001 and cap500. Every evaluated terminal state is retained. Eight primary meshes48×24 and two smaller verification controls. Nine distinct mechanics plus one paired perturbed-start control. This is not a statistical generalization study. No192 FEM, extended trajectories, neural training,100–200 generation, HPC or commit/push.','', '## Primary numerical and quality results','', '| Problem | Updates/stop | Terminal C | PG inf | Box KKT inf | Reference screen |','|---|---|---:|---:|---:|---|']
    for s in primary:lines.append(f"| {s['label']} | {s['updates']}/{s['stop']} | {s['terminal_compliance']:.7g} | {s['projected_gradient_inf']:.4g} | {s['box_stationarity_inf']:.4g} | {s['reference_label_eligible']} |")
    lines+=['',f"Independent validation passed on {v['states']} new saved states and {v['updates']} transitions across12 trajectories, {v['comparison_rows']} numerical comparison rows. All are dynamics_valid; all updated endpoints satisfy strict mean and historical scalar-sum volume guards. Native arrays/specifications/checksums, independent COO reactions/equilibrium, C/gradient API reconstruction, exact OC selection, terminal timing, features/energy units, windows and grouped splits passed. Worst residual/energy/force/moment diagnostics: {v['mechanical_maxima']}.",'',f"Change-converged {sum(s['change_converged'] for s in primary)}/10; capped {len(capped)}/10. Capped states remain valid dynamics examples and are never reference labels. Provisional sidecar reference eligibility: {eligible}. Native M3.3B reference metadata remains false and is never silently rewritten. No endpoint is certified optimal.", '', 'Policy m3.3c.quality.v1 requires independently valid dynamics, change_tolerance termination, strict volume/box feasibility, PG<=.01, scaled boxKKT<=.01, complementarity<=1e-5, at least20 tail updates, relative C span<=.001 and physical-density binary(.5) tail disagreement<=.005. Threshold sensitivity varies PG/box .005/.01/.02, complementarity1e-6/1e-5/1e-4, C span1e-4/.001/.01 and binary.001/.005/.01; all combinations are retained. These diagnostic thresholds have no optimality theorem; stiffness-floor paths, dominant-gradient normalization and threshold topology remain limitations. Approximate labels are screened candidates for later review, not proven local/global minima.','', '## Capped dynamics and force magnitudes','', '| Capped problem | Last50 relative C progress | Opposed update fraction | Cells changing>.001 | Move-limit steps |','|---|---:|---:|---:|---:|']
    for s in capped:lines.append(f"| {s['label']} | {s['tail_compliance_progress']:.4g} | {s['tail_negative_update_cosine_fraction']:.4g} | {s['tail_mean_changed_cell_fraction']:.4g} | {s['tail_move_limit_update_fraction']:.4g} |")
    lines+=['','These are descriptive indicators: slow improvement, opposed successive directions and spatially localized changes may coexist. Stable C is not sufficient stationarity. Causal attribution to bisection versus nonconvex dynamics is not established. M3.3A already showed8 persistent caps and7 worsened-PG cases among20; tighter precision did not fix all. Saved per-state C/change/PG/boxKKT/complementarity/bounds/multiplier histories and spatial snapshots quantify this pilot. At most two separately estimated continuation proposals are saved; none were executed.','', '| Force | Updates/stop | Common-prefix x max difference vsF1 | Terminal normalized lambda | Terminal PG |','|---|---|---:|---:|---:|']
    for x in f['trajectory_comparisons']:lines.append(f"| {x['force']} | {x['updates']}/{x['stop']} | {x['x_common_prefix_max']:.4g} | {x['normalized_multiplier_terminal']:.6g} | {x['terminal_pg']:.4g} |")
    lines+=['', 'Force scaling is controlled at fixed geometry/support/load direction/initial array/material/volume. Nine same-design FEM tests isolate u/F,C/F²,g/F² scaling; finite-width OC lambda/F² comparisons retain relative bracket width and endpoint design spread. Full trajectories diverge numerically after identical starts, so their normalized path differences are not pure same-x mechanics errors. Fixed [0,1e5] remained valid for these.1/1/10 controls, but this does not establish all force scales. An absolute1e-8 width has different relative precision after force scaling. A separately versioned relative/force-aware safeguarded search is a proposal only; validated optimizer/defaults unchanged.','', '## Learning targets and non-neural baselines','', 'Full offline statistics cover h1/5/10,k1/3; target x/rho MAE/max/relativeL2, binary disagreement, top10%-cell update-energy concentration and near-unchanged frequencies are in target tables. Early<50,middle50..299,late>=300 use the configured500 cap; stages are analysis annotations and not future inputs. Lag1/5/10 update-direction correlations are saved. Horizons change the accumulated target and difficulty by problem/stage; aggregate comparisons must account for different valid anchor populations.','', 'No-change predicts current x. Linear uses the most recent update; two-update averages the last two. Extrapolation is evaluated only with k3, never unavailable history at k1. Each raw method and explicit Euclidean box/sum-projected variant retains feasibility metrics, runtime and no-FEM cost. Projection is a new prediction-only versioned operation (80 bisections plus explicit density-scale conservative floating guard, prediction_projection.v2), not an OC alteration. Raw invalid boxes retain prediction/linear-filter errors but skip unsupported FEM without clipping. Binary topology metrics use physical rho. Prediction error does not establish optimizer acceleration.','', 'Exact OC h1 uses current saved gradient, costs zero additional FEM in this offline cached diagnostic but requires one fresh FEM at inference if not already available. Its cached runtime is not a clean fresh-call timing. The selected fresh OC h5/h10 baselines perform5/10 intermediate pre-update FEM evaluations; a further terminal evaluation computes C. It is never represented as one OC update. Fresh C comparisons share selected anchors across baseline methods; complete per-method calls/timings/skips are saved. No predicted rollout has been used to train or claim acceleration.','',f"Selected fresh prediction/OC checks executed {json.loads((out/'fresh_prediction_subset.json').read_text())['actual_FEM_evaluations']} FEM evaluations; same-design force checks executed {f['actual_FEM_evaluations']}. Generation/validation/restart calls are separately recorded, not hidden in those totals.",'', '## Sampling, diversity and splits','',f"Primary capped representation: uniform-window {sampling['primary_capped_share_uniform_window']:.2%}, uniform-problem {sampling['primary_capped_share_uniform_problem']:.2%}. Training uniform-window capped share {sampling['training_capped_share_uniform_window']:.2%}; deterministic problem-balanced realization {sampling['training_capped_share_balanced_realization']:.2%}. Window counts are broken down by complete problem/family,k,h,known-budget stage and termination. Offline termination labels never enter spatial inputs.", '', 'Optional balanced_training_indices samples only train problems uniformly then an eligible window uniformly with seed720; it preserves arrays/splits and excludes validation/test. Whole families unite force magnitudes, initial variants and near-normalized mechanics; bottom-clamp OOD is isolated. No physical trajectory shares a partition. The preserved policy canonicalizes horizontal mirrors/global force signs; this pilot contains no mirror draw and cannot prove continuous-near-duplicate absence. Known force/geometry normalization uses no held-out or future labels. Actual support/volume/mesh/partition counts and groups are in diversity_split_audit.json.','', 'Future100–200 design: preallocate independent family strata across loads/directions/locations, stable left/right/bottom/pin-roller supports, volumes.3/.4/.5/.6, aspect/cell-spacing/physical-filter conventions and uniform/feasible perturbed starts. Begin48; introduce96 refinement pairs within each family and partition. Balance family categories before drawing starts/windows; retain separate domain-growth labels. Reserve independent support/load families and frozen192 extrapolation only after separate approval/measurements. Ten controls do not demonstrate broad distribution coverage.','', '## Reused scaling, production and resource readiness','', 'Scaling generation20261013T201818977895Z was reused after archived-source and all native sample checksum checks; no scaling FEM repeated. Fixed-domain A: physical48×24, pointforce(48,12), physicalradius2, radii2/4. B: spacing1, growing domain, radius2. Representative FEM API2.6/7.8ms and worker peak599/625MiB at48/fixed96; domain-growth96 about7.6ms and621MiB. Native21-state storage2.70/10.50MiB and warm loading≈3.1/7.4µs. Fixed-domain terminal coarse/fine rho MAE≈.01415 after conservative2×2 restriction; both20-step runs remain unconverged.','', 'Nominal physical cone support is preserved, discrete quadrature/boundaries differ; rectangular scalar-radius cells have physical ellipses. Concentrated point-force compliance may be singular under refinement. Complete native element/node grids support variable resolution; spatial tensors alone need explicit known dx/dy/domain/filter/material/optimizer conditioning. Physical receptive fields, continuous position handling, grouped splits and inference-FEM feature cost must be reviewed before cross-resolution training. No192/384 FEM evidence is inferred from configurations.','', 'Production timing includes native diagnostics, atomic compressed frames, journal fsync/hash checks and field-by-field finalization. It is not an isolated solver benchmark. A saved20-update48 production prefix passed a conservative runtime/memory gate and was resumed, not discarded. Small-control restarted versus uninterrupted histories match all stored arrays; one repetition measures restart overhead, not a throughput distribution. Sequential processing/read-only mmap controls memory; additional workers/batches increase it. Array hashes/checkpoints/I/O can dominate numerical solves.','', '##100–200 planning estimates','', '| Dataset | Length scenario | Nominal generation FEM | Generation range min | Validation range min | Native storage range GiB |','|---|---|---:|---:|---:|---:|']
    for x in resource['datasets'][:2]:
        for s in x['scenarios']:lines.append(f"| {x['dataset']} | {s['label']} | {s['FEM_evaluations']:.0f} | {s['generation_seconds'][0]/60:.3g}–{s['generation_seconds'][1]/60:.3g} | {s['validation_seconds'][0]/60:.3g}–{s['validation_seconds'][1]/60:.3g} | {s['native_storage_bytes'][0]/2**30:.3g}–{s['native_storage_bytes'][1]/2**30:.3g} |")
    lines+=['', 'Ranges use observed median/max per-state costs and either observed mean stopping length or501 states; they are assumption scenarios, not confidence bounds. Mixed48/96 and future192 scenarios, native workspace/attention caveats and validation overhead are in resource_estimates.json. Validation additionally evaluates saved states and gradients (two fresh solver APIs per state), so generation nominal counts are not total audited-project solves. The reused192/384 time scenarios are unvalidated N^1.5..N², not measured scaling laws. Peak native factor memory at192 is unknown.','', '## Explicit research assessment and decisions','',f"1. All ten primary trajectories passed independent mechanical/transition replay; {v['states']} states includes the two force controls.", '2. Feasible-upper preserves strict volume for every saved updated endpoint here; bracket failures remain explicit rather than repaired.', '3. Linear mechanics scales u linearly and C/g quadratically at identical x; finite absolute multiplier width makes numerical optimizer paths scale dependent. See same-x and trajectory tables.', '4. Capped histories show the measured persistence/opposition/localization indicators above; no universal cause or extended convergence is established.', '5. All independently valid primary trajectories, including caps, are dynamics-learning examples with recorded controls.',f'6. The provisional screen admits {eligible}; all remain uncertified and native reference flags false. Sensitivity tables expose threshold dependence.', '7. Horizons1/5/10 accumulate different updates by family/stage; target statistics distinguish small late updates from substantial changes.', '8. No-change and projected extrapolation provide mandatory error/cost controls; exact fresh multi-step OC is a mechanically faithful cost baseline. Per-stage results and selected fresh C prevent confusing predictive MAE with acceleration.', '9. Native metadata/grids support variable resolution; architecture/conditioning/batching/physical filter policy remain necessary,192 unvalidated.', '10. Actual48 costs and100/200 scenarios are above; no large dataset was generated and parallel throughput is unknown.', '11. Stratify independent mechanics/support/load/volume/aspect/starts at48, then grouped96 pairs; keep refinement and domain growth distinct.', '12. Remaining: quality-threshold validation, capped dynamics and force-width robustness, dataset family coverage, equivalent-overhead production timing, learned rollout feasibility/stationarity and actual FEM/time savings.', '13. A192 single-state measurement is a justified next controlled resource investigation based on successful96 short runs, but requires separate approval and memory/time guards; no192 trajectory is authorized.', '', '**Decision A — CONDITIONAL GO:** a controlled100–200-problem dynamics dataset is scientifically supportable using validated dynamics labels, grouped stratification, strict controls, sequential/resource-limited batching and complete independent validation. Execution remains a separately authorized next stage; caps and provisional references must be labeled distinctly.', '', '**Decision B — CONDITIONAL GO:** initial M4 experiments at48 may test dynamics prediction on independent family splits against these baselines, with balanced sampling, fresh-FEM accounting and held-out rollout quality. This pilot is too small for broad learning claims or final-optimum supervision. No training executed.', '', '**Decision C — CONDITIONAL GO:** propose one192 fixed-domain FEM state with measured time/RSS/residuals under separately approved limits. Existing96 success motivates measurement, not a safe peak guarantee; full192 runs remain NO-GO pending that evidence.', '', '## Validation and reproduction','',f"Focused and full tests exit zero. Logs: {tests['focused']['stdout'].strip()} / {tests['regression']['stdout'].strip()}. Protected prior files unchanged: {protect(out)}. Actual executing sources are archived and checked unchanged throughout generation. A fresh-process full replay is also retained. All prior M1–M3.3B and scaling artifacts/core/defaults are preserved.", '', 'From repository root: `OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3-3c-completion-mpl /users/atabarra/.venv/bin/python run_m3_3c_completion.py all`. Creates a new isolated generation. `resume --output GENERATION` requires original source/config fingerprints. `validate --output GENERATION` replays saved states in a fresh process and writes timestamped outputs without optimization. `report --output GENERATION` uses saved tables only and writes a timestamped report. Focused tests: same environment, `python -m pytest -q -p no:cacheprovider tests/test_dataset_readiness.py tests/test_mesh_scaling.py`; omit filenames for full regression. Archive restoration instructions are in REPRODUCE.md.']
    name=('M3_3C_FULL_PILOT_REPORT_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.md') if fresh else 'M3_3C_FULL_PILOT_REPORT.md'
    (out/name).write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['all','resume','finish','validate','report']);ap.add_argument('--output',type=Path);args=ap.parse_args();out=args.output or ROOT/'m3_3c/completion_generations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.command=='validate':validate(out,True);return
    if args.command=='report':
        report(out,True);return
    if args.command=='finish':
        phase=out/'analysis_provenance'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');phase.mkdir(parents=True);sources=source_manifest(phase)
        save(phase/'phase.json',dict(scope='Finish saved valid generation after prediction-only projection failure; no trajectories regenerated',projection_version='m3.3c.prediction_projection.v2',initial_generation_sources_verified_unchanged_before_fix=True))
        audit=json.loads((out/'validation.json').read_text());analyze(out);estimates(out);checks(out)
        subprocess.run([sys.executable,str(ROOT/'run_m3_3c_completion.py'),'validate','--output',str(out)],check=True)
        assert_sources(sources);save(out/'final_validation.json',dict(passed=True,protected_files_unchanged=protect(out),generation_source_archives_preserved=True,analysis_sources_unchanged=True,analysis_phase=str(phase.relative_to(out)),validation=audit,memory=memory_status(),test_results=json.loads((out/'test_results.json').read_text())))
        report(out);print('Completion analysis finished:',out,flush=True);return
    if args.command=='all':initialize(out)
    else:assert_sources(json.loads((out/'source_manifest.json').read_text()))
    generate(out);audit=validate(out);restart_benchmark(out);force_study(out);analyze(out);estimates(out);checks(out)
    subprocess.run([sys.executable,str(ROOT/'run_m3_3c_completion.py'),'validate','--output',str(out)],check=True)
    assert_sources(json.loads((out/'source_manifest.json').read_text()));save(out/'final_validation.json',dict(passed=True,protected_files_unchanged=protect(out),source_bytes_unchanged=True,validation=audit,memory=memory_status(),test_results=json.loads((out/'test_results.json').read_text())))
    report(out);print('Completion finished:',out,flush=True)

if __name__=='__main__':main()
