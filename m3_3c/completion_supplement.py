"""Supplement completion evidence using saved states; no optimizer trajectories."""
import json,csv,gzip,sys,shutil,hashlib
from pathlib import Path
from collections import defaultdict,Counter
from time import perf_counter
import numpy as np
from topopt_utils.fea.trajectory_dataset import Trajectory,sample_problem,split_problems,file_hash
from topopt_utils.fea.robust_oc import trial
from topopt_utils.fea.mechanical_problem import DensityMap
from topopt_utils.fea.solver import FeaSolver
from run_m3_3c_completion import save,protect,configuration
from run_m3_3c import memory_status

def main(out):
    archive=out/'supplemental_provenance';archive.mkdir(exist_ok=False)
    for file in (Path(__file__),Path(__file__).with_name('completion_replay_snapshot.py')):shutil.copyfile(file,archive/(file.name+'.txt'))
    save(archive/'source_hashes.json',{f.name:file_hash(f) for f in (Path(__file__),Path(__file__).with_name('completion_replay_snapshot.py'))})
    manifest=json.loads((out/'manifest.json').read_text());entries=manifest['samples'];primary=entries[:10];ids={e['problem_id'] for e in primary};sameanchors=defaultdict(list);fd=[];precision=[];actual_fd_solves=0
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    caps=[e for e in primary if e['termination_reason']=='maxiter'];fig,axes=plt.subplots(len(caps),3,figsize=(15,4*len(caps)))
    for e in primary:
        tr=Trajectory(out/e['folder']);p=tr.problem;a=tr.arrays;T=len(a['x'])
        # Matched anchor population at all three horizons; never supplies future data as input.
        for t in range(T-10):
            for h in (1,5,10):
                dx=a['x'][t+h]-a['x'][t];dr=a['rho'][t+h]-a['rho'][t];phase='early' if t<50 else 'middle' if t<300 else 'late'
                sameanchors[(phase,h)].append(dict(mae=float(abs(dx).mean()),rho_mae=float(abs(dr).mean()),max=float(abs(dx).max()),relative_l2=float(np.linalg.norm(dx)/np.linalg.norm(a['x'][t+h])),near=bool(abs(dx).max()<.001)))
        # Objective derivatives, not volume-feasible optimizer directions: both spaces stay in box.
        solver=FeaSolver();mapping=DensityMap(p);d=.2+.8*np.random.default_rng(811).random(p.n);d/=max(d)
        for space,key in (('x','g_x'),('rho','g_rho')):
            z=np.asarray(a[space][0]);exact=float(a[key][0]@d)
            for step in (1e-3,3e-4):
                plus=mapping.forward(z+step*d) if space=='x' else z+step*d;minus=mapping.forward(z-step*d) if space=='x' else z-step*d
                tick=perf_counter();cp=solver.compliance(plus,*p.solver_args);cm=solver.compliance(minus,*p.solver_args);actual_fd_solves+=2
                approximate=(cp-cm)/(2*step);error=abs(approximate-exact)/max(abs(exact),1e-12)
                assert error<5e-5,(e['label'],space,step,error)
                fd.append(dict(problem_id=e['problem_id'],label=e['label'],space=space,step=step,relative_error=error,exact=exact,numerical=approximate,seconds=perf_counter()-tick))
        if e in caps:
            j=caps.index(e);tail=np.asarray(a['x'][-51:]);changes=np.abs(np.diff(tail,axis=0));axes[j,0].imshow(changes.max(axis=0).reshape(p.nely,p.nelx),origin='lower',extent=[0,p.width,0,p.height]);axes[j,0].set_title(e['label']+' max tail change')
            axes[j,1].plot(a['previous_change'][-51:],label='max change');axes[j,1].plot(a['projected_gradient_inf'][-51:],label='PG');axes[j,1].legend();axes[j,2].plot(a['compliance'][-51:]);axes[j,2].set_title('Last50 compliance')
            for t in range(T-11,T-1):
                x=a['x'][t];g=a['g_x'][t];diag=tr.transitions[t]['diagnostic'];low=trial(x,g,diag['multiplier_lower']);high=trial(x,g,diag['multiplier_upper']);spread=float(abs(low-high).max());actual=float(a['update_change'][t])
                precision.append(dict(label=e['label'],iteration=t,bracket_width=diag['bracket_width'],relative_bracket_width=diag['bracket_width']/max(diag['multiplier_upper'],1e-30),fixed_gradient_endpoint_design_spread=spread,actual_update_max=actual,spread_to_actual=spread/max(actual,1e-30),changed_cell_fraction=float(np.mean(abs(a['x'][t+1]-x)>.001))))
    fig.tight_layout();fig.savefig(out/'capped_localization.png');plt.close(fig)
    # Numerical baseline strength on all primary k3 windows and separately held-out partitions.
    groups=defaultdict(list)
    with gzip.open(out/'prediction_baselines.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            if r['problem_id'] in ids and int(r['k'])==3:
                for pop in ('all_primary','heldout' if r['partition']!='train' else 'train'):
                    groups[(pop,r['method'],int(r['h']),r['stage'])].append(r)
    baseline=[]
    for key,rs in groups.items():
        baseline.append(dict(population=key[0],method=key[1],h=key[2],stage=key[3],count=len(rs),x_mae_mean=float(np.mean([float(r['x_mae']) for r in rs])),x_relative_l2_mean=float(np.mean([float(r['x_relative_l2']) for r in rs])),rho_mae_mean=float(np.mean([float(r['rho_mae']) for r in rs])),box_feasible_fraction=float(np.mean([r['box_feasible']=='True' for r in rs])),volume_feasible_fraction=float(np.mean([r['volume_feasible']=='True' for r in rs]))))
    # Mirror grouping is verified with config-only synthetic counterpart, never FEM.
    c=configuration();specs=json.loads((out/'problems.json').read_text());mirror=json.loads(json.dumps(c['problems'][0]));mirror.update(label='mirror_grouping_probe',support='right_clamp',related_group='separate_declared_mirror_probe');mirror['loads'][0].update(position=[0.,3.],direction=[-1.,0.]);ms=sample_problem(mirror,990,c['controls'])[2]
    splits=split_problems(specs+[ms],577,['bottom_clamp']);assert splits['assignments'][ms['problem_id']]==splits['assignments'][specs[0]['problem_id']]
    # Account for preserved production prefix, which is separate from resumed session timing.
    through=json.loads((out/'throughput.json').read_text());gate=json.loads((out/'resource_gate.json').read_text());validation=json.loads((out/'validation.json').read_text());byid={x['problem_id']:x for x in validation['trajectories']};large=[x for x in through[:10] if x['mesh']==[48,24]]
    costs=[(x['session_seconds']+(gate['prefix_elapsed_seconds'] if x['problem_id']==gate['prefix_problem'] else 0))/x['state_count'] for x in large]
    lengths=[x['state_count'] for x in large];storage=[x['native_storage_bytes']/x['state_count'] for x in large];validate_cost=[byid[x['problem_id']]['seconds']/x['state_count'] for x in large]
    resources=[]
    for n in (100,200):
        for T,label in ((float(np.mean(lengths)),'observed_mean_length'),(501.,'all500caps')):
            resources.append(dict(dataset=f'{n}_at48',length=label,generation_FEM=n*T,one_full_validation_FEM=2*n*T,generation_minutes=[n*T*np.median(costs)/60,n*T*max(costs)/60],validation_minutes=[n*T*np.median(validate_cost)/60,n*T*max(validate_cost)/60],storage_GiB=[n*T*np.median(storage)/2**30,n*T*max(storage)/2**30]))
    scaling=json.loads((out/'scaling_integration.json').read_text());b48,b96=scaling['benchmarks'][:2];fr=b96['warm_median_seconds']['total']/b48['warm_median_seconds']['total'];sr=scaling['bounded'][1]['native_storage_bytes']/scaling['bounded'][0]['native_storage_bytes']
    for n in (100,200):
        resources.append(dict(dataset=f'{n}_half48_half96',length='all500caps',generation_FEM=n*501,one_full_validation_FEM=2*n*501,generation_minutes=[n*501*np.median(costs)*(1+fr)/2/60,n*501*max(costs)*(1+max(fr,sr))/2/60],validation_minutes=[n*501*np.median(validate_cost)*(1+fr)/2/60,n*501*max(validate_cost)*(1+max(fr,sr))/2/60],storage_GiB=[n*501*np.median(storage)*(1+sr)/2/2**30,n*501*max(storage)*(1+sr)/2/2**30],qualification='48 measured total costs multiplied by short96 FEM/storage ratios; not measured mixed full-history production throughput'))
    # Future192-containing population: extrapolated only, using prior unvalidated scenarios.
    future=next(x for x in scaling['larger_estimates'] if x['experiment']=='A' and x['mesh']==[192,96]);T=501.;n48,n96,n192=80,15,5
    resources.append(dict(dataset='100:80_at48_15_at96_5_at192',length='all500caps',generation_FEM=100*501,one_full_validation_FEM=2*100*501,generation_minutes=[(T*np.median(costs)*(n48+n96*fr)+n192*future['generation_501state_scenario_seconds'][0])/60,(T*max(costs)*(n48+n96*max(fr,sr))+n192*future['generation_501state_scenario_seconds'][1])/60],validation_minutes=[(T*np.median(validate_cost)*(n48+n96*fr)+n192*future['validation_501state_scenario_seconds'][0])/60,(T*max(validate_cost)*(n48+n96*max(fr,sr))+n192*future['validation_501state_scenario_seconds'][1])/60],storage_GiB=[(T*np.median(storage)*(n48+n96*sr)+n192*future['trajectory_500updates_storage_bytes'])/2**30,(T*max(storage)*(n48+n96*sr)+n192*future['trajectory_500updates_storage_bytes'])/2**30],peak_memory='192 factor/workspace peak unmeasured; no safe bound claimed',qualification='Unvalidated planning example only;192 has never undergone FEM in these stages; no execution authorization implied'))
    targets=[dict(stage=k[0],h=k[1],count=len(rs),x_mae_mean=float(np.mean([x['mae'] for x in rs])),rho_mae_mean=float(np.mean([x['rho_mae'] for x in rs])),x_max_mean=float(np.mean([x['max'] for x in rs])),near_unchanged_fraction=float(np.mean([x['near'] for x in rs]))) for k,rs in sameanchors.items()]
    q=json.loads((out/'quality_threshold_sensitivity.json').read_text());elig=[x['count'] for x in q]
    fresh=list(out.glob('fresh_validation_*.json'));prediction=json.loads((out/'fresh_prediction_subset.json').read_text());restart=json.loads((out/'restart_benchmark.json').read_text());force=json.loads((out/'force_scaling.json').read_text());S=validation['states']
    evidence=dict(objective_FD_checks=fd,objective_FD_FEM_evaluations=actual_fd_solves,capped_fixed_gradient_precision_probes=precision,matched_anchor_horizon_statistics=targets,primary_baselines=baseline,mirror_grouping_config_only_passed=True,quality_sensitivity_reference_count_range=[min(elig),max(elig)],prefix_inclusive_resource_estimates=resources,primary_plus_force_generation_seconds=sum(x['session_seconds'] for x in through)+gate['prefix_elapsed_seconds'],source_phase_note='Native generation/force/restart and first validation used original sources. Corrected prediction projection and safe output guard used separately archived analysis sources. No trajectories regenerated or repaired.',FEM_ledger=dict(generation=S,full_replays=len(fresh)+1,FEM_per_full_replay=2*S,restart_extra=restart['extra_benchmark_committed_FEM_evaluations'],force_common_design=force['actual_FEM_evaluations'],selected_prediction_and_OC=prediction['actual_FEM_evaluations'],objective_FD=actual_fd_solves,total_completed_experiment_FEM_excluding_regression_fixtures=S+2*S*(len(fresh)+1)+restart['extra_benchmark_committed_FEM_evaluations']+force['actual_FEM_evaluations']+prediction['actual_FEM_evaluations']+actual_fd_solves),protected_files_unchanged=protect(out),memory=memory_status(),resource_assumptions='Sequential same-threading production, measured median/max not confidence limits, prefix included, storage includes per-problem overhead, validation includes2 fresh solver calls/state.192 extrapolation inherited unvalidated costs/peak; no full dataset measured.')
    save(out/'completion_supplement.json',evidence)
    lines=['','## Completion audit: matched targets and numerical baselines','', 'Ten initial states additionally passed two centered step sizes in physical rho and design x (40 derivative checks,80 FEM calls). Directions are objective derivatives within the density box, not volume-feasible OC directions. No projection is active in these checks. Mirrors were checked via an exact config-only counterpart with a separate declared family; preserved grouping still united it with the axial case. No mirror FEM trajectory was added.','', '| Stage | Horizon | Common anchors | Mean x difference | Mean rho difference | Near unchanged(max<.001) |','|---|---:|---:|---:|---:|---:|']
    for r in sorted(targets,key=lambda x:(x['stage'],x['h'])):lines.append(f"| {r['stage']} | {r['h']} | {r['count']} | {r['x_mae_mean']:.5g} | {r['rho_mae_mean']:.5g} | {r['near_unchanged_fraction']:.2%} |")
    lines+=['','The table uses identical anchors for all horizons, eliminating unequal-endpoint population effects. Longer horizons accumulate larger average changes here, but nearly unchanged labels and direction correlations vary by stage; these are retrospective labels, not future inputs.','', '| Primary k3 baseline, h10 late | Mean x MAE | Box feasible | Volume feasible |','|---|---:|---:|---:|']
    for r in baseline:
        if r['population']=='all_primary' and r['h']==10 and r['stage']=='late':lines.append(f"| {r['method']} | {r['x_mae_mean']:.5g} | {r['box_feasible_fraction']:.2%} | {r['volume_feasible_fraction']:.2%} |")
    lines+=['','These comparisons use the same primary k3 windows. Raw extrapolation is often infeasible; projection improves errors while enforcing constraints. No-change has low average late error but can miss localized changes. Held-out-only baseline tables are separately saved. The selected fresh C checks compare actual mechanically evaluated designs and exact multi-step OC costs; they do not show learned acceleration or sustained predicted-rollout success.','',f"Across {len(precision)} last-ten-transition capped probes, maximum fixed-gradient bracket-endpoint design spread is {max(x['fixed_gradient_endpoint_design_spread'] for x in precision):.3g}; maximum spread/actual step is {max(x['spread_to_actual'] for x in precision):.3g}. This is much smaller than persistent changes on these snapshots, so one-step search width alone cannot account for the observed step magnitudes. It does not rule out accumulated sensitivity of the nonconvex path. capped_localization.png plots actual tail change maps beside C/change/PG. No extra continuation was executed.", '',f"Reference count across the provisional threshold grid: {min(elig)}–{max(elig)} of10. Stricter PG/box screens reject the small axial endpoint; threshold stability is evidence for a screening policy, not a certificate. Problem-balanced is not family-balanced: related force/start controls can upweight one physical family even while grouped splits prevent leakage. Future stratification should cap family variants or add a train-only family-first sampler.", '', '## Prefix-inclusive dataset budgets','', '| Population | Length scenario | Generation FEM | One validation FEM | Generation min | Validation min | Storage GiB |','|---|---|---:|---:|---:|---:|---:|']
    for r in resources:lines.append(f"| {r['dataset']} | {r['length']} | {r['generation_FEM']:.0f} | {r['one_full_validation_FEM']:.0f} | {r['generation_minutes'][0]:.3g}–{r['generation_minutes'][1]:.3g} | {r['validation_minutes'][0]:.3g}–{r['validation_minutes'][1]:.3g} | {r['storage_GiB'][0]:.3g}–{r['storage_GiB'][1]:.3g} |")
    lines+=['','This budget includes the separately preserved20-update production prefix; the original throughput session row for the resumed problem excludes it. Mixed and192-containing rows are planning scenarios. More workers, network storage, retained features/batches and GPU training alter memory/I/O substantially. Completion process peak includes analysis data/plots, not just solver memory; regression child peaks are not measured.192 sparse-factor memory remains unknown. No100–200 dataset or192 FEM was executed.','',f"Pilot/control generation wall total including prefix: {evidence['primary_plus_force_generation_seconds']:.3g}s. Numerical experiment FEM ledger (excluding regression temporary fixtures): {evidence['FEM_ledger']}. Original generation/validation sources remain byte-preserved; subsequent prediction projection v2, safe output-directory guard and analysis sources are separately archived. The roundoff failure and partial results remain in development_analysis, not hidden or called a FEM failure.",'', 'Reproduce this supplement only (new completion copy, to preserve the finalized report): `OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3-3c-completion-mpl PYTHONPATH=. /users/atabarra/.venv/bin/python m3_3c/completion_supplement.py GENERATION_COPY`. It evaluates80 derivative FEM calls, config-only mirror grouping, saved-state analyses and report additions; no optimization trajectories. Supplemental source bytes are archived before those derivative checks.']
    report=out/'M3_3C_FULL_PILOT_REPORT.md';report.write_text(report.read_text()+'\n'.join(lines)+'\n')
    print(json.dumps(dict(FDchecks=len(fd),FDsolves=actual_fd_solves,protected=evidence['protected_files_unchanged'],FEMledger=evidence['FEM_ledger'])))

if __name__=='__main__':main(Path(sys.argv[1]))
