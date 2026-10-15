#!/usr/bin/env python
"""M3.3A deterministic guarded OC pilot. Run from the repository root."""
import argparse,csv,hashlib,json,os,platform,shutil,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from topopt_utils.fea.oc_pilot import suite,run,load_case,replay_selected,topology
from topopt_utils.fea.robust_oc import OPTIONS,OCFailure
from topopt_utils.fea.convergence_reliability import first_stopping_update,stationarity_diagnostics
from topopt_utils.fea.consistent_trajectory import save_trajectory,load_trajectory
from topopt_utils.fea.mechanical_problem import array_digest
from topopt_utils.fea.trajectory_validation import comparison

ROOT=Path.cwd()
TOLS=(.001,.005,.01)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(path,data):
    def default(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False,default=default)+'\n')

def table(path,rows,empty_fields=None):
    fields=list(dict.fromkeys(k for r in rows for k in r)) or empty_fields or ['status']
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def protected():
    hashes=json.loads((ROOT/'m3_3/protected_before.json').read_text())
    changed=[p for p,h in hashes.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
    if changed:raise RuntimeError('Protected files modified: '+repr(changed))
    return len(hashes)


def source_snapshot(out):
    source_root=Path(__file__).resolve().parent
    paths=list(source_root.glob('*.py'))
    for name in ('topopt_utils','topo_transformer','inr_render','tests'):
        paths.extend((source_root/name).rglob('*.py'))
    manifest={}
    for path in sorted(set(paths)):
        name=path.relative_to(source_root);archive=out/'source_snapshots'/name
        if name.parts[0]=='tests':archive=archive.with_suffix('.py.txt')
        archive.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,archive)
        manifest[str(name)]=dict(sha256=digest(path),archive=str(archive.relative_to(out)),executing_import_root=str(source_root))
    dump(out/'source_manifest.json',manifest)
    return manifest


def source_check(manifest):
    source_root=Path(__file__).resolve().parent
    changed=[n for n,v in manifest.items() if digest(source_root/n)!=v['sha256']]
    if changed:raise RuntimeError('Executing source changed during generation: '+repr(changed))


def environment():
    return dict(utc=datetime.now(timezone.utc).isoformat(),python=sys.version,executable=sys.executable,platform=platform.platform(),
        versions={n:version(n) for n in ('numpy','scipy','matplotlib','torch','pypardiso','pytest')},
        threads={n:os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],text=True),
        module_paths={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items() if n.startswith('topopt_utils.fea') and getattr(m,'__file__',None)},
        cwd=str(ROOT),memory_scope='At most three current-case trajectories resident; <=1152 elements; no dense interaction matrices. Native peak RSS is not measured.')


def prefix_seconds(a,t):return float(a['evaluation_seconds'][:t+1].sum()+a['update_seconds'][:t].sum())


def differences(a,t,b,u):
    d=dict(relative_compliance_difference=float((a['compliance'][t]-b['compliance'][u])/b['compliance'][u]))
    for space in ('x','rho'):
        delta=a[space][t]-b[space][u]
        d.update({space+'_mae':float(np.mean(abs(delta))),space+'_max_difference':float(np.max(abs(delta))),space+'_relative_l2':float(np.linalg.norm(delta)/np.linalg.norm(b[space][u]))})
    return d


def plots(out,label,p,runs):
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for option,(a,m,search) in runs.items():
        t=a['iteration'];axes[0,0].plot(t,a['compliance'],label=option)
        axes[0,1].semilogy(t[1:],a['previous_change'][1:],label=option)
        axes[1,0].plot(t,a['design_volume']-p.volume_fraction,label=option)
        indices=np.unique(np.linspace(0,len(t)-1,min(45,len(t))).astype(int))
        pg=[stationarity_diagnostics(a['x'][i],a['g_x'][i],p.volume_fraction)['projected_gradient_inf'] for i in indices]
        axes[1,1].semilogy(t[indices],np.maximum(pg,1e-16),label=option)
    for ax,title in zip(axes.ravel(),('Compliance','Incoming max design change','Mean x minus target','Projected gradient infinity (sampled)')):
        ax.set_title(title);ax.set_xlabel('Completed OC updates');ax.legend(fontsize=7)
    for tol in TOLS:axes[0,1].axhline(tol,ls=':',color='gray',lw=.7)
    axes[1,0].axhline(0.,color='k',lw=.6);fig.suptitle(label);fig.tight_layout();fig.savefig(out/f'{label}_histories.png',dpi=130);fig.savefig(out/f'{label}_histories.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,len(runs),figsize=(4*len(runs),6),squeeze=False)
    for col,(option,(a,m,search)) in enumerate(runs.items()):
        for row,space in enumerate(('x','rho')):
            ax=axes[row,col];ax.imshow(a[space][-1].reshape(p.nely,p.nelx),origin='lower',extent=(0,p.width,0,p.height),vmin=0,vmax=1,cmap='gray_r')
            ax.set_title(f'{option}\n{space}, t={m["completed_updates"]}, {m["termination_reason"]}',fontsize=9);ax.set_xlabel('x coordinate');ax.set_ylabel('y coordinate')
    fig.suptitle(label);fig.tight_layout();fig.savefig(out/f'{label}_final_layouts.png',dpi=130);plt.close(fig)


def study(out,cap,manifest):
    convergence=[];histories=[];searches=[];validation=[];early_checks=[];topologies=[];paired=[];failures=[];run_manifest=[];problem_manifest=[]
    cases=suite(cap)
    for j,(p,x0,spec) in enumerate(cases):
        label=spec['label'];case=out/'problems'/spec['problem_id'];case.mkdir(parents=True)
        dump(case/'specification.json',spec);q,z,s=load_case(case/'specification.json')
        problem_manifest.append(dict(label=label,problem_id=spec['problem_id'],specification=str((case/'specification.json').resolve()),mechanical_id=p.problem_id))
        runs={}
        for option in OPTIONS:
            print(f'{j+1}/{len(cases)} {label} {option}',flush=True)
            try:
                a,m,search=run(p,x0,option)
                m.update(study_problem_id=spec['problem_id'],initial_design_fingerprint=array_digest(x0),executing_source_sha256=manifest['run_m3_3a.py']['sha256'])
                folder=case/'runs'/option;folder.parent.mkdir(exist_ok=True)
                save_trajectory(folder,p,a,m,dict(source_manifest='source_manifest.json',generation_id=out.name,executing_source_sha256=manifest['run_m3_3a.py']['sha256']))
                q,b,n=load_trajectory(folder)
                for key in a:np.testing.assert_array_equal(a[key],b[key])
                stops={tol:first_stopping_update(a['update_change'],tol) for tol in TOLS}
                indices=sorted(set([0,1,len(a['x'])//2,max(0,len(a['x'])-2),len(a['x'])-1]+[t for t in stops.values() if t is not None]))
                rows=replay_selected(q,x0,b,n,indices)
                validation.extend(dict(label=label,option=option,**r) for r in rows)
                strict_attained=[tol for tol in TOLS if stops[tol] is not None]
                ref_tol=min(strict_attained) if strict_attained else None
                ref=stops[ref_tol] if ref_tol else len(a['x'])-1
                for tol in TOLS:
                    hit=stops[tol];t=hit if hit is not None else len(a['x'])-1
                    diag=stationarity_diagnostics(a['x'][t],a['g_x'][t],p.volume_fraction)
                    if tol!=.001:
                        early,em,ed=run(p,x0,option,tol)
                        max_error=0.
                        for key in ('x','rho','u','compliance','g_rho','g_x'):
                            check=comparison(early[key],a[key][:len(early[key])],1e-8,'independent early stop '+key);max_error=max(max_error,check['relative_error'])
                        if em['completed_updates']!=t or em['termination_reason']!=('change_tolerance' if hit is not None else 'maxiter'):raise ValueError('Independent early stopping disagrees')
                        early_checks.append(dict(label=label,option=option,tolerance=tol,updates=t,FEM_solves=em['equilibrium_solves'],runtime_seconds=em['runtime_seconds'],max_relative_error=max_error,passed=True))
                    row=dict(label=label,problem_id=spec['problem_id'],option=option,tolerance=tol,first_hit=hit,updates=t,FEM_solves=t+1,common_run_runtime_seconds=m['runtime_seconds'],prefix_runtime_seconds=prefix_seconds(a,t),
                        termination_reason='change_tolerance' if hit is not None else 'maxiter',compliance=float(a['compliance'][t]),mean_x=float(a['design_volume'][t]),mean_rho=float(a['physical_volume'][t]),change=float(a['previous_change'][t]),
                        reference_tolerance=ref_tol,reference_index=ref,reference_change_converged=ref_tol is not None,strict_001_reference_available=stops[.001] is not None,**diag,**differences(a,t,a,ref))
                    convergence.append(row)
                    for threshold in (.4,.5,.6):
                        tc=topology(p,a['rho'][t],threshold);tr=topology(p,a['rho'][ref],threshold)
                        topologies.append(dict(label=label,option=option,tolerance=tol,threshold=threshold,components=tc['components'],reference_components=tr['components'],all_loads_supported=tc['all_loads_supported'],reference_all_loads_supported=tr['all_loads_supported'],
                            binary_rho_disagreement=float(np.mean((a['rho'][t]>threshold)!=(a['rho'][ref]>threshold))),binary_x_disagreement=float(np.mean((a['x'][t]>threshold)!=(a['x'][ref]>threshold))),largest_component_fraction=tc['largest_component_fraction']))
                for t in range(len(a['x'])):
                    histories.append(dict(label=label,option=option,iteration=t,compliance=float(a['compliance'][t]),change=None if t==0 else float(a['previous_change'][t]),mean_x=float(a['design_volume'][t]),mean_rho=float(a['physical_volume'][t]),volume_excess=float(a['design_volume'][t]-p.volume_fraction),
                        residual_relative=float(a['residual_relative_force'][t]),energy_relative=float(a['energy_compliance_relative'][t]),force_balance=float(a['force_balance_relative'][t]),moment_balance=float(a['moment_balance_relative'][t])))
                searches.extend(dict(label=label,**s) for s in search)
                run_manifest.append(dict(label=label,problem_id=spec['problem_id'],option=option,folder=str(folder.resolve()),updates=m['completed_updates'],termination_reason=m['termination_reason'],FEM_solves=m['equilibrium_solves'],runtime_seconds=m['runtime_seconds'],
                    max_volume_excess=float(np.max(a['design_volume']-p.volume_fraction)),strict_feasible_all_states=bool(np.all(a['design_volume']<=p.volume_fraction)),all_update_endpoints_feasible=bool(all(d['strict_feasible'] for d in search)),
                    bisection_iterations=sum(d['iterations'] for d in search),total_update_seconds=float(a['update_seconds'].sum()),max_equilibrium_residual=float(a['residual_relative_force'].max()),warnings=m['warnings']+(['positive volume excess observed'] if np.any(a['design_volume']>p.volume_fraction) else [])))
                runs[option]=(a,m,search)
                print(f'  {m["completed_updates"]} updates: {m["termination_reason"]}, C={a["compliance"][-1]:.8g}, {m["runtime_seconds"]:.3f}s',flush=True)
            except Exception as exc:
                record=dict(label=label,option=option,exception=type(exc).__name__,message=str(exc),diagnostic=getattr(exc,'diagnostic',{}),traceback=traceback.format_exc())
                failures.append(record);dump(case/f'{option}_failure.json',record);print('  FAILURE: '+str(exc),flush=True)
        if 'baseline_v1' in runs:
            a,ma,sa=runs['baseline_v1'];base_diag=stationarity_diagnostics(a['x'][-1],a['g_x'][-1],p.volume_fraction)
            for option,(b,mb,sb) in runs.items():
                if option=='baseline_v1':continue
                new_diag=stationarity_diagnostics(b['x'][-1],b['g_x'][-1],p.volume_fraction)
                row=dict(label=label,option=option,baseline_updates=ma['completed_updates'],new_updates=mb['completed_updates'],baseline_reason=ma['termination_reason'],new_reason=mb['termination_reason'],both_change_converged=ma['termination_reason']==mb['termination_reason']=='change_tolerance',
                    baseline_PG=base_diag['projected_gradient_inf'],new_PG=new_diag['projected_gradient_inf'],PG_difference=new_diag['projected_gradient_inf']-base_diag['projected_gradient_inf'],baseline_box_KKT=base_diag['box_stationarity_inf'],new_box_KKT=new_diag['box_stationarity_inf'],
                    runtime_ratio=mb['runtime_seconds']/ma['runtime_seconds'],update_time_ratio=float(b['update_seconds'].mean()/a['update_seconds'].mean()),baseline_mean_x=float(a['design_volume'][-1]),new_mean_x=float(b['design_volume'][-1]),**differences(b,-1,a,-1))
                for threshold in (.4,.5,.6):
                    row[f'binary_rho_difference_{threshold}']=float(np.mean((a['rho'][-1]>threshold)!=(b['rho'][-1]>threshold)))
                    ta=topology(p,a['rho'][-1],threshold);tb=topology(p,b['rho'][-1],threshold)
                    row[f'components_changed_{threshold}']=ta['components']!=tb['components'];row[f'paths_changed_{threshold}']=ta['loaded_paths']!=tb['loaded_paths']
                paired.append(row)
        if runs:plots(out/'figures',label,p,runs)
        # Checkpoint manifests updated after each case for recoverable provenance.
        dump(out/'trajectory_manifest.json',run_manifest);dump(out/'problem_manifest.json',problem_manifest)
        source_check(manifest);protected()
    for name,rows in [('optimizer_comparison',convergence),('histories',histories),('multiplier_search',searches),('numerical_validation',validation),('early_stop_validation',early_checks),('topology_comparison',topologies),('paired_variant_comparison',paired)]:table(out/(name+'.csv'),rows)
    table(out/'failed_cases.csv',[{k:v for k,v in r.items() if k!='traceback'} for r in failures],['label','option','exception','message','diagnostic']);dump(out/'failed_cases.json',failures)
    return dict(convergence=convergence,paired=paired,runs=run_manifest,failures=failures,validation_rows=len(validation),selected_state_count=sum(r['quantity']=='rho' for r in validation),selected_transition_count=sum(r['quantity']=='OC_transition' for r in validation),early_stop_checks=len(early_checks))


def tests(out):
    logs=out/'test_logs';logs.mkdir(exist_ok=True);results={}
    for name,extra in [('focused',['tests/test_robust_oc.py']),('regression',[])]:
        cmd=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*extra];start=perf_counter()
        result=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (logs/(name+'.log')).write_text('Command: '+' '.join(cmd)+'\n'+result.stdout+f'\nExit code: {result.returncode}\n')
        results[name]=dict(exit_code=result.returncode,wall_seconds=perf_counter()-start,stdout=result.stdout)
        print(name+': '+result.stdout.strip(),flush=True)
        if result.returncode:raise RuntimeError(name+' tests failed; see log')
    return results


def report(out,data,test_results):
    paired=data['paired'];runs=data['runs'];aggregates={}
    for option in ('tight_midpoint_v1','feasible_upper_v1'):
        rows=[r for r in paired if r['option']==option]
        is_faster=lambda r:(r['new_reason']=='change_tolerance' and (r['baseline_reason']=='maxiter' or r['new_updates']<r['baseline_updates']))
        slower=lambda r:(r['baseline_reason']=='change_tolerance' and (r['new_reason']=='maxiter' or r['new_updates']>r['baseline_updates']))
        aggregates[option]=dict(paired_cases=len(rows),faster_change_stopping=sum(is_faster(r) for r in rows),slower_change_stopping=sum(slower(r) for r in rows),
            both_capped=sum(r['baseline_reason']==r['new_reason']=='maxiter' for r in rows),both_change_converged=sum(r['both_change_converged'] for r in rows),
            PG_improved=sum(r['PG_difference']<-1e-8 for r in rows),PG_worsened=sum(r['PG_difference']>1e-8 for r in rows),
            any_binary_rho_change_at_05=sum(r['binary_rho_difference_0.5']>0 for r in rows),meaningful_binary_change_above_5_percent=sum(r['binary_rho_difference_0.5']>.05 for r in rows),
            paths_changed_at_05=sum(r['paths_changed_0.5'] for r in rows),components_changed_at_05=sum(r['components_changed_0.5'] for r in rows),
            median_runtime_ratio=float(np.median([r['runtime_ratio'] for r in rows])),median_guarded_update_time_ratio=float(np.median([r['update_time_ratio'] for r in rows])),
            median_absolute_relative_C_difference=float(np.median([abs(r['relative_compliance_difference']) for r in rows])),maximum_absolute_relative_C_difference=max(abs(r['relative_compliance_difference']) for r in rows))
    summary=dict(schema_version='m3.3a.validation.v1',problem_count=len(json.loads((out/'problem_manifest.json').read_text())),run_count=len(runs),failures=len(data['failures']),aggregates=aggregates,
        feasible_upper_update_runs=sum(r['all_update_endpoints_feasible'] for r in runs if r['option']=='feasible_upper_v1'),feasible_upper_run_count=sum(r['option']=='feasible_upper_v1' for r in runs),
        max_positive_volume_excess={o:max(r['max_volume_excess'] for r in runs if r['option']==o) for o in OPTIONS},
        change_converged_by_option={o:sum(r['termination_reason']=='change_tolerance' for r in runs if r['option']==o) for o in OPTIONS},
        selected_states=data['selected_state_count'],selected_transitions=data['selected_transition_count'],numerical_validation_rows=data['validation_rows'],independent_early_stop_checks=data['early_stop_checks'],
        tests=test_results,protected_count=protected(),source_unchanged_during_generation=True,
        interpretation='change convergence is not KKT or optimality; capped endpoints are censored for stopping comparisons and explicitly identified in quality comparisons')
    dump(out/'validation_summary.json',summary)
    lines=['# M3.3A — Robust OC validation across controlled problems','',
      f"Actual pilot: {summary['problem_count']} specified problems, {summary['run_count']} saved common trajectories, three optimizer options, and a strict 500-update cap without extensions. Failed runs: {summary['failures']}. All outputs are generation-specific. No prior numerical core, outputs or global defaults changed.",'',
      '## Optimizer audit and bracket proof','',
      'Baseline uses [0,1e5], width 1e-4, move .1, box [.001,1], sqrt(max(-g/lambda,0)) and the last tested midpoint. The budget is sum(x)<=N*target. It does not verify endpoint feasibility. For finite inputs and lambda>0, the negative-gradient square-root factor is nonincreasing; clipping by fixed move/box bounds preserves this monotonicity. Zero or positive gradients select the lower reachable value. The explicit right limit at lambda=0 handles zero gradients without 0/0.',
      'A move-limited budget can be unattainable when mean(max(.001,x-.1)) exceeds target. Large sensitivities can make the fixed 1e5 endpoint infeasible. The versioned dispatcher checks both conditions and rejects them with diagnostics, without expanding the historical bracket or rescaling densities. Feasible-upper selection is valid only after that invariant is established. If the zero right limit is already feasible, a slack-budget/no-root state is recorded. Floating-point stagnation or nonfinite inputs fail explicitly.',
      'Baseline selection calls the untouched helper and verifies its returned array. Both midpoint options retain the historical scalar-sum test. The feasible-upper option conservatively requires both that sum test and the reported NumPy mean test; reduction-roundoff differences are recorded. This explicit implementation choice prevents claiming feasibility from a different reduction. No post-update repair occurs.','',
      '## Problem specifications and validated scope','',
      'The manifest covers 12x6,24x12 and mostly48x24 meshes; unit or oblique nodal loads on left/right/top/bottom boundaries; left/right/bottom clamps and a bottom pin/roller pair; volume .3/.4/.5/.6; elongated/square/rectangular-element domains; uniform and seeded feasible perturbed starts. All retain p=3, E0=1, nu=.3, unit thickness, Emin=.001 and the existing boundary-renormalized density filter. Complete nodal coordinates, forces, fixed DOFs, initial arrays, ordering, material and optimizer controls are hashed in m3.3a.problem.v1. Problem IDs cover this full specification.',
      'Filter radius remains 2 element units. Physical support semi-axes are radius*dx and radius*dy; rectangular-element support is elliptical. This is a documented consequence of the unchanged kernel, not fixed-physical-radius mesh refinement. Loads on constrained DOFs and rigid-body-unstable supports are rejected by the existing specification. Every evaluated state checks equilibrium, force/moment balance and energy. Selected states independently reconstruct stiffness via COO assembly and compare the solver compliance/gradient API. No unsupported new physics was introduced.','',
      '## Numerical precision, feasibility and aggregate behavior','',
      '| Option | Change-converged / runs | Maximum signed mean-x excess |','|---|---:|---:|']
    for o in OPTIONS:lines.append(f"| {o} | {summary['change_converged_by_option'][o]}/{sum(r['option']==o for r in runs)} | {summary['max_positive_volume_excess'][o]:+.6g} |")
    lines+=['',f"Feasible-upper update endpoints passed both reductions in {summary['feasible_upper_update_runs']}/{summary['feasible_upper_run_count']} saved runs. Positive excess means strict infeasibility, even below the declared 1e-4 research feasibility screen. Initial feasibility is checked separately. Tightening to 1e-8 reduces bracket width from 30 to 44 bisections for an active root; tightening midpoint selection alone does not guarantee a feasible returned midpoint.",'',
      '| New option vs baseline | Faster stopping | Slower stopping | Both capped | PG improved | PG worsened | Any binary rho change (.5) | >5% binary change | Median wall ratio |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for o,r in aggregates.items():lines.append(f"| {o} | {r['faster_change_stopping']} | {r['slower_change_stopping']} | {r['both_capped']} | {r['PG_improved']} | {r['PG_worsened']} | {r['any_binary_rho_change_at_05']} | {r['meaningful_binary_change_above_5_percent']} | {r['median_runtime_ratio']:.4g} |")
    lines+=['','PG comparisons classify endpoint changes with an absolute 1e-8 diagnostic deadband and include explicitly labeled capped endpoints; they are not proof of convergence. Faster stopping means reaching .001 in fewer updates or attaining it while baseline is capped. Both-capped cases are censored, not equal convergence times. Component/path changes and binary differences at .4/.5/.6 are retained, using actual support nodes and loaded-node-adjacent elements rather than a cantilever-only heuristic. Ersatz stiffness means binary paths do not establish mechanical disconnection.','',
      '## Per-problem strict-target results','',
      '| Problem | Baseline updates / stop | Tight midpoint updates / stop | Feasible upper updates / stop |','|---|---|---|---|']
    for case in json.loads((out/'problem_manifest.json').read_text()):
        rs={r['option']:r for r in runs if r['label']==case['label']}
        values=[f"{rs[o]['updates']} / {rs[o]['termination_reason']}" if o in rs else 'FAILED' for o in OPTIONS]
        lines.append('| '+case['label']+' | '+' | '.join(values)+' |')
    lines+=['','## Tolerance, stationarity and quality conventions','',
      'For each option, first strict crossings of .01/.005/.001 come from one common .001-target trajectory. Independent relaxed runs verify complete x/rho/u/C/g_rho/g_x prefixes, stop indices/reasons and terminal evaluations. Each uninterrupted endpoint uses t+1 FEM solves. Prefix profiling excludes the outgoing update at the endpoint. All runs retain the same starting x. A nonuniform start is explicit in the outer specification, not inferred from mesh size.',
      'The primary within-option design reference is its strictest attained change tolerance. If none is attained, the recorded capped endpoint is explicitly nonconverged. Between-option tables compare recorded strict-target endpoints and flag both-change-converged pairs; a capped endpoint is not a high-quality reference. Signed C differences, separate x/rho MAE/max/relative-L2 and binary differences are retained. Different endpoints do not establish different local minima without an optimality analysis.',
      'Stationarity reuses M3.2: s=max|g_x| and PG=x-Proj_box_sum(x-g_x/s). A nonnegative sum-constraint multiplier minimizes squared box stationarity: lower-bound residual=min(g+lambda,0), upper=max(g+lambda,0), interior=g+lambda. Report infinity/RMS residuals divided by s, complementarity=lambda*abs(mean(x)-target)/s, bound and primal/dual feasibility. Dominant gradients affect this scaling. The OC move box is not the optimization feasible set. .001 change convergence alone is never a KKT certificate. Provisional screens PG<=.01 and mean excess<=1e-4 are diagnostics, not guarantees.','',
      '## Timing and overhead','',
      'Wall times include FEM/state diagnostics and guarded OC updates, excluding serialization, replay, plots and tests. Runs are sequential with unchanged MKL16 and OpenBLAS1. Baseline instrumentation independently reconstructs the bisection and calls the original helper to verify array equality; that extra audit overhead is included, so wall/update ratios are guarded-pilot costs and cannot be presented as a clean production benchmark. The 44-versus-30 scalar iterations quantify algorithmic overhead. More precise production timing requires removing duplicated audit work uniformly and repeated matched-state timing. No hardware-independent speed claim is made.','',
      '## Failures, replay and reproducibility','',
      f"Failures: {summary['failures']}, retained in failed_cases.csv/JSON and per-case traceback files. Numerical validation reconstructed {summary['selected_states']} selected states and {summary['selected_transitions']} transitions ({summary['numerical_validation_rows']} comparison rows); independently checked {summary['independent_early_stop_checks']} relaxed runs. Hash checks and physical reconstruction are separate. Source manifest archives actual loaded source bytes before execution and verifies they stay unchanged throughout the generation. Executable .py snapshots preserve package structure; archived tests use .py.txt to avoid unintended pytest collection. Environment/module paths, configuration and numerical array digests are recorded.",
      f"Protected files unchanged: {summary['protected_count']}. Focused and regression logs record actual stdout and exit codes. No M3.3B/C, neural models, randomized large datasets, HPC/SLURM, commit or push.",'',
      '## Scientific assessment and next-data recommendation','',
      'Use the explicitly versioned feasible_upper_v1 for the next controlled training-trajectory pilot when the fixed bracket passes its invariant checks. Its justification is verified numerical feasibility; it is not a universally faster or more stationary optimizer. Retain the historical baseline for replay and report any fixed-bracket failure rather than return an infeasible density. Keep .001 as a provisional change target and a 500-update cap with independent stationarity/feasibility flags; relaxed endpoints remain useful labeled intermediate states, not certified optimum labels.',
      'Before 100–200 trajectories: investigate capped/slower and worse-stationarity cases, decide a quality/stopping policy beyond design change, benchmark production overhead without duplicate audit loops, test load-scale/near-zero sensitivity limitations of an absolute multiplier tolerance and the fixed bracket, broaden feasible perturbations and problem-level holdouts, and determine how threshold connectivity and stiffness-floor load paths affect label suitability. No claim of generality beyond this deterministic population or certified local/global optimality is made.']
    lines+=['','### Cases with slower stopping or worse endpoint stationarity','', '| Problem | Option | Baseline/new stop | Baseline/new PG infinity | Signed C difference |','|---|---|---|---:|---:|']
    for r in paired:
        if r['PG_difference']>1e-8 or (r['baseline_reason']=='change_tolerance' and (r['new_reason']=='maxiter' or r['new_updates']>r['baseline_updates'])):
            lines.append(f"| {r['label']} | {r['option']} | {r['baseline_updates']} {r['baseline_reason']} / {r['new_updates']} {r['new_reason']} | {r['baseline_PG']:.5g} / {r['new_PG']:.5g} | {r['relative_compliance_difference']:+.5g} |")
    (out/'M3_3A_OC_VALIDATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['all','validate']);parser.add_argument('--output',type=Path);parser.add_argument('--maxiter',type=int,default=500);args=parser.parse_args()
    if not 1<=args.maxiter<=500:parser.error('Pilot cap must be between 1 and 500; no automatic extensions')
    protected()
    if args.command=='validate':
        if args.output is None:parser.error('validate requires existing generation --output')
        out=args.output;rows=[]
        for e in json.loads((out/'trajectory_manifest.json').read_text()):
            p,a,m=load_trajectory(e['folder']);case=Path(e['folder']).parent.parent;_,x,s=load_case(case/'specification.json')
            indices=[0,1,len(a['x'])//2,max(0,len(a['x'])-2),len(a['x'])-1]
            rows.extend(dict(label=e['label'],option=e['option'],**r) for r in replay_selected(p,x,a,m,indices))
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');table(out/f'fresh_replay_{stamp}.csv',rows);print('Fresh process replay passed:',len(rows),'comparisons;',protected(),'protected files unchanged');return
    out=args.output or ROOT/'m3_3/generations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir();source=source_snapshot(out);dump(out/'environment.json',environment())
    dump(out/'configuration.json',dict(schema_version='m3.3a.study.v1',cap=args.maxiter,tolerances=TOLS,options=OPTIONS,problem_count=20,strict_volume='sum and reported mean',PG_screen=.01,mean_volume_screen=1e-4,thresholds=[.4,.5,.6]))
    data=study(out,args.maxiter,source);source_check(source)
    test_results=tests(out);summary=report(out,data,test_results);source_check(source);protected()
    dump(ROOT/'m3_3/latest_generation.json',dict(generation=str(out.resolve()),report=str((out/'M3_3A_OC_VALIDATION_REPORT.md').resolve()),source_manifest_sha256=digest(out/'source_manifest.json'),validation_summary_sha256=digest(out/'validation_summary.json')))
    print('Completed generation:',out,flush=True)
    print(json.dumps(summary['aggregates'],indent=2),flush=True)

if __name__=='__main__':main()
