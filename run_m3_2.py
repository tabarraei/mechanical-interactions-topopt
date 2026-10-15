#!/usr/bin/env python
"""M3.2 small deterministic convergence/restart/OC reliability study."""
import argparse
from dataclasses import replace
from datetime import datetime,timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_m2_spectral import write_csv,serial,sha256
from topopt_utils.fea.consistent_trajectory import load_trajectory,save_trajectory,evaluate_state
from topopt_utils.fea.trajectory_validation import comparison,validate_trajectory
from topopt_utils.fea.convergence_reliability import (TOLERANCES,run_trajectory,first_stopping_update,stationarity_diagnostics,final_design_difference,oc_bisection_variant,topology_diagnostics)
from topopt_utils.topopt_2d import __oc_update_design__

ROOT=Path(__file__).resolve().parent
STATIONARITY_INF_TARGET=.01
VOLUME_TARGET=1e-4


def protected_check():
    data=json.loads((ROOT/'m3_2/protected_before.json').read_text())
    changed=[p for p,h in data.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=h]
    if changed:raise RuntimeError(f'Protected files changed: {changed}')
    return len(data)


def provenance():
    names=['run_m3_2.py','topopt_utils/fea/convergence_reliability.py','tests/test_convergence_reliability.py','topopt_utils/fea/mechanical_problem.py','topopt_utils/fea/consistent_trajectory.py','topopt_utils/fea/trajectory_validation.py','topopt_utils/topopt_2d.py','topopt_utils/fea/solver.py','topopt_utils/fea/q4.py']
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),source_fingerprints={n:sha256(ROOT/n) for n in names},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        versions={n:version(n) for n in ['numpy','scipy','matplotlib','pypardiso','pytest']},thread_environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','PYPARDISO_MKL_RT']},
        state_timing='t completed OC updates, state evaluated before outgoing update; terminal state includes one extra FEM evaluation',
        stopping='strict max(abs(x_next-x)) < tolerance; check incoming transition at evaluated terminal x',
        unchanged_settings='M3.1 mechanics, uniform x=.5, p=3, emin=.001, filter radius=2, move=.1, x bounds [.001,1], sum(x) constraint, absolute lambda bracket width=1e-4 on [0,1e5]',
        acceptance_targets=dict(compliance_absolute_relative=.01,volume_violation_mean=VOLUME_TARGET,projected_gradient_inf=STATIONARITY_INF_TARGET,stationarity_ratio_to_strict=2.,solve_reduction_fraction=.2,binary_disagreement_fraction=.05),
        topology='rho>.5 with four-neighbor components and right loaded-node-adjacent solid connection to left; also thresholds .4/.6. Ersatz stiffness keeps all elements mechanically connected.')


def source_snapshot(out,metadata):
    for name,h in metadata['source_fingerprints'].items():
        dst=out/'source_snapshots'/(name+'.txt');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
        if sha256(dst)!=h:raise RuntimeError('Source fingerprint drift')


def generate_references(out,cap=500):
    manifest=json.loads((ROOT/'m3_1/trajectory_manifest.json').read_text());entries=[]
    for old in manifest['trajectories']:
        base,original,old_meta=load_trajectory(old['folder']);p=replace(base,maxiter=cap,tolerance=.001)
        print(f'Common reference {old["domain"]}: unchanged OC, cap={cap}',flush=True)
        a,m=run_trajectory(p)
        if m['termination_reason']=='maxiter':
            # Explicit bounded extension, only if the measured pilot cost is small.
            print(f'  cap reached: last100 min change={a["update_change"][-100:].min():.6g}; runtime={m["generation_seconds"]:.3f}s',flush=True)
            initial=out/'initial_cap_references'/old['domain'];initial.parent.mkdir(parents=True,exist_ok=True);save_trajectory(initial,p,a,m,provenance())
            if cap==500 and m['generation_seconds']<60:
                print('  justified extension to 1000: cheap deterministic case, strict reference needed; no setting other than budget changes',flush=True)
                extended=replace(p,maxiter=1000)
                b,n=run_trajectory(extended,start_x=a['x'][-1],start_iteration=cap,previous_change=a['previous_change'][-1])
                merged={}
                transition={'update_change','update_seconds','bisection_iterations'}
                for key in a:
                    if key in transition:merged[key]=np.concatenate((a[key],b[key]))
                    elif key=='elapsed_seconds':merged[key]=np.concatenate((a[key],b[key][1:]+m['generation_seconds']))
                    else:merged[key]=np.concatenate((a[key],b[key][1:]))
                m=dict(n,start_iteration=0,state_count=len(merged['x']),generation_seconds=m['generation_seconds']+n['generation_seconds'],
                    equilibrium_solve_count=m['equilibrium_solve_count']+n['equilibrium_solve_count'],problem_id=extended.problem_id,
                    extension_note='500-update cap insufficient; explicit 1000-update extension. The shared checkpoint state was evaluated twice; actual solve count includes it.')
                a,p=merged,extended
        folder=out/'references'/old['domain'];folder.parent.mkdir(parents=True,exist_ok=True);save_trajectory(folder,p,a,m,provenance())
        # Actual numerical equality of the pre-existing M3.1 prefix, not just hashes.
        for key in ('x','rho','u','compliance','g_rho','g_x'):
            comparison(a[key][:len(original[key])],original[key],1e-8,'M3.1/reference common prefix')
        entries.append(dict(domain=old['domain'],folder=str(folder.resolve()),m3_1_folder=old['folder'],problem_id=p.problem_id,updates=m['completed_updates'],termination_reason=m['termination_reason']))
    (out/'reference_manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
    return entries


def state_at(arrays,index):return {k:arrays[k][index] for k in ('x','rho','u','compliance','g_rho','g_x')}


def prefix_runtime(a,t):return float(a['evaluation_seconds'][:t+1].sum()+a['update_seconds'][:t].sum())


def analyze(out,entries):
    summaries=[];comp_hist=[];change_hist=[];volume_hist=[];station=[];replays=[];bis_rows=[];designs=[];variant_rows=[];prefix_checks=[];tails=[];references={}
    for entry in entries:
        domain=entry['domain'];p,a,m=load_trajectory(entry['folder']);base,old,oldm=load_trajectory(entry['m3_1_folder']);shape=(p.nely,p.nelx)
        print(f'Analyze {domain}: reference stopped {m["completed_updates"]}, {m["termination_reason"]}',flush=True)
        validate_trajectory(p,a,m)
        stops={tol:first_stopping_update(a['update_change'],tol) for tol in TOLERANCES}
        available=[tol for tol in TOLERANCES if stops[tol] is not None]
        reference_tol=min(available) if available else None
        ref_index=stops[reference_tol] if reference_tol is not None else len(a['x'])-1
        ref=state_at(a,ref_index);ref_station=stationarity_diagnostics(ref['x'],ref['g_x'],p.volume_fraction)
        references[domain]=dict(tolerance=reference_tol,iteration=ref_index,converged=reference_tol is not None,cap=p.maxiter)
        for t in range(len(a['x'])):
            comp_hist.append(dict(domain=domain,iteration=t,compliance=a['compliance'][t]))
            change_hist.append(dict(domain=domain,iteration=t,incoming_design_change=a['previous_change'][t]))
            volume_hist.append(dict(domain=domain,iteration=t,mean_x=a['design_volume'][t],mean_rho=a['physical_volume'][t],volume_excess=a['design_volume'][t]-p.volume_fraction))
        for tol in TOLERANCES:
            hit=stops[tol];t=hit if hit is not None else len(a['x'])-1;current=state_at(a,t)
            diag=stationarity_diagnostics(current['x'],current['g_x'],p.volume_fraction)
            difference=final_design_difference(current,ref,shape)
            # Independent relaxed early-stop run; strict common reference is itself
            # an actual strict early-stop run and need not be duplicated.
            if tol!=.001:
                early_problem=replace(p,tolerance=tol)
                b,n=run_trajectory(early_problem)
                maxerr=0.
                for key in ('x','rho','u','compliance','g_rho','g_x'):
                    check=comparison(b[key],a[key][:len(b[key])],1e-8,'independent early stop sequence')
                    maxerr=max(maxerr,check['relative_error'])
                if n['completed_updates']!=t or n['termination_reason']!=('change_tolerance' if hit is not None else 'maxiter'):raise ValueError('Independent stop mismatch')
                folder=out/'early_stop_runs'/domain/f'tol_{tol:g}';folder.parent.mkdir(parents=True,exist_ok=True);save_trajectory(folder,early_problem,b,n,provenance())
                runtime=n['generation_seconds'];solves=n['equilibrium_solve_count']
                prefix_checks.append(dict(domain=domain,tolerance=tol,updates=t,max_relative_replay_error=maxerr,passed=True))
            else:runtime=m['generation_seconds'];solves=m['equilibrium_solve_count']
            same_path=difference['relaxed_loaded_path']==difference['reference_loaded_path'];same_components=difference['relaxed_components']==difference['reference_components']
            pg_ok=diag['projected_gradient_inf']<=STATIONARITY_INF_TARGET and diag['projected_gradient_inf']<=max(2*ref_station['projected_gradient_inf'],.001)
            solve_saved=1-(t+1)/(ref_index+1)
            accepted=reference_tol is not None and hit is not None and difference['relative_absolute_compliance_difference']<=.01 and diag['volume_violation']<=max(VOLUME_TARGET,ref_station['volume_violation']) and same_path and same_components and difference['rho_binary_disagreement_fraction']<=.05 and pg_ok and solve_saved>=.2
            row=dict(domain=domain,tolerance=tol,oc_updates=t,first_satisfying_update=hit,equilibrium_solves=solves,total_runtime_seconds=runtime,prefix_measured_runtime_seconds=prefix_runtime(a,t),
                final_compliance=float(current['compliance']),mean_x=float(current['x'].mean()),mean_rho=float(current['rho'].mean()),maximum_design_change=float(a['previous_change'][t]),
                termination_reason='change_tolerance' if hit is not None else 'maxiter',volume_excess=diag['volume_excess'],volume_violation=diag['volume_violation'],strict_volume_feasible=diag['strict_volume_feasible'],
                reference_tolerance=reference_tol,reference_updates=ref_index,reference_converged=reference_tol is not None,fraction_reference_updates_retained=t/ref_index,
                solve_reduction_fraction=solve_saved,measured_prefix_runtime_reduction_fraction=1-prefix_runtime(a,t)/prefix_runtime(a,ref_index),
                projected_gradient_inf=diag['projected_gradient_inf'],provisional_stationarity_acceptable=pg_ok,provisional_candidate_acceptance=bool(accepted),**difference)
            summaries.append(row);station.append(dict(domain=domain,tolerance=tol,iteration=t,**diag));designs.append(dict(domain=domain,tolerance=tol,iteration=t,**difference))
            threshold_rows=[]
            for threshold in (.4,.5,.6):
                td=topology_diagnostics(current['rho'],shape,threshold);tr=topology_diagnostics(ref['rho'],shape,threshold)
                threshold_rows.append(dict(domain=domain,tolerance=tol,threshold=threshold,relaxed_components=td['component_count'],reference_components=tr['component_count'],relaxed_loaded_path=td['loaded_path'],reference_loaded_path=tr['loaded_path']))
            tails.extend(threshold_rows)
        # Use saved M3.1 checkpoints, explicitly extending their remaining budget
        # to the same reference cap, rather than resetting the iteration counter.
        for k in (10,25,50,75):
            b,n=run_trajectory(p,start_x=old['x'][k],start_iteration=k,previous_change=old['previous_change'][k])
            record=dict(domain=domain,checkpoint=k,extended_budget=p.maxiter,final_iteration=n['completed_updates'],termination_reason=n['termination_reason'],equilibrium_solves=n['equilibrium_solve_count'])
            for key in ('x','rho','compliance','g_rho','g_x'):
                check=comparison(b[key],a[key][k:],1e-8,'checkpoint continuation '+key)
                record[key+'_absolute_error']=check['absolute_error'];record[key+'_relative_error']=check['relative_error']
            comparison(b['update_change'],a['update_change'][k:],1e-8,'checkpoint OC transitions')
            if n['termination_reason']!=m['termination_reason'] or n['completed_updates']!=m['completed_updates']:raise ValueError('Restart stopping mismatch')
            record['topology_identical']=bool(np.array_equal(b['rho'][-1]>.5,a['rho'][-1]>.5));record['passed']=True;replays.append(record)
        selected=sorted(set([0,10,25,50,75]+[i for i in stops.values() if i is not None]))
        for t in selected:
            x=a['x'][t];g=a['g_x'][t];baseline=__oc_update_design__(x,g,p.volume_fraction)
            baseline_next=evaluate_state(p,baseline)
            for width in (1e-2,1e-4,1e-6,1e-8):
                for selection in ('last_midpoint','feasible_upper'):
                    next_x,diagnostic=oc_bisection_variant(x,g,p.volume_fraction,width,selection)
                    if width==1e-4 and selection=='last_midpoint':np.testing.assert_array_equal(next_x,baseline)
                    next_state=evaluate_state(p,next_x)
                    bis_rows.append(dict(domain=domain,iteration=t,**diagnostic,next_compliance=next_state['compliance'],next_compliance_relative_to_baseline=(next_state['compliance']-baseline_next['compliance'])/baseline_next['compliance'],
                        max_update_difference_from_baseline=float(np.max(abs(next_x-baseline))),next_physical_volume=next_state['physical_volume']))
        # Independent full trajectories for explicitly named optional variants.
        for name,selection in [('tight_midpoint_1e-8','last_midpoint'),('feasible_upper_1e-8','feasible_upper')]:
            variant=dict(width=1e-8,selection=selection);b,n=run_trajectory(p,variant=variant)
            folder=out/'optional_variant_runs'/domain/name;folder.parent.mkdir(parents=True,exist_ok=True);save_trajectory(folder,p,b,n,provenance())
            vs=stationarity_diagnostics(b['x'][-1],b['g_x'][-1],p.volume_fraction)
            variant_rows.append(dict(domain=domain,variant=name,updates=n['completed_updates'],termination_reason=n['termination_reason'],runtime_seconds=n['generation_seconds'],equilibrium_solves=n['equilibrium_solve_count'],
                mean_x=float(b['x'][-1].mean()),mean_rho=float(b['rho'][-1].mean()),volume_excess=vs['volume_excess'],strict_volume_feasible=vs['strict_volume_feasible'],projected_gradient_inf=vs['projected_gradient_inf'],
                total_bisection_iterations=int(b['bisection_iterations'].sum()),**final_design_difference(state_at(b,-1),ref,shape)))
        plot_domain(out,domain,p,a,stops)
    for name,rows in [('convergence_summary',summaries),('compliance_histories',comp_hist),('density_change_histories',change_hist),('volume_constraint_histories',volume_hist),('stationarity_summary',station),
        ('checkpoint_replay_summary',replays),('oc_bisection_study',bis_rows),('final_design_comparison',designs),('optional_oc_trajectory_comparison',variant_rows),('early_stop_replay_summary',prefix_checks),('topology_threshold_sensitivity',tails)]:write_csv(out/(name+'.csv'),rows)
    metadata=dict(provenance=provenance(),references=references,protected_files_unchanged=True,protected_file_count=protected_check(),
        stationarity_convention='s=max|g_x|, PG=x-Proj_box_sum(x-g_x/s); estimate nonnegative lambda minimizing box-KKT squared residual, retain normalized complementarity/feasibility separately; not OC move constraints',
        timing='Independent relaxed-run wall times exclude serialization/plotting. Prefix estimates sum measured equilibrium/state-diagnostic and update durations; no outgoing update charged at terminal. Shared reference timings exclude archive I/O; optional variants measured independently. No runtime threshold inference from different hardware.',
        checkpoint_convention='M3.1 saved checkpoint x and incoming change, with absolute iteration retained and budget explicitly extended to reference cap; no hidden OC history. Compare complete suffix and stopping decisions, no cross-BLAS bitwise claim.',
        topology_convention='four-neighbor thresholded physical rho at .5, plus .4/.6 sensitivity; loaded-node adjacent right-edge solid element path to left; binary connectivity is not mechanical connectivity with stiffness floor',
        optional_variant_convention='problem.json retains unchanged baseline spec. Run metadata.variant is the explicit optimizer-only override; these runs are separate experiments, never passed off as baseline or validated by the baseline OC replay rule.',
        primary_scope='Three deterministic problems, not randomized dataset or ML training. 500 initial updates, explicit bounded extension if inexpensive/necessary; no neural networks, M3.3, HPC, commits or pushes.')
    (out/'reproducibility.json').write_text(json.dumps(serial(metadata),indent=2,allow_nan=False)+'\n')
    write_report(out,summaries,station,replays,bis_rows,variant_rows,metadata)
    late_evolution(out,entries)
    return summaries


def plot_domain(out,domain,p,a,stops):
    for field,ylabel,filename,log in [('compliance','Compliance','compliance',False),('previous_change','Max design-variable change','density_change',True),('design_volume','Mean design x','volume',False)]:
        fig,ax=plt.subplots(figsize=(7,4));getattr(ax,'semilogy' if log else 'plot')(a['iteration'][1:] if log else a['iteration'],a[field][1:] if log else a[field])
        for tol,t in stops.items():
            if t is not None:ax.axvline(t,ls='--',label=f'tol {tol:g}: {t} updates')
        if field=='previous_change':
            for tol in TOLERANCES:ax.axhline(tol,ls=':',color='gray')
        if field=='design_volume':ax.axhline(p.volume_fraction,ls=':',color='k')
        ax.set_xlabel('Completed OC updates / evaluated state');ax.set_ylabel(ylabel);ax.legend();fig.tight_layout();fig.savefig(out/f'{domain}_{filename}.png',dpi=160);fig.savefig(out/f'{domain}_{filename}.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for col,tol in enumerate((.01,.005,.001)):
        t=stops[tol] if stops[tol] is not None else len(a['x'])-1
        for row,space in enumerate(('x','rho')):
            axes[row,col].imshow(a[space][t].reshape(p.nely,p.nelx),origin='lower',extent=(0,p.width,0,p.height),vmin=0,vmax=1,cmap='gray_r');axes[row,col].set_title(f'{space}, tol={tol:g}, t={t}'+(' (maxiter)' if stops[tol] is None else ''));axes[row,col].set_xlabel('x coordinate');axes[row,col].set_ylabel('y coordinate')
    fig.tight_layout();fig.savefig(out/f'{domain}_final_designs.png',dpi=160);fig.savefig(out/f'{domain}_final_designs.pdf');plt.close(fig)


def write_report(out,rows,station,replays,bis,variants,metadata):
    lines=['# M3.2 — Convergence tolerance and optimization reliability','',
        'The study reuses the three mechanically consistent M3.1 axial nodal-load cantilevers. All FEM/SIMP/filter/interaction/HVP/OC routines and M1–M3.1 results are byte-preserved. Geometry, supports, force, initial x, material, filter, volume and OC controls are unchanged; baseline experiments vary only stopping tolerance and update budget.','',
        '## Inspected stopping and top88 distinction','',
        'Here change=max(abs(x_next-x)) and stopping requires change<tolerance. State t has t completed updates; terminal x is additionally evaluated, giving t+1 equilibrium solves in an uninterrupted run. OC uses move=.1, design lower bound=.001, lambda bracket [0,1e5], absolute bracket width=1e-4, and a sum(x) volume budget.','',
        'The original top88 appendix uses a .01 change threshold but also move=.2, zero design lower bound, Emin=1e-9, a [0,1e9] bracket and relative bracket-width test. Its standard problem is a half MBB beam; density-filter mode tests sum(xPhys) and filters its volume derivative. Thus copying only its stopping threshold does not reproduce that optimizer. Source: [DTU top88 paper appendix](https://www.topopt.mek.dtu.dk/-/media/subsites/topopt/apps/dokumenter-og-filer-til-apps/topopt88.pdf?hash=E80FAB2808804A29FFB181CA05D2EEFECAA86686&la=da).','',
        '## Actual baseline stopping points and terminal quality','',
        '| Domain | Tolerance | Updates | FEM solves | Runtime s | Final C | mean x | mean rho | Final change | Stop reason |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['domain']} | {r['tolerance']:g} | {r['oc_updates']} | {r['equilibrium_solves']} | {r['total_runtime_seconds']:.4g} | {r['final_compliance']:.9g} | {r['mean_x']:.9g} | {r['mean_rho']:.9g} | {r['maximum_design_change']:.5g} | {r['termination_reason']} |")
    lines+=['','A common strict trajectory determines first crossings. Independent relaxed early-stop runs match the complete common prefixes and terminal evaluations, not only the stopping indices. The strict common reference itself is an actual strict early-stop run. Timings are measured with the recorded thread settings and exclude file/figure writes. Prefix-based savings avoid treating timer noise between independent runs as an optimization speedup.','',
        '| Domain | Tolerance | Reference tolerance | Signed relative C difference | Absolute relative C difference | Solve reduction | Prefix time reduction | rho MAE | rho max difference | rho relative L2 | x MAE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['domain']} | {r['tolerance']:g} | {r['reference_tolerance']} | {r['relative_compliance_difference']:+.5g} | {r['relative_absolute_compliance_difference']:.5g} | {r['solve_reduction_fraction']:.1%} | {r['measured_prefix_runtime_reduction_fraction']:.1%} | {r['rho_mean_absolute_difference']:.5g} | {r['rho_max_absolute_difference']:.5g} | {r['rho_relative_l2_difference']:.5g} | {r['x_mean_absolute_difference']:.5g} |")
    lines+=['','Full separate x/rho max, MAE, relative L2 and binary disagreement metrics are in `final_design_comparison.csv`; visual similarity is not treated as numerical equality. A signed negative compliance difference can arise from approximate OC feasibility or a different stopping point, and is not evidence that a relaxed run is more optimal.','',
        '## Constrained stationarity, complementarity and feasibility','',
        metadata['stationarity_convention']+'. The sum constraint multiplier convention differs by N from a mean-constraint multiplier. At a lower bound g+lambda must be nonnegative, at an upper bound nonpositive, and at an interior variable zero. Volume multiplier is estimated to minimize these violations; it is not inferred from the previous OC multiplier or an unconstrained gradient norm. Complementarity uses lambda*abs(mean(x)-target)/s.','',
        '| Domain | Tolerance | PG infinity | PG RMS | Box KKT infinity | Complementarity scaled | Mean x excess | Strictly feasible |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in station:lines.append(f"| {r['domain']} | {r['tolerance']:g} | {r['projected_gradient_inf']:.5g} | {r['projected_gradient_rms']:.5g} | {r['box_stationarity_inf']:.5g} | {r['volume_complementarity_scaled']:.5g} | {r['volume_excess']:+.5g} | {r['strict_volume_feasible']} |")
    lines+=['','A density-change crossing does not certify KKT convergence. Provisional reasonable-stationarity acceptance requires PG infinity <=.01 and <=max(2*strict PG infinity,.001). Scale is max absolute filtered gradient at each state, so values are scale-invariant under load-amplitude changes but can be influenced by a dominant loaded-region gradient. Raw gradient scale and multiplier, box residual, primal/dual feasibility and complementarity are retained in the CSV. These thresholds are declared research diagnostics, not universal optimality tolerances.','',
        '## Connectivity and information retained','',
        '| Domain | Tolerance | Updates retained | Binary rho disagreement | Components relaxed/ref | Loaded path relaxed/ref | Candidate acceptance |','|---|---:|---:|---:|---|---|---|']
    for r in rows:lines.append(f"| {r['domain']} | {r['tolerance']:g} | {r['fraction_reference_updates_retained']:.1%} | {r['rho_binary_disagreement_fraction']:.1%} | {r['relaxed_components']}/{r['reference_components']} | {r['relaxed_loaded_path']}/{r['reference_loaded_path']} | {r['provisional_candidate_acceptance']} |")
    lines+=['','Connectivity is measured at rho>.5 using four-neighbor image components, including a support-to-loaded-adjacent-element path. Threshold .4/.6 sensitivity is saved separately. No mechanical disconnection claim is made in the presence of Ersatz stiffness. Provisional acceptance uses <=1% absolute relative compliance difference, mean volume violation <=1e-4 (or strict-reference excess if larger), equal component/path status, <=5% binary disagreement, the stationarity criteria above and >=20% fewer FEM solves. These operational definitions make the proposed research targets explicit; they do not establish held-out-problem reliability.','',
        '## OC volume-bisection study','',
        'The baseline formula is untouched. On identical saved states, absolute widths 1e-2,1e-4,1e-6,1e-8 and last-midpoint versus feasible-upper-endpoint selection are compared. The 1e-4/last-midpoint result is checked array-identical to the original helper. `oc_bisection_study.csv` includes volume excess, scalar iteration counts/runtime, next-state compliance and maximum update changes. No rescaling of densities or replacement of the x volume budget is performed.','',
        '| Domain | Optional variant | Updates | Stop | C difference vs baseline reference | Mean x excess | PG infinity | Total bisection iterations | Runtime s |','|---|---|---:|---|---:|---:|---:|---:|---:|']
    for r in variants:lines.append(f"| {r['domain']} | {r['variant']} | {r['updates']} | {r['termination_reason']} | {r['relative_compliance_difference']:+.5g} | {r['volume_excess']:+.5g} | {r['projected_gradient_inf']:.5g} | {r['total_bisection_iterations']} | {r['runtime_seconds']:.4g} |")
    lines+=['','Optional optimizer overrides live explicitly in run metadata.variant while problem.json retains the canonical baseline problem specification; variants are separate experiments, not baseline trajectories. The feasible-upper choice removes positive volume excess at the tested target, whereas tightening alone does not mathematically guarantee strict feasibility. Trajectory effects, including different local minima or stopping points, are compared rather than assumed negligible.','',
        '## Checkpoint reliability','',
        f"All {len(replays)} M3.1 checkpoint continuations (10,25,50,75 per domain) passed complete-suffix x/rho/C/physical-gradient/design-gradient and OC-change comparisons, stopping-index/reason checks and binary final-layout checks. The saved incoming change and absolute iteration are required. Budget is explicitly extended from M3.1's 100 to the common reference cap, not reset at restart. Tolerances are relative 1e-8 plus dimension-scaled float64 roundoff; source and BLAS/MKL settings are recorded. Cross-platform bitwise identity is not claimed.",'',
        '## Recommendations and limits','']
    for tol in (.01,.005):
        selected=[r for r in rows if r['tolerance']==tol]
        lines.append(f"Tolerance {tol:g}: provisional acceptance on {sum(r['provisional_candidate_acceptance'] for r in selected)}/{len(selected)} tested problems.")
    good01=all(r['provisional_candidate_acceptance'] for r in rows if r['tolerance']==.01)
    good005=all(r['provisional_candidate_acceptance'] for r in rows if r['tolerance']==.005)
    candidate=.01 if good01 else .005 if good005 else None
    lines+=['',f"Default candidate for intermediate-dynamics training trajectories: {candidate if candidate is not None else 'no relaxed universal default accepted by the declared targets; retain .001 pilot reference stopping and explicit quality flags'}. Even an accepted relaxed endpoint omits measured late density evolution; keep a controlled subset of strict full trajectories rather than train dynamics only on truncated examples.",
        'For high-quality reference labels, use .001 as the stricter stopping target together with stationarity, feasibility and layout diagnostics. A .001 crossing alone is not a proof of optimality or a high-quality label. No global dataset defaults are changed by this study.',
        'Maximum iteration policy: start at 500 updates. If exhausted, label maxiter, inspect history/cost/stationarity and extend explicitly only when justified. This entry point permits one documented extension to 1000 for a cheap deterministic case; no automatic unbounded loops or convergence relabeling.',
        'OC volume enforcement: retain the verified baseline for historical replay. Prefer an explicitly opted-in feasible-upper endpoint with tighter 1e-8 absolute width when strict numerical feasibility is required, subject to the reported trajectory effects and further problem-level validation. Tightening alone is insufficient to guarantee strict feasibility. Do not silently replace the existing helper.',
        'Only three deterministic cantilevers and thresholded connectivity proxies are tested. Training intermediate dynamics and labeling approximate final optima are different tasks; omitted update fraction and measured late evolution matter even when C changes little. No neural network, large random dataset, M3.3, HPC submission, commit or push occurred.','',
        f"Protected files unchanged: {metadata['protected_file_count']}. Numerical invariants, replay details, complete raw histories, source snapshots and reproducibility metadata are provided. Focused/full test logs and the final audit record actual test counts and times."]
    (out/'M3_2_CONVERGENCE_REPORT.md').write_text('\n'.join(lines)+'\n')


def late_evolution(out,entries):
    """Read-only diagnostics for omitted dynamics, including capped strict tails."""
    import csv
    rows=[]
    for entry in entries:
        p,a,m=load_trajectory(entry['folder']);shape=(p.nely,p.nelx)
        stops={tol:first_stopping_update(a['update_change'],tol) for tol in TOLERANCES}
        available=[tol for tol in TOLERANCES if stops[tol] is not None]
        primary=stops[min(available)] if available else len(a['x'])-1
        for tol in TOLERANCES:
            start=stops[tol] if stops[tol] is not None else len(a['x'])-1
            for label,end in [('strictest_attained_tolerance',primary),('full_recorded_strict_history',len(a['x'])-1)]:
                if end<start:continue
                tail=a['rho'][start:end+1];changes=np.diff(tail,axis=0)
                td=[topology_diagnostics(r,shape) for r in tail]
                initial=td[0]
                row=dict(domain=entry['domain'],tolerance=tol,horizon=label,start_iteration=start,end_iteration=end,
                    horizon_termination_reason=m['termination_reason'] if label=='full_recorded_strict_history' else 'change_tolerance' if available else 'maxiter',
                    omitted_updates=end-start,remaining_relative_compliance_improvement=float((a['compliance'][start]-a['compliance'][end])/a['compliance'][end]),
                    remaining_rho_mean_total_variation=float(np.abs(changes).sum()/p.n),
                    maximum_rho_excursion_from_relaxed=float(np.max(abs(tail-tail[0]))),
                    maximum_remaining_rho_step=float(np.max(abs(changes))) if len(changes) else 0.,
                    maximum_remaining_x_step=float(np.max(a['update_change'][start:end])) if end>start else 0.,
                    component_count_min=min(d['component_count'] for d in td),component_count_max=max(d['component_count'] for d in td),
                    states_with_changed_component_count=sum(d['component_count']!=initial['component_count'] for d in td),
                    states_with_changed_loaded_path=sum(d['loaded_path']!=initial['loaded_path'] for d in td),
                    maximum_binary_rho_disagreement=float(np.max(np.mean((tail>.5)!=(tail[0]>.5),axis=1))))
                rows.append(row)
    write_csv(out/'late_evolution_summary.csv',rows)
    report=out/'M3_2_CONVERGENCE_REPORT.md'
    text=report.read_text().split('\n## Remaining dynamics and capped-reference distinction')[0]
    text+='\n## Remaining dynamics and capped-reference distinction\n\n'
    capped=[f"{e['domain']} ({e['updates']} updates)" for e in entries if e['termination_reason']=='maxiter']
    text+='The primary comparison uses the strictest attained change tolerance separately for each domain. Capped strict baseline histories: '+(', '.join(capped) if capped else 'none')+'. A capped strict history supplies no converged 0.001 reference. Savings against an attained 0.005 reference must not be presented as savings against a converged 0.001 solution.\n\n'
    text+='Late evolution is measured over every omitted state, including transient component/path changes and accumulated physical-density variation. A small endpoint compliance difference does not imply that the omitted dynamics are redundant. `late_evolution_summary.csv` distinguishes the attained reference from the full recorded strict history, including capped histories.\n\n'
    text+='| Domain | Tolerance | Horizon | Omitted updates | Remaining C improvement | rho mean total variation | Max rho excursion | Component range | Changed path states |\n|---|---:|---|---:|---:|---:|---:|---|---:|\n'
    for r in rows:
        if r['tolerance'] in (.01,.005):
            text+=f"| {r['domain']} | {r['tolerance']:g} | {r['horizon']} | {r['omitted_updates']} | {r['remaining_relative_compliance_improvement']:.3%} | {r['remaining_rho_mean_total_variation']:.5g} | {r['maximum_rho_excursion_from_relaxed']:.5g} | {r['component_count_min']}–{r['component_count_max']} | {r['states_with_changed_loaded_path']} |\n"
    audit=out/'M3_2_FINAL_AUDIT.md'
    if audit.exists():text+='\n'+audit.read_text()
    report.write_text(text)


def report_saved(out,entries):
    """Rebuild documentation and tail diagnostics; no FEM or optimization."""
    import csv
    def read(name):
        result=[]
        for row in csv.DictReader((out/(name+'.csv')).open()):
            converted={}
            for k,v in row.items():
                if v in ('True','False'):converted[k]=v=='True'
                elif v=='':converted[k]=None
                else:
                    try:converted[k]=float(v) if any(c in v for c in '.eE') else int(v)
                    except ValueError:converted[k]=v
            result.append(converted)
        return result
    meta=json.loads((out/'reproducibility.json').read_text())
    write_report(out,read('convergence_summary'),read('stationarity_summary'),read('checkpoint_replay_summary'),read('oc_bisection_study'),read('optional_oc_trajectory_comparison'),meta)
    late_evolution(out,entries)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['all','references','analyze','report'])
    parser.add_argument('--output',type=Path,default=ROOT/'m3_2')
    parser.add_argument('--reference-manifest',type=Path)
    parser.add_argument('--maxiter',type=int,default=500)
    args=parser.parse_args();protected_check();out=args.output
    if (args.command!='report' and (out/'M3_2_CONVERGENCE_REPORT.md').exists()) or (args.command not in ('analyze','report') and (out/'reference_manifest.json').exists()):out=out/'generations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=True)
    if args.command!='report':source_snapshot(out,provenance())
    if args.command in ('all','references'):entries=generate_references(out,args.maxiter)
    else:
        if args.reference_manifest is None and args.command=='report':args.reference_manifest=out/'reference_manifest.json'
        if args.reference_manifest is None:parser.error('analyze requires --reference-manifest')
        entries=json.loads(args.reference_manifest.read_text())
    if args.command=='report':report_saved(out,entries)
    elif args.command!='references':analyze(out,entries)
    protected_check();print('Output:',out,flush=True)

if __name__=='__main__':main()
