#!/usr/bin/env python
"""Small mechanically consistent M3.1 family; independent saved-state validation."""
import argparse
import csv
from datetime import datetime,timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from topopt_utils.fea.mechanical_problem import cantilever_problem,SCHEMA_VERSION,array_digest
from topopt_utils.fea.consistent_trajectory import generate_trajectory,save_trajectory,load_trajectory,representative_indices
from topopt_utils.fea.trajectory_validation import validate_trajectory,reconstruct_interaction,derivative_checks,TOLERANCES
from topopt_utils.fea.solver import FeaSolver
from analyze_m2_spectral import serial,write_csv,sha256

ROOT=Path(__file__).resolve().parent


def verify_protected():
    snapshot=json.loads((ROOT/'m3_1/protected_before.json').read_text())
    changed=[name for name,digest in snapshot.items() if not (ROOT/name).exists() or sha256(ROOT/name)!=digest]
    if changed:raise RuntimeError(f'Protected files changed: {changed}')
    return len(snapshot)


def provenance():
    names=['run_m3_1.py','topopt_utils/fea/mechanical_problem.py','topopt_utils/fea/consistent_trajectory.py','topopt_utils/fea/trajectory_validation.py','topopt_utils/fea/mechanical_audit.py',
        'topopt_utils/fea/solver.py','topopt_utils/fea/q4.py','topopt_utils/topopt_2d.py','tests/test_consistent_trajectory.py']
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),source_fingerprints={n:sha256(ROOT/n) for n in names if (ROOT/n).exists()},
        versions={n:version(n) for n in ('numpy','scipy','matplotlib','pytest','pypardiso')},python=sys.version,
        thread_environment={n:os.environ.get(n) for n in ('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYPARDISO_MKL_RT')},
        family='M1 axial single-nodal-load left-clamped cantilever; dx=dy=1, physical domain scales with element counts',random_seeds=dict(problem=0,random=10,smooth=14,localized=16,stiffness_probes=701))


def schema_document():
    return dict(problem_schema=SCHEMA_VERSION,trajectory_schema='m3.1.trajectory.v1',problem_storage='canonical problem.json; all arrays cross-checked against geometry/loads derived from authoritative settings',
        trajectory_storage='trajectory.npz + trajectory.json, float64 numerical arrays, int64 iteration; no pickle execution; problem data saved once per trajectory',
        definitions=dict(x='OC design variables; volume constraint on mean(x)',rho='filtered physical density F x; no projection',u='row-major nodal DOFs alternating x/y',
            g_rho='physical-density compliance gradient',g_x='F.T g_rho',P='physical-density mechanical term A.T solve(K,A)',H='H_rho=2P-diag(D); H_x=F.T H_rho F',
            k_probe_actions='Dirichlet K applied to two deterministic ndof-dimensional probes, not full K storage',previous_change='incoming transition metric; NaN at initial state',update_change='outgoing OC metric for each transition, T-1 values'),
        shapes=dict(x=['T','nely*nelx'],rho=['T','nely*nelx'],u=['T','2*(nelx+1)*(nely+1)'],g_rho=['T','N'],g_x=['T','N'],compliance=['T'],solver_returned=['T'],solver_finite=['T'],
            k_diagonal=['T','ndof'],k_probe_actions=['T','ndof',2],k_frobenius=['T'],iteration=['T'],update_change=['T-1'],previous_change=['T']),
        array_types=dict(numerical='float64',iteration='int64',solver_flags='Boolean',initial_previous_change='NaN sentinel; all later transition values finite'),
        filter_jacobian='F=diag(1/Hs)G, Hs=G*1; rho=F*x, g_x=F.T*g_rho, H_x=F.T*H_rho*F; projection inactive',
        bounds=dict(x=[.001,1.],rho='same bounds by convex normalized filter, accepting only float64 roundoff at bounds'),
        units='Consistent arbitrary length/force units; E0=1, plane stress, implicit unit thickness; compliance in force*displacement; densities dimensionless',
        timing='State t before update; terminal x after last update separately evaluated. T = completed_updates+1.',
        continuation='No hidden OC history. Need x, completed updates, previous change and settings; cross-library bitwise continuation not guaranteed.',tolerances=TOLERANCES)


def choose_output(base,marker='M3_1_VALIDATION_REPORT.md'):
    base=Path(base)
    if (base/marker).exists():base=base/'generations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    base.mkdir(parents=True,exist_ok=True)
    return base


def generate_family(out,meshes,maxiter=100,tolerance=.001):
    provenance_data=provenance();entries=[]
    for name,digest in provenance_data['source_fingerprints'].items():
        source=ROOT/name
        if sha256(source)!=digest:raise RuntimeError('Source changed during provenance capture')
        target=out/'source_snapshots/generation'/(name+'.txt')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    (out/'trajectories').mkdir(exist_ok=True)
    for nx,ny in meshes:
        p=cantilever_problem(nx,ny,maxiter=maxiter,tolerance=tolerance)
        print(f'Generating {nx}x{ny}, domain {p.width:g}x{p.height:g}',flush=True)
        arrays,meta=generate_trajectory(p)
        folder=save_trajectory(out/'trajectories'/f'{nx}x{ny}',p,arrays,meta,provenance_data)
        entries.append(dict(domain=f'{nx}x{ny}',problem_id=p.problem_id,folder=str(folder.resolve()),state_count=meta['state_count'],completed_updates=meta['completed_updates'],termination_reason=meta['termination_reason'],generation_seconds=meta['generation_seconds']))
    (out/'trajectory_manifest.json').write_text(json.dumps(dict(schema_version='m3.1.manifest.v1',trajectories=entries,provenance=provenance_data),indent=2)+'\n')
    (out/'problem_schema.json').write_text(json.dumps(schema_document(),indent=2)+'\n')
    return [Path(e['folder']) for e in entries]


def validate_family(folders,out,with_interactions=True,with_derivatives=True):
    from topopt_utils.fea.mechanical_audit import fallback_mismatch_comparison
    mismatch_rows=[]
    comparisons=[];histories=[];interaction_rows=[];gradients=[];hvps=[];skips=[];summaries=[]
    start=perf_counter()
    for folder in folders:
        p,a,m=load_trajectory(folder);domain=f'{p.nelx}x{p.nely}';solver=FeaSolver()
        print(f'Validating {domain}: {len(a["x"])} saved states',flush=True)
        try:
            rows,history,summary=validate_trajectory(p,a,m,solver)
            comparisons.extend(dict(domain=domain,**r) for r in rows);histories.extend(dict(domain=domain,**r) for r in history)
            if with_interactions:
                mismatch_rows.extend(dict(domain=domain,rho_state=0,**r) for r in fallback_mismatch_comparison(p,a['rho'][0],solver))
            summary.update(domain=domain,problem_id=p.problem_id,volume_fraction_target=p.volume_fraction,trajectory_folder=str(folder.resolve()),selected_iterations=representative_indices(len(a['x'])),final_design_volume=float(a['design_volume'][-1]),final_physical_volume=float(a['physical_volume'][-1]),
                final_compliance=float(a['compliance'][-1]),initial_compliance=float(a['compliance'][0]),final_volume_bisection_uncertainty=float(history[-2]['oc_volume_uncertainty']),final_change=float(a['update_change'][-1]),legacy_last_evaluated_compliance=float(a['compliance'][-2]),terminal_vs_legacy_compliance_relative=float((a['compliance'][-1]-a['compliance'][-2])/a['compliance'][-2]),terminal_rho_change_norm=float(np.linalg.norm(a['rho'][-1]-a['rho'][-2])))
            for t in summary['selected_iterations'] if with_interactions else []:
                print(f'  exact interactions/derivatives state {t}',flush=True)
                matrix,diag,hessian,diagnostic=reconstruct_interaction(p,a,t,solver)
                target=out/'interactions'/domain;target.mkdir(parents=True,exist_ok=True)
                file=target/f'state_{t:04d}.npz'
                if file.exists():raise FileExistsError(f'Refusing interaction overwrite: {file}')
                np.savez_compressed(file,P=matrix,D=diag,H_rho=hessian,rho=a['rho'][t],x=a['x'][t],iteration=t,problem_id=p.problem_id)
                (file.with_suffix('.json')).write_text(json.dumps(serial(dict(problem_id=p.problem_id,space='physical rho',rho_fingerprint=array_digest(a['rho'][t]),x_fingerprint=array_digest(a['x'][t]),source_trajectory=str(folder.resolve()),diagnostics=diagnostic)),indent=2)+'\n')
                interaction_rows.append(dict(domain=domain,**diagnostic))
                if with_derivatives:
                    gs,hs,ss=derivative_checks(p,a,t,matrix,diag,solver)
                    gradients.extend(dict(domain=domain,**r) for r in gs);hvps.extend(dict(domain=domain,**r) for r in hs);skips.extend(dict(domain=domain,**r) for r in ss)
            summaries.append(summary);plot_history(p,a,out)
        except Exception as exc:
            if hasattr(exc,'gradient_rows'):
                write_csv(out/f'{domain}_failed_gradient_verification.csv',exc.gradient_rows)
                write_csv(out/f'{domain}_failed_hessian_verification.csv',exc.hvp_rows)
            (out/f'{domain}_failure.json').write_text(json.dumps(dict(domain=domain,error=repr(exc),trajectory_folder=str(folder.resolve())),indent=2)+'\n')
            raise
    for name,rows in [('legacy_mismatch_effects',mismatch_rows),('reconstruction_comparison',comparisons),('trajectory_history',histories),('interaction_reconstruction',interaction_rows),('gradient_verification',gradients),('hessian_vector_verification',hvps)]:
        if rows:write_csv(out/(name+'.csv'),rows)
    (out/'skipped_derivative_checks.json').write_text(json.dumps(serial(skips),indent=2)+'\n')
    summary=dict(schema_version='m3.1.validation.v1',problems=summaries,provenance=provenance(),tolerances=TOLERANCES,protected_file_count=verify_protected(),protected_files_unchanged=True,
        validation_seconds=perf_counter()-start,gradient_step_rows=len(gradients),hvp_step_rows=len(hvps),interaction_states=len(interaction_rows),
        optimization_converged=all(s['termination_reason']=='change_tolerance' for s in summaries),
        validation_scope='all saved FEM states/transitions'+ (' + exact interactions and both-space FD derivatives' if with_interactions else '; no interaction/derivative recomputation in this command'),
        fd_gradient_individually_failed_steps=sum(not r['passed'] for r in gradients),fd_hvp_individually_failed_steps=sum(not r['passed'] for r in hvps),
        fd_acceptance='At least two of four distinct step sizes must satisfy error+roundoff tolerances in every tested direction and space; every coarse/fine failure is retained, not filtered.',
        skipped_checks=skips,full_acceptance=with_interactions and with_derivatives and len(summaries)==3 and {s['domain'] for s in summaries}=={'12x6','24x12','48x24'})
    (out/'validation_summary.json').write_text(json.dumps(serial(summary),indent=2,allow_nan=False)+'\n')
    (out/'problem_schema.json').write_text(json.dumps(schema_document(),indent=2)+'\n')
    write_report(out,summary,comparisons,histories,interaction_rows,gradients,hvps)
    return summary


def plot_history(p,a,out):
    domain=f'{p.nelx}x{p.nely}';t=a['iteration'];fig,axes=plt.subplots(2,2,figsize=(10,6))
    axes[0,0].plot(t,a['compliance']);axes[0,0].set_ylabel('Compliance')
    axes[0,1].plot(t,a['design_volume'],label='mean x');axes[0,1].plot(t,a['physical_volume'],label='mean rho');axes[0,1].axhline(p.volume_fraction,color='k',ls='--');axes[0,1].legend();axes[0,1].set_ylabel('Volume')
    axes[1,0].semilogy(t[1:],a['previous_change'][1:]);axes[1,0].axhline(p.tolerance,color='k',ls='--');axes[1,0].set_ylabel('Incoming max design change')
    axes[1,1].semilogy(t,a['residual_relative_force']);axes[1,1].set_ylabel('Free equilibrium relative residual')
    for ax in axes.flat:ax.set_xlabel('Completed OC updates / evaluated state')
    fig.suptitle(domain);fig.tight_layout();fig.savefig(out/f'{domain}_history.png',dpi=150);fig.savefig(out/f'{domain}_history.pdf');plt.close(fig)
    indices=representative_indices(len(t));fig,axes=plt.subplots(1,len(indices),figsize=(3*len(indices),3),squeeze=False)
    for ax,i in zip(axes[0],indices):
        ax.imshow(a['rho'][i].reshape(p.nely,p.nelx),origin='lower',extent=(0,p.width,0,p.height),vmin=0,vmax=1,cmap='gray_r');ax.set_title(f't={i}');ax.set_xlabel('x');ax.set_ylabel('y')
    fig.tight_layout();fig.savefig(out/f'{domain}_density_evolution.png',dpi=150);fig.savefig(out/f'{domain}_density_evolution.pdf');plt.close(fig)


def write_report(out,s,comparisons,histories,interactions,gradients,hvps):
    lines=['# M3.1 — Mechanically consistent trajectories and exact interactions','',
        'The old mismatch arose in orchestration: unrelated trajectories were selected by mesh shape, or fallback optimization used a 1×2 domain and four-node element loads, while exact interaction evaluation used dx=dy=1 and a single nodal load. The numerical FEM/SIMP/HVP formulation itself is preserved. See `M3_1_PIPELINE_AUDIT.md` at the M3.1 root for source-level discrepancies and corrections.','',
        'The new canonical specification is used throughout optimization, serialization, reconstruction and interactions. Complete geometry, connectivity, supports, nodal loads/full force vector, material/filter/OC settings and source fingerprints are saved once per problem. Arrays and numerical reconstruction are checked independently; hashes alone are not validation.','',
        'The family matches the verified M1 axial cantilever: left-edge x/y constraints and a unit +x load at the right-middle node. Domains are 12×6,24×12,48×24 with dx=dy=1. These are domain-size changes, not mesh refinements. Seed zero is provenance; this deterministic family has no randomized problem sampling.','',
        '## Actual trajectories','','| Domain | Evaluated states | OC updates | Stop reason | Initial C | Final C | Final mean x | Final mean rho | Final change |','|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for p in s['problems']:
        lines.append(f"| {p['domain']} | {p['completed_updates']+1} | {p['completed_updates']} | {p['termination_reason']} | {p['initial_compliance']:.9g} | {p['final_compliance']:.9g} | {p['final_design_volume']:.9g} | {p['final_physical_volume']:.9g} | {p['final_change']:.4g} |")
    lines+=['','States are evaluated before their outgoing OC update; the terminal updated x is additionally evaluated without changing any update or stopping rule. Relative to the legacy return value this adds one labeled terminal evaluation, rather than attaching the updated x to an old u/C/gradient. The old numerical routines remain unchanged. OC constraints mean(x), not mean(rho), and finite multiplier bisection can leave a small signed budget discrepancy; it is reported below. No monotonic compliance constraint is imposed.','',
        'Terminal versus legacy last-evaluated compliance (extra evaluation only): '+ '; '.join(f"{p['domain']} {p['terminal_vs_legacy_compliance_relative']:+.6g} relative" for p in s['problems']), '', '## Numerical reconstruction and equilibrium','','| Domain | Worst saved-u relative discrepancy | Worst saved-gradient discrepancy | Max free residual / force | Max backward residual | Max energy discrepancy | Final volume excess |','|---|---:|---:|---:|---:|---:|---:|']
    for p in s['problems']:
        domain=p['domain'];cr=[r for r in comparisons if r['domain']==domain];hr=[r for r in histories if r['domain']==domain]
        get=lambda q:max(r['relative_error'] for r in cr if r['quantity']==q)
        lines.append(f"| {domain} | {get('u'):.3g} | {get('solver_gradient'):.3g} | {max(r['residual_relative_force'] for r in hr):.3g} | {max(r['residual_backward_error'] for r in hr):.3g} | {max(r['energy_compliance_relative'] for r in hr):.3g} | {hr[-1]['volume_excess']:.6g} |")
    lines+=['','Independent COO element assembly checks every stiffness entry against the solver Dirichlet matrix at every state. Saved diagonal, norm and two operator probes provide numerical stiffness-state checks without storing K for every iteration. Unconstrained assembly provides support reactions, global force/moment balance and u.T K u versus f.T u. All saved states and OC transitions are replay-checked, not only the selected interaction states.','',
        '## Exact interactions','','| Domain | State t | P raw minimum eigenvalue | P symmetry error | RHS solve residual | P/D HVP identity worst relative error |','|---|---:|---:|---:|---:|---:|']
    for r in interactions:
        identity=max((v['identity_relative_error'] for v in hvps if v['domain']==r['domain'] and v['iteration']==r['iteration']),default=np.nan)
        lines.append(f"| {r['domain']} | {r['iteration']} | {r['raw_minimum_eigenvalue']:.4g} | {r['symmetry_relative_error']:.3g} | {r['rhs_equilibrium_relative_error']:.3g} | {identity:.3g} |")
    lines+=['','P and H are stored unmodified. Only accepted tiny asymmetry is averaged for the diagnostic symmetric eigenvalue solve; raw symmetry/minimum/count/tolerance are recorded. No PSD clipping is applied. The existing dense-memory guard is honored. P=A.T solve(K,A), D=s\'\'(rho)*element strain energy and H_rho=2P-diag(D) hold for the unchanged SIMP law s=.001+.999*rho^p. No explicit inverse is formed.','',
        '## Independent derivative verification','','| Domain | Space | Families checked | Direction/state checks | Worst best-step gradient relative error | Worst best-step HVP relative error |','|---|---|---|---:|---:|---:|']
    for p in s['problems']:
        for space in ('rho','x'):
            gs=[r for r in gradients if r['domain']==p['domain'] and r['space']==space];hs=[r for r in hvps if r['domain']==p['domain'] and r['space']==space]
            groups=sorted({(r['iteration'],r['family']) for r in gs})
            bestg=[min((r['relative_error'] for r in gs if (r['iteration'],r['family'])==key and np.isfinite(r['relative_error'])),default=np.nan) for key in groups]
            besth=[min(r['relative_error'] for r in hs if (r['iteration'],r['family'])==key) for key in groups]
            if gs:lines.append(f"| {p['domain']} | {space} | {', '.join(sorted({r['family'] for r in gs}))} | {len(groups)} | {np.nanmax(bestg):.3g} | {max(besth):.3g} |")
    lines+=['','Four centered step sizes are retained per feasible direction, along with absolute/relative errors, reference norms, tolerances and roundoff allowances. At least two step sizes must pass per direction and variable space; the table summarizes best steps but acceptance does not rely on one arbitrary step. Initial/early states have unrestricted directions; boundary states explicitly restrict directions to entries farther than .001 from bounds. Empty/zero restricted probes are listed in `skipped_derivative_checks.json`. Density perturbations and x perturbations are tested separately. Actual state changes use rho differences in rho space and x differences in design space.','',
        f"Individual FD steps outside tolerance: gradient {s['fd_gradient_individually_failed_steps']}, HVP {s['fd_hvp_individually_failed_steps']}. These observations remain in the CSV, including coarse-step truncation and fine-step cancellation behavior.",'',
        '## Variable interpretation and acceptance','',
        'No projection is active. rho=F x uses the existing boundary-renormalized filter; F is not symmetric at boundaries despite a symmetric kernel. g_x=F.T g_rho and H_x=F.T H_rho F. P_rho is the physical-density mechanical term; it is not the full compliance Hessian, and not automatically a filtered design-space operator. A future nonlinear projection requires an additional mapping-curvature term, which is not implemented here.','',
        'Numerical thresholds are quantity-specific and recorded in `problem_schema.json`/`validation_summary.json`: relative reconstruction 1e-9, stiffness/filter 1e-12, replay 1e-8, force-relative equilibrium 1e-7 plus backward residual 1e-12, energy 1e-8, symmetry 1e-10. PSD tolerance is 100*N*eps*max(abs(eigenvalues)). Gradient/HVP FD relative allowances are 3e-5/2e-3 plus step-dependent float64 cancellation allowances. All relative criteria scale with the physical/reference quantities rather than a unit-sized absolute floor; residual force normalization and backward error distinguish conditioning from solve error.','',
        f"Mechanical validation scope complete for all three requested configurations: {s['full_acceptance']}. Optimization convergence is a separate status: {all(p['termination_reason']=='change_tolerance' for p in s['problems'])}. Protected source/artifact fingerprints unchanged: {s['protected_file_count']}. Reconstruction, derivative and transition checks are actual numerical checks, not inferred from test collection. Test counts/times are recorded separately in the final audit and test logs.",'',
        'Warnings and limits:','']
    for p in s['problems']:
        for warning in p['warnings']:lines.append(f"- {p['domain']}: {warning}")
    lines+=['- Historical M1–M2.4 trajectories remain mismatched; this phase creates new consistent data and does not retroactively certify old optimizer curvature.',
        '- These three deterministic cantilevers are correctness fixtures, not a diverse training corpus or a held-out generalization benchmark.',
        '- Physical mean rho may differ from the design budget; finite OC bisection effects are retained. Filter radius is in element-count units; no fixed-domain refinement claim is made.',
        '- Dense exact matrices are restricted to selected snapshots and existing memory guards; native MKL allocations and analysis temporaries exceed nominal dense arrays alone.',
        '- No learned model, optimizer-improvement claim, randomized large dataset, HPC job or M3.2 experiment is implemented.','',
        '## Scientific conclusions and next phase','',
        '1. The original mismatch was caused by geometry/load definitions and shape-only trajectory reuse in the orchestration layer, not evidence of a faulty verified interaction formula.',
        '2. A canonical saved specification and aligned x/rho/FEM recorder eliminate those differences for the new family, with unchanged FEM, filter and OC routines.',
        '3. New saved trajectories are self-consistent when every state/transition reconstruction check above passes; termination and volume warnings remain visible.',
        '4. Exact P/D/H can be reconstructed exclusively from saved metadata and rho, with no optimizer rerun or inverse assembly.',
        '5–6. Where derivative rows are present, centered compliance-gradient and gradient-HVP checks in both variable spaces validate the analytical derivatives over multiple feasible steps; P/D actions independently match the matrix-free HVP.',
        '7. Filtering changes derivative space through F.T and F; no projection is active. Physical P/H labels remain distinct from design-space derivatives.',
        '8. Terminal states are not converged optimum labels: all three recorded runs reached maxiter. These records are suitable as mechanically verified pilot fixtures for future ML data pipelines, not sufficient by themselves for training/generalization claims. Data splits must group complete problems/trajectories and align the ML filter with this specification.',
        '9. Remaining limitations are deterministic-problem diversity, approximate OC volume enforcement, dense interaction scaling, legacy artifact provenance, and platform-dependent bitwise behavior.',
        '10. M3.2 should first investigate change convergence and OC volume-bisection sensitivity on these consistent problems, preserving explicit tolerance/termination labels. Then benchmark spatial versus local-plus-modal approximations on these consistent trajectories, then assess H_x actions and actual optimization rollouts at matched storage/application/amortized construction budgets. Problem-level held-out cases and independent physical benchmark tests should precede architectural superiority claims. This is a recommendation only; M3.2 is not started.']
    (out/'M3_1_VALIDATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    append_detailed_report(out,s,gradients,hvps)


def append_detailed_report(out,s,gradients,hvps):
    lines=['','## Worst observed finite-difference steps','',
        'All steps are retained, including coarse-step failures. Passing the derivative checks requires at least two acceptable distinct steps for every direction/space/state.','',
        '| Domain | Space | Gradient absolute | Gradient relative | HVP absolute | HVP relative |','|---|---|---:|---:|---:|---:|']
    for problem in s['problems']:
        for space in ('rho','x'):
            gs=[r for r in gradients if r['domain']==problem['domain'] and r['space']==space]
            hs=[r for r in hvps if r['domain']==problem['domain'] and r['space']==space]
            if not gs:continue
            gr=max((r['relative_error'] for r in gs if np.isfinite(r['relative_error'])),default=np.nan)
            lines.append(f"| {problem['domain']} | {space} | {max(r['absolute_error'] for r in gs):.4g} | {gr:.4g} | {max(r['absolute_error'] for r in hs):.4g} | {max(r['relative_error'] for r in hs):.4g} |")
    mismatch_file=out/'legacy_mismatch_effects.csv'
    if mismatch_file.exists():
        lines+=['','## Controlled quantification of the legacy mismatch','',
            'The same initial uniform rho is evaluated under four separately specified mechanics cases. This isolates geometry and load representation effects; it does not reconstruct unknown historical optimizer problems or regenerate old datasets.','',
            '| Domain | Mechanics case | Compliance | Ratio to consistent M1 |','|---|---|---:|---:|']
        with mismatch_file.open() as stream:
            for r in csv.DictReader(stream):
                lines.append(f"| {r['domain']} | {r['case']} | {float(r['compliance']):.7g} | {float(r['compliance_ratio_to_consistent']):.6g} |")
    lines+=['','## Final OC budget resolution','',
        '| Domain | Signed final mean(x) excess | Last multiplier-bisection volume bracket width |','|---|---:|---:|']
    for p in s['problems']:
        lines.append(f"| {p['domain']} | {p['final_design_volume']-p.get('volume_fraction_target',.5):+.6g} | {p['final_volume_bisection_uncertainty']:.6g} |")
    lines+=['',
        'The target lies within every saved-transition replay multiplier bracket. The final volume error is retained rather than corrected. Small positive excesses mean the strict budget is not exactly satisfied, despite correct replay of the unchanged OC algorithm. Physical rho is not the constrained variable.','',
        'Per-step failures are retained and quantified rather than discarded.']
    failed_sizes={}
    for row in gradients:
        if row['passed'] is False or row['passed']=='False':
            failed_sizes[row['step']]=failed_sizes.get(row['step'],0)+1
    lines.append('Gradient FD failures by step size: '+str(failed_sizes)+'. HVP failures: '+str(sum(row['passed'] is False or row['passed']=='False' for row in hvps))+'.')
    with (out/'M3_1_VALIDATION_REPORT.md').open('a') as stream:stream.write('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['all','generate','validate','interactions','report'])
    parser.add_argument('--output',type=Path,default=ROOT/'m3_1')
    parser.add_argument('--trajectory',type=Path,action='append',help='Existing trajectory folder; may be repeated')
    parser.add_argument('--manifest',type=Path,help='Existing trajectory_manifest.json')
    parser.add_argument('--meshes',default='12x6,24x12,48x24')
    parser.add_argument('--maxiter',type=int,default=100);parser.add_argument('--tolerance',type=float,default=.001)
    args=parser.parse_args();verify_protected()
    if args.command=='report':
        # Reporting reads only saved numerical results; never invokes FEM/optimization.
        summary=json.loads((args.output/'validation_summary.json').read_text())
        def read(name):
            file=args.output/(name+'.csv')
            if not file.exists():return []
            rows=list(csv.DictReader(file.open()))
            for row in rows:
                for key,value in row.items():
                    try:row[key]=float(value) if any(c in value.lower() for c in '.en') else int(value)
                    except ValueError:pass
            return rows
        target=choose_output(args.output)
        if target!=args.output:
            files=list(args.output.glob('*.csv'))
            files.extend(args.output/name for name in ('validation_summary.json','problem_schema.json','skipped_derivative_checks.json') if (args.output/name).exists())
            for file in files:shutil.copyfile(file,target/file.name)
            (target/'report_source.json').write_text(json.dumps(dict(results_source=str(args.output.resolve()),input_fingerprints={p.name:sha256(p) for p in files}),indent=2)+'\n')
        write_report(target,summary,read('reconstruction_comparison'),read('trajectory_history'),read('interaction_reconstruction'),read('gradient_verification'),read('hessian_vector_verification'))
        print(target/'M3_1_VALIDATION_REPORT.md');return
    marker='trajectory_manifest.json' if args.command=='generate' else 'M3_1_VALIDATION_REPORT.md'
    out=choose_output(args.output,marker)
    if args.command in ('all','generate'):
        meshes=[tuple(map(int,t.split('x'))) for t in args.meshes.split(',')]
        folders=generate_family(out,meshes,args.maxiter,args.tolerance)
    else:
        folders=args.trajectory or []
        if args.manifest:folders.extend(Path(e['folder']) for e in json.loads(args.manifest.read_text())['trajectories'])
        if not folders:parser.error('Specify --trajectory or --manifest for independent validation')
    if args.command!='generate':
        summary=validate_family(folders,out,with_interactions=args.command!='validate',with_derivatives=args.command!='validate')
        print(json.dumps(serial(dict(output=str(out),full_acceptance=summary['full_acceptance'],problems=len(summary['problems']),protected_files_unchanged=True)),indent=2))
    verify_protected()

if __name__=='__main__':main()
