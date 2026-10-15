"""Post-generation native-state timing and documentation audit; no OC generation."""
import json,sys,resource,platform,shutil
from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import numpy as np
from topopt_utils.fea.trajectory_dataset import Trajectory,file_hash
from topopt_utils.fea.dataset_storage import atomic_json
from topopt_utils.fea.solver import FeaSolver
from run_m3_3c import fingerprints,filter_audit,specification,memory_status

def main(out):
 rows=json.loads((out/'bounded_trajectories.json').read_text());records=[];worst={}
 for r in rows:
  tr=Trajectory(out/r['folder']);p=tr.problem;solver=FeaSolver();stages=[]
  for index in (0,tr.metadata['state_count']-1):
   times=[]
   for _ in range(5):
    c,g,t=solver.compliance(tr.arrays['rho'][index],*p.solver_args,sensitivities=True,profile=True);times.append(t)
   np.testing.assert_allclose(c,tr.arrays['compliance'][index],rtol=1e-10)
   np.testing.assert_allclose(g,tr.arrays['g_rho'][index],rtol=1e-10,atol=1e-11)
   stages.append(dict(iteration=index,samples=times,warm_medians={k:float(np.median([t[k] for t in times[1:]])) for k in times[0]}))
  for key in ('residual_relative_force','residual_backward_error','energy_compliance_relative','force_balance_relative','moment_balance_relative'):
   worst[key]=max(worst.get(key,0),float(tr.arrays[key].max()))
  records.append(dict(experiment=r['experiment'],mesh=r['mesh'],profiles=stages,initial_mean_rho=float(tr.arrays['rho'][0].mean()),terminal_mean_rho=float(tr.arrays['rho'][-1].mean()),raw_terminal_multiplier=r['oc_multipliers'][-1],mean_constraint_terminal_multiplier=p.n*r['oc_multipliers'][-1]))
 # Only filter/config operations at larger meshes; no larger FEM, features or OC.
 filters=[]
 for nx in (192,384):
  p,_,_=specification(nx,'A');filters.append(dict(mesh=[nx,nx//2],audit=filter_audit(p)))
 deps={}
 for name in ('numpy','scipy','pypardiso','torch','pytest','matplotlib'):
  try:deps[name]=version(name)
  except Exception:deps[name]='unavailable'
 atomic_json(out/'supplemental_audit.json',dict(native_state_profiles=records,mechanical_maxima=worst,larger_configuration_filter_checks=filters,dependency_versions=deps,memory=memory_status(),protected_files_unchanged=fingerprints(),source_sha256=file_hash(__file__),scope='Existing states only: repeated initial/terminal FEM profiles; 192/384 filter/config checks only; no extra optimization'))
 shutil.copyfile(__file__,out/'supplemental_audit.py.txt');shutil.copyfile(Path(__file__).with_name('replay_snapshot.py'),out/'replay_snapshot.py.txt')
 report=out/'M3_3C_MESH_SCALING_REPORT.md'
 lines=['','## Supplemental measured comparison','', 'This post-generation audit profiles the already saved initial/terminal states (one warm-up plus four retained repetitions); no extra OC updates. Exact supplemental source bytes and dependency versions are preserved separately. Native generation time includes setup, checkpoint hashing and diagnostics: the48 run is first/cold, while96 runs follow runtime initialization, so their generation-time ratio is not a clean production scaling benchmark.','', '| Case | Filter ms | OC ms | No-FEM features ms | Cached full features ms | Serialization s | Warm random slice us |','|---|---:|---:|---:|---:|---:|---:|']
 benches=[]
 for r in rows:
  b=json.loads((out/f"benchmark_{r['experiment']}_{r['mesh'][0]}.json").read_text());benches.append(b)
  lines.append(f"| {r['experiment']} {r['mesh']} | {b['filter']['median_seconds']*1000:.4g} | {b['oc']['median_seconds']*1000:.4g} | {b['features']['no_fem_v1']['median_seconds']*1000:.4g} | {b['features']['fresh_full_v1']['median_seconds']*1000:.4g} | {r['serialization_seconds']:.4g} | {r['random_access']['median_seconds']*1e6:.4g} |")
 lines+=['','| Case | Initial FEM API ms | Terminal FEM API ms | Terminal raw lambda | Terminal N*lambda | Terminal mean rho |','|---|---:|---:|---:|---:|---:|']
 for r in records:
  lines.append(f"| {r['experiment']} {r['mesh']} | {r['profiles'][0]['warm_medians']['total']*1000:.4g} | {r['profiles'][1]['warm_medians']['total']*1000:.4g} | {r['raw_terminal_multiplier']:.5g} | {r['mean_constraint_terminal_multiplier']:.5g} | {r['terminal_mean_rho']:.7g} |")
 base,fine=benches[:2];r0,r1=rows[:2]
 lines+=['',f"Fixed-domain48→96: DOFs {fine['ndof']/base['ndof']:.3g}×; upper structural nnz {fine['upper_nnz']/base['upper_nnz']:.3g}×; representative warm FEM API {fine['warm_median_seconds']['total']/base['warm_median_seconds']['total']:.3g}×; filter {fine['filter']['median_seconds']/base['filter']['median_seconds']:.3g}×; native21-state storage {r1['native_storage_bytes']/r0['native_storage_bytes']:.3g}×; full replay {r1['validation_seconds']/r0['validation_seconds']:.3g}×. Worker peak grows from {base['peak_rss_mib']:.2f} to {fine['peak_rss_mib']:.2f} MiB; import baselines dominate, so peak ratio cannot identify factor-memory scaling.", '', 'Both20-step fixed-domain cases retain final maximum change≈.1 (the move limit), so neither meets .001 stopping. C changes33.2645→6.04197 at48 and36.7691→6.50750 at96. The uniform96 initial C is identical between A/B because square-cell KE depends on aspect ratio rather than common length scaling; subsequent densities differ because physical/index filter support differs. This does not establish equal continuum compliance. Final conservative rho MAE=.0141545, max=.17867; fine subcell unresolved RMS=.0423405. The smooth-field filter/restriction MAE=.000135401 demonstrates discrete differences even without optimization.', '', '| Larger case | DOFs | Estimated upper nnz | Solve scenario s/state |501-state native storage MiB |100–200 generation scenario h |','|---|---:|---:|---:|---:|---:|']
 for e in json.loads((out/'larger_mesh_estimates.json').read_text()):
  lines.append(f"| {e['experiment']} {e['mesh']} | {e['ndof']} | {e['upper_structural_nnz']} | {e['solve_scenario_seconds'][0]:.4g}–{e['solve_scenario_seconds'][1]:.4g} | {e['trajectory_500updates_storage_bytes']/2**20:.4g} | {e['dataset_100_200_generation_scenario_hours'][0]:.4g}–{e['dataset_100_200_generation_scenario_hours'][1]:.4g} |")
 lines+=['', 'Planning storage scales the entire21-state native sample, including per-problem JSON, by state count and N; that conservatively repeats one-time configuration overhead and is not an exact file-size prediction. N^1.5..N² time and peak-memory scenarios are deliberately unvalidated; their endpoints are assumptions, not confidence intervals. The peak-memory scenarios scale import overhead too and therefore are crude conservative budgeting examples. Structural CSR nnz is analytical at192/384, not a measured factor nnz. No dense interaction matrices were formed. Filter-only checks at192/384 verify cone weights, constant preservation and adjoints without FEM or feature tensors.', '', 'Worst mechanical diagnostics over63 saved states: '+', '.join(f'{k}={v:.3g}' for k,v in worst.items())+'.', '', 'Reproduce this supplement only: `OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python m3_3c/supplemental_audit.py m3_3c/generations/GENERATION_ID`. It adds measurements/report text to that new M3.3C generation; preserve existing completed outputs and use a copied report directory for repeats. For exact preserved supplement, invoke `supplemental_audit.py.txt` with the repository on PYTHONPATH.']
 report.write_text(report.read_text()+'\n'.join(lines)+'\n')
 print(json.dumps(dict(protected=fingerprints(),mechanical_maxima=worst)))

if __name__=='__main__':main(Path(sys.argv[1]))
