"""Fresh-worker numerical calls using unchanged validated implementation."""
import json,time,os
from pathlib import Path
from collections import defaultdict
import numpy as np
from m3_4b.monitoring import atomic,memory,digest

def execute(out,index,task,result,work=None,fault=None):
 from run_m3_4b import binding,check_authorization
 from topopt_utils.fea.dataset_storage import generate_sample
 from topopt_utils.fea.trajectory_dataset import Trajectory,load_problem,features,PROFILES
 from topopt_utils.fea.production_dataset import stream_replay,OwnedSolver
 from topopt_utils.fea.dataset_readiness import endpoint_analysis,quality_flags,POLICY
 from topopt_utils.fea.mechanical_problem import DensityMap
 from run_m3_4 import fault_hooks
 m=binding(out);check_authorization(out,m);job=m['jobs'][index];spec=json.loads((out/job['spec']).read_text());root=Path(work) if work else out/'native';started=time.perf_counter();before=memory()
 if task=='generate':
  if fault:fault_hooks(fault)
  from topopt_utils.fea import dataset_storage as storage
  totals=defaultdict(float);calls=defaultdict(int);fem=0
  for name in ('atomic_frame','atomic_json','file_hash','enrich_state','update'):
   original=getattr(storage,name)
   def measured(*a,_name=name,_old=original,**kw):
    tick=time.perf_counter()
    try:return _old(*a,**kw)
    finally:totals[_name]+=time.perf_counter()-tick;calls[_name]+=1
   setattr(storage,name,measured)
  answer=generate_sample(root,spec,m['source_id']);tr=Trajectory(answer['folder']);fem=calls['enrich_state'];sample=tr.folder
  # Filesystem interval is separately labeled; exact inclusive wrappers never alter math.
  finalization_span=(sample/'metadata.json').stat().st_mtime-(sample/'journal.json').stat().st_mtime
  data=dict(status=answer['status'],states=tr.metadata['state_count'],updates=tr.metadata['completed_updates'],FEM_evaluations_this_attempt=fem,OC_updates_this_attempt=calls['update'],inclusive_timing_seconds=dict(totals),timing_calls=dict(calls),journal_to_metadata_publication_seconds=finalization_span,native_bytes=sum(f.stat().st_size for f in sample.rglob('*') if f.is_file()),folder=str(sample),termination_reason=tr.metadata['termination_reason'])
 elif task=='validate':
  tr=Trajectory(root/'samples'/job['problem_id']);tick=time.perf_counter();data=stream_replay(tr);data['numerical_replay_seconds']=time.perf_counter()-tick
  q=endpoint_analysis(tr);q['reference_screen_candidate']=q['reference_label_eligible'];q['reference_label_eligible']=False;data['quality']=q;data['feature_profiles']={}
  for profile in PROFILES:
   tick=time.perf_counter()
   for t in sorted({0,tr.metadata['state_count']//2,tr.metadata['state_count']-1}):
    f=features(tr.problem,tr.arrays['x'][t],t,profile,tr.state(t) if PROFILES[profile]['fresh_fem'] else None)
    if f['element'].shape!=(len(PROFILES[profile]['element']),tr.problem.nely,tr.problem.nelx) or f['node'].shape!=(len(PROFILES[profile]['node']),tr.problem.nely+1,tr.problem.nelx+1):raise ValueError('Native feature shapes')
    np.testing.assert_allclose(f['element'][1].ravel(),tr.arrays['rho'][t],rtol=1e-12,atol=1e-12)
    for d in (0,1):np.testing.assert_allclose(f['node'][d].ravel()*spec['force_scale'],tr.problem.forces[d::2],rtol=1e-12,atol=1e-15)
    if profile=='fresh_full_v1':np.testing.assert_allclose(f['element'][-1].sum()*spec['force_scale']**2,tr.arrays['compliance'][t]/2,rtol=1e-8)
   data['feature_profiles'][profile]=dict(cached_extraction_seconds=time.perf_counter()-tick,inference_fresh_FEM_required=PROFILES[profile]['fresh_fem'])
  p=tr.problem;dm=DensityMap(p);x=tr.arrays['x'][0];d=np.random.default_rng(340).normal(size=p.n);d/=np.max(abs(d));h=1e-5
  with OwnedSolver() as solver:
   plus=solver.compliance(dm.forward(x+h*d),*p.solver_args);minus=solver.compliance(dm.forward(x-h*d),*p.solver_args)
   _,_,profile=solver.compliance(dm.forward(x),*p.solver_args,sensitivities=True,profile=True)
  fd=(plus-minus)/(2*h);analytic=float(tr.arrays['g_x'][0]@d);relative=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-12)
  if relative>2e-5:raise ValueError('Centered design gradient FD failed')
  data['finite_difference']=dict(h=h,analytic=analytic,centered=fd,relative_error=relative,extra_FEM=2,direction='Initial seeded infinity-normalized objective derivative, not volume feasible')
  data['primitive_FEM_profile']=profile;data['extra_primitive_FEM']=1;data['FEM_evaluations_this_attempt']=data['independent_FEM_evaluations']+3
  data['threshold_sensitivity']=[dict(pg_box=t,candidate=quality_flags(q,dict(POLICY,pg=t,box_kkt=t))['reference_label_eligible']) for t in (.005,.01,.02)]
  data['array_hashes']={k:digest(tr.folder/e['file']) for k,e in tr.metadata['arrays'].items()}
  data['maxima']={k:float(np.max(tr.arrays[k])) for k in ('residual_relative_force','energy_compliance_relative','force_balance_relative','moment_balance_relative','volume_excess','historical_sum_excess')}
  # Search only old manifests for exact problem IDs; no cap assumption for unmatched cases.
  prior=out.parents[2]/'m3_4/generations/20261014T000500998078Z/pilot/samples'/job['problem_id']
  if prior.exists():
   old=Trajectory(prior)
   for k in tr.arrays:np.testing.assert_allclose(tr.arrays[k],old.arrays[k],rtol=1e-10,atol=1e-12)
   data['historical_equivalence']=dict(matched=True,folder=str(prior),all_arrays=True,rtol=1e-10,atol=1e-12)
  else:data['historical_equivalence']=dict(matched=False,statement='Independent numerical reconstruction; no historical equivalence claimed')
 else:raise ValueError('Unknown real worker task')
 atomic(result,dict(problem_id=job['problem_id'],job_index=index,task=task,before=before,after=memory(),worker_self_hwm_mib=memory().get('VmHWM',0),worker_seconds=time.perf_counter()-started,**data))
