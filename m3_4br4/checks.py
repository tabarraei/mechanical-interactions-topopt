"""Native feature/FD and fixed-comparison recovery checks; no optimizer redesign."""
import sys,json
from pathlib import Path
import numpy as np
from .common import OLD,job
from m3_4br2.memory import load_trajectory,compare_native_trajectories,frame_allclose
from m3_4b.monitoring import atomic,memory,digest

def compare(out,index,current,control,result_name):
 out=Path(out);m,j,spec=job(index);a=load_trajectory(current);b=load_trajectory(control)
 if a.spec!=b.spec or a.metadata['source_id']!=b.metadata['source_id']:raise ValueError('Recovery binding differs')
 if a.metadata['termination_reason']!=b.metadata['termination_reason']:raise ValueError('Stopping differs')
 answer=compare_native_trajectories(a,b)
 if len(a.transitions)!=len(b.transitions):raise ValueError('Transition count')
 for ta,tb in zip(a.transitions,b.transitions):
  for k in ('iteration','next_iteration','optimizer','valid'):
   if ta[k]!=tb[k]:raise ValueError('Transition bookkeeping')
  for k in ('maximum_change','selected_multiplier'):np.testing.assert_allclose(ta[k],tb[k],rtol=1e-10,atol=1e-12)
  if ta['diagnostic'].keys()!=tb['diagnostic'].keys():raise ValueError('Diagnostic key set')
  for k,va in ta['diagnostic'].items():
   vb=tb['diagnostic'][k]
   if isinstance(va,(str,bool)):
    if va!=vb:raise ValueError('Categorical diagnostic '+k)
   else:np.testing.assert_allclose(va,vb,rtol=1e-10,atol=1e-12)
 atomic(out/result_name,dict(**answer,states=a.metadata['state_count'],transitions=len(a.transitions),source_id=m['source_id'],termination=a.metadata['termination_reason'],delta_digests_match=all(ta['delta_digest']==tb['delta_digest'] for ta,tb in zip(a.transitions,b.transitions)),**memory()))

def features_fd(out,index,folder,result_name):
 from topopt_utils.fea.trajectory_dataset import features,PROFILES,feature_schema
 from topopt_utils.fea.mechanical_problem import DensityMap
 from topopt_utils.fea.production_dataset import OwnedSolver
 from topopt_utils.fea.dataset_readiness import endpoint_analysis
 m,j,spec=job(index);tr=load_trajectory(folder);p=tr.problem;profiles={}
 for profile in PROFILES:
  for t in sorted({0,tr.metadata['state_count']//2,tr.metadata['state_count']-1}):
   fs=features(p,tr.arrays['x'][t],t,profile,tr.state(t) if PROFILES[profile]['fresh_fem'] else None)
   if fs['element'].shape!=(len(PROFILES[profile]['element']),p.nely,p.nelx) or fs['node'].shape!=(len(PROFILES[profile]['node']),p.nely+1,p.nelx+1):raise ValueError('Feature grid')
   np.testing.assert_allclose(fs['element'][1].ravel(),tr.arrays['rho'][t],rtol=1e-12,atol=1e-12)
   for d in (0,1):np.testing.assert_allclose(fs['node'][d].ravel()*spec['force_scale'],p.forces[d::2],rtol=1e-12,atol=1e-15)
   if profile=='fresh_full_v1':np.testing.assert_allclose(fs['element'][-1].sum()*spec['force_scale']**2,tr.arrays['compliance'][t]/2,rtol=1e-8)
  profiles[profile]=dict(extraction_FEM_calls=0,input_source='saved mechanically validated native history',inference_fresh_FEM_required=PROFILES[profile]['fresh_fem'])
 dm=DensityMap(p);x=tr.arrays['x'][0];d=np.random.default_rng(340).normal(size=p.n);d/=np.max(abs(d));h=1e-5
 with OwnedSolver() as solver:
  plus=solver.compliance(dm.forward(x+h*d),*p.solver_args);minus=solver.compliance(dm.forward(x-h*d),*p.solver_args)
  _,_,profile=solver.compliance(dm.forward(x),*p.solver_args,sensitivities=True,profile=True)
 fd=(plus-minus)/(2*h);analytic=float(tr.arrays['g_x'][0]@d);relative=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-12)
 if relative>2e-5:raise ValueError('Unchanged FD gate failed')
 prior=OLD.parents[2]/'m3_4/generations/20261014T000500998078Z/pilot/samples'/j['problem_id']
 historical=dict(matched=False)
 if prior.exists():historical=dict(matched=True,folder=str(prior),**compare_native_trajectories(tr,load_trajectory(prior)))
 quality=endpoint_analysis(tr);quality['reference_screen_candidate']=quality['reference_label_eligible'];quality['reference_label_eligible']=False
 atomic(Path(out)/result_name,dict(passed=True,problem_id=j['problem_id'],profiles=profiles,feature_schema=feature_schema(),FD=dict(relative_error=relative,analytic=analytic,centered=fd,h=h),primitive_profile=profile,extra_FEM_calls=3,extra_OC_calls=0,historical=historical,quality=quality,**memory()))
if __name__=='__main__':
 if sys.argv[1]=='compare':compare(sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[5],sys.argv[6])
 else:features_fd(sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[5])
