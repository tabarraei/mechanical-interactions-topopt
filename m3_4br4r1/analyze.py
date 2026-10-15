"""One-pass retained comparison and math analysis; explicitly forbids FEM imports."""
import sys,importlib.abc,json,os,shutil
from pathlib import Path
class NoPhysicsImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('topopt_utils','pypardiso','torch','m3_4b','m3_4br2','m3_4br3'):
   raise ImportError('R4-R1 comparator-only guard prohibits physics imports: '+fullname)
sys.meta_path.insert(0,NoPhysicsImports())
import numpy as np
from .policy import compare,canonical,EvidenceError
from .evidence import ROOT,R4,R3,PID,POINTS,read,digest,verify_baseline,load_native,independent_certificate,exact_checkpoint

def safe(value):return json.loads(json.dumps(value,default=lambda x:x.item(),allow_nan=False))
def write(out,name,value):(Path(out)/name).write_text(json.dumps(safe(value),indent=2,allow_nan=False)+'\n')

def state66_analysis(control,restart):
 a=control['arrays'];b=restart['arrays'];t=66;u=a['u'][t];v=b['u'][t];d=v-u;delta=np.hypot(d[0::2],d[1::2]);scale=np.hypot(u[0::2],u[1::2]);budget=1e-12+1e-10*abs(u);bad=np.flatnonzero(abs(d)>budget);nodes=np.asarray(control['spec']['mechanical']['node_coordinates']);indices=np.argsort(delta)[-20:][::-1]
 spec=control['spec'];fixed=np.asarray(spec['mechanical']['fixed_dofs']);mask=np.ones(len(u),bool);mask[fixed]=False
 vals={}
 for label,folder in [('control',R4/'validation/recovery_control_replay'),('restart',R4/'validation/replay_metadata_before_rename')]:
  row=json.loads((folder/'states.jsonl').read_text().splitlines()[t]);vals[label]=dict(primary=row['primary'],native_certificates={k:float((a if label=='control' else b)[k][t]) for k in ('residual_absolute','residual_relative_force','residual_backward_error','energy_compliance_absolute','energy_compliance_relative','force_balance_relative','moment_balance_relative')})
  with np.load(folder/f'state_operands_{t:05d}.npz') as z:
   vals[label]['retained_independent_displacement_difference']=dict(first_solve_max=float(np.max(abs(z['replay_u']-z['saved_u']))),API_solve_max=float(np.max(abs(z['api_u']-z['saved_u']))))
 return dict(state=t,maximum_absolute_component_difference=float(np.max(abs(d))),absolute_vector_difference=float(np.linalg.norm(d)),relative_vector_difference=float(np.linalg.norm(d)/np.linalg.norm(u)),control_vector_norm=float(np.linalg.norm(u)),different_DOFs=int(np.count_nonzero(d)),total_DOFs=len(u),free_DOFs=int(np.sum(mask)),historical_failing_DOFs=bad.tolist(),historical_failures=[dict(DOF=int(i),node=int(i//2),component='x' if i%2==0 else 'y',position=nodes[i//2].tolist(),restart=float(v[i]),control=float(u[i]),difference=float(d[i]),old_allowance=float(budget[i])) for i in bad],node939=dict(control=u[1878:1880].tolist(),restart=v[1878:1880].tolist(),error=float(delta[939]),physical_nodal_scale=float(scale[939]),proposed_allowance=float(1e-12+1e-10*scale[939])),top_node_differences=[dict(node=int(i),position=nodes[i].tolist(),error=float(delta[i]),control_scale=float(scale[i])) for i in indices],spatial_bounds=nodes[np.flatnonzero(delta)].min(axis=0).tolist()+nodes[np.flatnonzero(delta)].max(axis=0).tolist(),retained_independent_evidence=vals,load_vector_available=True,full_stiffness_matrices_saved=False,available_stiffness_data='Diagonal, Frobenius norm and two probe actions; insufficient to reconstruct complete K uniquely',energy_norm_difference_certified=False,backward_error_certificates='Existing individual-state scalar certificates retained; not a bound on K-weighted restart/control displacement difference',pre_interruption_proof='State66 original frame hash is verified against reconstructed final bytes within162-state committed prefix before final metadata interruption',upstream_cause='Unknown; no scheduling/nondeterminism attribution proved')

def run(out):
 out=Path(out);verify_baseline(out)
 archive=out/'source_snapshots';archive.mkdir(exist_ok=False);source={}
 for p in (ROOT/'m3_4br4r1').glob('*.py'):
  dest=archive/(p.name+'.txt');shutil.copyfile(p,dest);source[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(out)))
 write(out,'executing_source_manifest.json',source)
 control=load_native(R4/'recovery/control/samples'/PID);control['certificate']=independent_certificate(control,R4/'validation/recovery_control_replay');matrices=[];prefix=[];decompositions=[]
 for point in POINTS:
  try:
   side=load_native(R4/'recovery'/point/'samples'/PID);side['certificate']=independent_certificate(side,R4/'validation'/('replay_'+point));proof=exact_checkpoint(side,point);prefix.append(dict(point=point,proof=proof))
   if side['certificate']['environment']!=control['certificate']['environment']:raise EvidenceError('Rejected','Retained solver environment mismatch')
   result=compare(control,side,proof);write(out,f'proposal_{point}.json',result)
   old=read(R4/'checks'/('recovery_diagnosis_'+point+'.json'))
   if result['historical_comparator'] is not None and result['historical_comparator']!=old['all_fixed_native_comparisons_passed']:raise EvidenceError('Rejected','Historical comparison reproduction differs')
   volume=[]
   if result.get('residuals'):
    for field,rows in result['residuals'].items():
     for t,r in enumerate(rows):
      if not r['historical_passed']:
       entry=dict(point=point,field=field,state=t,primary_design_passed=result['checks']['x']['historical_passed'],strict_feasibility=True,**r);volume.append(entry);decompositions.append(entry)
   matrix=dict(point=point,historical='Accepted' if old['production_gate_pass'] else 'Rejected',proposed=result['status'],reason=result['reason'],checkpoint=result['checkpoint_integrity'],independent_mechanical_OC=result['independent_correctness'],stopping_feasibility='Accepted' if result['independent_correctness']=='Accepted' else result['status'],primary_design=result['checks'].get('x'),primary_physical_gradient=result['checks'].get('g_rho'),primary_design_gradient=result['checks'].get('g_x'),displacement=dict(historical=result['checks'].get('u',{}).get('historical_passed'),proposed=all(r['passed'] for r in result.get('displacements',[])) if result.get('displacements') else None,maximum_local_ratio=max((r['local_maximum_ratio'] for r in result.get('displacements',[])),default=None)),OC_actions=result.get('action_comparison'),historical_sum_excess=dict(historical=result['checks'].get('historical_sum_excess',{}).get('historical_passed'),proposed=all(r['passed'] for r in result.get('residuals',{}).get('historical_sum_excess',[])) if result.get('residuals') else None,failing_historical_states=[v['state'] for v in volume if v['field']=='historical_sum_excess']),bookkeeping='Accepted' if result['status']=='Accepted' else result['status'],evidence_gaps=[],original_frame_bytes_reconstructed_exactly=True,environment=side['certificate']['environment'])
   matrices.append(matrix)
   if point=='metadata_before_rename':write(out,'state66_displacement_analysis.json',state66_analysis(control,side))
  except EvidenceError as e:matrices.append(dict(point=point,proposed=e.status,reason=str(e),evidence_gaps=[str(e)] if e.status=='Indeterminate' else [],historical='Unchanged retained R4 failure/pass'))
 write(out,'R4_R1_ACCEPTANCE_MATRIX.json',matrices);write(out,'committed_checkpoint_integrity.json',prefix);write(out,'all_failed_volume_decompositions.json',decompositions)
 write(out,'execution_accounting.json',dict(new_FEM_solves=0,new_optimization_trajectories=0,new_OC_updates=0,replayed_FEM_states=0,real_interruptions_repeated=0,comparators_only=True,source_import_guard=True,comparison_environment=dict(python=sys.version,numpy=np.__version__,OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'),OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS')),thread_settings_changed=False,production_gate_changed=False))
 print(json.dumps([(r['point'],r['proposed'],r['reason']) for r in matrices]),flush=True)
if __name__=='__main__':run(sys.argv[1])
