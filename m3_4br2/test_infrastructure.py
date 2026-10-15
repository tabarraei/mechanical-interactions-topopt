import json,hashlib
from pathlib import Path
import numpy as np
import pytest
from m3_4br2.memory import load_trajectory,digest,frame_allclose
ROOT=Path(__file__).resolve().parents[1]
GEN=ROOT/'m3_4b/generations/20261014T013017Z'

def test_readonly_completed_reuse(monkeypatch):
 from topopt_utils.fea import dataset_storage as storage
 job=json.loads((GEN/'execution_manifest.json').read_text())['jobs'][0]
 source=json.loads((ROOT/'m3_4b/authorization.json').read_text())['source_id']
 spec=json.loads((GEN/job['spec']).read_text());folder=GEN/'native/samples'/job['problem_id'];before=digest(folder/'metadata.json')
 def forbidden(*a,**k):raise AssertionError('Completed reuse must not solve/update')
 monkeypatch.setattr(storage,'FeaSolver',forbidden);monkeypatch.setattr(storage,'update',forbidden)
 answer=storage.generate_sample(GEN/'native',spec,source)
 assert answer['status']=='completed_reused' and digest(folder/'metadata.json')==before
 with pytest.raises(ValueError):storage.generate_sample(GEN/'native',spec,'wrong source')

def test_streaming_loader_equal_to_historical():
 from topopt_utils.fea.trajectory_dataset import Trajectory
 job=json.loads((GEN/'execution_manifest.json').read_text())['jobs'][0];folder=GEN/'native/samples'/job['problem_id']
 a=Trajectory(folder);b=load_trajectory(folder)
 assert a.metadata==b.metadata and a.spec==b.spec
 from m3_4br2.memory import compare_native_trajectories
 assert compare_native_trajectories(a,b)['passed']

def test_synthetic_atomic_orphan_journal(tmp_path):
 # Exercises real atomic writers and checks journal truth; does not claim real
 # generate_sample recovery was executed by this synthetic protocol.
 from topopt_utils.fea.dataset_storage import atomic_json,atomic_frame
 (tmp_path/'frames').mkdir();frame=tmp_path/'frames/00000.npz';atomic_frame(frame,dict(x=np.ones(4)))
 journal=dict(binding=dict(source_id='source',spec_sha='spec'),states=[dict(file='frames/00000.npz',sha256=digest(frame))],updates=[])
 atomic_json(tmp_path/'journal.json',journal)
 orphan=tmp_path/'frames/00001.npz';atomic_frame(orphan,dict(x=np.zeros(4)))
 loaded=json.loads((tmp_path/'journal.json').read_text());committed={e['file'] for e in loaded['states']};assert len(loaded['states'])==1
 assert str(orphan.relative_to(tmp_path)) not in committed and digest(frame)==loaded['states'][0]['sha256']
 # Model interruption leaves orphan uncommitted. Recovery quarantine simulation.
 (tmp_path/'quarantined').mkdir();orphan.rename(tmp_path/'quarantined'/orphan.name)
 assert digest(frame)==loaded['states'][0]['sha256'] and not orphan.exists()
 with (tmp_path/'journal.json.tmp').open('w') as f:f.write('{')
 assert json.loads((tmp_path/'journal.json').read_text())==journal

def test_replay_is_validation_only_and_selectable():
 import ast
 from m3_4br2 import replay
 tree=ast.parse(Path(replay.__file__).read_text())
 names={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
 assert not names.intersection({'generate_sample','generate_trajectory','optimize'})
 assert {'evaluate_state','audit_transition','update','load_trajectory'}<=names
 assert 'choices=[POLICY,\'historical\']' in Path(replay.__file__).read_text()

def test_corrected_manifest_grouping_and_labels():
 out=ROOT/'m3_4br2/generations/20261014T124054Z';m=json.loads((out/'corrected_planning_manifest.json').read_text());comparison=json.loads((out/'split_comparison.json').read_text())
 assert len(m['jobs'])==164 and len({e['problem_id'] for e in m['jobs']})==164
 assert all(e['ood_axis']!='ID_interpolated_location_family' for e in m['jobs'])
 assert all(p['resolved'] and p['disposition'] for p in comparison['pairs']) and len(comparison['pairs'])==21
 assert comparison['counts']==dict(train=84,validation=10,test_id=20,test_ood=50)
 assert m['status']=='planning_only_not_authorized'

def test_instrumented_replay_synthetic_no_fem(tmp_path,monkeypatch):
 from types import SimpleNamespace
 from scipy.sparse import eye
 from m3_4br2 import replay
 from m3_4br2.test_audit import fixture
 from topopt_utils.fea import production_dataset as prod, dataset_storage as storage,consistent_trajectory as consistent,robust_oc as oc
 from topopt_utils.fea.mechanical_problem import DensityMap,array_digest
 s,r,target=fixture(4);x=s['x'];a={k:np.stack([v,v]) for k,v in dict(x=x,rho=x,u=np.zeros(8),compliance=np.array(1.),g_rho=s['g_rho'],g_x=s['g_x'],k_diagonal=np.ones(8),k_probe_actions=np.ones((8,2))).items()}
 a.update(iteration=np.arange(2),has_incoming_update=np.array([False,True]),previous_change=np.zeros(2),update_change=np.zeros(1))
 p=SimpleNamespace(tolerance=.001,maxiter=1,ndof=8,fixed_dofs=[0],solver_args=(),volume_fraction=.4,nely=1,nelx=4,dx=1,dy=1)
 tr=SimpleNamespace(problem=p,arrays=a,initial=x,metadata=dict(state_count=2,termination_reason='change_tolerance',source_id='synthetic-only',arrays={}),spec=dict(problem_id='synthetic-only'),folder=tmp_path,transitions=[dict(delta_digest=array_digest(x-x),selected_multiplier=1.,diagnostic=s['diagnostic'])])
 class Solver:
  def __enter__(self):return self
  def __exit__(self,*a):pass
  def compliance(self,*a,**k):return 1.,s['g_rho']
  def get_connectivity(self,*a):return None
  def get_mesh_pattern(self,*a):return None
  def get_KE(self,*a):return None
  def assemble_dirichlet(self,*a):return eye(8,format='csr')
 monkeypatch.setattr(replay,'certify',lambda *a:dict(passed=True,synthetic=True));monkeypatch.setattr(replay,'load_trajectory',lambda folder:tr);monkeypatch.setattr(prod,'OwnedSolver',Solver)
 monkeypatch.setattr(storage,'verify_state',lambda *a:None)
 monkeypatch.setattr(consistent,'evaluate_state',lambda p,x,solver:{**{k:v[0] for k,v in a.items() if k not in ('iteration','has_incoming_update','previous_change','update_change')},'solver_returned':True,'solver_finite':True})
 monkeypatch.setattr(consistent,'independent_stiffness',lambda *a:eye(8,format='csr'));monkeypatch.setattr(consistent,'symmetric_csr',lambda a:a);monkeypatch.setattr(consistent,'stiffness_probes',lambda p:np.ones((8,2)))
 monkeypatch.setattr(DensityMap,'__init__',lambda self,p:None);monkeypatch.setattr(DensityMap,'transpose',lambda self,g:g)
 monkeypatch.setattr(oc,'update',lambda *a:(x.copy(),s['diagnostic'].copy()));monkeypatch.setattr(oc,'trial',lambda *a:x.copy())
 out=tmp_path/'replay';replay.run(tmp_path,out,'m3.4b.oc_diagnostic_audit.v2')
 summary=json.loads((out/'summary.json').read_text());rows=[json.loads(line) for line in (out/'transitions.jsonl').read_text().splitlines()]
 assert summary['passed'] and summary['states']==2 and len(rows)==1 and rows[0]['passed']
 assert rows[0]['identity']['native_source']=='synthetic-only'
 with np.load(out/'operands_00000.npz') as operands:
  assert {'saved_x','replay_x','saved_rho','replay_g_rho','saved_g_x','replay_action'}<=set(operands.files)
 assert len(list(out.glob('state_operands_*.npz')))==2

def test_real_cached_operands_audit_without_fem():
 from m3_4br2.audit import audit_transition
 from topopt_utils.fea.robust_oc import update
 out=ROOT/'m3_4br2/generations/20261014T124054Z';prior=out/'independent_replay'
 if not prior.exists():pytest.skip('Only available after real worker fragment')
 row=json.loads((prior/'transitions.jsonl').read_text().splitlines()[0]);states=json.loads((prior/'states.jsonl').read_text().splitlines()[0])
 folder=GEN/'native/samples'/row['identity']['problem_id'];tr=load_trajectory(folder)
 with np.load(prior/row['operand_file']) as op:
  s={k:op['saved_'+k].copy() for k in ('x','rho','g_rho','g_x','action')};r={k:op['replay_'+k].copy() for k in s}
 action,d=update(r['x'],r['g_x'],tr.problem.volume_fraction,'feasible_upper_v1');np.testing.assert_array_equal(action,r['action'])
 s.update(diagnostic=tr.transitions[0]['diagnostic'],selected_multiplier=tr.transitions[0]['selected_multiplier']);r.update(diagnostic=d,selected_multiplier=d['multiplier_upper'])
 result=audit_transition(s,r,tr.problem.volume_fraction,states['primary']);assert result['passed'],result

def test_validation_restart_uses_cached_state_without_fem(tmp_path,monkeypatch):
 import shutil
 from m3_4br2 import replay
 test_instrumented_replay_synthetic_no_fem(tmp_path,monkeypatch)
 prior=tmp_path/'prefix';prior.mkdir();original=tmp_path/'replay'
 row=json.loads((original/'transitions.jsonl').read_text().splitlines()[0]);row['passed']=False;row['reason']='Diagnostic keys'
 (prior/'transitions.jsonl').write_text(json.dumps(row)+'\n');(prior/'states.jsonl').write_text((original/'states.jsonl').read_text().splitlines()[0]+'\n')
 shutil.copyfile(original/row['operand_file'],prior/row['operand_file']);shutil.copyfile(original/'state_operands_00000.npz',prior/'state_operands_00000.npz')
 continuation=tmp_path/'continued';replay.run(tmp_path,continuation,'m3.4b.oc_diagnostic_audit.v2',prior)
 summary=json.loads((continuation/'summary.json').read_text())
 assert summary['cached_states']==1 and summary['FEM_calls_this_fragment']==2 and summary['FEM_calls']==4
 assert json.loads((continuation/'cached_transition_00000.json').read_text())['passed']
 assert len(list(continuation.glob('state_operands_*.npz')))==1

def test_conditioned_gradient_certificate_rejects_local_error():
 from m3_4br2.gradient_certificate import certify
 from topopt_utils.fea.mechanical_problem import DensityMap
 from topopt_utils.fea.trajectory_validation import comparison
 out=ROOT/'m3_4br2/generations/20261014T124054Z';folder=GEN/'native/samples/m3_3b_d7d625d32faa15712edf39c5';tr=load_trajectory(folder)
 evidence=out/'independent_replay_continuation/state_operands_00006.npz'
 if not evidence.exists():pytest.skip('Retained actual transition6 not yet available')
 with np.load(evidence) as z:
  rho=z['saved_rho'];u=z['saved_u'];g=z['saved_g_rho'];gx=z['saved_g_x'];p=tr.problem
 assert certify(p,rho,u,g,gx)['passed']
 bad=g.copy();bad[1487]+=1e-12;badgx=DensityMap(p).transpose(bad)
 assert comparison(bad,g,1e-9,enforce=False)['passed']
 assert not certify(p,rho,u,bad,badgx)['passed']
