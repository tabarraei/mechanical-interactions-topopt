"""Planning-only corruption tests. No FEM/OC, trajectories or model training."""
import copy,collections,importlib
import numpy as np
import pytest
from .plan import Audit,compatibility,dispatch,risk,ORDER
@pytest.fixture(scope='session')
def audit():return Audit()
@pytest.fixture(scope='session')
def batch(audit):return audit.build()
def test_authoritative_counts_and_inventory(audit):
 assert len(audit.jobs)==164 and audit.family_count==36 and len(audit.completed)==12
 assert collections.Counter(j['partition'] for j in audit.jobs)==dict(train=84,validation=10,test_id=20,test_ood=50)
 assert set(audit.inventory)==audit.completed and len(audit.jobs)-len(audit.completed)==152
 assert {j['plan_index'] for j in audit.rows if j['execution_status']=='complete_independently_validated'}=={2,7,8,12,17,21,35,44,53,58,144,161}
def test_batch_is_frozen_planning_only(audit,batch):
 assert audit.validate(batch) and batch['selected_count']==30 and batch['partition_counts']==dict(train=15,validation=5,test_id=5,test_ood=5)
 assert batch['family_count']==12 and all(j['execution_authorized'] is False for j in batch['jobs'])
 assert sum(j['resource']['capacity_qualified'] for j in batch['jobs'])==25
@pytest.mark.parametrize('attack',['duplicate_id','original12_reuse','reassign_partition','family','near_neighbor','additional_relationship','source','spec','channels','mesh','promote_unqualified_resource','validation','unauthorized_dispatch','heldout_normalization','future_normalization','capped_reference','unknown_termination','count','limit','executable_schema'])
def test_reject_corrupt_proposals(audit,batch,attack):
 b=copy.deepcopy(batch);r=b['jobs'][0]
 if attack=='duplicate_id':b['jobs'][1]['problem_id']=r['problem_id']
 elif attack=='original12_reuse':r['problem_id']=next(iter(audit.completed))
 elif attack=='reassign_partition':r['partition']='validation';b['partition_counts']=dict(collections.Counter(x['partition'] for x in b['jobs']))
 elif attack=='family':r['family']='fabricated family';b['family_count']=len({x['family'] for x in b['jobs']})
 elif attack=='near_neighbor':b['near_neighbor_reconciliation'].pop()
 elif attack=='additional_relationship':b['additional_control_relationships'].append(dict(undocumented_conflict=True))
 elif attack=='source':r['source_binding']={}
 elif attack=='spec':r.pop('problem_spec')
 elif attack=='channels':r['feature_schemas'].append('future_sensitivity')
 elif attack=='mesh':r['configuration']['mesh']=[192,96]
 elif attack=='promote_unqualified_resource':b['jobs'][-1]['resource']['capacity_qualified']=True
 elif attack=='validation':r['validation_requirements'].remove('m3.4b.oc_diagnostic_audit.v2')
 elif attack=='unauthorized_dispatch':r['execution_authorized']=True
 elif attack=='heldout_normalization':b['normalization']['allowed_problem_ids'].append(next(j['problem_id'] for j in b['jobs'] if j['partition']=='test_ood'))
 elif attack=='future_normalization':b['normalization']['fit_state_policy']='all_future_states'
 elif attack=='capped_reference':r['reference_label_eligible']=True
 elif attack=='unknown_termination':r['termination_reason']='change_tolerance'
 elif attack=='count':b['selected_count']-=1
 elif attack=='limit':b['limits']['worker_rss_mib']=1025
 elif attack=='executable_schema':b['planning_only']=False
 with pytest.raises(ValueError):audit.validate(b)
@pytest.mark.parametrize('shape',[[192,96],[384,192]])
def test_unsupported_resolution_never_qualified(shape):
 r=risk(dict(configuration=dict(mesh=shape,support='left_clamp')));assert r['classification']=='unsupported_high_resolution' and not r['capacity_qualified']
@pytest.mark.parametrize('authorized',[False,True])
def test_dispatch_cannot_be_enabled(authorized,batch):
 b=copy.deepcopy(batch);b['execution_authorized']=authorized
 with pytest.raises(PermissionError,match='Planning-only'):dispatch(b)
def test_all21_dispositions_preserved(audit,batch):
 assert len(batch['near_neighbor_reconciliation'])==21
 assert collections.Counter(r['resolution'] for r in batch['near_neighbor_reconciliation'])=={'held-location extrapolation':6,'held-fold direction cluster grouped':3,'controlled direction OOD':6,'explicit composite OOD':6}
 for r in batch['near_neighbor_reconciliation']:
  assert r['resolved_in_R2'] and not r['independence_claim'] and r['disposition']
def test_family_controls_selected(audit,batch):
 indices={j['plan_index'] for j in batch['jobs']};assert {50,98,122,10,130,138,162,145,151}.issubset(indices)
 for i,j,allow in [(98,50,{'fixed_dofs'}),(122,2,{'loads','force_vector'}),(151,145,{'loads','force_vector'})]:
  a,b=[audit.specs[audit.jobs[n]['problem_id']]['mechanical'] for n in (i,j)]
  assert {k for k in a if a[k]!=b[k]}.issubset(allow)
def test_loader_contract_synthetic_all_modes_no_training(audit,batch):
 c=compatibility(audit,batch);assert c['tensor_contract_compatible'] and not c['current_release_loader_compatible']
 assert c['synthetic_probe_count']==120 and c['real_new_trajectories_tested']==c['new_FEM']==c['new_OC']==c['new_model_runs']==0
@pytest.mark.parametrize('module',['topopt_utils.fea.solver','m3_4br4.worker','torch'])
def test_forbidden_expensive_imports(module):
 with pytest.raises(ImportError,match='(forbids|prohibits)'):importlib.import_module(module)

def test_resource_scenarios_do_not_invent_qualifications(audit,batch):
 from .resources import measurements,costs
 m=measurements(audit);s=costs(audit,batch,m);assert not s['memory_safety_certified'] and not s['new_benchmarks_run']
 assert s['required_free_disk_scenario_bytes']>2*s['all_cap_base_footprint_bytes']>0
 assert all(p['raw_native_array_bytes']>0 and p['raw_independent_operand_bytes']>0 for p in s['projections'])
 assert all(p['replay_seconds_scenario'] is None for p in s['projections'] if p['mesh'] in ([64,32],[48,16]))
 assert all(g['FEM']==g['states'] and g['self_wall_seconds']>0 for g in m['actual_new_generation'])
