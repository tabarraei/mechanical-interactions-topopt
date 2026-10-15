"""Synthetic schema controls only; no FEM/OC/production imports or dispatch."""
import copy,json,hashlib,sys
from pathlib import Path
import pytest
from m3_5b2s.alias_schema import *
REPORT=Path(__file__).parent/'reports/20261015_protocol_v1'
@pytest.fixture
def base():
 b=json.loads((REPORT/'alias_bindings.json').read_text());a=json.loads((REPORT/'historical_alias_proposal.json').read_text());return a,b

def complete_fixture(a):
 a=copy.deepcopy(a);c=a['family_crosswalk'];c.update(reviewed=True,review_identity='SYNTHETIC_NOT_APPROVAL',review_timestamp='synthetic-fixture',review_reference='synthetic-fixture')
 refs={}
 for kind,r in a['certificates'].items():
  p='synthetic-only/'+kind;h=hashlib.sha256(p.encode()).hexdigest();r.update(status='referenced_unverified',path=p,sha256=h);refs[p]=h
 return a,refs

def test_default_proposal_has_no_authority(base):
 a,b=base;assert validate_draft(a,b)['execution_authorized'] is False
 with pytest.raises(ValueError):validate_reference_design(a,b,{})

@pytest.mark.parametrize('attack',['historical_id','selected_id','equivalence','initial','load','support','OC','partition','reference','manifest','authorize','source','seed','eligible','count','unknown','approved'])
def test_adversarial_immutable_proposals(base,attack):
 a,b=base;a=copy.deepcopy(a)
 if attack=='historical_id':a['historical']['id']='wrong'
 elif attack=='selected_id':a['selected']['id']='wrong'
 elif attack=='equivalence':a['mechanical_equivalence_sha256']='0'*64
 elif attack=='initial':a['historical']['spec']['initial_x'][0]+=.001
 elif attack=='load':a['historical']['spec']['mechanical']['loads'][0]['fx']=1
 elif attack=='support':a['historical']['spec']['mechanical']['fixed_dofs']=[]
 elif attack=='OC':a['historical']['spec']['optimizer']['tolerance']=.01
 elif attack=='partition':a['historical']['partition']='validation'
 elif attack=='reference':a['reference_optimum']=True
 elif attack=='manifest':a['manifest_sha256']='0'*64
 elif attack=='authorize':a['execution_authorized']=True
 elif attack=='source':a['historical']['source_id']=None
 elif attack=='seed':a['seeds']['historical']=0
 elif attack=='eligible':a['release_eligible']=True
 elif attack=='count':a['coverage']['transition_weight_multiplier']=2
 elif attack=='unknown':a['dispatch']='run'
 elif attack=='approved':a['reuse_approved']=True
 with pytest.raises(ValueError):validate_draft(a,b)

@pytest.mark.parametrize('attack',['missing_certificate','bad_hash','unreviewed_crosswalk','incomplete_restart','other_case_restart'])
def test_evidence_reference_design_rejects(base,attack):
 a,b=base;a,refs=complete_fixture(a)
 if attack=='missing_certificate':a['certificates'].pop('independent_v2_v3')
 elif attack=='bad_hash':a['certificates']['independent_v2_v3']['sha256']='0'*64
 elif attack=='unreviewed_crosswalk':a['family_crosswalk'].update(reviewed=False,review_identity=None,review_timestamp=None,review_reference=None)
 elif attack=='incomplete_restart':a['certificates']['original_interrupted_prefix'].update(status='missing',path=None,sha256=None)
 elif attack=='other_case_restart':a['certificates']['target_restart_control']['historical_id']='m3_3b_d891dbfe4a407a7bca945669'
 with pytest.raises(ValueError):validate_reference_design(a,b,refs)

def test_synthetic_references_do_not_claim_acceptance(base):
 a,b=base;a,refs=complete_fixture(a);r=validate_reference_design(a,b,refs);assert r['numerically_verified'] is False and r['release_eligible'] is False

@pytest.mark.parametrize('attack',['duplicate_entry','duplicate_transitions','double_family_weight','cross_partition'])
def test_registry_release_design(attack):
 e=dict(physical_history_key='one-history',logical_selected_id=SELECTED,partition='train',transitions=500,transition_weight_multiplier=1,family_weight_key='one-history');entries=[e]
 if attack=='duplicate_entry':entries.append(dict(e,logical_selected_id=HISTORICAL))
 elif attack=='duplicate_transitions':e['transition_weight_multiplier']=2
 elif attack=='double_family_weight':e['family_weight_key']='second-family-copy'
 else:e['partition']='test_id'
 with pytest.raises(ValueError):check_release_design(entries)

def test_single_history_counts_once():
 r=check_release_design([dict(physical_history_key='one',logical_selected_id=SELECTED,partition='train',transitions=500,transition_weight_multiplier=1,family_weight_key='one')]);assert r==dict(histories=1,transitions=500,family_contributions=1,released=False)

def test_strict_json():
 with pytest.raises(ValueError):strict_load('{"x":0,"x":1}')
 with pytest.raises(ValueError):strict_load('{"x":NaN}')

def test_no_numerical_or_production_imports():
 assert not any(n.startswith(('topopt_utils','m3_4br3.replay','m3_5b2.runner','torch','pypardiso')) for n in sys.modules)
