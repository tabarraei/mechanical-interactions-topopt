"""Non-executable alias proposal schema checks. No production imports or adoption API."""
import hashlib,json
HISTORICAL='m3_3b_d88cb10eb219fb432141045c'
SELECTED='m3_3b_84d8d729bc9abf70d17abec2'
MANIFEST='6605348244ec0e6935892898dc7bd82f51d3205219736f20bd1e336d97b66d08'
EQUIVALENCE='55363ad4631c5ec5da1470857fd380c25c4d9b3734a1b5f83d704a54262ea841'
SCHEMA='m3.5b2s.historical_alias.proposal.v1'
CERTIFICATES={'independent_v2_v3','supplemental_features_FD','final_checkpoint','original_interrupted_prefix','target_restart_control','policy_scope_decision'}
FIELDS={'schema_version','manifest_sha256','historical','selected','mechanical_equivalence_sha256','initial_design_digest','seeds','family_crosswalk','physical_history_key','coverage','certificates','approval','reuse_approved','release_eligible','execution_authorized','reference_optimum','termination','status'}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def equivalent(s):
 m=dict(s['mechanical']);m.pop('seed');return canonical(dict(mechanical=m,initial_x=s['initial_x'],optimizer=s['optimizer']))
def exact(d,keys,label):require(isinstance(d,dict) and set(d)==set(keys),label+' fields')
def strict_load(text):
 def pairs(items):
  d={}
  for k,v in items:require(k not in d,'Duplicate JSON field');d[k]=v
  return d
 return json.loads(text,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def validate_draft(a,b):
 """Validate immutable proposal provenance against an independently pinned binding.
 This does not verify numerical certificates, issue authority or release anything.
 """
 exact(a,FIELDS,'alias');require(a['schema_version']==SCHEMA and a['manifest_sha256']==MANIFEST,'Frozen manifest/schema mutation')
 for key in ('reuse_approved','release_eligible','execution_authorized','reference_optimum'):require(a[key] is False,'Alias cannot authorize adoption/execution/reference')
 require(a['status']=='unapproved_proposal' and a['termination']=='maxiter','Capped termination/status')
 for side,pid in [('historical',HISTORICAL),('selected',SELECTED)]:
  exact(a[side],{'id','spec','spec_path','spec_file_sha256','spec_canonical_sha256','source_id','source_manifest_path','source_manifest_sha256','family','partition','location','artifact_hashes'},side)
  require(a[side]['id']==pid and a[side]==b[side],'Wrong '+side+' identity/spec/source/partition/artifacts')
  require(a[side]['partition']=='train' and a[side]['source_id'],'Source/partition')
  require(canonical(a[side]['spec'])==a[side]['spec_canonical_sha256'],'Spec digest')
 require(a['seeds']=={'historical':703,'selected':0},'Original seed provenance')
 require(a['historical']['spec']['mechanical']['seed']==703 and a['selected']['spec']['mechanical']['seed']==0,'Seed/spec')
 require(a['mechanical_equivalence_sha256']==EQUIVALENCE==equivalent(a['historical']['spec'])==equivalent(a['selected']['spec']),'Mechanical equivalence')
 require(a['initial_design_digest']==b['initial_design_digest'] and a['historical']['spec']['initial_x']==a['selected']['spec']['initial_x'],'Initial design')
 require(a['physical_history_key']==b['physical_history_key'],'One physical history identity')
 exact(a['coverage'],{'states','transitions','released_histories','transition_weight_multiplier'},'coverage')
 require(a['coverage']=={'states':501,'transitions':500,'released_histories':0,'transition_weight_multiplier':1},'Duplicate transition counting')
 exact(a['family_crosswalk'],{'historical_family','selected_family','historical_id','selected_id','scope','reviewed','review_identity','review_timestamp','review_reference'},'crosswalk')
 c=a['family_crosswalk'];require(c['historical_family']=='right_vertical' and c['selected_family']=='left_clamp:vertical:loc0.5:aspect2' and c['historical_id']==HISTORICAL and c['selected_id']==SELECTED and c['scope']=='this_history_only_no_global_family_union','Family crosswalk')
 require(type(c['reviewed']) is bool,'Crosswalk reviewed type')
 if c['reviewed']:require(all(isinstance(c[k],str) and c[k] for k in ('review_identity','review_timestamp','review_reference')),'Missing crosswalk review metadata')
 else:require(all(c[k] is None for k in ('review_identity','review_timestamp','review_reference')),'Unreviewed crosswalk metadata')
 exact(a['approval'],{'identity','timestamp','reference'},'approval');require(all(v is None for v in a['approval'].values()),'No alias approval may be issued here')
 exact(a['certificates'],CERTIFICATES,'certificates')
 for kind,r in a['certificates'].items():
  exact(r,{'status','path','sha256','historical_id','selected_id','scope'},'certificate')
  require(r['historical_id']==HISTORICAL and r['selected_id']==SELECTED,'Other trajectory certificate')
  require(r['status'] in ('missing','pending_policy_review','referenced_unverified'),'Certificate status')
  require(r['scope']==kind,'Certificate scope')
  if r['status']=='referenced_unverified':require(isinstance(r['path'],str) and r['path'] and isinstance(r['sha256'],str) and len(r['sha256'])==64 and all(c in '0123456789abcdef' for c in r['sha256']),'Certificate hash/reference')
  else:require(r['path'] is None and r['sha256'] is None,'Missing certificate cannot have evidence')
 return {'schema_consistent':True,'reuse_approved':False,'release_eligible':False,'execution_authorized':False}
def validate_reference_design(a,b,artifact_hashes):
 """Stricter hypothetical integration fixture check, never actual acceptance.
 Caller-provided hashes are schema-test inputs, not a trust store or numerical proof.
 """
 validate_draft(a,b);require(a['family_crosswalk']['reviewed'],'Unreviewed family crosswalk')
 for kind,r in a['certificates'].items():
  require(r['status']=='referenced_unverified','Missing acceptance/restart/policy certificate '+kind)
  require(artifact_hashes.get(r['path'])==r['sha256'],'Invalid certificate hash '+kind)
 return {'reference_design_consistent':True,'numerically_verified':False,'release_eligible':False}
def check_release_design(entries):
 """Identity/count invariants on synthetic proposed records, no release writer."""
 seen=set();logical=set();transitions=0;families={}
 for e in entries:
  require(e['physical_history_key'] not in seen and e['logical_selected_id'] not in logical,'Duplicate release entry/history')
  seen.add(e['physical_history_key']);logical.add(e['logical_selected_id'])
  require(e['partition']=='train' and e['transitions']==500 and e['transition_weight_multiplier']==1,'Duplicate counting/cross-partition alias')
  require(e['family_weight_key']==e['physical_history_key'],'Duplicated family weight')
  families[e['family_weight_key']]=1;transitions+=500
 return {'histories':len(seen),'transitions':transitions,'family_contributions':sum(families.values()),'released':False}
