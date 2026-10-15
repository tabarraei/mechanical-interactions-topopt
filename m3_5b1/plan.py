"""Reconcile immutable planning/release evidence and freeze a nonexecutable proposal."""
import collections,copy,json,subprocess
from pathlib import Path
import numpy as np
from m3_5a.io import ROOT,RELEASE,R2,R4,OLD,read,digest,canonical,array_digest,check
from m3_5a.loader import Release
from m3_5a.features import schema,extract,collate,crop_predictions
ML=ROOT/'m3_5a/results/20261015_initial_pilot'
PLAN=R2/'corrected_planning_manifest.json'
PAIR=R2/'split_pair_resolution_details.json'
PROPOSAL=ROOT/'m3_4br4r1/results/20261015T020242Z'
APPROVED={ROOT/'m3_4br4r1/policy.py':'c74a81c5477622aaccce4c0866ebbda190dfe4d01fa74f985cd38d15cd31b120',ROOT/'m3_4br4r1/evidence.py':'fd30fdb7eb9392ce77d6a7203d5ee162f62b8d7c1d66416b31863e7be2349761',PROPOSAL/'R4_R1_ACCEPTANCE_MATRIX.json':'417686f60b4e1462cf76cb84264d14ad07640c5bb259422f524b83827e1a4881'}
LIMITS=dict(worker_rss_mib=1024,parent_rss_mib=768,combined_rss_mib=1792,worker_timeout_seconds=900,concurrency=1)
ORDER=[0,1,3,9,10,11,32,33,34,50,64,65,66,67,72,73,74,75,80,81,82,83,94,122,162,98,130,138,145,151]
REQUIREMENTS=['source_config_exact','no_duplicate_any_release','original_FEM_SIMP_filter_OC','strict_box_move_volume_mean_and_sum','primary_mechanics_compliance_sensitivities','selected_OC_action_and_multiplier','m3.4b.oc_diagnostic_audit.v2','m3.4b.transition_change_audit.v3','restart_A_exact_checkpoint','restart_B_independent_correctness','restart_C_m3.4b.restart_acceptance.v2','original_stopping_first_point','complete_states_and_transitions','feature_FD_and_matched_history_when_applicable','output_hashes_and_atomic_recovery','bounded_monitor_HWM_timeout','no_reference_label_even_change_tolerance','quarantine_and_stop_on_failure']
INTEGRATION=['new_authorization_scoped_runner_adapter','new_expansion_release_validation_adapter','train_only_initial_prefix_normalization','immutable_global_completed_ID_registry','fresh_environment_disk_source_preflight']

def equivalence(spec):
 p=copy.deepcopy(spec['mechanical']);p.pop('seed',None)
 return canonical(dict(mechanical=p,initial_x=spec['initial_x'],optimizer=spec['optimizer']))

def risk(j):
 nx,ny=j['configuration']['mesh'];support=j['configuration']['support']
 if nx>96 or ny>48:return dict(classification='unsupported_high_resolution',capacity_qualified=False,stage=99,qualification='No192/384 executable production capability; separate authorization and full qualification required')
 if [nx,ny]==[96,48]:return dict(classification='unverified_full_cap_96',capacity_qualified=False,stage=5,qualification='Only193-state96 replay and supplemental memory are measured. Full501-state output/operand/loader residency and relevant factorization HWM must be qualified within unchanged limits; no stopping change and no extrapolated safety theorem')
 if [nx,ny]==[64,32]:return dict(classification='unverified_64_spacing',capacity_qualified=False,stage=4,qualification='No native64 production envelope established. First validate selected paired coarse162; then explicitly authorized bounded64 qualification with full independent replay, archive and FD memory')
 if [nx,ny]==[48,16] or support=='dual_pin':return dict(classification='unverified_new_support_or_geometry',capacity_qualified=False,stage=3,qualification='Comparable/smaller grid is plausible, not a qualified bound for new support/aspect. Bounded exact selected sentinel generation/replay/FD under separate future authority; full cap accounting and worker HWM mandatory')
 return dict(classification='demonstrated_grid_envelope_only',capacity_qualified=True,stage=1 if j['partition']=='train' else 2,qualification='48x24/48x32 up to501 states demonstrated; this exact new configuration is unexecuted. Fresh source/disk/env and per-worker HWM/RSS/time validation remain mandatory; sampled RSS is not a universal peak bound')

class Audit:
 def __init__(self):
  self.manifest=read(PLAN);self.jobs=self.manifest['jobs'];self.pairs=read(PAIR);check(len(self.jobs)==164,'Plan count');self.by_id={j['problem_id']:j for j in self.jobs};check(len(self.by_id)==164,'Duplicate plan IDs')
  for p,h in APPROVED.items():check(digest(p)==h,'Approved policy hash '+str(p))
  # Bind original corrected-plan artifacts to the unchanged released integrity tree.
  for p in (PLAN,PAIR,R2/'M3_4B_R2_SPLIT_REVIEW.md'):
   expected=read(R2/'artifact_manifest.json')[str(p.relative_to(R2))];check(digest(p)==expected,'R2 planning artifact '+str(p))
  self.release=Release();self.completed=set(self.release.by_id);self.specs={};self.rows=[];self.source_id=read(OLD/'execution_manifest.json')['source_id'];self.source_manifest_sha256=digest(OLD/'source_manifest.json')
  self.bindings={str(p.relative_to(ROOT)):digest(p) for p in (PLAN,PAIR,OLD/'source_manifest.json',OLD/'execution_manifest.json',RELEASE/'accepted_dataset_index.json',RELEASE/'artifact_manifest.json',RELEASE/'explicit_policy_approval.json',ML/'artifact_manifest.json',ML/'feature_schemas.json',ML/'normalization.json',ML/'final_readiness.json')}
  self.bindings.update({str(p.relative_to(ROOT)):h for p,h in APPROVED.items()})
  mlmanifest=read(ML/'artifact_manifest.json')
  for name,h in mlmanifest.items():check(digest(ML/name)==h,'M3.5A artifact '+name)
  for name,e in read(ML/'final_source_manifest.json').items():check(digest(ROOT/name)==e['sha256'] and digest(ML/e['archive'])==e['sha256'],'M3.5A source/archive '+name)
  for name,e in read(R4/'executing_source_manifest.json').items():check(digest(ROOT/name)==e['sha256'],'Production/replay source '+name)
  self.execution_sources={name:e['sha256'] for name,e in read(R4/'executing_source_manifest.json').items()}
  self.execution_sources.update({str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'m3_4br4approved/continue_batch.py',ROOT/'m3_4b/monitoring.py',ROOT/'m3_5a/loader.py',ROOT/'m3_5a/features.py',ROOT/'m3_5a/normalization.py']})
  signatures={};families=collections.defaultdict(set)
  for i,j in enumerate(self.jobs):
   path=R2/j['spec'];s=read(path);check(canonical(s)==j['spec_sha256'],'Canonical specification');check(digest(path)==digest(OLD/j['spec']),'Original spec bytes')
   check(s['problem_id']==j['problem_id'] and s['related_group']==j['family'] and s['configuration']==j['configuration'],'Spec ID/family/configuration')
   body={k:v for k,v in s.items() if k not in ('problem_id','label','related_group','support_pattern','configuration')};check('m3_3b_'+canonical(body)[:24]==j['problem_id'],'Original computed problem ID')
   self.specs[j['problem_id']]=s;p=s['mechanical'];c=j['configuration'];x=np.asarray(s['initial_x']);nx,ny=p['nelx'],p['nely'];N=(nx+1)*(ny+1)
   check([nx,ny]==c['mesh'] and [p['width'],p['height']]==c['domain'],'Mesh/domain');check(x.shape==(nx*ny,) and array_digest(x)==s['initial_digest'],'Initial design identity')
   check(np.isfinite(x).all() and x.min()>=.001 and x.max()<=1 and x.mean()<=p['volume_fraction'] and sum(x)<=len(x)*p['volume_fraction'],'Initial strict feasibility')
   check(np.asarray(p['node_coordinates']).shape==(N,2) and np.asarray(p['element_centers']).shape==(nx*ny,2),'Physical grid metadata')
   check(len(p['force_vector'])==2*N and len(set(p['fixed_dofs']))==len(p['fixed_dofs']),'Forces/supports shape')
   force=np.zeros(2*N)
   for load in p['loads']:
    n=load['node_id'];check(0<=n<N and load['coordinates']==p['node_coordinates'][n],'Exact point load node');force[2*n:2*n+2]+=[load['fx'],load['fy']]
   check(np.array_equal(force,p['force_vector']) and float(np.linalg.norm(force))==s['force_scale'],'Force representation')
   check(np.isfinite(force).all() and s['optimizer']==read(OLD/j['spec'])['optimizer'],'Original optimizer')
   sig=equivalence(s);check(sig not in signatures,'Configuration-equivalent duplicate');signatures[sig]=j['problem_id'];families[j['family']].add(j['partition'])
   status='complete_independently_validated' if j['problem_id'] in self.completed else 'unexecuted'
   self.rows.append(dict(plan_index=i,**copy.deepcopy(j),spec_file=str(path.relative_to(ROOT)),spec_file_sha256=digest(path),configuration_equivalence_sha256=sig,declared_source_id=j.get('source_id'),source_binding=dict(original_execution_source_id=self.source_id,source_manifest_sha256=self.source_manifest_sha256,note='Planning entry has no declared source_id; governing original77 numerical sources/archive are separately verified, not inferred as an entry field'),actual=dict(mesh=[nx,ny],aspect=p['width']/p['height'],domain=[p['width'],p['height']],spacing=[p['dx'],p['dy']],support_pattern=s['support_pattern'],fixed_dofs_sha256=canonical(p['fixed_dofs']),loads=p['loads'],force_scale=s['force_scale'],volume=p['volume_fraction'],initial_digest=s['initial_digest'],initial_summary=dict(minimum=float(x.min()),maximum=float(x.max()),count=len(x)),filter=p['filter'],physical_filter_semi_axes=s['filter_physical_semi_axes'],material=p['material'],simp=p['simp'],optimizer=s['optimizer']),execution_status=status,validation_status='complete' if status.startswith('complete') else 'not_evaluated'))
  check(all(len(v)==1 for v in families.values()),'Cross-partition declared-family leakage');self.family_count=len(families);self.signatures=signatures
  check(self.completed.issubset(self.by_id),'Release reconciliation');self.release.close()
  self.inventory=self.inventory_histories();check(set(self.inventory)==self.completed,'Additional historical executions require reconciliation')
  self.pair_audit=self.reconcile_pairs(set(ORDER))
 def inventory_histories(self):
  files=subprocess.run(['rg','--files','-uuu'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.splitlines();found={}
  for name in files:
   p=ROOT/name
   if p.name in ('metadata.json','journal.json') and p.parent.name in self.by_id:
    check(p.name!='journal.json' or 'incomplete' not in p.parts,'Unreconciled incomplete planned execution')
    if p.name=='metadata.json':
     meta=read(p);check(meta['problem_id']==p.parent.name,'Historical metadata identity');found.setdefault(meta['problem_id'],[]).append(dict(path=name,states=meta.get('state_count'),source_id=meta.get('source_id'),role='released' if str(p.parent) in {e['folder'] for e in self.release.entries} else 'historical_or_recovery_fixture_not_released'))
  return found
 def reconcile_pairs(self,selected_indices):
  selected={self.jobs[i]['problem_id'] for i in selected_indices};pool=selected|self.completed;families={j['family']:j['partition'] for j in self.jobs};rows=[]
  check(len(self.pairs)==21 and [p['index'] for p in self.pairs]==list(range(1,22)),'21 pair coverage')
  for p in self.pairs:
   check([families[f] for f in p['original_families']]==p['corrected_partitions'],'Corrected pair folds');check(bool(p['disposition']) and not p['same_related_family'],'Pair disposition/family')
   for v in p['a_variants']+p['b_variants']:
    j=self.by_id[v['problem_id']];check(j['spec_sha256']==v['spec_canonical_sha256'] and j['family']==v['family'] and j['partition']==v['partition'],'Pair variants reconcile')
   groups=[[v['problem_id'] for v in p[key] if v['problem_id'] in pool] for key in ('a_variants','b_variants')]
   rows.append(dict(index=p['index'],families=p['original_families'],original_partitions=p['original_partitions'],corrected_partitions=p['corrected_partitions'],resolution=p['resolution'],disposition=p['disposition'],risk=p['risk'],both_sides_present_in_combined_pool=all(groups),combined_variant_ids=groups,source_sha256=digest(PAIR),resolved_in_R2=True,independence_claim=False))
  return rows
 def additional_relationships(self,selected):
  pool=selected|self.completed;old={frozenset(p['original_families']) for p in self.pairs};groups=collections.defaultdict(list);result=[]
  for pid in sorted(pool):
   s=self.specs[pid];p=s['mechanical'];key=canonical(dict(mesh=[p['nelx'],p['nely']],domain=[p['width'],p['height']],fixed=p['fixed_dofs'],positions=[l['coordinates'] for l in p['loads']],volume=p['volume_fraction'],initial=s['initial_digest'],filter=p['filter'],optimizer=s['optimizer']))
   groups[key].append(pid)
  for ids in groups.values():
   for i,a in enumerate(ids):
    for b in ids[:i]:
     ja,jb=self.by_id[a],self.by_id[b]
     if ja['partition']==jb['partition'] or frozenset([ja['family'],jb['family']]) in old:continue
     fa=np.asarray(self.specs[a]['mechanical']['force_vector']);fb=np.asarray(self.specs[b]['mechanical']['force_vector'])
     if np.array_equal(fa,fb):continue
     result.append(dict(problem_ids=[a,b],families=[ja['family'],jb['family']],partitions=[ja['partition'],jb['partition']],relationship='Same geometry/support/load position/volume/initial/filter/optimizer; different force direction. Additional matched physics relationship beyond historical21; not an independent random draw.',unit_force_dot=float(fa@fb/np.linalg.norm(fa)/np.linalg.norm(fb)),disposition='Retain original partitions as explicit controlled direction relationship; never claim universal family independence or random IID. Freeze before observing outcomes.'))
  return result
 def rationale(self,i):
  if i in (0,1,3):return 'Complete four baseline volume levels in already released horizontal-left family (original plan2 provides volume.5); defer perturbed157 and reuse existing96 plan144.'
  if i in (9,10,11):return 'Complete vertical-left baseline volumes using released plan8(.3); supply exact volume controls for validation72..75 and direction-OOD122.'
  if i in (32,33,34):return 'Complete bottom-vertical baseline volumes with released35(.6), enabling matched-volume held-location test80..83.'
  if i==50:return 'Exact volume.5 pin-roller horizontal control for dual-pin98; only constrained DOF set changes in that comparison.'
  if i in (64,65,66,67):return 'Add a full four-volume pin-roller oblique training family and a same-volume/location-stratum control for held-location test94.'
  if i in (72,73,74,75):return 'Validation vertical-left family at held location.75; complete coarse volume quartet; extrapolation beyond training.5, never interpolation or independent-ID.'
  if i in (80,81,82,83):return 'Test_id administrative fold: bottom-vertical held-location.75 volume quartet; explicit extrapolation despite test_id name.'
  if i==94:return 'Pin-roller oblique held-location.75 test at volume.5; family-consistent one-volume slice matched to training66; no multi-volume claim for this held family.'
  if i==98:return 'Dual-pin support OOD at volume.5 matched to selected pin-roller50, same exact nodal load/domain/mesh/material/filter/initial/optimizer.'
  if i==122:return 'Alternate force direction(.8,-.6) OOD at existing exact position; matched volume.5 horizontal-left2 and vertical-left10 controls; oblique.5 control18 deferred.'
  if i==130:return 'Aspect3 geometry OOD: same width/unit spacing/relative boundary-load location, changed height and element count; geometry plus boundary-relative physical load-position confounding disclosed.'
  if i in (162,138):return 'Select both volume.5 coarse48 and fine64 members of the same OOD family; only paired comparison isolates discretization at fixed physical geometry/load/filter.'
  if i==145:return 'Training96 fixed-domain refinement matched to selected coarse10; needed to distinguish seen-resolution validation151 from unseen resolution. Deferred pending full-cap96 resource qualification.'
  if i==151:return 'Complete all five variants of held vertical-left validation family (four volumes plus96 volume.5); unseen family at resolution already present in released training144. Resource qualification required.'
  raise ValueError('Missing rationale')
 def build(self):
  rows=[];future_root='m3_5b1/future_execution/expansion30_v1'
  for order,i in enumerate(ORDER,1):
   a=self.rows[i];j=self.jobs[i];r=risk(j)
   rows.append(dict(plan_index=i,problem_id=j['problem_id'],family=j['family'],partition=j['partition'],ood_axis=j['ood_axis'],configuration=copy.deepcopy(j['configuration']),spec_sha256=j['spec_sha256'],problem_spec=dict(path=a['spec_file'],sha256=a['spec_file_sha256'],canonical_sha256=j['spec_sha256']),source_binding=a['source_binding'],runner_source_fingerprints=copy.deepcopy(self.execution_sources),rationale=self.rationale(i),execution_order=order,resource=r,required_preflight=INTEGRATION+[r['qualification']],validation_requirements=REQUIREMENTS.copy(),feature_schemas=[schema('A')['version'],schema('B')['version']],expected_output_root=future_root+'/native/samples/'+j['problem_id'],execution_status='unexecuted',validation_status='not_evaluated',termination_reason=None,reference_label_eligible=False,execution_authorized=False))
  b=dict(schema_version='m3.5b1.proposed_expansion.v1',planning_only=True,execution_authorized=False,production_dispatch_supported=False,planning_manifest_sha256=digest(PLAN),release_index_sha256=digest(RELEASE/'accepted_dataset_index.json'),source_binding=dict(original_execution_source_id=self.source_id,source_manifest_sha256=self.source_manifest_sha256),approved_restart_policy=dict(name='m3.4b.restart_acceptance.v2',category='C only; A/B unchanged',files={str(p.relative_to(ROOT)):h for p,h in APPROVED.items()}),selected_count=len(rows),partition_counts=dict(collections.Counter(j['partition'] for j in rows)),family_count=len({j['family'] for j in rows}),limits=LIMITS.copy(),normalization=dict(fit_partition='train',fit_state_policy='initial_state_only',allowed_problem_ids=sorted(self.completed|{j['problem_id'] for j in rows if j['partition']=='train'}),held_out_fit_allowed=False,fit_performed=False),near_neighbor_reconciliation=copy.deepcopy(self.pair_audit),additional_control_relationships=self.additional_relationships({j['problem_id'] for j in rows}),jobs=rows)
  self.validate(b);return b
 def validate(self,b):
  check(b.get('schema_version')=='m3.5b1.proposed_expansion.v1' and b.get('planning_only') is True,'Planning schema');check(b.get('execution_authorized') is False and b.get('production_dispatch_supported') is False,'Unauthorized production dispatch')
  check(b['planning_manifest_sha256']==digest(PLAN) and b['release_index_sha256']==digest(RELEASE/'accepted_dataset_index.json'),'Frozen parent bindings');check(b['limits']==LIMITS,'Resource limits')
  rows=b['jobs'];check(20<=len(rows)<=30 and b['selected_count']==len(rows),'Selection counts');ids=[j['problem_id'] for j in rows];check(len(set(ids))==len(ids),'Duplicate IDs');check(not set(ids)&self.completed,'Reuse original12 as new')
  check(b['partition_counts']==dict(collections.Counter(j['partition'] for j in rows)) and b['family_count']==len({j['family'] for j in rows}),'Partition/family counts')
  check(sorted(j['execution_order'] for j in rows)==list(range(1,len(rows)+1)),'Execution ordering')
  for r in rows:
   check(r['problem_id'] in self.by_id,'Unknown planned configuration');j=self.by_id[r['problem_id']];a=self.rows[r['plan_index']]
   for k in ('problem_id','family','partition','spec_sha256','configuration','ood_axis'):check(r[k]==j[k] and a[k]==j[k],'Original family/partition/configuration '+k)
   spec=r.get('problem_spec');check(spec and spec['path']==a['spec_file'] and spec['sha256']==a['spec_file_sha256'] and spec['canonical_sha256']==j['spec_sha256'],'Missing/incorrect problem spec')
   check(r.get('source_binding')==a['source_binding'] and r.get('runner_source_fingerprints')==self.execution_sources,'Missing/incorrect source fingerprints')
   check(r.get('execution_authorized') is False,'Unauthorized row dispatch');check(r['resource']==risk(j),'Unsupported/unqualified mesh risk');check(r['resource']['classification']!='unsupported_high_resolution','Unapproved high resolution')
   check(r['feature_schemas']==[schema('A')['version'],schema('B')['version']],'Unsupported feature channels')
   check(r['validation_requirements']==REQUIREMENTS and all(x in r['required_preflight'] for x in INTEGRATION),'Missing validation/preflight requirements')
   check(r['reference_label_eligible'] is False and r['termination_reason'] is None and r['execution_status']=='unexecuted' and r['validation_status']=='not_evaluated','Capped/reference/status treatment')
  expected_pairs=self.reconcile_pairs({j['plan_index'] for j in rows});check(b['near_neighbor_reconciliation']==expected_pairs,'Undocumented near-neighbor conflict')
  check(b['additional_control_relationships']==self.additional_relationships(set(ids)),'Undocumented additional mechanical relationship')
  n=b['normalization'];allowed=sorted(self.completed|{r['problem_id'] for r in rows if r['partition']=='train'})
  check(n['fit_partition']=='train' and n['fit_state_policy']=='initial_state_only' and n['allowed_problem_ids']==allowed and n['held_out_fit_allowed'] is False and n['fit_performed'] is False,'Held-out/future normalization')
  return True

def dispatch(*args,**kwargs):raise PermissionError('Planning-only manifest cannot authorize or dispatch production; a separately approved, reviewed runner adapter is required')

class SyntheticHistory:
 """Interface fixture only: no saved target, equilibrium or gradient claim."""
 def __init__(self,spec):
  self.specs={spec['problem_id']:spec};p=spec['mechanical'];n=p['nelx']*p['nely'];nd=2*(p['nelx']+1)*(p['nely']+1);x=np.asarray(spec['initial_x']);self.arrays=dict(x=np.stack([x,x]),rho=np.stack([x,x]),u=np.zeros((2,nd)),g_rho=np.zeros((2,n)),g_x=np.zeros((2,n)),compliance=np.zeros(2))
 def open(self,pid):return self.arrays

def compatibility(audit,batch):
 rows=[];samples={m:[] for m in ('A','B')}
 for j in batch['jobs']:
  s=audit.specs[j['problem_id']];fake=SyntheticHistory(s);p=s['mechanical'];H,W=p['nely'],p['nelx']
  for mode in ('A','B'):
   for t in (0,1):
    x=extract(fake,j['problem_id'],t,mode,3,None);check(x['element_history'].shape==(3,3 if mode=='A' else 7,H,W) and x['node_history'].shape==(3,0 if mode=='A' else 2,H+1,W+1),'Element/node schema');check(not x['element_history'][~x['history_valid']].any(),'Padding');check(np.all(x['history_indices'][x['history_valid']]<=t),'Causality')
   samples[mode].append(dict(inputs=x,target=np.zeros((1,H,W),np.float32),proof=dict(problem_id=j['problem_id'],transition=1)))
  rows.append(dict(problem_id=j['problem_id'],mesh=[W,H],mode_A_layout=True,mode_B_layout=True,causal_masks=True,exact_static_forces_supports=True,release_loader_compatible=False,required_change='New verified expansion release adapter; existing Release hardcodes12 entries, original counts and train-only',normalization='Not fitted; synthetic probe uses analytical-only scales',evidence_kind='Synthetic history interface check using real static spec; no real FEM/OC state/transition/trajectory validation'))
 # Batch at most3 different shapes to keep this planning probe bounded.
 for mode,ss in samples.items():
  for start in range(0,len(ss),3):
   subset=ss[start:start+3];b=collate(subset);p=crop_predictions(b['target'],b);check(all(x.shape==s['target'].shape for x,s in zip(p,subset)),'Mixed mesh collation/cropping')
 return dict(tensor_contract_compatible=True,current_release_loader_compatible=False,normalizer_requires_train_only_expansion_adapter=True,real_new_trajectories_tested=0,synthetic_probe_count=len(rows)*4,rows=rows,feature_schema_changed=False,new_FEM=0,new_OC=0,new_model_runs=0)
