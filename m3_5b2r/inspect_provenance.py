"""Saved-byte/provenance investigation only. No project numerical imports or solves."""
import json,hashlib,gzip,csv,collections,io,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'m3_5b2r/reports/20261015_provenance_v1'
H=ROOT/'m3_3c/completion_generations/20261013T203017221805Z'
PID='m3_3b_d88cb10eb219fb432141045c';SID='m3_3b_84d8d729bc9abf70d17abec2';F=H/'samples'/PID
PLAN=ROOT/'m3_5b1/plans/20261015_family_expansion_v2/M3_5B1_SELECTED_BATCH.json'
EXPECTED='6605348244ec0e6935892898dc7bd82f51d3205219736f20bd1e336d97b66d08'
inspected={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def note(p):
 p=Path(p);inspected[str(p.relative_to(ROOT))]=sha(p);return p
def read(p):return json.loads(note(p).read_text())
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(name,v):
 with (OUT/name).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def arrhash(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def eq(s):
 m=dict(s['mechanical']);m.pop('seed');return canonical(dict(mechanical=m,initial_x=s['initial_x'],optimizer=s['optimizer']))
def main():
 # Extend the prior baseline with every M3.5B-2 source/report, before any new report writes.
 prior=read(ROOT/'m3_5b2/reports/20261015_infrastructure_v1/protected_before.json')
 protected={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'m3_5b2').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 protected.update(prior)
 assert all((ROOT/p).is_file() and sha(ROOT/p)==h for p,h in protected.items()),'Prior protected integrity'
 save('protected_before.json',protected)
 m=read(PLAN);assert sha(PLAN)==EXPECTED and m['execution_authorized'] is False and all(j['execution_authorized'] is False for j in m['jobs'])
 jobs={j['plan_index']:j for j in m['jobs']};a=read(F/'problem.json');b=read(ROOT/jobs[10]['problem_spec']['path']);fine=read(ROOT/jobs[145]['problem_spec']['path'])
 assert a['problem_id']==PID and b['problem_id']==SID
 topdiff=[k for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)];mdiff=[k for k in a['mechanical'] if a['mechanical'][k]!=b['mechanical'][k]]
 assert mdiff==['seed'] and a['initial_x']==b['initial_x'] and a['optimizer']==b['optimizer']
 omitted={k:dict(historical=a.get(k),selected=b.get(k)) for k in topdiff if k!='mechanical'}
 fingerprint=eq(a);assert fingerprint==eq(b)=='55363ad4631c5ec5da1470857fd380c25c4d9b3734a1b5f83d704a54262ea841'
 for spec in (a,b):
  body={k:v for k,v in spec.items() if k not in ('problem_id','label','related_group','support_pattern','configuration')}
  assert 'm3_3b_'+canonical(body)[:24]==spec['problem_id']
 meta=read(F/'metadata.json');journal=read(F/'journal.json');trans=read(F/'transitions.json');arrays={};native_checks={}
 for k,e in meta['arrays'].items():
  p=note(F/e['file']);z=np.load(p,mmap_mode='r',allow_pickle=False);assert sha(p)==e['sha256'] and list(z.shape)==e['shape'] and str(z.dtype)==e['dtype'];arrays[k]=z;native_checks[k]=dict(sha256=sha(p),shape=list(z.shape),dtype=str(z.dtype))
 assert sha(F/'problem.json')==meta['problem_sha256'] and sha(F/'transitions.json')==meta['transitions_sha256']
 assert np.array_equal(arrays['x'][0],np.asarray(a['initial_x'])) and arrhash(arrays['x'][0])==a['initial_digest']==b['initial_digest']
 assert meta['state_count']==501 and meta['completed_updates']==len(trans)==500 and meta['termination_reason']=='maxiter'
 assert np.array_equal(arrays['iteration'],np.arange(501))
 fields=[k for k in meta['arrays'] if k!='update_change'];frame_failed=[];update_failed=[]
 for t,e in enumerate(journal['states']):
  buf=io.BytesIO();np.savez_compressed(buf,**{k:arrays[k][t] for k in fields})
  if e['file']!=f'frames/{t:05d}.npz' or hashlib.sha256(buf.getbuffer()).hexdigest()!=e['sha256']:frame_failed.append(t)
 for t,e in enumerate(journal['updates']):
  h=hashlib.sha256((json.dumps(trans[t],indent=2,allow_nan=False)+'\n').encode()).hexdigest()
  if e['file']!=f'updates/{t:05d}.json' or h!=e['sha256']:update_failed.append(t)
 assert not frame_failed and not update_failed and journal['binding']==dict(problem_id=PID,spec_sha=canonical(a),source_id=meta['source_id'])
 stored=dict(states=501,updates=500,termination='maxiter',native_hashes_verified=native_checks,initial_exact=True,initial_array_digest=a['initial_digest'],initial_min=float(arrays['x'][0].min()),initial_max=float(arrays['x'][0].max()),final_update_change=float(arrays['update_change'][-1]),minimum_update_change=float(arrays['update_change'].min()),first_stop_preserved=bool(np.all(arrays['update_change']>=.001)),max_mean_excess=max(float(x.mean()-.5) for x in arrays['x']),max_numpy_sum_excess=max(float(x.sum()-576) for x in arrays['x']),max_builtin_sum_excess=max(float(sum(x)-576) for x in arrays['x']),native_bounds=[float(arrays['x'].min()),float(arrays['x'].max())],exact_final_journal=dict(frames=501,updates=500,reconstructed_frames_match=True,reconstructed_updates_match=True,interrupted_prefix_receipt_available=False,scope='Serialization/hash consistency only, no numerical validation or historical before/after proof'))
 save('native_integrity_checks.json',stored)
 cats={
  'mesh_geometry':['nelx','nely','width','height','dx','dy','node_coordinates','connectivity','element_dofs','element_centers','ordering','coordinate_frame'],
  'material':['material'],'simp':['simp'],'filter':['filter'],'boundary':['fixed_dofs'],'loads':['loads'],'volume':['volume_fraction'],'mechanical_optimizer':['optimizer'],'numerical_model':['units']}
 checks={category:dict(fields=ks,exactly_equal=all(a['mechanical'].get(k)==b['mechanical'].get(k) for k in ks)) for category,ks in cats.items()}
 assert all(v['exactly_equal'] for v in checks.values())
 save('specification_comparison.json',dict(historical_id=PID,selected_id=SID,full_spec_canonical_hashes=[canonical(a),canonical(b)],full_spec_file_hashes=[sha(F/'problem.json'),sha(ROOT/jobs[10]['problem_spec']['path'])],mechanical_field_differences={'seed':[703,0]},top_level_differences=topdiff,excluded_provenance_differences=omitted,component_checks=checks,initial_design_exact=True,optimizer_exact=True,equivalence_sha256=fingerprint,fingerprint_construction='SHA256 canonical JSON {mechanical excluding seed, initial_x, optimizer}; canonical uses sorted keys, compact separators, allow_nan=False',excluded_top_level_fields=[k for k in a if k not in ('mechanical','initial_x','optimizer')],excluded_other_field_equality={k:a.get(k)==b.get(k) for k in a if k not in ('mechanical','initial_x','optimizer','problem_id','label','related_group','configuration')},seed_semantics='Recorded in mechanical identity; uniform initialization uses volume-slack, no RNG. Perturbed initialization uses seed but is not selected here. Subsequent FEM/OC has no problem-seed dependency. Fixed seeded stiffness probes are diagnostics, not optimization randomness.',fine_case=dict(plan_index=145,problem_id=fine['problem_id'],family=fine['related_group'],partition=jobs[145]['partition'],coarse_plan=10,coarse_mesh=[48,24],fine_mesh=[96,48],same_domain=[48.,24.],coarse_dx=1.,fine_dx=.5,coarse_radius=2.,fine_radius=4.,physical_filter_axes=[2.,2.],discrete_filter_caveat='Physical support preserved; discrete operators and concentrated load discretization differ, so not mechanically equivalent.')))
 sources=read(H/'source_manifest.json');sourcechecks={}
 for p,e in sources.items():
  archive=note(H/e['archive']);assert sha(archive)==e['sha256'];sourcechecks[p]=dict(sha256=e['sha256'],archive=e['archive'],archive_verified=True,current_match=sha(ROOT/p)==e['sha256']);note(ROOT/p)
 assert canonical(sources)==meta['source_id']
 save('historical_source_verification.json',dict(source_id=meta['source_id'],manifest_canonical_sha256=canonical(sources),source_count=len(sources),sources=sourcechecks,current_differences=[p for p,e in sourcechecks.items() if not e['current_match']],interpretation='Archived generation source is authoritative. Later documented analysis-only changes are not silently substituted. Numerical core source hashes match current copies.'))
 replay=[]
 for name in ('validation','fresh_validation_20261013T205322382070Z','fresh_validation_20261013T205838529387Z'):
  summary=read(H/(name+'.json'));counts=collections.Counter();failures=[];indices=collections.defaultdict(set)
  p=note(H/(name+'_comparisons.csv.gz'))
  with gzip.open(p,'rt') as f:
   for row in csv.DictReader(f):
    if row['problem_id']!=PID:continue
    counts[row['quantity']]+=1;indices[row['quantity']].add(int(row['iteration']))
    if row['passed']!='True' or float(row['absolute_error'])>float(row['absolute_tolerance']):failures.append(row)
  assert sum(counts.values())==7011 and not failures
  replay.append(dict(summary=name+'.json',passed_recorded=summary['passed'],target_summary=next(x for x in summary['trajectories'] if x['problem_id']==PID),quantity_counts=counts,quantity_iteration_ranges={k:[min(v),max(v)] for k,v in indices.items()},retained_comparison_records_consistent=True,failures=failures,scope='Old scalar/action comparison records; no independent numerical replay run in this investigation'))
 save('historical_replay_evidence.json',replay)
 supplement=read(H/'completion_supplement.json');fd=[r for r in supplement['objective_FD_checks'] if r['problem_id']==PID];save('retained_FD_evidence.json',fd)
 for n in ('manifest.json','config.json','environment.json','resource_gate.json','completion_audit.json','final_validation.json','restart_benchmark.json','diversity_split_audit.json','endpoint_quality.json','REPRODUCE.md','M3_3C_FULL_PILOT_REPORT.md'):
  note(H/n)
 splits=read(H/'splits.json');group=next(g for g in splits['groups'] if PID in g['problem_ids']);historical_specs=read(H/'problems.json')
 neighbors=[dict(problem_id=s['problem_id'],label=s['label'],partition=splits['assignments'][s['problem_id']]['partition'],family=s['related_group'],mechanical_equivalent=eq(s)==fingerprint) for s in historical_specs if s['related_group']==a['related_group']]
 selected_relatives=[dict(plan_index=j['plan_index'],problem_id=j['problem_id'],family=j['family'],partition=j['partition'],configuration=j['configuration']) for j in m['jobs'] if j['configuration']['support']=='left_clamp' and j['configuration']['direction']=='vertical']
 exact_selected_matches=[]
 for j in m['jobs']:
  spec=read(ROOT/j['problem_spec']['path'])
  if eq(spec)==fingerprint:exact_selected_matches.append(dict(index=j['plan_index'],partition=j['partition'],problem_id=j['problem_id']))
 assert exact_selected_matches==[dict(index=10,partition='train',problem_id=SID)]
 save('family_partition_comparison.json',dict(historical=dict(problem_id=PID,family=a['related_group'],assignment=splits['assignments'][PID],group=group),selected=dict(problem_id=SID,family=b['related_group'],partition=jobs[10]['partition']),historical_relatives=neighbors,selected_relatives=selected_relatives,exact_selected_matches=exact_selected_matches,alias_does_not_add_exact_cross_partition_leakage=True,duplicate_if_both_ids_counted=True,family_names_identical=False,required_provenance='Preserve both original family labels; explicit reviewed crosswalk only. One physical history counts once, never publish both IDs as separate training samples.',near_neighbor_interpretation='Existing location0.75 validation versus0.5 training is disclosed location extrapolation; fine151 shares validation family72..75, fine145 shares training family10. Alias adds no new mechanics beyond planned10; no universal independence claim. Historical right_vertical groups also include quarter-load volume0.3 and a perturbed start; importing that entire broader family would require separate split/leakage audit.'))
 # Inventory exact evidence filename classes without invoking the production validator.
 targetfiles=[str(p.relative_to(ROOT)) for p in F.rglob('*') if p.is_file()]
 raw_files=[p for p in targetfiles if any(t in Path(p).name for t in ('state_operands','transition_operands','states.jsonl','transitions.jsonl','summary.json'))]
 save('acceptance_evidence_matrix.json',dict(current_acceptance_satisfied=False,reuse_eligible_now=False,rows=[
 dict(gate='Source/spec/native integrity',historical='67 original archived sources, exact native array metadata/checksums and ID/source binding',finding='Verified read-only',gap='New reviewed alias must preserve original source_id; cannot claim selected production source_id'),
 dict(gate='Completion/termination',historical='501 states/500 updates/maxiter',finding='Counts and saved first-stop metadata consistent',gap='No optimum/reference claim permitted'),
 dict(gate='Independent FEM/displacement/equilibrium/compliance/sensitivity',historical='Three older full replay summaries and7011 comparison rows per target; saved primary/COO diagnostics',finding='Existing old-contract pass records consistent',gap='No retained actual independent/API displacement and gradient operands or current ordered raw v2/v3 certificate; old pass is not current acceptance'),
 dict(gate='FD sensitivity',historical=fd,finding='Four recorded target directional checks in x/rho',gap='Point checks supplement, do not replace full trajectory independent acceptance'),
 dict(gate='OC/action/multiplier',historical='500 saved-gradient action comparisons; archived validator additionally checks selected multiplier and diagnostics',finding='Old evidence exists; diagnostic/multiplier checks were not individually exported as current certificates',gap='Current complete independent action/multiplier operands/strict feasibility certificates missing'),
 dict(gate='Volume',historical='Native design arrays and stored volume diagnostic fields',finding='Read-only stored-array feasibility values in native_integrity_checks.json',gap='Current independent action feasibility record still required'),
 dict(gate='v2 residual/v3 incoming',historical='No current-version records found in historical generation',finding='Missing',gap='No version relabeling; exact actual operands needed to apply unchanged audits'),
 dict(gate='A exact checkpoint',historical='Full final journal remains;501 frame/500 transition hashes reconstructed exactly',finding='Final serialization integrity verified',gap='No retained original20-update interrupted journal receipt and before/after hash proof for this target'),
 dict(gate='B independent correctness',historical='Older complete replay records',finding='Not current certificate',gap='Current full raw independent evidence required'),
 dict(gate='C restart/control',historical='Target resource_gate documents21-state prefix resumed; restart_benchmark is DIFFERENT axial-small ID',finding='Target was resumed but has no exact same-spec paired control/C evidence',gap='Approved hash-bound restartC cannot borrow another problem control; no missing runs performed'),
 dict(gate='Resource acceptance',historical='Historical2048MiB monolithic screen; final analysis HWM1924.77MiB, isolated replay737.02MiB',finding='Historical observations, different measurement scope',gap='No original1024/768/1792 worker/parent/combined monitored certificate for this target; observed analysis peak is not attributed to this target worker'),
 dict(gate='Release/adoption provenance',historical='M3.3C completed pilot, not original12 release',finding='Not accepted expansion entry',gap='Reviewed versioned alias/adoption contract, one-history registry semantics and separate explicit approval required')],existing_raw_operand_files_in_target=raw_files,native_file_inventory=targetfiles))
 # Hash the governing acceptance/seed-source text, without importing it.
 for n in ('PROJECT_CONTEXT.md','AGENTS.md','m3_5b2/common.py','m3_5b2/registry.py','m3_5b2/validation.py','m3_5b2/authorization.py','m3_4br4r1/evidence.py','m3_4br4r1/policy.py','m3_3c/completion_replay_snapshot.py','m3_5b1/plans/20261015_family_expansion_v2/M3_5B1_SPLIT_REVIEW.md'):
  if (ROOT/n).exists():note(ROOT/n)
 plan164=read(ROOT/'m3_4br2/generations/20261014T124054Z/corrected_planning_manifest.json');save('planning_family_members.json',[dict(plan_index=i,**j) for i,j in enumerate(plan164['jobs']) if j['family']==b['related_group']])
 # Report prose kept separate from scientific/runtime source.
 from report_text import write_reports
 write_reports(OUT,stored,sourcechecks)
 save('inspected_files_sha256.json',inspected)
 changes=[p for p,h in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h];assert not changes
 assert sha(PLAN)==EXPECTED
 save('integrity_final.json',dict(passed=True,protected_files=len(protected),changed=changes,frozen_manifest_sha256=sha(PLAN),historical_arrays_verified=len(arrays),historical_sources_verified=len(sources),no_FEM=0,no_OC=0,no_new_replays=0,no_training=0,no_authorization_created=True,no_historical_artifact_changes=True))
 scripts={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')};save('investigation_source_sha256.json',scripts)
 save('artifact_manifest.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
 print(json.dumps(dict(status='INSUFFICIENT_ACCEPTANCE_EVIDENCE',mechanical_equivalence=True,original_files_unchanged=len(protected),new_FEM=0,new_OC=0)),flush=True)
if __name__=='__main__':main()
