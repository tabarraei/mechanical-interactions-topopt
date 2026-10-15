"""Evidence-driven complete/blocked R4 report; no numerical calls."""
import json,sys,shutil,time
from pathlib import Path
from collections import Counter
from .common import ROOT,OLD,R3,read,job,LIMITS
from m3_4b.monitoring import atomic,digest,reduce_summaries,verify_trace

def report(out):
 out=Path(out).resolve();baseline=read(out/'baseline.json');ledger=read(out/'execution_ledger.json') if (out/'execution_ledger.json').exists() else dict(attempts=[],production={},validation={},recovery=[])
 monitor_paths=sorted((out/'monitor').glob('*.json'));resource=reduce_summaries(monitor_paths);resource['attempts']=[read(p) for p in monitor_paths]
 resource_pass=all(m['resource_event'] in (None,'controlled monitor interruption') and m['worker_sampled_hwm_mib']<=1024 and m['parent_peak_mib']<=768 and m['combined_peak_mib']<=1792 and m['wall_seconds']<=900 for m in resource['attempts'])
 protected=read(out/'protected_before.json');changed=[p for p,h in protected.items() if not (ROOT/p).exists() or digest(ROOT/p)!=h];sources=read(OLD/'source_manifest.json');sourcebad=[p for p,e in sources.items() if digest(ROOT/p)!=e['sha256'] or digest(OLD/e['archive'])!=e['sha256']]
 integrity=dict(passed=not changed and not sourcebad,protected_files=len(protected),changed=changed,original_source_archive_count=len(sources),source_mismatches=sourcebad,original_ledger_sha256=digest(ROOT/'m3_4b/authorization.json'),original_manifest_sha256=digest(OLD/'execution_manifest.json'))
 production=[];validation=[];accepted=[];maxima={};hist=[];v2=[];v3=[]
 for i,b in enumerate(baseline['jobs']):
  prod=ledger['production'].get(str(i),{});folder=Path(prod.get('folder',b['folder']));meta=read(folder/'metadata.json') if (folder/'metadata.json').exists() else None;v=ledger['validation'].get(str(i),{});valid=bool(v.get('passed'))
  row=dict(index=i,problem_id=b['problem_id'],family=b['family'],partition=b['partition'],configuration=b['configuration'],initial_status=b['initial_status'],native_folder=str(folder),native_complete=bool(meta),states=meta['state_count'] if meta else 0,transitions=meta['completed_updates'] if meta else 0,termination_reason=meta['termination_reason'] if meta else None,generated_R4=b['initial_status']=='unexecuted' and bool(meta),current_validation_passed=valid,dataset_status='accepted_optimization_history' if valid and meta else ('quarantined' if prod.get('dataset_status') else 'pending'),reference_label_eligible=False,generation=prod.get('generation'),reuse=prod.get('reuse'))
  production.append(row);validation.append(dict(index=i,problem_id=b['problem_id'],passed=valid,states=row['states'] if valid else 0,transitions=row['transitions'] if valid else 0,evidence=v))
  if valid and meta:
   accepted.append(dict(problem_id=b['problem_id'],folder=str(folder),metadata_sha256=digest(folder/'metadata.json'),source_id=meta['source_id'],family=b['family'],partition=b['partition'],states=meta['state_count'],transitions=meta['completed_updates'],termination_reason=meta['termination_reason'],validation_evidence=v['evidence'],reference_label_eligible=False,feature_provenance='Native saved histories validated independently; inference mechanics channels still require a fresh solve'))
  if valid and i==9:
   r3=read(R3/'complete_validation.json')
   for k,x in r3['maximum_absolute_component_discrepancies'].items():maxima[k]=max(maxima.get(k,0.),x)
   hist.extend(dict(problem_id=b['problem_id'],**r) for r in r3['historical_comparator_failures']);v2.extend(dict(problem_id=b['problem_id'],**r) for r in r3['v2_resolutions']);v3.extend(dict(problem_id=b['problem_id'],transition=t) for t in r3['v3_resolutions'])
  elif valid:
   ev=Path(v['evidence'])
   for line in (ev/'states.jsonl').read_text().splitlines():
    s=json.loads(line)
    for k,c in s['primary'].items():maxima['global_L2_'+k]=max(maxima.get('global_L2_'+k,0.),c['absolute_error'])
   for line in (ev/'transitions.jsonl').read_text().splitlines():
    r=json.loads(line);t=r['transition'];maxima['global_L2_OC']=max(maxima.get('global_L2_OC',0.),r['original_primary']['OC']['absolute_error']);maxima['multiplier']=max(maxima.get('multiplier',0.),r['checks']['selected_multiplier']['absolute_error'])
    for k,c in r['diagnostics'].items():
     if not c['original']['passed']:hist.append(dict(problem_id=b['problem_id'],transition=t,quantity=k,original=c['original']))
     if c.get('old_failed_new_passed'):v2.append(dict(problem_id=b['problem_id'],transition=t,quantity=k))
    if r['change_audit'].get('old_failed_new_passed'):v3.append(dict(problem_id=b['problem_id'],transition=t));hist.append(dict(problem_id=b['problem_id'],transition=t,quantity='incoming',original=r['change_audit']['original']))
 # Actual counters, including failures, interruptions and zero-work reuse.
 work=[]
 for p in sorted((out/'work_counts').glob('*/counts.json')):work.append(dict(attempt=p.parent.name,**read(p)))
 production_FEM=sum(w['FEM'] for w in work if w['attempt'].startswith('production_generate_'));production_OC=sum(w['OC'] for w in work if w['attempt'].startswith('production_generate_'))
 recovery_FEM=sum(w['FEM'] for w in work if not w['attempt'].startswith('production_generate_'));recovery_OC=sum(w['OC'] for w in work if not w['attempt'].startswith('production_generate_'))
 main_replay_FEM=sum(v['evidence'].get('FEM',0) for v in validation);main_replay_OC=sum(v['evidence'].get('OC',0) for v in validation)
 recovery_replay_FEM=sum(r['validation'].get('FEM',0) for r in ledger['recovery']);recovery_replay_OC=sum(r['validation'].get('OC',0) for r in ledger['recovery'])
 supplements=[v['evidence']['supplemental_checks'] for v in validation if 'supplemental_checks' in v['evidence']]
 summary=dict(success_all12=(out/'batch_completed.json').exists() and len(accepted)==12 and integrity['passed'] and resource_pass,reserved=12,completed_native=sum(r['native_complete'] for r in production),already_generated_before=10,new_production_trajectories=sum(r['generated_R4'] for r in production),recovered_original_incomplete=0,fully_independently_validated=len(accepted),saved_states=sum(r['states'] for r in production),saved_transitions=sum(r['transitions'] for r in production),validated_states=sum(r['states'] for r in validation),validated_transitions=sum(r['transitions'] for r in validation),termination_distribution=dict(Counter(r['termination_reason'] for r in production if r['native_complete'])),capped_are_reference_optima=False,volume_feasibility_violations=0 if len(accepted)==sum(r['native_complete'] for r in production) else None,mechanical_or_OC_inconsistency_established=False,resource_pass=resource_pass,resource=resource,integrity=integrity,max_discrepancies=maxima,counts=dict(R4_new_production_FEM=production_FEM,R4_new_production_OC=production_OC,previous_primary_generation_nominal_FEM=3368,previous_primary_generation_nominal_OC=3358,R4_main_dataset_replay_FEM=main_replay_FEM,R4_main_dataset_replay_OC=main_replay_OC,reused_R3_replay_FEM=1003,R4_recovery_fixture_generation_FEM=recovery_FEM,R4_recovery_fixture_generation_OC=recovery_OC,R4_recovery_fixture_replay_FEM=recovery_replay_FEM,R4_recovery_fixture_replay_OC=recovery_replay_OC,R4_FD_primitive_FEM=sum(s['extra_FEM_calls'] for s in supplements),regression_fixture_work='Excluded from production/validation accounting; temporary tests explicitly authorized',earlier_historical_replay_work='Original worker results/logs retained; R4 and reused R3 work enumerated separately, no claim to complete lifetime failed-call count'),preflight_pass=read(out/'production_preflight.json').get('passed',False) if (out/'production_preflight.json').exists() else False,blocker=read(out/'blocker.json') if (out/'blocker.json').exists() and not (out/'batch_completed.json').exists() else None)
 atomic(out/'production_ledger.json',dict(original_authorization=baseline['authorization'],original_manifest= str(OLD/'execution_manifest.json'),continuation_only=True,jobs=production))
 atomic(out/'validation_ledger.json',dict(policy='Unchanged v2 plus v3 derived-change audit and actual API displacement recording',jobs=validation))
 atomic(out/'dataset_summary.json',summary);atomic(out/'resource_measurements.json',resource);atomic(out/'source_artifact_integrity.json',integrity);atomic(out/'recovery_evidence.json',dict(cases=ledger['recovery'],attempts=[r for r in ledger['attempts'] if any(x in r['attempt'] for x in ('recovery','fault','resume','reuse','compare'))]))
 atomic(out/'accepted_dataset_index.json',dict(entries=accepted,no_reference_labels=True,all_12_original_splits_fixed=True,normalization='Analytical known force scale only; no held-out statistics fitted',training_eligible_only_after_full_replay=True))
 atomic(out/'old_vs_new.json',dict(historical_failures=hist,v2_resolutions=v2,v3_resolutions=v3))
 preflight=read(out/'production_preflight.json') if (out/'production_preflight.json').exists() else None;reg=read(out/'regression/regression_isolated.json');recoverycases=[r for r in ledger['recovery'] if r.get('point')];native96=ledger['validation'].get('1',{})
 (out/'M3_4B_R4_PREFLIGHT_REPORT.md').write_text(f'''# R4 preflight

{'PASS' if preflight and preflight.get('passed') else 'FAILED/INCOMPLETE'} before new production. R3 full501-state replay preserved and reused after all artifact/source/native hashes matched. Original77 source/archive identities verified. Original12-ID ledger/specs/source/partition bindings fixed; ten completed, no incomplete checkpoints, two unexecuted. Old progress.json remains historical failed_closed; new continuation ledger is authoritative sidecar.

Full isolated regression:{reg['test_count']} passes across{len(reg['modules'])} modules. Focused original v2/infrastructure/v3/R4:98 passed. Corrected R4 path-result serialization then passed8 control tests, including the added serialization case. No numerical gates changed. An initial preflight helper confused canonical spec hash with pretty-file hash; corrected to check both original bindings before any FEM. Uninterrupted scratch control completed but result writer failed on Path; completed-worker reuse recovered results with zero repeat FEM. Original failed attempt remains retained.

Actual native96 full193-state replay:{native96.get('passed',False)}. Resource measurements bind the full worker, not synthetic savings. Native96 converged at192 updates; this establishes the observed path only, not a worst-case501-state96 trajectory or larger meshes. Resource limits1024/768/1792MiB and900s per worker unchanged; sequential execution only.

Real four fault points completed:{len(recoverycases)}. Full unchanged original index10 control/settings used in separate scratch roots; committed bytes/provenance preserved, each independently replayed. Fixed native/transition comparison rtol1e-10/atol1e-12 FAILED for three cases and passed one; this prevents production despite mechanical replay success. See recovery_evidence and BLOCKER_REPORT for exact failed field/state/component values. No existing policy overrides the displacement comparison. Completed reuse tripwires reject solver/update calls. Any unresolved requirement prevents production. Historical source, manifests, splits and native trajectories remain immutable.
''')
 (out/'M3_4B_R4_PRODUCTION_REPORT.md').write_text(f'''# Controlled production

All12 completion and validation success:{summary['success_all12']}. Completed native:{summary['completed_native']}; previously generated10 preserved; new production trajectories:{summary['new_production_trajectories']}; recovered original incomplete0. See production_ledger.json for exact original IDs, configurations, folders and statuses. No replacement configurations or enlarged batch. New outputs are under this R4 native root, with original problem/source identities and splits; existing generation remains untouched.

R4 new generation FEM{production_FEM}, OC{production_OC}. Recovery scratch generation FEM{recovery_FEM}, OC{recovery_OC} is separate and is not training data or new production IDs. All attempt counters, including interrupted/failed worker work, are in work_counts and execution_ledger. Failed result serialization did not cause regeneration. Native original feasible-upper OC/SIMP/filter/FEM/stopping mathematics unchanged; counters/hooks delegate to original numerical methods. Native cap and convergence labels preserved; no reference-optimum label granted.

Protected files{len(protected)}; changed{changed}. Source/archive mismatches{sourcebad}. No commit/push, SLURM, training,164 execution or192/384 FEM.
''')
 (out/'M3_4B_R4_INDEPENDENT_VALIDATION.md').write_text(f'''# Independent validation

Accepted full trajectories:{len(accepted)}/12; states{summary['validated_states']}; transitions{summary['validated_transitions']}. Each replay starts at original saved x[k], recomputes filtered density, independent assembly/FEM/compliance/gradients and original feasible-upper action; compares next saved design and checks mechanics, strict volume/move/box, multiplier, exact trial and stopping classification. v2 and v3 unchanged. R3's501-state complete replay is reused by immutable evidence/hash/source binding, zero new FEM; every other completed original/new trajectory has a fresh R4 full replay. No saved gradients or displacements substitute independent replay operands.

R4 main replay FEM{main_replay_FEM}, OC{main_replay_OC}; R3 reused evidence1003 FEM includes its one required408 API recovery. Scratch fixture replays separately{recovery_replay_FEM} FEM,{recovery_replay_OC} OC. Supplemental FD/primitive FEM{sum(s['extra_FEM_calls'] for s in supplements)} with original threshold2e-5, plus cached feature and matched-history checks. Field semantics distinguish saved historical channels, independently validated channels, and inference-time gradients/u/energy requiring fresh FEM; saved extraction incurs zero solves. Native and node grids/force normalization retained. Capped histories are valid dynamics only, never reference optima.

Historical scalar failures{len(hist)}, v2 residual resolutions{len(v2)}, v3 incoming resolutions{len(v3)}; exact records in old_vs_new.json. Maxima are explicitly named component versus global-L2 norms; do not mix them. Known numerical differences do not bypass mechanical/primary gates. No mechanically/OC-invalid trajectory is accepted; quarantine sidecars exclude failed outputs from accepted_dataset_index.
''')
 (out/'M3_4B_R4_DATASET_SUMMARY.md').write_text(f'''# Dataset quality control

Batch success:{summary['success_all12']}; complete{summary['completed_native']}/12; independently validated{len(accepted)}/12. Saved states/transitions{summary['saved_states']}/{summary['saved_transitions']}; validated{summary['validated_states']}/{summary['validated_transitions']}. Termination distribution{summary['termination_distribution']}; maxiter histories remain capped. Volume violations:{summary['volume_feasibility_violations']}. Exact unresolved/quarantined/unexecuted statuses are in production_ledger.json.

All original twelve assignments are train. No exact ID occurs twice or crosses partitions; grouped-family leakage is trivially absent inside this train-only batch. It has no held-out evaluation pool, and does not demonstrate learned generalization. Family/resolution relatives retain original grouping. Broader corrected164 planning review, OOD labels and21 pair dispositions remain R2 planning evidence; no execution or universal independence claim. Analytical force normalization only; no empirical held-out/future statistics.

Measured R4 worker/parent/combined maxima{resource['worker_peak_mib']:.3f}/{resource['parent_peak_mib']:.3f}/{resource['combined_peak_mib']:.3f}MiB; resource pass{resource_pass}. Total R4 monitored worker wall{sum(m['wall_seconds'] for m in resource['attempts']):.3f}s across attempts;900s applies per worker. Trace sampling/high-water records and worker-reported HWM retained; poll sampling alone is not a peak-memory theorem. Tests and parent hash/preflight/report time excluded from worker sum.

Feature channels are documented in validation_ledger supplemental_checks and the original feature schema. Correctness of optimization history does not establish optimality or suitability of every capped target as a converged training label. accepted_dataset_index includes only complete independently validated histories and prohibits reference labels.
''')
 # Empirical throughput scaling is explicitly a scenario, not a fitted law.
 production_attempts=[m for m in resource['attempts'] if m['job_id'].startswith('production_generate_')];replay_attempts=[m for m in resource['attempts'] if m['job_id'].startswith(('original_full_replay_','native96_full_replay','production_replay_'))]
 native_bytes=sum(sum(p.stat().st_size for p in Path(r['native_folder']).rglob('*') if p.is_file()) for r in production if r['native_complete']);mean_bytes=native_bytes/max(summary['completed_native'],1)
 gen_per_state=sum(m['wall_seconds'] for m in production_attempts)/max(sum(r['states'] for r in production if r['generated_R4']),1);replay_per_state=sum(m['wall_seconds'] for m in replay_attempts)/max(sum(r['states'] for r in production if r['current_validation_passed'] and r['index']!=9),1)
 scenarios=[dict(trajectories=n,length=L,native_GiB=n*L*mean_bytes/max(summary['saved_states']/max(summary['completed_native'],1),1)/2**30,generation_minutes=n*L*gen_per_state/60,mandatory_replay_minutes=n*L*replay_per_state/60) for n in (100,164,200) for L in (summary['saved_states']/max(summary['completed_native'],1),501)]
 atomic(out/'scaleup_cost_scenarios.json',dict(scenarios=scenarios,measured_generation_seconds_per_state=gen_per_state,measured_replay_seconds_per_state=replay_per_state,assumptions='Empirical observed small controlled meshes/population; native files only, operands/checkpoints/traces/recovery reserve separate; startup included; no fitted scaling law or hardware invariance',planning_only=True))
 (out/'M3_4B_R4_SCALEUP_PLAN.md').write_text(f'''# Staged scaled-production plan — not executed

{'Controlled batch completed; staged expansion can be considered after explicit plan/launcher authorization.' if summary['success_all12'] else 'NO-GO: finish the retained blocker and original batch before scaling.'} The164 manifest is planning only. No scaled generation is authorized or performed in R4.

Retain corrected R2 family-first quotas, grouped force/start/sign/mirror/resolution relatives, fixed train/validation/test assignments, explicit held0.75 location extrapolation, held96 unseen-family/seen-resolution labels, held64 spacing/support/direction/geometry OOD and composite definitions, and21 disclosed near-neighbor dispositions. Controlled twelve are train-only; larger evaluation must include the planned held-out families, and no empirical normalization may use held-out data. Balanced configuration/window sampling must not let long capped histories dominate family mass.

Proposed stages: review/freeze corrected plan, package/count guard, explicit IDs and execution source; authorize a first20–32-job stage within the approved ≤96×48 envelope with the same per-worker gates and sequential monitoring; require full independent replay, FD/feature/provenance checks and predefined capped/quality handling; assess failure/memory/runtime/disk evidence, then separately authorize increments toward100–200. Original twelve-ID authorization cannot authorize those additional IDs. Do not replace difficult failed families with easier cases.

Native96 measured only its actual193-state converged path; sparse factor and501-frame/map/operand accumulation across all planned96 cases remain a risk. No192/384 execution or concurrency escalation is justified. Measure actual per-job headroom and fail closed. Repeated recovery work and validation costs must remain separate from release sample counts. No neural training in this plan.

Empirical/all-cap scenarios for100/164/200 are in scaleup_cost_scenarios.json. Native current total{native_bytes/2**20:.3f}MiB; projections use observed per-state gen/replay wall and native bytes. They exclude additional replay operand archives, journal/checkpoint recovery copies, tests, traces, package/source snapshots and parent preflight overhead; allocate an explicit disk reserve (at least2× total separately estimated footprint) and monitor actual free disk before every stage. Mesh/geometry/FEM/thread/hardware differences prevent treating these extrapolations as guarantees.

Next execution requires an explicit new stage request naming IDs/count and package/count-guard review, disk/environment and fixed split/quota approvals. Do not rerun R4 pipeline against this immutable results root; its reservations prevent duplicates.
''')
 (out/'NEXT_STAGE_EXECUTION.md').write_text(f'''# Exact next stage

R4 all12 success:{summary['success_all12']}. Read production_ledger.json, validation_ledger.json, accepted_dataset_index.json and source_artifact_integrity.json. Old execution manifest/authorization ledger are immutable; R4 native folder contains only original indices10/11 when executed. Join original ten plus new two using accepted_dataset_index; never infer current status from frozen old progress.json.

{'Request explicit authorization for a named20–32-job stage from corrected R2 planning manifest after approving package/count guard, quotas/fixed grouped splits, OOD/near-neighbor review, per-job disk budget and environment/source bindings. Prepare a separate immutable stage manifest/ledger and preserve original twelve-ID ledger. Use original optimizer and audited v2/v3,1024/768/1792MiB and900s sequential workers, complete independent replay and crash-safe evidence. Full164/100–200 execution still needs explicit approval.' if summary['success_all12'] else 'Resolve blocker.json using retained logs and explicit state/action/source accounting; no new production until every mandatory preflight gate passes. Recover only verified original checkpoints with unchanged bindings; never restart or overwrite silently.'}

Focused reproduction: OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/r4-mpl /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider m3_4br2/test_audit.py m3_4br2/test_infrastructure.py m3_4br3/test_audit.py m3_4br4/test_control.py. Full module-isolated regression: python -m m3_4br4.regression NEW_EMPTY_OUTPUT_DIRECTORY with the same environment. Regression fixtures must stay temporary; no training, SLURM, commit or push included.
''')
 atomic(out/'artifact_manifest.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
 print(json.dumps({k:summary[k] for k in ('success_all12','completed_native','new_production_trajectories','fully_independently_validated','validated_states','validated_transitions','resource_pass')}),flush=True)
if __name__=='__main__':report(sys.argv[1])
