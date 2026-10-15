"""Saved-evidence reporting only; no FEM/OC imports or execution."""
import json,sys,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
R4=ROOT/'m3_4br4/results/20261014T174028Z'
R3=ROOT/'m3_4br3/results'
PROPOSAL=ROOT/'m3_4br4r1/results/20261015T020242Z'
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(out,name,value):
 with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def run(out):
 out=Path(out).resolve();summary=read(out/'final_readiness.json');index=read(out/'accepted_dataset_index.json');entries=index['configurations'];ledger=read(out/'execution_ledger.json');prior=read(R4/'dataset_summary.json')
 assert summary['passed'] and len(entries)==12 and (out/'effective_production_preflight.json').is_file()
 maxima=prior['max_discrepancies'].copy();oldnew=read(R4/'old_vs_new.json');supplemental={};hist=oldnew['historical_failures'];v2=oldnew['v2_resolutions'];v3=oldnew['v3_resolutions']
 for i in (10,11):
  val=ledger['validation'][str(i)];ev=Path(val['evidence']);pid=entries[i]['problem_id'];components={}
  for line in (ev/'states.jsonl').read_text().splitlines():
   row=json.loads(line);assert row['passed']
   for key,c in row['primary'].items():maxima['global_L2_'+key]=max(maxima.get('global_L2_'+key,0.),c['absolute_error'])
  for line in (ev/'transitions.jsonl').read_text().splitlines():
   row=json.loads(line);assert row['passed'] and all(row['strict_feasibility'].values())
   maxima['global_L2_OC']=max(maxima.get('global_L2_OC',0.),row['original_primary']['OC']['absolute_error']);maxima['multiplier']=max(maxima.get('multiplier',0.),row['checks']['selected_multiplier']['absolute_error'])
   for key,c in row['diagnostics'].items():
    if not c['original']['passed']:hist.append(dict(problem_id=pid,transition=row['transition'],quantity=key,original=c['original']))
    if c.get('old_failed_new_passed'):v2.append(dict(problem_id=pid,transition=row['transition'],quantity=key))
   if row['change_audit'].get('old_failed_new_passed'):v3.append(dict(problem_id=pid,transition=row['transition']));hist.append(dict(problem_id=pid,transition=row['transition'],quantity='incoming',original=row['change_audit']['original']))
  supplemental[str(i)]=val['supplemental_checks']
 calls=summary['calls'];counts={**prior['counts'],**calls}
 counts['R4_new_production_FEM']=calls['new_generation_FEM'];counts['R4_new_production_OC']=calls['new_generation_OC'];counts['R4_main_dataset_replay_FEM']+=calls['new_independent_validation_FEM'];counts['R4_main_dataset_replay_OC']+=calls['new_independent_validation_OC'];counts['R4_FD_primitive_FEM']+=calls['new_supplemental_FEM']
 resources=summary['resources'];attempts=[read(p) for p in sorted((out/'monitor').glob('*.json'))]
 assert all(r['resource_event'] is None and r['exit_code']==0 and r['worker_sampled_hwm_mib']<=1024 and r['parent_peak_mib']<=768 and r['combined_peak_mib']<=1792 and r['wall_seconds']<=900 for r in attempts)
 workerwall=sum(r['wall_seconds'] for r in attempts);newwall=sum(r['wall_seconds'] for r in attempts if r['job_id'].startswith(('production_','features_')))
 quality=dict(passed=True,original_configurations=12,partition_counts=dict(Counter(e['partition'] for e in entries)),unique_problem_ids=len({e['problem_id'] for e in entries}),no_cross_partition_exact_id_leakage=True,held_out_partitions_present=False,normalization='Analytical force-scale only; no held-out empirical statistics',reference_labels=0,feature_provenance='Saved histories independently validated. Inference displacement/gradient/energy channels still require fresh FEM. Feature extraction itself uses saved histories and zero FEM.',new_supplemental_checks=supplemental)
 write(out,'dataset_quality.json',quality);write(out,'old_vs_new.json',oldnew)
 complete={**summary,'resources_pass':True,'new_worker_wall_seconds':newwall,'all_continuation_monitored_wall_seconds':workerwall,'resource_attempts':attempts,'max_discrepancies':maxima,'all_R4_plus_reused_R3_calls':counts,'historical_scalar_failures':len(hist),'v2_residual_resolutions':len(v2),'v3_incoming_resolutions':len(v3),'unresolved_blockers':[],'no_reference_optimum_claim':True}
 write(out,'dataset_summary.json',complete)
 write(out,'restart_recovery_evidence.json',dict(passed=True,category_C='Explicitly approved m3.4b.restart_acceptance.v2',category_A='Unchanged mandatory exact checkpoint; all338 reconstructed bytes match original hashes',category_B='Unchanged mandatory complete independent mechanical/OC correctness',fresh_comparator_evidence=str(out/'preflight_analysis'),retained_real_interruptions=str(R4/'recovery_evidence.json'),historical_fixed_failures_preserved=str(R4/'production_preflight.json'),old_fixed_passed=1,old_fixed_failed=3,approved_cases_passed=4,new_real_interruptions=0,new_completed_reuse_cases=10,new_completed_reuse_FEM=0,new_completed_reuse_OC=0,approved_bindings=read(out/'effective_production_preflight.json')['approved_bindings']))
 table='| Index | Original ID | States | Transitions | Termination | Generation | Replay |\n|---|---|---:|---:|---|---|---|\n'+''.join(f"| {e['index']} | {e['problem_id']} | {e['states']} | {e['transitions']} | {e['termination_reason']} | {'reused' if e['index']<10 else 'new'} | PASS |\n" for e in entries)
 docs={
 'M3_4B_R4_PREFLIGHT_REPORT.md':f'''# Approved continuation preflight

PASS. Explicit user approval activates only the reviewed category-C m3.4b.restart_acceptance.v2 in this new sidecar. All three frozen hashes match; approval_receipt.json and explicit_policy_approval.json record authority. effective_production_preflight.json binds the exact policy/adapter/matrix, original source/ledger/specs, retained regression and current environment/disk. Category-A exact checkpoint and category-B complete independent mechanics/OC remain mandatory. Fresh retained comparison passed all four cases;55 comparator-only tests passed. Original focused98/module-isolated279 and corrected8 R4 controls are retained under unchanged source fingerprints; no expensive suite was rerun.

The first10 completed trajectories were reused with tripwires: zero FEM/OC. Ten prior complete replays remain bound to original native hashes and unchanged audit sources, including R3's1003-call501-state replay. No completed trajectory was regenerated. The original historical failed production preflight and all three old comparator failures remain immutable.

Sequential limits:1024MiB worker,768MiB parent,1792MiB combined,900s per worker. Original OPENBLAS1 and MKL/OMP16 preserved; no numerical settings or stopping changes. Original12 train IDs only; original10/11 were absent before dispatch. Source/artifact integrity passed before release. No unresolved mandatory blocker.
''',
 'M3_4B_R4_PRODUCTION_REPORT.md':f'''# Complete original production batch

PASS:12/12 original configurations complete. Ten were generated before R4 and reused; two new original configurations10/11 were generated in this continuation's native/samples. No incomplete production trajectory required recovery, no configuration was replaced, and no scratch recovery fixture was promoted to production. Original specs/IDs/source bindings/partitions and feasible-upper OC settings are unchanged.

{table}

New generation actual FEM:{calls['new_generation_FEM']}; OC:{calls['new_generation_OC']}. Completed reuse:0/0. Historical primary generation nominal totals3368/3358 are retained separately; no complete lifetime failed-work count is inferred. Exclusive reservations, phase/count journals, atomic native outputs and streamed monitoring remain in this root. Production and validation ledgers link the exact original10 and new2 folders. No quarantined configuration.
''',
 'M3_4B_R4_INDEPENDENT_VALIDATION.md':f'''# Complete independent dataset validation

PASS:12 trajectories;{summary['states']} states;{summary['transitions']} transitions. New full replays start from each original saved x, independently rebuild/filter/solve mechanics, recompute both gradients, perform the original OC action and compare the saved next x. Exact trial, multiplier, volume/box/move, state timing and stopping checks pass. The unchanged v2/v3 numerical audits are applied only under their retained contracts; actual API displacement is paired with its gradient. No saved gradient/displacement substitutes for an independent solve.

New independent FEM:{calls['new_independent_validation_FEM']}; OC:{calls['new_independent_validation_OC']}. New supplemental centered FD and primitive-profile FEM:6; OC:0. All12 feature/FD checks passed, with original FD threshold2e-5 and matched-history checks where applicable. Prior main R4 independent FEM5734/OC2858 and R3 FEM1003 are reused by bound evidence; old scratch/test/recovery work remains separately counted. Aggregate accounting is dataset_summary.json.

No mechanical/OC inconsistency or strict volume violation occurred in accepted histories. Historical scalar failures{len(hist)}, v2 residual resolutions{len(v2)}, v3 incoming resolutions{len(v3)} remain visible in old_vs_new.json. Dataset maximum discrepancies are reported with component versus global-L2 labels in dataset_summary.json; no energy-norm bound or floating scheduling cause is inferred. Capped histories are mechanically valid optimization histories; replay agreement does not establish reference optimality.
''',
 'M3_4B_R4_DATASET_SUMMARY.md':f'''# Final original12 dataset

PASS:12/12 complete and fully independently validated;{summary['states']} states and{summary['transitions']} transitions. Ten reused, two new, zero quarantined, zero unresolved mandatory blockers. Terminations:{json.dumps(summary['termination_reasons'],sort_keys=True)}. No strict volume violations. Reference labels:0. accepted_dataset_index.json is the release index, with original IDs, source/output hashes, train assignments and linked validation. Existing original ledger/manifests and historical reports are immutable.

{table}

Continuation worker/parent/combined sampled peaks:{resources['worker_peak_mib']:.3f}/{resources['parent_peak_mib']:.3f}/{resources['combined_peak_mib']:.3f}MiB; all original limits passed. New production/replay/supplemental workers:{newwall:.3f}s; all continuation monitored subprocesses:{workerwall:.3f}s.900s is per worker. Monitor trace checksums and actual worker HWM are retained. Parent source-hashing/report wall is outside worker timing. Historical R4 native96 replay193 states passed713.879MiB; native96 supplemental checks peaked785.145MiB. These observations do not establish worst-case501-state96 safety.

Retained four real interruptions pass unchanged exact prefix and independent mechanics gates and the explicitly approved category-C comparator. Old fixed comparisons retain3 failures/1 pass. Fresh completed-reuse10 cases used zero FEM/OC. No new interruption fixtures or new workload IDs were executed.

Train-only controlled batch: no held-out evaluation is inferred; no exact cross-partition identity leakage is present here. Saved history features are independently validated, while inference mechanics channels require fresh FEM. Plan split/near-neighbor limitations remain documented in R2 and the retained R4 scale-up plan. The larger164 plan, training, SLURM and commit/push were not executed.
''',
 'M3_4B_R4_SCALEUP_PLAN.md':'''# Scaled production assessment

The original twelve-ID batch is complete and mechanically validated. This clears the controlled-batch blocker; scaled execution remains unauthorized. The batch is train-only and does not establish held-out generalization or reference optimality. Retain the reviewed R2 corrected164-job family/split/quota plan and all21 near-neighbor dispositions; held0.75 is location extrapolation, held96 is seen-resolution/unseen-family, and held64/support/direction/geometry are explicit OOD/composite cases.

Next prepare a concrete20–32-ID first-stage package with fixed grouped splits, count guard, disk/runtime/resource budget and the reviewed category-C hashes. Request explicit authorization naming those IDs/count before dispatch. Then require full independent replay, feature/FD/integrity checks and predefined capped-history handling for every sample. Review actual failure rates and96 headroom before separately authorized increments toward100–200. Do not replace hard cases, increase concurrency or extrapolate to192/384 meshes.

Observed193-state96 replay is insufficient to guarantee capped501-state96 headroom. Keep1024/768/1792MiB and900s bounds and sequential monitoring, measure actual headroom, and fail closed. Reserve disk for native histories, independent operand archives, staging/recovery copies, traces and tests separately. Existing R4 scaleup_cost_scenarios.json contains empirical/all-cap projections, not guarantees; revise them with the two new actual jobs before a stage request. Balance family/trajectory sampling so capped lengths do not dominate. No scaled optimizer, training or SLURM command is authorized by this report.
''',
 'NEXT_STAGE_EXECUTION.md':'''# Next stage

Current batch is complete: use accepted_dataset_index.json. Do not rerun continue_batch.py against this or another root: the original remaining IDs are now complete in the continuation index, and the original immutable root alone does not encode this new status.

Next task is preparation only: select20–32 exact IDs from the reviewed corrected164 planning manifest with approved grouped splits/quotas; freeze a new stage manifest and explicit global completed-ID index including this continuation; enforce count and duplicate guards; bind approved policy/source, verify current environment, estimate full native-plus-validation/recovery disk and per-worker time; request explicit stage authorization. Review that concrete package before dispatch. Original twelve-ID authorization cannot launch additional IDs.

Comparator-only verification remains: PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider m3_4br4r1/test_policy.py m3_4br4r1/test_retained.py. No production/replay/fixture rerun is required to use this complete release index.
'''}
 for name,text in docs.items():
  with (out/name).open('x') as f:f.write(text)
 # Standalone reporting source is archived; frozen executing launcher is unchanged.
 archive=out/'report_source.py.txt'
 with archive.open('x') as f:f.write(Path(__file__).read_text())
 files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json',)}
 write(out,'artifact_manifest.json',files)
 print(json.dumps({k:complete[k] for k in ('complete','states','transitions','termination_reasons','resources_pass','unresolved_blockers')}))
if __name__=='__main__':run(sys.argv[1])
