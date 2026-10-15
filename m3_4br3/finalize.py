"""Summarize actual retained evidence without changing native artifacts."""
import json,shutil
from pathlib import Path
import numpy as np
from m3_4br2.memory import digest,load_trajectory
from m3_4br2.preflight import ROOT,OLD,FAILED
from m3_4b.monitoring import reduce_summaries

def main(out):
 out=Path(out).resolve();r2=ROOT/'m3_4br2/generations/20261014T124054Z';new=out/'independent_replay_completed'
 def read(p):return json.loads(p.read_text())
 def write(name,data):(out/name).write_text(json.dumps(data,indent=2,allow_nan=False))
 fragments=[r2/x for x in ('independent_replay','independent_replay_continuation','independent_replay_final')]+[out/'independent_replay',new]
 states=[json.loads(line) for f in fragments for line in (f/'states.jsonl').read_text().splitlines()]
 indices=[r['state'] for r in states];unique=sorted(set(indices));assert indices==list(range(len(indices)))
 transitions=[read(new/f'cached_transition_{t:05d}.json') for t in range(409)]+[json.loads(line) for line in (new/'transitions.jsonl').read_text().splitlines()]
 summary=read(new/'summary.json') if (new/'summary.json').exists() else None
 tr=load_trajectory(OLD/'native/samples'/FAILED)
 peaks=reduce_summaries([r2/'replay_monitor'/n for n in ('r2_single_replay.json','r2_remaining_states.json','r2_final_states.json')]+[out/'monitor'/n for n in ('r3_remaining_states.json','r3_recover408.json','r3_complete409_500.json','r3_complete_retry.json')])
 history=[read(r2/'replay_monitor'/n) for n in ('r2_single_replay.json','r2_remaining_states.json','r2_final_states.json')]+[read(out/'monitor'/n) for n in ('r3_remaining_states.json','r3_recover408.json','r3_complete409_500.json','r3_complete_retry.json')]
 maxima={k:0. for k in ('design_vector','compliance','sensitivity_physical','sensitivity_design','OC_action','multiplier')}
 state_files={t:next(f/f'state_operands_{t:05d}.npz' for f in fragments if (f/f'state_operands_{t:05d}.npz').exists()) for t in unique}
 for t,f in state_files.items():
  with np.load(f) as z:
   for label,k in [('design_vector','x'),('compliance','compliance'),('sensitivity_physical','g_rho'),('sensitivity_design','g_x')]:maxima[label]=max(maxima[label],float(np.max(abs(z['saved_'+k]-z['replay_'+k]))))
 for f in list(state_files.values())+[out/'recovered_state408/state_operands_00408.npz']:
  with np.load(f) as z:
   for label,k in [('sensitivity_physical','g_rho'),('sensitivity_design','g_x')]:maxima[label]=max(maxima[label],float(np.max(abs(z['api_'+k]-z['saved_'+k]))))
 for r in transitions:
  maxima['multiplier']=max(maxima['multiplier'],r.get('checks',{}).get('selected_multiplier',{}).get('absolute_error',0.))
 for f in fragments+[out/'cached_state41',out/'recovered_state408']:
  for op in f.glob('operands_*.npz'):
   with np.load(op) as z:maxima['OC_action']=max(maxima['OC_action'],float(np.max(abs(z['saved_action']-z['replay_action']))))
 old_fail=[];v2_res=[];v3_res=[]
 for r in transitions:
  t=r['transition'];c=r.get('change_audit',{})
  if c.get('old_failed_new_passed'):v3_res.append(t);old_fail.append(dict(transition=t,quantity='incoming',original=c['original']))
  for key,d in r.get('diagnostics',{}).items():
   if not d['original']['passed']:old_fail.append(dict(transition=t,quantity=key,original=d['original']))
   if d.get('old_failed_new_passed'):v2_res.append(dict(transition=t,quantity=key))
 complete=bool(summary and summary['passed'] and len(states)==501 and len(transitions)==500 and all(r['passed'] for r in transitions))
 resource_pass=all(h['resource_event'] is None for h in history) and peaks['worker_peak_mib']<=1024 and peaks['parent_peak_mib']<=768 and peaks['combined_peak_mib']<=1792 and sum(h['wall_seconds'] for h in history)<=900
 if summary:resource_pass=resource_pass and summary['worker_memory'].get('VmHWM',0)<=1024
 result=dict(complete=complete,passed=complete,total_saved_states=501,independently_validated_states=len(states),independently_validated_transitions=sum(r['passed'] for r in transitions),state_indices=unique,FEM_calls=2*len(states)+1,necessary_repeated_API_FEM_calls=1,unique_independent_states=len(states),duplicate_independent_states=len(indices)-len(unique),original_trajectory_OC_actions=500,independently_reconstructed_actions=len(transitions),R3_cache_OC_reconstructions=42+43+409,R3_state41_cache_preparation_OC_calls=2,R3_new_state_OC_calls=(409-42)+(500-409),R3_recovery408_OC_calls=1,historical_comparator_failures=old_fail,v2_resolutions=v2_res,v3_resolutions=v3_res,maximum_absolute_component_discrepancies=maxima,volume_feasibility_violations=sum(not all(r.get('strict_feasibility',{}).values()) for r in transitions),stopping_consistency=all(r['change_audit']['saved_stop']==r['change_audit']['replay_stop'] for r in transitions),original_termination_reason=tr.metadata['termination_reason'],reference_labels_changed=False,resource_pass=resource_pass,resource=peaks,worker_wall_seconds=sum(h['wall_seconds'] for h in history),unresolved=[] if complete else ['Replay incomplete or failed; inspect immutable logs'])
 write('complete_validation.json',result);write('old_vs_new.json',dict(historical_failures=old_fail,v2_resolutions=v2_res,v3_resolutions=v3_res));write('resource_measurements.json',dict(summary=peaks,fragments=history,cumulative_seconds=result['worker_wall_seconds']))
 expected=read(out/'protected_before.json');changes=[p for p,h in expected.items() if not (ROOT/p).exists() or digest(ROOT/p)!=h]
 originals=read(OLD/'source_manifest.json');mismatches=[p for p,e in originals.items() if digest(ROOT/p)!=e['sha256'] or digest(OLD/e['archive'])!=e['sha256']]
 integrity=dict(files_checked=len(expected),changed=changes,original_source_archive_count=len(originals),source_mismatches=mismatches,ledger_sha256=digest(ROOT/'m3_4b/authorization.json'),passed=not changes and not mismatches)
 write('protected_artifact_integrity.json',integrity)
 reg=read(out/'regression_revision2/regression_isolated.json');gates=dict(complete_independent_replay=complete,v2_and_v3_regression=reg['passed'],protected_integrity=integrity['passed'],resources=resource_pass,native96_revised_memory_evidence=False,real_crash_restart_evidence=False,remaining_reserved_two_complete=False,planning_manifest='R2 schema/split review retained; no execution approval',authorization='Original twelve-ID ledger unchanged; R3 permits validation only')
 write('production_readiness.json',dict(decision_A=complete,decision_B=False,full_dataset=False,gates=gates))
 (out/'M3_4B_R3_TEST_REPORT.md').write_text(f'''# Actual test outcomes

Focused final:28 passed (focused_revision3.log); earlier24 and26 passes retained. Module-isolated complete regression:{reg['test_count']} passed; {len(reg['modules'])} modules, all exits successful:{reg['passed']}. Original temporary fixtures only; no production optimization. adversarial_outcomes.json records expected/actual, historical scalar outcomes and rejection reasons. R2 adversarial tests also rerun unchanged. Genuine errors including primary design/action failure, indexing, missing operands, shape/nonfinite, incorrect diagnostic and threshold crossing reject. Initial development test run had 23 passes, one fixture-path failure and one JSON serialization teardown error; preserved in focused_initial.log and corrected before replay. Regression log basename collision and pre-worker cache serialization error are preserved in development_history.md and launch_attempt2.log; completed regression modules were not repeated. Original15 synthetic harness passed separately in original_synthetic.log. No numerical policy was changed to remedy those harness errors. Final two cache-gate tests added after the complete270-test regression; all28 current R3 tests passed. Original/R2 modules remained byte-identical. The failed cached-only completion used adapted rather than historical primary evidence;43 cached OC actions and zero FEM retained and charged; corrected cache uses original_primary. See development_history.md.
''')
 (out/'M3_4B_R3_COMPLETE_REPLAY_REPORT.md').write_text(f'''# Complete independent replay

Decision A: {'PASS' if complete else 'FAIL'}. Saved states501; independent states{len(states)}; validated transitions{result['independently_validated_transitions']}/500; FEM API calls{result['FEM_calls']}; repeated independent states{result['duplicate_independent_states']}. R3 resumed at42 after revalidating original0..41 cached independent operands. A failed cross-solve gradient certificate at408 exposed missing actual API displacement bookkeeping. R3 records actual API solve displacement and retains unmodified v2 certificate. Exactly one necessary repeated API FEM call recovered408; first state408 solve was reused, and final worker starts409 after cached-prefix validation. Original failed408 evidence remains immutable. Total calls1003 versus1002 minimum; no hidden repeats. Each original saved x seeds its own FEM/OC action. Native trajectory, optimizer and reference metadata untouched.

Historical scalar failures:{len(old_fail)} comparisons; v2 residual resolutions:{len(v2_res)} comparisons; v3 incoming resolutions:{len(v3_res)} transitions. See old_vs_new.json for exact indices and original tolerances. All500 actions independently reconstructed. Extra R3 validation OC work:494 prefix recomputations across three workers (43 in a failed cached-only attempt),2 state41 cache preparations,1 state408 recovery action and458 new actions (including original failed408 action); this is validation, not free-running optimization. R2 retained prior counts separately in its replay_status.json.

Maximum absolute component discrepancies:{json.dumps(maxima)}. These are component maxima; unchanged primary global norms are in each state/transition record. Volume feasibility violations:{result['volume_feasibility_violations']}; stopping classifications preserved:{result['stopping_consistency']}. Original termination:{tr.metadata['termination_reason']}. The failed cross-pair certificate cannot retroactively be labeled a pass; actual missing API displacement was unavailable. Correctly paired recovered408 and subsequent states passed unchanged v2. Replay agreement does not establish convergence or reference optimality.

Resources pass:{resource_pass}; worker/parent/combined sampled maxima:{peaks['worker_peak_mib']:.3f}/{peaks['parent_peak_mib']:.3f}/{peaks['combined_peak_mib']:.3f}MiB. Cumulative monitored worker wall:{result['worker_wall_seconds']:.3f}s versus900s. Streaming traces and their checksums, per-state phases and checkpoints retained. Monitor CPU/write overhead and fragment wall timings in resource_measurements.json; recovery validation is part of R3 worker time, not separately isolated as an independent benchmark.
''')
 (out/'M3_4B_R3_PRODUCTION_READINESS.md').write_text(f'''# Production readiness

A. Existing501-state trajectory fully independently validated: {complete}.

Repaired validation instrumentation pairs each API sensitivity with the displacement from its own solve; no core or v2 certificate formula changed.

B. Infrastructure ready for explicitly authorized controlled production continuation: NO-GO. Complete replay alone cannot establish readiness. Revised96×48 native-memory evidence and real interrupted/restarted generation checks remain absent. Remaining two reserved jobs have not been generated. Full164 planning manifest execution remains unauthorized; split review/schema validity is retained R2 evidence, not an execution gate completion.

Regression passed:{reg['passed']}; integrity passed:{integrity['passed']}; resources passed:{resource_pass}. Numerical core, v2 and historical comparator unchanged; native feasibility/stopping/reference labels unchanged. All original{len(originals)} source/archive identities match; {len(expected)} protected files checked, changes:{changes}.

Before full training data: explicitly authorize controlled continuation/recovery work, verify completed-worker reuse with tripwires, perform real crash/restart controls, measure full revised96 replay headroom, complete remaining reserved jobs and applicable feature/FD/matched-history checks, independently validate the controlled batch, approve planning/splits/quotas/count/disk/environment/package/launcher gates and authorize the full workload. No training,HPC,192/384,new production trajectories,commit or push in R3.
''')
 (out/'CONTINUATION_PROTOCOL.md').write_text('''# Exact next stage — no execution in R3

Keep original m3_4b/generations/20261014T013017Z and twelve-ID authorization ledger. Bind R3 executing_manifest_revision4.json, complete_validation.json, native array hashes and complete v3 sidecars. Do not rerun validated states, overwrite native status/reference labels, reset ledger or initiate a replacement generation. Original capped result remains capped.

Next request must explicitly authorize controlled recovery/native96 validation and original reserved continuation, naming original IDs, isolated scratch controls and unchanged1024/768/1792MiB and900s gates. Follow m3_4br2/continue_protocol.md four fault points and its uninterrupted scratch reference protocol with separate immutable outputs; bind original source identities, commit-prefix hashes, explicit duplicate work accounting and v3 validation sidecars. Completed native controls must use completed_reused with FEM/update tripwires. Run no remaining generation until prerequisite native96 memory/recovery evidence is reviewed and continuation is authorized.

The remaining two reserved configurations require original specs/source and complete independent replay after generation, plus applicable feature/FD and matched-history checks. Full164 execution separately requires approved splits, quotas/count/disk/environment/launcher and explicit workload authorization. No training or SLURM included.

For a failed validation fragment, preserve monitor/phase/state/action evidence and reservation. Only a separately recorded validation recovery may revalidate a unique contiguous cached prefix after source/policy/hash checks, start from first unevaluated saved state, and charge cumulative runtime and every repeated FEM if dependency invalidation demands it. Do not rerun launch.py against immutable results.
''')
 write('artifact_manifest.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
 print(json.dumps({k:result[k] for k in ('complete','independently_validated_states','independently_validated_transitions','FEM_calls','resource_pass')}))
if __name__=='__main__':main(__import__('sys').argv[1])
