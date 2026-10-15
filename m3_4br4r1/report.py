"""Documentation/package only. Uses final retained results; no physics calls."""
import json,sys,shutil
from pathlib import Path
from .evidence import ROOT,R4,R3,read,digest

def write(out,name,obj):(out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def main(out):
 out=Path(out).resolve();final=out/'final_analysis'
 for p in final.iterdir():
  if p.is_file():shutil.copyfile(p,out/p.name)
 matrix=read(out/'R4_R1_ACCEPTANCE_MATRIX.json');state=read(out/'state66_displacement_analysis.json');vol=read(out/'all_failed_volume_decompositions.json');prefix=read(out/'committed_checkpoint_integrity.json');outcomes=read(out/'synthetic_outcomes.json');log=(out/'final_comparator_tests.log').read_text()
 if '55 passed' not in log or 'failed' in log.lower():raise ValueError('Final comparator-only tests')
 if not all(x['actual']==x['expected'] for x in outcomes):raise ValueError('Adversarial outcomes')
 allpass=all(r['proposed']=='Accepted' for r in matrix);assert len(matrix)==4
 before=read(out/'protected_before.json');changed=[p for p,h in before.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
 integrity=dict(passed=not changed,protected_files=len(before),changed=changed,baseline=read(out/'baseline_integrity.json'),zero_new_FEM=True,zero_new_OC=True,production_gate_modified=False)
 write(out,'source_artifact_integrity.json',integrity)
 if not integrity['passed']:raise ValueError('Protected source/artifact changed')
 # Spatial data exported directly from retained u, not a solve or reconstruction.
 import numpy as np
 from .evidence import PID
 ctrl=R4/'recovery/control/samples'/PID;other=R4/'recovery/metadata_before_rename/samples'/PID
 u=np.load(ctrl/'arrays/u.npy',mmap_mode='r')[66];v=np.load(other/'arrays/u.npy',mmap_mode='r')[66];d=v-u;nodes=np.asarray(read(ctrl/'problem.json')['mechanical']['node_coordinates']);delta=np.hypot(d[0::2],d[1::2]);budget=1e-12+1e-10*np.hypot(u[0::2],u[1::2]);old=abs(d)>1e-12+1e-10*abs(u)
 np.savetxt(out/'state66_spatial_differences.csv',np.column_stack([np.arange(len(nodes)),nodes,d[0::2],d[1::2],delta,budget,delta/budget,old[0::2],old[1::2]]),delimiter=',',header='node,x,y,delta_ux,delta_uy,nodal_error,nodal_allowance,nodal_ratio,old_x_failed,old_y_failed',comments='')
 rows=['| Fixture | Historical | Proposed | Exact checkpoint | Independent mechanics/OC | Historical sum failures |','|---|---|---|---|---|---:|']
 for r in matrix:rows.append(f"| {r['point']} | {r['historical']} | {r['proposed']} | {r['checkpoint']} | {r['independent_mechanical_OC']} | {len(r['historical_sum_excess']['failing_historical_states'])} |")
 (out/'R4_R1_RESTART_FAILURE_ANALYSIS.md').write_text(f'''# Retained restart failure analysis

Historical failures are preserved: frame-before-journal54 historical_sum_excess states; committed-update2 has24; metadata-before-rename has4 plus displacement at state66/DOF1879. Transition-before-journal passed. Every native field and all four histories were evaluated, with no new FEM, OC update, trajectory, or interruption.

For all82 failed sum comparisons, exact binary-rational sums separate operand changes from reduction and subtraction errors. Every recorded diagnostic reproduces its own historical Python sum and common rounded Nv*. Both designs pass unchanged native x comparison, box and strict NumPy-mean plus sequential-sum feasibility. Original target and stiffness/OC floors unchanged. Files: all_failed_volume_decompositions.json, proposal_*.json.

First state3 (frame-before-journal): restart-control residual difference−3.694822225952521e-12; exact vector-sum difference−3.826638658721571e-12; restart/control reduction errors1.7805308009238852e-12/1.6487143681548355e-12. Subtraction error is exactly0 here and in all82 failed sums. The residual is ill-conditioned relative to two much larger sums, but cancellation introduces no extra subtraction error in these retained cases. Underlying design operands differ; this is not an identical-array reduction-order demonstration. The output field cancellation_amplification is explicitly the relative residual change caused by exact operand difference, not a solver condition-number estimate.

State66: maximum component discrepancy{state['maximum_absolute_component_difference']:.17g}; full-vector relative discrepancy{state['relative_vector_difference']:.17g}; {state['different_DOFs']}/{state['total_DOFs']} DOFs differ (all{state['free_DOFs']} free DOFs). Only1879 fails the historical component threshold. Differences are spatially distributed rather than confined to that DOF; CSV includes every node and raw components.

DOF1879 is y at node939, position(8,19). Control node vector[2.473875333121175,0.0010037396847259814]; restart[2.47387533312221,0.0010037396859255414]. Local vector error1.5841735681208536e-12 versus the proposed1e-12+1e-10×2.4738755367477077=2.4838755367477076e-10. The isolated historical failure scales y by its small component while the physical displacement magnitude is about2.474. This explains the acceptance-contract discrepancy, not the upstream origin of the floating values.

At state66 both retained independent first/API displacements equal their own native displacement exactly. Individual force residuals≈1.55e-14/1.62e-14; backward-error certificates≈2.51e-18/2.62e-18; energy/compliance relative errors≈6.96e-14/8.97e-14. Source/configuration/load/support identities match exactly; replay environment records match. State66 is inside the162-state pre-interruption committed prefix; reconstructed frame SHA256 equals its original hash, proving the difference preceded metadata interruption.

Full K matrices are not saved. Forces are available, alongside K diagonals, Frobenius norms and two probe actions; those do not uniquely determine K. No new assembly, solve, pairwise energy norm or new backward-error guarantee was computed. Individual retained mechanical certificates support correctness of each history; the proposed local vector criterion separately defines numerical equivalence. Solver/thread scheduling cause remains unknown.
''')
 (out/'R4_R1_RESTART_ACCEPTANCE_SPEC.md').write_text('''# Proposed m3.4b.restart_acceptance.v2

Status: PROPOSED ONLY. Three categories must independently pass. Missing mandatory evidence yields Indeterminate; detected inconsistency/corruption yields Rejected; only all satisfied requirements yield Accepted. Historical comparator, OC residual audit v2 and incoming-change audit v3 remain unchanged.

A — Exact checkpoint integrity. Verify original source/configuration binding, pre-interruption journal bytes/hash, prefix state/transition order and entries. Reconstruct committed NPZ/JSON bytes from retained final arrays/transitions with the original deterministic serialization and require original SHA256 exactly. Require immutable committed-field digests against the proposal's actual inputs. Even a one-ULP committed design change fails. Original frame files were removed by documented finalization, but their exact recorded serialized content is recovered by matching hashes; this is serialization/comparison, not trajectory regeneration. No numerical allowance applies to any committed byte or identity.

B — Independently validated correctness. Verify frozen original source/archive, native files and full independent replay summaries, every state/transition comparison and actual saved/replayed operands. All current global reconstruction/gradient/OC/multiplier/assembly checks, strict box/move/mean/sequential-sum feasibility, physical residual/energy/balance thresholds and v2/v3 prerequisites remain mandatory. Source/configuration identities include exact loads/supports/material/filter/optimizer and force scale. Retain cap versus change-tolerance classification, first stopping point, incoming indexing and original count. No outcome changes reference eligibility. Missing certificates/operands fail closed, never get inferred from a passing control comparison.

C — Restart/control equivalence. Retain historical component criterion |a-b|≤1e-12+1e-10|b| on every ordinary native field, including x/rho/compliance/physical and design gradients/stiffness summaries/primary actions (next-state x). Each transition's original numerical/categorical diagnostics, selected upper and timing are checked. Only derived volume residuals and displacement receive the explicit alternatives below. Preserve the entire original pass/fail matrix.

Displacement: for every physical node, require ||u_restart,node−u_control,node||2≤1e-12+1e-10||u_control,node||2; both x/y components remain included. Require the unchanged independent-contract global u comparison (rtol1e-9 with its existing dimension/scale roundoff term) as well. Homogeneous constrained DOFs remain exactly zero. The per-node criterion prevents a large localized error from hiding in the global norm. It uses the historical relative coefficient and absolute floor, applied to a physical two-component vector, not the largest observed error or an arbitrary10×/100× multiplier. The ideal nodal criterion is invariant under an orthogonal change of Cartesian basis. A nearly zero component is controlled by its local vector magnitude; a nearly zero node is controlled by the absolute floor. This is an explicit engineering equivalence contract, not a claim that every scalar component or K-energy norm has the old relative accuracy. Tests reject localized perturbations which pass the global norm. Physical matrices need not be reconstructed because no new energy-norm claim is made.

Volume: recompute each diagnostic exactly using its original binary64 reduction/subtraction convention; any inconsistent saved scalar is Rejected. Independently require every design comparison and strict feasibility. Let S_a,S_b be exact rational sums of actual binary64 vectors; n their size, u=2^-53 and gamma_k=ku/(1-ku). For historical sum residuals, bound |r_a-r_b| by |S_a-S_b|+gamma_(n−1)(S_a+S_b)+u(|s_a|+|s_b|+2|rounded Nv*|), where s_a,s_b are actual rounded sequential sums. Mean residuals use |S_a-S_b|/n+gamma_n(S_a+S_b)/n plus the analogous subtraction allowance. Positive normal densities and supported binary64 layout/length avoid overflow/underflow; exact common target rounding cancels. Exact rational comparisons establish the bound; reporting rounds upward. Also decompose measured summation and subtraction errors and verify them against their allowances. A residual bound cannot waive primary-vector, mechanical, endpoint, feasibility, checkpoint or stopping failure. No arbitrary scale multiplier is introduced.

Trust model: the retained evidence adapter verifies the frozen R3/R4 artifact manifests, original source/archive fingerprints, native checksums, replay identities and actual retained operands. Synthetic certificate stubs in unit tests are test inputs, not evidence of FEM correctness. New histories require their own complete independent validation and frozen provenance before the proposal can apply.

Activation: explicit user approval must name this policy/source revision and category-C replacement. This package is not imported by production and does not rewrite old preflight/gate status. No approval is implied by successful proposal tests.
''')
 (out/'R4_R1_RETAINED_EVIDENCE_TESTS.md').write_text(f'''# Retained evidence outcomes

All four real fixtures Accepted under the proposed contract. Historical results remain three Rejected and one Accepted. All category-A exact hashes, category-B complete independent replay evidence, source/configuration/environment, strict feasibility and stopping checks pass. No mandatory evidence gap remains for this proposed contract; a pairwise K-energy guarantee is unavailable and is not claimed or required.

{chr(10).join(rows)}

Exact338 serialized checkpoint hash checks:171 committed frames and167 committed transition JSON files. All original interrupted journal hashes/entries/order match; immutable prefix field digests prevent later proposal inputs from changing committed data. State66 original bytes verify its pre-fault existence. Original compressed working frames were intentionally deleted by native finalization; exact reconstructed hash equality supplies their retained-content proof.

Maximum displacement local-error/allowance ratio across each complete history: frame-before-journal0.010858147080834225; transition-before-journal0.0016500280057252918; committed-update20.011458442271424759; metadata-before-rename0.009020680043498664. Coefficients were specified from the original criterion and physical node grouping, not fitted to those ratios. Check full per-state records in proposal_*.json.

Source manifests, old native and independent operand records, and all R3/R4 artifacts are bound in baseline_integrity.json. No expensive historical worker was rerun; only comparator evaluations and deterministic serialization executed. Final results are also preserved under final_analysis; earlier development comparator outputs remain distinct.
''')
 (out/'R4_R1_ADVERSARIAL_TEST_REPORT.md').write_text('''# Comparator-only adversarial validation

Final run:55 passed in18.20s (final_comparator_tests.log). Only new NumPy/comparator synthetic and retained-evidence tests were executed. Original FEM/generator/full regression suites were not rerun; R4's historical279-pass suite remains retained evidence.

The30-row synthetic acceptance/rejection/Indeterminate table is synthetic_outcomes.json. Tests cover exact/modified checkpoints, wrong state/transition/source/config/load/boundary identity, small/large/mean-preserving design shifts, incorrect actions/gradients/displacements, global-norm-hidden localized corruption, small distributed differences, strict infeasibility, stopping, missing certificate/displacement/checkpoint/binding, nonfinite/shape/dtype, residual cancellation/incorrect scalars, OC multipliers and mechanical limits. Missing mandatory evidence is Indeterminate, never Accepted. Genuine-error cases are Rejected. Additional tests verify node rotation invariance, near-zero-node protection, exact certificate bindings and zero-support displacement.

Retained tests cover all four real proposal outcomes, a single flipped compressed checkpoint byte in each fixture, a one-ULP committed design mutation, localized displacement corruption which passes the global norm but fails the nodal guard, missing real independent displacement operands and missing checkpoint payload binding. Original data were never modified: corruptions exist only in temporary bytes or synthetic in-memory copies. A deliberate in-memory certificate-digest substitution is used solely to isolate the numerical local guard; immutable real certificate files/manifest are not forged.

A strict one-ULP budget violation cannot borrow residual allowance. Two equally erroneous histories cannot borrow an unchanged global comparison to override the original backward-error limit. No changed tolerance was selected to encompass observed errors. No FEM/OC imports or calls in the analysis runner (explicit import guard).
''')
 decision='PASS' if allpass and integrity['passed'] else 'INDETERMINATE'
 policy_sha=digest(ROOT/'m3_4br4r1/policy.py');collector_sha=digest(ROOT/'m3_4br4r1/evidence.py');matrix_sha=digest(out/'R4_R1_ACCEPTANCE_MATRIX.json')
 write(out,'policy_decision.json',dict(decision_A=decision,decision_B='READY_FOR_EXPLICIT_POLICY_APPROVAL' if decision=='PASS' else 'BLOCKED',policy='m3.4b.restart_acceptance.v2',policy_sha256=policy_sha,evidence_adapter_sha256=collector_sha,acceptance_matrix_sha256=matrix_sha,fixtures_accepted=sum(r['proposed']=='Accepted' for r in matrix),historical_failures_preserved=True,production_gate_activated=False,explicit_policy_approval_required=True,original_jobs_10_11_launched=False,all_other_preflight_evidence='R4 verified gates retained; fresh integrity/environment/resource checks still required at continuation dispatch'))
 (out/'R4_R1_POLICY_APPROVAL_REQUEST.md').write_text(f'''# Concrete policy approval request

Decision A — {decision}. Recommend approving m3.4b.restart_acceptance.v2 as a scientifically defensible replacement for category-C restart/control numerical equivalence. Exact checkpoint integrity and complete independent mechanical/OC correctness remain separate mandatory gates. This is a local-vector engineering comparison plus a derived-residual forward-error contract; it is not a proof of pairwise energy-norm error or the solver's floating cause.

Decision B — {'READY_FOR_EXPLICIT_POLICY_APPROVAL' if decision=='PASS' else 'BLOCKED'}. All four retained fixtures pass the proposed contract;55 comparator-only tests pass; old comparison failures remain visible. Original production gate remains authoritative and failed until explicit approval. No production GO is issued by this report.

Approval scope: category-C replacement only, with frozen implementation policy.py SHA256 {policy_sha}, retained adapter evidence.py SHA256 {collector_sha}, and acceptance matrix SHA256 {matrix_sha}. Read RESTART_ACCEPTANCE_SPEC and the complete matrix. No FEM/OC/material/filter/thread/stopping change, no reference label change, no new workload IDs, no164 execution/training/SLURM/commit/push.

The user request explicitly states: “This task authorizes analysis, implementation, and testing of a proposed versioned restart-acceptance policy. It does not automatically authorize replacing the mandatory production gate or launching production.” Therefore this package does not activate the gate or dispatch original#10/#11. Approval must explicitly name this replacement; merely acknowledging the report is not approval.

After policy approval and rechecking all other mandatory preflight requirements, original#10/#11 can proceed under the existing R4 twelve-ID authorization. No new production authorization for those same IDs is inferred as necessary, but execution is outside this R4-R1 evidence-only task. Keep the original ledger and all historical reports immutable and use a new continuation sidecar root.
''')
 (out/'MINIMAL_PRODUCTION_CONTINUATION.md').write_text('''# Minimal continuation — preparation only

1. Obtain explicit approval of the exact m3.4b.restart_acceptance.v2 policy/source hashes in POLICY_APPROVAL_REQUEST. Write an approval receipt in a NEW continuation root; preserve original R4 production_preflight.json (failed historically), execution manifest and twelve-ID ledger. Do not silently import/install this proposal as the gate.

2. Bind the approval receipt, R4-R1 acceptance matrix/proofs, original77 numerical sources/archive identities, native arrays and old R3/R4 replay records. Recheck current environment/source/output hashes, disk and unchanged1024/768/1792MiB and900s sequential worker envelope. Category A and B remain mandatory; replace only category C by the explicitly approved policy in a new effective-preflight sidecar. No replay of already verified states or real fault experiments is required unless a verified dependency invalidates retained evidence. Original nine R4 full replays plus R3 complete501-state evidence remain reusable.

3. Use original execution_manifest jobs indices10 (m3_3b_93ad168499a05bed721257ad) and11 (m3_3b_59815dbc9ef20642f546f22e), their original specs/source IDs/loads/supports/domain/material/filter/volume/multiplier/feasible-upper/stopping settings and fixed train assignments. Never run a fresh plan/all command, reset ledger, expand IDs or overwrite old samples. Isolated scratch recovery histories remain test evidence; do not silently promote them into production samples.

4. Dispatch original generation in bounded monitored sequential workers to NEW_CONTINUATION_ROOT/native. Reviewable generation argv: python -m m3_4br4.worker NEW_CONTINUATION_ROOT INDEX NEW_CONTINUATION_ROOT/native production_INDEX none generate, for INDEX10 then11. This must be wrapped by the unchanged m3_4b.monitoring.run with original limits, exclusive immutable reservations and approval/effective-preflight checks; never execute the raw argv without that envelope. Existing completed10 must take reuse with solver/OC tripwires and zero numerical work.

5. After each new trajectory, dispatch bounded full independent replay: python -m m3_4br3.replay --trajectory NEW_CONTINUATION_ROOT/native/samples/ORIGINAL_ID --output NEW_CONTINUATION_ROOT/validation/ORIGINAL_ID --policy m3.4b.transition_change_audit.v3. Then execute original feature/FD/matched-history checks through m3_4br4.checks (3 extra FEM per new configuration), retaining counts, checksums, gate results, memory and stopping. Quarantine/exclude any invalid trajectory. Release an accepted dataset index only after all12 complete and full independent replay evidence covers every saved state and transition. Capped histories stay capped, reference flags stay false.

6. Reassess readiness for a separately authorized scaled20–32 stage/100–200 plan only after the twelve-ID batch completes. The164 planning manifest and training remain unexecuted and unauthorized here. No SLURM/commit/push.

Comparator-only reproduction: PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider m3_4br4r1/test_policy.py m3_4br4r1/test_retained.py. No thread-setting overrides are required or applied. Retained analysis reproduction may use python -m m3_4br4r1.analyze NEW_EMPTY_OUTPUT_DIRECTORY; its import guard prohibits all physics/generation modules. Do not rerun a production launcher from old immutable R4 roots.
''')
 archive=out/'final_source_snapshots';archive.mkdir(exist_ok=False);source={}
 for p in (ROOT/'m3_4br4r1').glob('*.py'):
  dest=archive/(p.name+'.txt');shutil.copyfile(p,dest);source[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(out)))
 write(out,'final_source_manifest.json',source)
 write(out,'artifact_manifest.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
 print(json.dumps(dict(decision_A=decision,decision_B='READY_FOR_EXPLICIT_POLICY_APPROVAL' if decision=='PASS' else 'BLOCKED',fixtures=4,tests=55,protected_files=len(before),changed=changed,new_FEM=0,new_OC=0,activated=False)),flush=True)
if __name__=='__main__':main(sys.argv[1])
