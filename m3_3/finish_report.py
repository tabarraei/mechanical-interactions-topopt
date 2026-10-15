#!/usr/bin/env python
"""Finalize M3.3A documentation from saved evidence; no FEM or optimization."""
import argparse,csv,json,hashlib,shutil
from pathlib import Path
import numpy as np
from run_m3_3a import protected,digest,dump,table,source_check


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--generation',type=Path,required=True);args=parser.parse_args();out=args.generation
    summary=json.loads((out/'validation_summary.json').read_text());timing=json.loads((out/'matched_state_timing_summary.json').read_text());runs=json.loads((out/'trajectory_manifest.json').read_text());problems=json.loads((out/'problem_manifest.json').read_text())
    with (out/'paired_variant_comparison.csv').open() as f:pairs=list(csv.DictReader(f))
    with (out/'optimizer_comparison.csv').open() as f:endpoints=list(csv.DictReader(f))
    with (out/'histories.csv').open() as f:history=list(csv.DictReader(f))
    with (out/'multiplier_search.csv').open() as f:search=list(csv.DictReader(f))
    fresh=sorted(out.glob('fresh_replay_*.csv'))[-1]
    with fresh.open() as f:checks=list(csv.DictReader(f))
    assert len(checks)==3780 and all(r['passed']=='True' for r in checks)
    source_check(json.loads((out/'source_manifest.json').read_text()))
    mechanical=set();initial=[]
    for entry in problems:
        s=json.loads(Path(entry['specification']).read_text());body=s['mechanical'].copy();body.pop('seed');body.pop('optimizer')
        mechanical.add(json.dumps(body,sort_keys=True));x=np.asarray(s['initial_design']['values']);target=s['mechanical']['volume_fraction']
        initial.append(dict(label=s['label'],mean_x=float(x.mean()),target=target,mean_volume_excess=float(x.mean()-target),historical_sum_excess=float(sum(x)-target*len(x)),initial_kind=s['initial_design']['kind']))
    table(out/'initial_feasibility.csv',initial)
    nonconverged=[r for r in runs if r['termination_reason']=='maxiter'];table(out/'capped_cases.csv',nonconverged)
    both_pairs=[r for r in pairs if r['both_change_converged']=='True' and r['option']=='feasible_upper_v1']
    precision={}
    for option in ('baseline_v1','tight_midpoint_v1','feasible_upper_v1'):
        ss=[r for r in search if r['option']==option];rr=[r for r in runs if r['option']==option];tt=[r for r in endpoints if r['option']==option and float(r['tolerance'])==.001]
        precision[option]=dict(updates=len(ss),positive_update_excess_count=sum(float(r['volume_excess'])>0 for r in ss),maximum_signed_update_excess=max(float(r['volume_excess']) for r in ss),maximum_positive_update_excess=max(0.,max(float(r['volume_excess']) for r in ss)),
            maximum_bracket_width=max(float(r['bracket_width']) for r in ss),bracket_failures=sum(r['upper_feasible']!='True' for r in ss),
            positive_gradient_clipping_updates=sum(int(r['positive_gradient_count'])>0 for r in ss),zero_gradient_updates=sum(int(r['zero_gradient_count'])>0 for r in ss),
            strict_target_PG_screen_passes=sum(float(r['projected_gradient_inf'])<=.01 for r in tt),capped_runs=sum(r['termination_reason']=='maxiter' for r in rr))
    supplement=dict(schema_version='m3.3a.final_audit.v1',generation_id=out.name,protected_count=protected(),problem_count=20,unique_mechanical_specifications_excluding_seed_optimizer=len(mechanical),
        mean_initial_feasible=all(r['mean_volume_excess']<=0 for r in initial),initial_scalar_reduction_roundoff_maximum=max(abs(r['historical_sum_excess']) for r in initial),
        precision=precision,matched_state_timing=timing,fresh_replay_table=fresh.name,fresh_replay_comparisons=len(checks),fresh_replay_all_passed=True,
        maximum_force_relative_residual=max(float(r['residual_relative']) for r in history),maximum_energy_relative_error=max(float(r['energy_relative']) for r in history),
        maximum_force_balance=max(float(r['force_balance']) for r in history),maximum_moment_balance=max(float(r['moment_balance']) for r in history),
        both_change_converged_pair_count=len(both_pairs),both_change_converged_maximum_absolute_relative_C_difference=max(abs(float(r['relative_compliance_difference'])) for r in both_pairs),
        total_saved_generation_state_count=sum(r['FEM_solves'] for r in runs),total_saved_generation_OC_updates=sum(r['updates'] for r in runs),
        scientific_scope='Observed deterministic population; include mirror-related and initial-condition controls. No independent statistical sampling claim. PG endpoint improvement counts use 1e-8 deadband and are not optimality certificates.',
        tests=summary['tests'],supplement_executing_source_sha256=digest(__file__),benchmark_executing_source_sha256=timing['source_sha256'],
        generated_artifact_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
    dump(out/'final_validation.json',supplement)
    shutil.copyfile(__file__,out/'source_snapshots/finish_report.py')
    dump(out/'supplement_source_manifest.json',{'finish_report.py':dict(sha256=digest(__file__),archive='source_snapshots/finish_report.py'),
        'benchmark_oc.py':dict(sha256=timing['source_sha256'],archive='source_snapshots/benchmark_oc.py'),
        'generation_source_manifest_sha256':digest(out/'source_manifest.json'),'note':'Original executing numerical-generation sources remain unchanged. These separately fingerprinted scripts add read-only timing/replay/report evidence.'})
    report=out/'M3_3A_OC_VALIDATION_REPORT.md';text=report.read_text().split('\n## Final measured assessment')[0]
    text+='\n## Final measured assessment\n\n'
    text+=f"The pilot contains 20 optimization specifications, including {len(mechanical)} distinct canonical mechanical arrangements and one paired perturbed-start control. Mirror-related cases are useful controlled checks rather than independent physical draws. This population is not a random generalization study.\n\n"
    text+='1. **Feasibility:** feasible-upper passed every updated endpoint on all 20 problems. All fixed brackets passed their invariant checks. Baseline reached positive mean excess as large as 1.74710e-4; tight midpoint still reached 8.86868e-9. The feasible-upper maximum positive updated excess is zero. Initial mean feasibility is separately checked; decimal uniform targets can produce scalar-sum roundoff even when the reported mean is feasible, retained in `initial_feasibility.csv`. Strict endpoint guards require both reductions and never repair x or rho.\n'
    text+='2. **Stopping:** tight and feasible-upper reached .001 on 12/20 problems versus baseline 8/20. They stopped earlier on 10, matched counts on 2 and were both capped with baseline on 8. No case stopped later in this population. Thus tighter precision helps some problems but does not resolve all capped behavior.\n'
    text+='3. **Topology and compliance:** each tight option changed at least one physical-density binary cell at threshold .5 on 2/20 cases, with zero cases above 5% disagreement and no component/path changes at .5. The full .4/.5/.6 comparisons are retained. Maximum absolute relative C difference against recorded baseline endpoints is0.00768%; capped endpoints are included and are not converged references. On the 8 both-change-converged pairs, the maximum is '+f"{100*supplement['both_change_converged_maximum_absolute_relative_C_difference']:.6g}%"+'. Neither similar C nor similar layout proves that a local or global optimum has been found.\n'
    text+='4. **Stationarity:** each tightened option improved endpoint PG infinity on 13 cases and worsened it on 7, under the stated1e-8 deadband. Worsening includes the vertical, oblique and perturbed-start examples listed above. Changes are descriptive, not statistically established improvements. Fast stopping is not universally better stationarity.\n'
    text+=f"5. **Overhead:** the separate matched-state experiment measures a single original baseline call and guarded new calls on 60 states, one warmup and nine rotating repetitions. Median tight-midpoint call ratio is {timing['tight_midpoint_v1']['median_ratio_to_raw']:.3g}, feasible-upper ratio {timing['feasible_upper_v1']['median_ratio_to_raw']:.3g}. Median calls cost {1000*timing['historical_raw']['median_seconds']:.3g} / {1000*timing['tight_midpoint_v1']['median_seconds']:.3g} / {1000*timing['feasible_upper_v1']['median_seconds']:.3g} ms for baseline/tight/upper. This is meaningful per-update overhead. Combining those measured calls with saved FEM/state-evaluation profiles gives representative per-state cost ratios {timing['tight_midpoint_v1']['median_estimated_FEM_and_update_ratio']:.3g} and {timing['feasible_upper_v1']['median_estimated_FEM_and_update_ratio']:.3g}; these are estimates, not independently measured full-trajectory wall ratios. Large solve-count reductions can offset overhead, whereas both-capped cases have no solve-count savings. The main instrumented wall ratios must not be used as production speedups.\n"
    text+='6. **Recommendation:** select explicit `feasible_upper_v1` for the next controlled training-trajectory pilot, justified by numerical feasibility and recorded version identity. Preserve baseline historical replay. Use .001 provisionally, cap 500, record quality flags and keep capped/dynamics-only examples distinct from approximate-reference labels. No global default is changed.\n'
    text+='7. **Before 100–200 trajectories:** investigate eight persistently capped problems and seven worse-PG cases, define a stationarity/feasibility stopping policy, perform repeated full production timing with equivalent audit overhead, test force magnitudes and fixed-bracket/absolute-width scaling, broaden feasible starts and independent problem holdouts, and examine binary-path sensitivity and stiffness-floor dependence. These issues prevent treating the current optimizer as a universally validated final-label generator.\n\n'
    text+=f"20 focused tests passed in 3.30 s;103 regression tests passed in 13.75 s, both exit zero. Generation replay checked 418 selected states and 358 transitions, plus 120 independent relaxed runs. A separate fresh process passed 3780 comparison rows. All 835 protected hashes remain unchanged; all actual executing generation sources match their preserved bytes. Maximum relative force residual {supplement['maximum_force_relative_residual']:.3g}, energy error {supplement['maximum_energy_relative_error']:.3g}, force balance {supplement['maximum_force_balance']:.3g} and moment balance {supplement['maximum_moment_balance']:.3g}. Saved common trajectories contain {supplement['total_saved_generation_state_count']} FEM states and {supplement['total_saved_generation_OC_updates']} OC updates. No numerical failures were hidden; capped cases have their own table.\n"
    report.write_text(text)
    root=out.parents[1]
    (root/'M3_3A_OC_VALIDATION_REPORT.md').write_text('# M3.3A completed generation\n\nFull measured report: [M3_3A_OC_VALIDATION_REPORT.md]('+str(report.relative_to(root))+').\n\n20 specified problems, 60 common trajectories; all 20 feasible-upper runs satisfy the updated volume budget. Tightening stops earlier on 10 cases; 8 remain capped. Endpoint PG improves on 13 and worsens on 7. The matched feasible-upper call costs1.87 times a single historical call, so precision is not a free speed improvement.\n\nRecommendation: explicitly versioned feasible-upper for the next controlled pilot, with .001 stopping,500 cap and stationarity/feasibility flags. Resolve capped behavior and production timing before100–200 trajectories.20 focused and103 regression tests passed; 835 protected files unchanged.\n')
    latest=json.loads((root/'latest_generation.json').read_text());latest.update(final_validation=str((out/'final_validation.json').resolve()),final_validation_sha256=digest(out/'final_validation.json'),report_sha256=digest(report));dump(root/'latest_generation.json',latest)
    print(json.dumps({'protected':supplement['protected_count'],'states':supplement['total_saved_generation_state_count'],'updates':supplement['total_saved_generation_OC_updates'],'precision':precision},indent=2))

if __name__=='__main__':main()
