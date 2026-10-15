"""Measured envelopes and structural storage scenarios, never a RAM safety extrapolation."""
from pathlib import Path
import numpy as np
from m3_5a.io import ROOT,RELEASE,R4,read,check
from .plan import risk,LIMITS

def measurements(audit):
 rows=[]
 for e in audit.release.entries:
  s=audit.specs[e['problem_id']];v=e['validation'];monitor=v.get('monitor');sup=v['supplemental_checks'];meta=read(Path(e['folder'])/'metadata.json')
  rows.append(dict(problem_id=e['problem_id'],mesh=s['configuration']['mesh'],domain=s['configuration']['domain'],support=s['support_pattern'],states=e['states'],termination=e['termination_reason'],native_array_bytes=sum((Path(e['folder'])/q['file']).stat().st_size for q in meta['arrays'].values()),replay_wall_seconds=monitor['wall_seconds'] if monitor else None,replay_worker_HWM_mib=max(monitor['worker_sampled_hwm_mib'],v['summary']['worker_memory'].get('VmHWM',0)) if monitor else None,supplemental_worker_HWM_mib=sup.get('VmHWM'),replay_source='R3 logical fragments; separate ledger' if monitor is None else v['evidence']))
 gen=[]
 for i in (10,11):
  c=read(RELEASE/'work_counts'/f'production_generate_{i}'/'result.json');m=read(RELEASE/'monitor'/f'production_generate_{i}.json')
  gen.append(dict(index=i,mesh=[48,32],states=c['states'],FEM=c['FEM'],OC=c['OC'],self_wall_seconds=c['wall_seconds'],monitored_wall_seconds=m['wall_seconds'],worker_HWM_mib=max(m['worker_sampled_hwm_mib'],c.get('VmHWM',0)),native_bytes=c['native_bytes']))
 return dict(released=rows,actual_new_generation=gen,historical_R4_resource_summary=read(R4/'resource_measurements.json'),approved_continuation_resources=read(RELEASE/'resource_measurements.json'),limits=LIMITS,generation_zero_projection_not_used='Historical R4 pre-continuation scaleup scenarios had0 generation time because no new production had run; use actual two continuation workers instead.')

def costs(audit,batch,measured):
 template=read(Path(audit.release.entries[0]['folder'])/'metadata.json')['arrays'];projections=[]
 gens=[g['self_wall_seconds']/g['states'] for g in measured['actual_new_generation']]
 for row in batch['jobs']:
  p=audit.specs[row['problem_id']]['mechanical'];n=p['nelx']*p['nely'];ndof=2*(p['nelx']+1)*(p['nely']+1);T=p['optimizer']['maxiter']+1;raw=0
  for key,meta in template.items():
   count=n if key in ('x','rho','g_rho','g_x') else ndof if key in ('u','k_diagonal') else ndof*2 if key=='k_probe_actions' else 1
   raw+=count*(T-1 if key=='update_change' else T)*np.dtype(meta['dtype']).itemsize+256
  # NPZ compressed archives cannot exceed these raw payload scenarios by large
  # compression-dependent factors; reserve extra headers/JSON independently.
  states=(10*n+3*ndof+2)*8*T;actions=10*n*8*(T-1);headers_json=(64*1024)*T
  same=[r['replay_wall_seconds']/r['states'] for r in measured['released'] if r['mesh']==[p['nelx'],p['nely']] and r['replay_wall_seconds'] is not None]
  projections.append(dict(problem_id=row['problem_id'],mesh=[p['nelx'],p['nely']],max_states=T,raw_native_array_bytes=int(raw),raw_independent_operand_bytes=int(states+actions),checkpoint_and_JSON_allowance_bytes=int(raw+headers_json),replay_seconds_scenario=[T*min(same),T*max(same)] if same else None,replay_scenario_basis='Observed same-grid rates including imports; not safety guarantee' if same else 'Unmeasured grid; no fitted timing claim',generation_seconds_small48x32_rate_reference=[T*min(gens),T*max(gens)],generation_reference_is_same_grid=[p['nelx'],p['nely']]==[48,32],resource_classification=row['resource']['classification']))
 footprint=sum(r['raw_native_array_bytes']+r['raw_independent_operand_bytes']+r['checkpoint_and_JSON_allowance_bytes'] for r in projections);reserve=2*footprint+1024**3
 import shutil
 return dict(projections=projections,all_cap_base_footprint_bytes=footprint,required_free_disk_scenario_bytes=reserve,free_disk_bytes_at_planning=shutil.disk_usage(ROOT).free,assumptions=['All selected jobs retain original500 update cap; actual early stops reduce storage','Native plus independent operands plus simultaneous staging frames plus64KiB/state JSON allowance','2x base plus1GiB for additional recovery copies,traces,source/checkpoints; recompute actual free disk before execution','Archive compression is not treated as a guaranteed saving','Memory is not extrapolated from disk/per-state storage or sampled RSS','Same-grid throughput rates are descriptive; boundary/load/material/factorization/condition changes can alter time','Generation rate available only from two48x32 continuation histories; other shapes marked reference-only'],memory_safety_certified=False,new_benchmarks_run=False)
