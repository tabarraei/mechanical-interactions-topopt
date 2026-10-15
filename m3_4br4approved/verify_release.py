"""Final read-only evidence aggregation and integrity verification; zero physics."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def run(out):
 out=Path(out).resolve();summary=read(out/'dataset_summary.json');entries=read(out/'accepted_dataset_index.json')['configurations'];maxima={};witness={};coverage=[]
 def add(c,label,pid,t):
  value=c['absolute_error']
  if value>maxima.get(label,-1):maxima[label]=value;witness[label]=dict(problem_id=pid,state_or_transition=t,absolute_error=value,absolute_tolerance=c.get('absolute_tolerance'),relative_error=c.get('relative_error'))
 def state(row,pid):
  assert row['passed']
  for k,c in row['primary'].items():add(c,'global_L2_'+k,pid,row['state'])
 def transition(row,pid):
  assert row['passed']
  assert all(row['strict_feasibility'].values())
  add(row['original_primary']['OC'],'global_L2_OC',pid,row['transition']);add(row['checks']['selected_multiplier'],'multiplier',pid,row['transition'])
 for e in entries:
  pid=e['problem_id'];val=e['validation'];T=e['states']
  if e['index']!=9:
   ev=Path(val['evidence']);states=[json.loads(l) for l in (ev/'states.jsonl').read_text().splitlines()];trans=[json.loads(l) for l in (ev/'transitions.jsonl').read_text().splitlines()]
  else:
   r2=ROOT/'m3_4br2/generations/20261014T124054Z';r3=ROOT/'m3_4br3/results';fragments=[r2/p for p in ('independent_replay','independent_replay_continuation','independent_replay_final')]+[r3/'independent_replay',r3/'independent_replay_completed']
   states=[json.loads(l) for f in fragments for l in (f/'states.jsonl').read_text().splitlines()]
   assert [s['state'] for s in states]==list(range(T))
   recovery=json.loads((r3/'recovered_state408/states.jsonl').read_text().strip());assert recovery['state']==408;states[408]=recovery
   ev=r3/'independent_replay_completed';trans=[read(ev/f'cached_transition_{t:05d}.json') for t in range(409)]+[json.loads(l) for l in (ev/'transitions.jsonl').read_text().splitlines()]
  assert [s['state'] for s in states]==list(range(T)) and [r['transition'] for r in trans]==list(range(T-1))
  for row in states:state(row,pid)
  for row in trans:transition(row,pid)
  coverage.append(dict(index=e['index'],states=len(states),transitions=len(trans),passed=True))
 # Keep historical R3 component maxima explicit and separate from aggregate norms.
 result=dict(passed=True,states=sum(r['states'] for r in coverage),transitions=sum(r['transitions'] for r in coverage),coverage=coverage,maximum_absolute_global_L2_errors=maxima,witnesses=witness,R3_only_absolute_component_maxima=read(ROOT/'m3_4br3/results/complete_validation.json')['maximum_absolute_component_discrepancies'],R3_state408='Recovered actual API pairing; original failed cross-pair certificate remains immutable',new_FEM=0,new_OC=0)
 (out/'maximum_discrepancies.json').write_text(json.dumps(result,indent=2)+'\n')
 summary['max_discrepancies']=maxima;summary['maximum_discrepancy_evidence']='maximum_discrepancies.json';summary['R3_only_component_maxima']=result['R3_only_absolute_component_maxima']
 (out/'dataset_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 p=out/'M3_4B_R4_INDEPENDENT_VALIDATION.md';s=p.read_text().replace('Dataset maximum discrepancies are reported with component versus global-L2 labels in dataset_summary.json;','Aggregate absolute global-L2 maxima across all12 accepted replays are in maximum_discrepancies.json with witnesses and complete coverage. The retained R3 component maxima are labeled R3-only separately;')
 s+='\nAggregate maxima (global L2): compliance '+str(maxima['global_L2_compliance'])+'; displacement '+str(maxima['global_L2_u'])+'; physical gradient '+str(maxima['global_L2_g_rho'])+'; design gradient '+str(maxima['global_L2_g_x'])+'; OC action '+str(maxima['global_L2_OC'])+'. All associated unchanged primary limits passed.\n';p.write_text(s)
 protected=read(out/'protected_before.json');assert all((ROOT/p).is_file() and digest(ROOT/p)==h for p,h in protected.items())
 source=read(out/'executing_source_manifest.json');assert all(digest(ROOT/p)==e['sha256'] and digest(out/e['archive'])==e['sha256'] for p,e in source.items())
 before=read(out/'artifact_manifest.json');changes=[p for p,h in before.items() if digest(out/p)!=h]
 assert set(changes)=={'dataset_summary.json','M3_4B_R4_INDEPENDENT_VALIDATION.md'}
 verification=dict(passed=True,protected_files=len(protected),protected_changed=0,executing_sources=len(source),executing_source_archives_match=True,aggregate_state_records=result['states'],aggregate_transition_records=result['transitions'],new_FEM=0,new_OC=0,report_enrichment_files=changes)
 (out/'final_verification.json').write_text(json.dumps(verification,indent=2)+'\n');(out/'verify_release_source.py.txt').write_text(Path(__file__).read_text())
 (out/'artifact_manifest.json').write_text(json.dumps({str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'},indent=2)+'\n')
 print(json.dumps(verification))
if __name__=='__main__':run(sys.argv[1])
