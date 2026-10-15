"""Read-only bindings; reservation is R4 sidecar, original ledger immutable."""
import json,sys,shutil
from pathlib import Path
from .common import ROOT,OLD,R3,job
from m3_4b.monitoring import digest,atomic

def inspect(out):
 out=Path(out);baseline=json.loads((out/'baseline.json').read_text());source=json.loads((OLD/'source_manifest.json').read_text())
 bad=[k for k,e in source.items() if digest(ROOT/k)!=e['sha256'] or digest(OLD/e['archive'])!=e['sha256']]
 if bad:raise ValueError('Original source/archive '+str(bad))
 r3sources=json.loads((R3/'executing_manifest_revision4.json').read_text());relevant=['audit.py','replay.py','operands.py','test_audit.py','recover408.py']
 for name in relevant:
  e=r3sources['m3_4br3/'+name]
  if digest(ROOT/'m3_4br3'/name)!=e['sha256'] or digest(R3/e['archive'])!=e['sha256']:raise ValueError('R3 policy/source '+name)
 r3=json.loads((R3/'complete_validation.json').read_text());assert r3['complete'] and r3['independently_validated_states']==501 and r3['independently_validated_transitions']==500 and r3['FEM_calls']==1003
 for name,h in json.loads((R3/'artifact_manifest.json').read_text()).items():
  if digest(R3/name)!=h:raise ValueError('R3 evidence '+name)
 for i,b in enumerate(baseline['jobs']):
  m,j,spec=job(i)
  if b['metadata']:
   meta=b['metadata'];folder=Path(b['folder']);assert meta['source_id']==m['source_id'] and meta['problem_id']==j['problem_id']
   if json.loads((folder/'problem.json').read_text())!=spec:raise ValueError('Saved specification identity')
   for k,e in meta['arrays'].items():
    if digest(folder/e['file'])!=e['sha256']:raise ValueError('Saved native array '+k)
   for file,key in [('problem.json','problem_sha256'),('transitions.json','transitions_sha256')]:
    if digest(folder/file)!=meta[key]:raise ValueError('Saved metadata binding')
  elif b['incomplete_exists']:raise ValueError('Unexpected original incomplete stage requires explicit safe recovery review')
 if shutil.disk_usage(out).free<3*2**30:raise ValueError('Insufficient disk for bounded fixtures/native/replay')
 records={};archive=out/'source_snapshots';archive.mkdir(exist_ok=False)
 for p in list((ROOT/'m3_4br4').glob('*.py'))+list((ROOT/'m3_4br3').glob('*.py'))+list((ROOT/'m3_4br2').glob('*.py')):
  dest=archive/(p.parent.name+'_'+p.name+'.txt');shutil.copyfile(p,dest);records[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(out)))
 atomic(out/'source_manifest.json',records)
 atomic(out/'baseline_binding_checks.json',dict(passed=True,original_source_archive_count=len(source),R3_complete_replay_bound=True,original_ids=[j['problem_id'] for j in baseline['jobs']],completed_before=10,unexecuted_before=2,original_ledger_sha256=digest(ROOT/'m3_4b/authorization.json'),original_manifest_sha256=digest(OLD/'execution_manifest.json'),fixed_partitions={j['problem_id']:j['partition'] for j in baseline['jobs']},protected_files=baseline['protected_files'],free_disk_bytes=shutil.disk_usage(out).free))
if __name__=='__main__':inspect(sys.argv[1])
