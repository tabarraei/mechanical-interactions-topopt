"""Read-only gates and source archive before the single authorized replay."""
import json,sys,os,shutil,platform
from pathlib import Path
from .memory import digest,load_trajectory
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'m3_4b/generations/20261014T013017Z'
FAILED='m3_3b_d7d625d32faa15712edf39c5'
LIMITS=dict(worker_rss_mib=1024,parent_rss_mib=768,combined_rss_mib=1792,worker_timeout_seconds=900,poll_seconds=.05)

def check(out):
 before=json.loads((out/'protected_before.json').read_text());changed=[p for p,h in before.items() if not (ROOT/p).exists() or digest(ROOT/p)!=h]
 source=json.loads((OLD/'source_manifest.json').read_text());mismatches=[name for name,e in source.items() if digest(ROOT/name)!=e['sha256'] or digest(OLD/e['archive'])!=e['sha256']]
 tr=load_trajectory(OLD/'native/samples'/FAILED);ledger=json.loads((ROOT/'m3_4b/authorization.json').read_text());assert FAILED in ledger['problem_ids'] and ledger['source_id']==tr.metadata['source_id']
 regression=json.loads((out/'regression_isolated.json').read_text())
 memory=json.loads((out/'memory_results.json').read_text())
 matrix=json.loads((out/'acceptance_matrix.json').read_text());matrix_pass=all(row['expected']==row['actual'] for row in matrix)
 archive=out/'source_snapshots';archive.mkdir(exist_ok=True);records={}
 for p in sorted((ROOT/'m3_4br2').glob('*.py')):
  dest=archive/(p.name+'.txt')
  if dest.exists() and dest.read_bytes()!=p.read_bytes():raise ValueError('Archive changed; use separately recorded revision')
  if not dest.exists():dest.write_bytes(p.read_bytes())
  records[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(out)))
 (out/'executing_source_manifest.json').write_text(json.dumps(records,indent=2))
 checks=dict(protected_unchanged=not changed,original_sources_unchanged=not mismatches,full_isolated_regression_passed=regression['passed'],all_adversarial_expectations_passed=matrix_pass,synthetic_memory_passed=all(memory[k]['monitor']['resource_event'] is None for k in ('whole','frame')),ledger_binding=True,disk_free_gt_1GiB=shutil.disk_usage(out).free>2**30,thread_env=os.environ.get('OPENBLAS_NUM_THREADS')=='1',native_size_allowed=tr.problem.n==1536)
 result=dict(passed=all(checks.values()),checks=checks,changed=changed,source_mismatches=mismatches,trajectory=FAILED,states=tr.metadata['state_count'],ledger_sha256=digest(ROOT/'m3_4b/authorization.json'),limits=LIMITS,sources=records,environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS','unchanged solver16')),test_fixture_exception='User answered option1, allowing existing temporary regression fixtures only')
 (out/'replay_prerequisites.json').write_text(json.dumps(result,indent=2));print(json.dumps(checks));return result
if __name__=='__main__':check(Path(sys.argv[1]))
