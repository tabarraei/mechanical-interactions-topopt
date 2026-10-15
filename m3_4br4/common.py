"""R4 paths, immutable reservations and unchanged resource envelope."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'m3_4b/generations/20261014T013017Z'
R3=ROOT/'m3_4br3/results'
LIMITS=dict(worker_rss_mib=1024,parent_rss_mib=768,combined_rss_mib=1792,worker_timeout_seconds=900,poll_seconds=.05)
def read(p):return json.loads(Path(p).read_text())
def results():return Path(read(ROOT/'m3_4br4/latest_results.json')['path'])
def job(index):
 m=read(OLD/'execution_manifest.json');j=m['jobs'][index];auth=read(ROOT/'m3_4b/authorization.json')
 if j['problem_id'] not in auth['problem_ids'] or auth['source_id']!=m['source_id']:raise ValueError('Original reservation')
 from m3_4b.monitoring import digest
 import hashlib
 spec=read(OLD/j['spec']);canonical=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 if canonical!=j['spec_sha256']:raise ValueError('Original canonical spec checksum')
 binding=read(OLD/'manifest_binding.json')
 if digest(OLD/j['spec'])!=binding['files'][j['spec']]:raise ValueError('Original spec file checksum')
 return m,j,read(OLD/j['spec'])
