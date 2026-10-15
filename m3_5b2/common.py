"""Standard-library-only immutable selection and content-addressed bindings."""
import json,hashlib,os,sys,platform,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'m3_5b1/plans/20261015_family_expansion_v2/M3_5B1_SELECTED_BATCH.json'
PLAN_SHA='6605348244ec0e6935892898dc7bd82f51d3205219736f20bd1e336d97b66d08'
ORIGINAL=ROOT/'m3_4br4approved/results/20261015_approved_continuation/accepted_dataset_index.json'
ORIGINAL_SHA='303d5570d78ecbf1276459d99be6ed1420ccfb49b8f6c6224f38019bfe024a5a'
OLD=ROOT/'m3_4b/generations/20261014T013017Z'
LIMITS=dict(worker_rss_mib=1024,parent_rss_mib=768,combined_rss_mib=1792,worker_timeout_seconds=900,poll_seconds=.05)
BLOCKERS={98,130,138,145,151}
PARTITIONS={'train','validation','test_id','test_ood'}
def check(ok,msg):
 if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text(),object_pairs_hook=unique_pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+s)))
def unique_pairs(pairs):
 d={}
 for k,v in pairs:check(k not in d,'Duplicate JSON key '+k);d[k]=v
 return d
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def atomic(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');p.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def inside(base,name):
 p=(Path(base)/name).resolve();check(p.is_relative_to(Path(base).resolve()),'Path traversal');return p
def equivalent(s):
 p=dict(s['mechanical']);p.pop('seed',None);return canonical(dict(mechanical=p,initial_x=s['initial_x'],optimizer=s['optimizer']))
def environment():return dict(python=sys.version,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','pypardiso')},threads={n:os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')})
class Selection:
 def __init__(self):
  check(digest(PLAN)==PLAN_SHA,'Frozen manifest SHA256');self.manifest=read(PLAN);check(self.manifest['execution_authorized'] is False,'Frozen false flags');self.jobs={j['problem_id']:j for j in self.manifest['jobs']};check(len(self.jobs)==30,'Selected30')
  check(digest(ORIGINAL)==ORIGINAL_SHA,'Original release hash');self.original=read(ORIGINAL)['configurations'];self.original_ids={e['problem_id'] for e in self.original};self.specs={}
  for pid,j in self.jobs.items():
   check(j['execution_authorized'] is False and pid not in self.original_ids,'Frozen authorization/original12')
   p=inside(ROOT,j['problem_spec']['path']);check(digest(p)==j['problem_spec']['sha256'],'Immutable spec bytes');s=read(p);check(canonical(s)==j['spec_sha256'] and s['problem_id']==pid and s['related_group']==j['family'],'Immutable spec ID/family');self.specs[pid]=s
  self.production_sources={n:e['sha256'] for n,e in read(OLD/'source_manifest.json').items()};self.validation_sources=self.manifest['jobs'][0]['runner_source_fingerprints'].copy()
  self.validation_sources.update(self.manifest['approved_restart_policy']['files'])
 def verify_sources(self):
  for n,h in {**self.production_sources,**self.validation_sources}.items():check(digest(ROOT/n)==h,'Production/validation source hash '+n)
  for n,e in read(OLD/'source_manifest.json').items():check(digest(OLD/e['archive'])==e['sha256'],'Original source archive '+n)
  return True
 def implementation_sources(self):return {str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).parent.glob('*.py')}

def minimum_disk_bytes(selection,ids):
 """Conservative full-cap structural disk reserve, NOT a RAM qualification."""
 total=0
 for pid in ids:
  p=selection.specs[pid]['mechanical'];n=p['nelx']*p['nely'];nd=2*(p['nelx']+1)*(p['nely']+1);T=p['optimizer']['maxiter']+1
  native=(4*n+4*nd+64)*8*T;operands=(20*n+3*nd+2)*8*T;staging=native+65536*T;total+=native+operands+staging
 return 2*total+1024**3
