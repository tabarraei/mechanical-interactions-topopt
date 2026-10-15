"""Training-initial-prefix-only mechanical RMS; analytical A scaling."""
import numpy as np
from .io import check,canonical,read
VERSION='m3.5a.normalization.initial_prefix.v1'
class Normalization:
 def __init__(self,record):
  self.record=record;self.identifier=canonical(record);self.scales=record['scales']
  check(record['version']==VERSION and record['fit_state_policy']=='initial_state_only' and record['targets_used'] is False,'Forbidden normalization policy')
  check(set(self.scales)=={'u','g_rho','g_x','compliance'} and all(np.isfinite(v) and v>0 for v in self.scales.values()),'Invalid normalization scales')
 def validate(self,release):
  r=self.record;check(r['release_sha256']==release.index_sha256,'Normalization release')
  check(r['fit_samples'] and len({x['problem_id'] for x in r['fit_samples']})==len(r['fit_samples']),'Duplicate normalization examples')
  for x in r['fit_samples']:
   check(x['problem_id'] in release.by_id and release.by_id[x['problem_id']]['partition']=='train' and x['state']==0,'Normalization forbidden data/future state')
   check(x['native_hashes']==release.by_id[x['problem_id']]['array_hashes'],'Normalization provenance')
  return True
 @classmethod
 def fit(cls,release,fit_samples=None):
  requested=fit_samples if fit_samples is not None else [(pid,0) for pid in release.by_id]
  check(len(requested)==len(set(requested)),'Duplicate normalization examples');sums={k:0. for k in ('u','g_rho','g_x','compliance')};counts={k:0 for k in sums};rows=[]
  for pid,t in requested:
   check(pid in release.by_id and release.by_id[pid]['partition']=='train' and t==0,'Normalization forbidden data/future state')
   a=release.open(pid);F=release.specs[pid]['force_scale']
   for k in sums:
    v=np.asarray(a[k][0],float)/(F if k=='u' else F**2);check(np.isfinite(v).all(),'Nonfinite normalization data');sums[k]+=float(np.sum(v*v));counts[k]+=v.size
   rows.append(dict(problem_id=pid,state=0,native_hashes=release.by_id[pid]['array_hashes']))
  check(bool(rows),'Empty normalization fit');rms={k:float(np.sqrt(sums[k]/counts[k])) for k in sums};scales={k:(v if v>=1e-12 else 1.) for k,v in rms.items()}
  obj=cls(dict(version=VERSION,release_sha256=release.index_sha256,fit_state_policy='initial_state_only',fit_samples=rows,targets_used=False,scales=scales,raw_RMS=rms,counts=counts,nearly_zero_fallback={k:v<1e-12 for k,v in rms.items()},vector_rule='Shared zero-centered joint RMS for both Cartesian u components; force vector normalized jointly by known norm; directions unchanged',analytical_rules='x,rho dimensionless unchanged; previous_delta/move; coordinates/domain; t/declared cap; force/F; mechanics first divided byF orF^2',provisional=True))
  obj.validate(release);return obj
 @classmethod
 def load(cls,path,release):
  obj=cls(read(path));obj.validate(release);return obj
