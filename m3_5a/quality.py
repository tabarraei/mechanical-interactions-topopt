"""Streaming saved-array diagnostics and complete transition/causality checks."""
import sys,time,json,resource
from pathlib import Path
import numpy as np
from .loader import Release,TrajectoryDataset
from .normalization import Normalization
from .io import read,write,check

def summary(v):
 a=np.asarray(v,float);return dict(count=len(a),minimum=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),maximum=float(a.max()),quantiles={str(q):float(np.quantile(a,q)) for q in (.01,.1,.25,.75,.9,.99)})
def rss():
 values={}
 for l in Path('/proc/self/status').read_text().splitlines():
  if l.startswith(('VmRSS:','VmHWM:')):values[l.split(':')[0]]=int(l.split()[1])/1024
 return values

def run(config,out):
 cfg=read(config);out=Path(out);out.mkdir(parents=True,exist_ok=False);before=rss();start=time.perf_counter()
 with Release(cfg['release_index'],cfg['release_sha256'],cfg['release_manifest_sha256']) as r:
  verified=rss();norm=Normalization.load(cfg['normalization'],r);ds={m:TrajectoryDataset(r,m,cfg['history'],norm) for m in ('A','B')};rows=[];rms=[];inf=[];cos=[];phases={k:[] for k in ('early','middle','late')};hist={k:np.zeros(40,np.int64) for k in ('x','rho')};bins=np.linspace(0,1,41);density={k:dict(count=0,sum=0.,minimum=1.,maximum=0.) for k in hist};nonfinite=0;seen=set();target_seconds=feature_seconds=0.;feature_bytes_max=0;bytes_native=0
  for e in r.entries:
   pid=e['problem_id'];spec=r.specs[pid];a=r.open(pid);T=e['states'];p=spec['mechanical'];local=[];prev=None;nonfinite_here=0;load0=time.perf_counter()
   for k,v in a.items():
    for t in range(0,len(v),32):nonfinite_here+=int(np.count_nonzero(~np.isfinite(v[t:t+32])))
   nonfinite+=nonfinite_here
   for name in hist:
    for frame in a[name]:
     hist[name]+=np.histogram(frame,bins)[0];density[name]['count']+=frame.size;density[name]['sum']+=float(frame.sum());density[name]['minimum']=min(density[name]['minimum'],float(frame.min()));density[name]['maximum']=max(density[name]['maximum'],float(frame.max()))
   for t in range(T-1):
    tick=time.perf_counter();delta=r.target(pid,t);target_seconds+=time.perf_counter()-tick;check((pid,t) not in seen,'Duplicate example');seen.add((pid,t));mag=float(np.sqrt(np.mean(delta**2)));mx=float(np.max(abs(delta)));local.append(mag);rms.append(mag);inf.append(mx);phases[('early','middle','late')[min(2,3*t//(T-1))]].append(mag)
    if prev is not None:
     den=float(np.linalg.norm(delta)*np.linalg.norm(prev));cos.append(float((delta.ravel()@prev.ravel())/den) if den>1e-20 else 0.)
    prev=delta
    # Complete pair audit in both modes; model feature construction never reads
    # post-update fields. t+1 is read only by the supervised target function.
    tick=time.perf_counter()
    for mode,d in ds.items():
     s=d.sample(pid,t);x=s['inputs'];check(s['proof']['transition']==t and s['proof']['next_state']==t+1,'Wrong pair');check(np.array_equal(s['target'][0],delta.astype(np.float32)),'Target mismatch');check(np.all(x['history_indices'][x['history_valid']]<=t),'Future input time');check(not x['element_history'][~x['history_valid']].any() and not x['node_history'][~x['history_valid']].any(),'History padding')
     feature_bytes_max=max(feature_bytes_max,sum(v.nbytes for v in x.values() if isinstance(v,np.ndarray))+s['target'].nbytes)
    feature_seconds+=time.perf_counter()-tick
   native=sum((Path(e['folder'])/v['file']).stat().st_size for v in read(Path(e['folder'])/'metadata.json')['arrays'].values());bytes_native+=native
   rows.append(dict(problem_id=pid,mesh=[p['nelx'],p['nely']],domain=[p['width'],p['height']],support=spec['support_pattern'],related_group=spec['related_group'],partition=e['partition'],states=T,transitions=T-1,termination=e['termination_reason'],reference=False,nonfinite=nonfinite_here,update_RMS=summary(local),native_array_bytes=native,scan_seconds=time.perf_counter()-load0))
  check(len(seen)==3697 and sum(e['states'] for e in rows)==3709 and nonfinite==0,'Complete finite coverage')
  for k,v in density.items():v['mean']=v.pop('sum')/v['count']
  result=dict(passed=True,trajectories=rows,usable_trajectories=12,usable_transitions=3697,saved_states=3709,termination_counts={k:sum(e['termination']==k for e in rows) for k in ('change_tolerance','maxiter')},trajectory_length=summary([e['states'] for e in rows]),update_RMS=summary(rms),update_Linf=summary(inf),near_zero=dict(RMS_threshold=1e-5,count=sum(v<1e-5 for v in rms),fraction=sum(v<1e-5 for v in rms)/len(rms),below_stop_change_count=sum(v<.001 for v in inf)),phase_update_RMS={k:summary(v) for k,v in phases.items()},phase_definition='posthoc thirds of observed transition count; quality diagnostics only, never an input',density=density,density_histograms=dict(edges=bins.tolist(),counts={k:v.tolist() for k,v in hist.items()},weighting='Every saved elemental value; fine meshes receive more weight'),adjacent_update_cosine=summary(cos),nonfinite_values=nonfinite,duplicate_examples=0,causal_examples_checked_per_mode=len(seen),release_sha256=r.index_sha256,normalization_id=norm.identifier,features_by_mode={m:__import__('m3_5a.features',fromlist=['schema']).schema(m) for m in ('A','B')},performance=dict(before=before,after_provenance_verification=verified,after_all_loading=rss(),release_verify_seconds=r.verification_seconds,target_construction_seconds=target_seconds,both_mode_feature_seconds=feature_seconds,maximum_single_sample_bytes=feature_bytes_max,total_native_array_bytes=bytes_native,wall_seconds=time.perf_counter()-start,cache='One mmap trajectory at a time; no eager all-trajectory tensors',benchmarks_include='Cold/warm filesystem mix; hashes, QC, causal extraction; not an isolated raw disk benchmark'),new_FEM_calls=0,new_OC_calls=0)
  write(out/'quality.json',result);write(out/'verified_inputs.json',dict(files=r.verified_files,artifact_manifest_bindings=r.bindings));print(json.dumps({k:result[k] for k in ('passed','usable_trajectories','usable_transitions','near_zero','performance')}),flush=True)
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
