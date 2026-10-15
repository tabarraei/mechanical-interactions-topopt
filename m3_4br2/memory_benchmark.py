"""Fresh synthetic comparisons under existing resource monitoring limits."""
import argparse,json,os,sys,time,resource
from pathlib import Path
import numpy as np
from .memory import frame_allclose

def worker(mode,out):
 # Resident base fields are identical between modes; biggest historical field.
 a=np.full((193,9506,2),.3);b=a.copy();start=time.perf_counter()
 if mode=='whole':np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
 else:frame_allclose(a,b)
 data=dict(mode=mode,shape=list(a.shape),array_mib=a.nbytes/2**20,peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,comparison_seconds=time.perf_counter()-start,synthetic=True)
 Path(out).write_text(json.dumps(data,indent=2))

def run(out):
 from m3_4b.monitoring import run
 limits=dict(worker_rss_mib=1024,parent_rss_mib=768,combined_rss_mib=1792,worker_timeout_seconds=900,poll_seconds=.05)
 result={}
 for mode in ('whole','frame'):
  file=out/f'memory_{mode}.json';s=run([sys.executable,'-m','m3_4br2.memory_benchmark','--mode',mode,'--output',str(file)],out/'memory_monitors',mode,'synthetic96',limits)
  result[mode]=dict(measurement=json.loads(file.read_text()),monitor=s)
 result['saved_peak_mib']=result['whole']['measurement']['peak_mib']-result['frame']['measurement']['peak_mib'];result['native96_safety_proved']=False
 (out/'memory_results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v['measurement'] for k,v in result.items() if isinstance(v,dict)}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['whole','frame']);ap.add_argument('--output',required=True);a=ap.parse_args()
 if a.mode:worker(a.mode,a.output)
 else:run(Path(a.output))
