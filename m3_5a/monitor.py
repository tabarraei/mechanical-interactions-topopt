"""Single sequential bounded subprocess; never imports production/physics."""
import json,os,time,subprocess
from pathlib import Path
from .io import write,digest,check
LIMITS=dict(worker_mib=1024,parent_mib=768,combined_mib=1792,seconds=900)
def memory(pid=None):
 try:
  return {s.split(':')[0]:int(s.split()[1])/1024 for s in Path(f'/proc/{pid or os.getpid()}/status').read_text().splitlines() if s.startswith(('VmRSS:','VmHWM:'))}
 except FileNotFoundError:return {}
def run(command,out,identifier,seconds=900):
 out=Path(out);out.mkdir(exist_ok=True,parents=True);start=time.perf_counter();limits={**LIMITS,'seconds':seconds};write(out/(identifier+'_reservation.json'),dict(command=command,limits=limits));peak=dict(worker_mib=0.,worker_hwm_mib=0.,parent_mib=0.,combined_mib=0.);count=0;reason=None
 trace=out/(identifier+'.jsonl');log=out/(identifier+'.log')
 with trace.open('x') as f,log.open('x') as g:
  child=subprocess.Popen(command,stdout=g,stderr=subprocess.STDOUT,env=os.environ.copy())
  try:
   while child.poll() is None:
    w=memory(child.pid);p=memory();elapsed=time.perf_counter()-start;row=dict(seconds=elapsed,worker_mib=w.get('VmRSS',0),worker_hwm_mib=w.get('VmHWM',0),parent_mib=p.get('VmRSS',0));row['combined_mib']=row['worker_mib']+row['parent_mib']
    for k in peak:peak[k]=max(peak[k],row[k])
    f.write(json.dumps(row)+'\n');f.flush();count+=1
    if any(row[k]>limits[k] for k in ('worker_mib','parent_mib','combined_mib')):reason='Memory gate'
    if elapsed>seconds:reason='Time gate'
    if reason:child.terminate();break
    time.sleep(.05)
  finally:
   if child.poll() is None:
    if reason is None:reason='Monitor interrupted'
    child.terminate()
    try:child.wait(timeout=2)
    except subprocess.TimeoutExpired:child.kill();child.wait()
  code=child.wait()
 result=dict(passed=code==0 and reason is None,exit_code=code,resource_event=reason,limits=limits,peak=peak,wall_seconds=time.perf_counter()-start,trace_sha256=digest(trace),trace_samples=count,log=str(log))
 write(out/(identifier+'.json'),result);check(result['passed'],'Failed closed '+identifier+': '+str(reason or code));return result
