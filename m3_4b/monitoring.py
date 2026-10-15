"""Constant-buffer subprocess monitoring and trace integrity, standard library only."""
import json,os,time,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone

def stamp():return datetime.now(timezone.utc).isoformat()
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def memory(pid=None):
 try:
  data=Path(f'/proc/{pid or os.getpid()}/status').read_text()
  return {line.split(':')[0]:int(line.split()[1])/1024 for line in data.splitlines() if line.startswith(('VmRSS:','VmHWM:'))}
 except FileNotFoundError:return {}
def atomic(path,data):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def trace_rows(path,job_id=None):
 with Path(path).open() as f:
  for line in f:
   if not line.endswith('\n'):raise ValueError('Incomplete monitor tail; preserve as partial, do not call complete')
   row=json.loads(line)
   if job_id is not None and row['job_id']!=job_id:raise ValueError('Trace job attribution mismatch')
   yield row

def verify_trace(summary):
 if digest(summary['trace_path'])!=summary['trace_sha256']:raise ValueError('Monitor checksum mismatch')
 count=0
 for row in trace_rows(summary['trace_path'],summary['job_id']):count+=1
 if count!=summary['sample_count']:raise ValueError('Monitor count mismatch')
 return count

def run(command,directory,job_id,problem_id,limits,expected=0,env=None,interrupt_monitor=False):
 """Raw records never enter a list or returned summary. Partial traces survive failure."""
 directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
 trace=directory/(job_id+'.jsonl');summary_path=directory/(job_id+'.json');log=directory/(job_id+'.log')
 if trace.exists() or summary_path.exists():raise ValueError('Unique immutable attempt ID required')
 started=time.monotonic();utc=stamp();count=0;parent_before=memory().get('VmRSS',0);pp=parent_before;wp=wh=cp=0.;cpu0=time.process_time();io_seconds=0.;reason=None;interrupted=False
 with trace.open('x',buffering=1) as stream,log.open('x') as output:
  child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env)
  try:
   while child.poll() is None:
    tick=time.monotonic();wm=memory(child.pid);pm=memory();rss=wm.get('VmRSS',0);hwm=wm.get('VmHWM',rss);parent=pm.get('VmRSS',0);wp=max(wp,rss);wh=max(wh,hwm);pp=max(pp,parent);cp=max(cp,rss+parent)
    row=dict(job_id=job_id,problem_id=problem_id,seconds=tick-started,worker_rss_mib=rss,worker_hwm_mib=hwm,parent_rss_mib=parent)
    write_tick=time.monotonic();stream.write(json.dumps(row,separators=(',',':'))+'\n');count+=1
    if count%20==0:stream.flush();os.fsync(stream.fileno())
    io_seconds+=time.monotonic()-write_tick
    if rss>limits['worker_rss_mib'] or parent>limits['parent_rss_mib'] or rss+parent>limits['combined_rss_mib']:reason='RSS gate exceeded'
    if tick-started>limits['worker_timeout_seconds']:reason='Time gate exceeded'
    trigger=interrupt_monitor is True and count==3
    if isinstance(interrupt_monitor,str) and Path(interrupt_monitor).exists():
     try:trigger=len(json.loads(Path(interrupt_monitor).read_text())['states'])>=3
     except (json.JSONDecodeError,KeyError):trigger=False
    if trigger:reason='controlled monitor interruption';interrupted=True
    if reason:
     child.terminate()
     try:child.wait(timeout=2)
     except subprocess.TimeoutExpired:child.kill();child.wait()
     break
    time.sleep(limits['poll_seconds'])
  finally:
   if child.poll() is None:
    child.terminate()
    try:child.wait(timeout=2)
    except subprocess.TimeoutExpired:child.kill();child.wait()
   stream.flush();os.fsync(stream.fileno())
  code=child.wait()
 answer=dict(job_id=job_id,problem_id=problem_id,start_utc=utc,end_utc=stamp(),wall_seconds=time.monotonic()-started,exit_code=code,expected_exit=expected,worker_peak_rss_mib=wp,worker_sampled_hwm_mib=wh,parent_before_mib=parent_before,parent_after_mib=memory().get('VmRSS',0),parent_peak_mib=pp,combined_peak_mib=cp,resource_event=reason,controlled_interruption=interrupted,sample_count=count,trace_path=str(trace),trace_sha256=digest(trace),trace_bytes=trace.stat().st_size,log_path=str(log),worker_exited=not Path(f'/proc/{child.pid}').exists(),monitor_parent_cpu_seconds=time.process_time()-cpu0,monitor_write_seconds=io_seconds)
 atomic(summary_path,answer)
 if (reason and not interrupted) or (code!=expected and not interrupted):raise RuntimeError(f'Worker failed closed: {reason or code}; {log}')
 verify_trace(answer);return answer

def reduce_summaries(paths,verify=True):
 result=dict(jobs=0,samples=0,trace_bytes=0,parent_peak_mib=0.,worker_peak_mib=0.,combined_peak_mib=0.,monitor_cpu_seconds=0.,monitor_write_seconds=0.)
 for p in paths:
  s=json.loads(Path(p).read_text())
  if verify:verify_trace(s)
  result['jobs']+=1;result['samples']+=s['sample_count'];result['trace_bytes']+=s['trace_bytes'];result['parent_peak_mib']=max(result['parent_peak_mib'],s['parent_peak_mib']);result['worker_peak_mib']=max(result['worker_peak_mib'],s['worker_peak_rss_mib'],s['worker_sampled_hwm_mib']);result['combined_peak_mib']=max(result['combined_peak_mib'],s['combined_peak_mib']);result['monitor_cpu_seconds']+=s['monitor_parent_cpu_seconds'];result['monitor_write_seconds']+=s['monitor_write_seconds']
 return result
