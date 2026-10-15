"""Persistent per-entry accounting; limits charged before each numerical entry."""
import fcntl,time,os
from contextlib import contextmanager
from datetime import datetime,timezone
from .authorization import timestamp
from .common import *
class Budget:
 def __init__(self,root,p):self.root=Path(root);self.p=p;self.path=self.root/'work_ledger.json'
 @contextmanager
 def locked(self):
  with (self.root/'ledger.lock').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);d=read(self.path) if self.path.exists() else dict(binding_sha256=canonical(self.p['binding']),authorization_id=self.p['authorization_id'],counts={k:0 for k in WORK},stages={k:{q:0 for q in STAGES[k]} for k in STAGES})
   check(d['binding_sha256']==canonical(self.p['binding']) and d['authorization_id']==self.p['authorization_id'],'Ledger identity')
   yield d;atomic(self.path,d)
 def charge(self,k,stage=None,amount=1):
  check(amount>=0,'Negative work')
  check(datetime.now(timezone.utc)<timestamp(self.p['approval']['expires_at']),'Authorization expired before numerical entry')
  over=False
  with self.locked() as d:
   over=d['counts'][k]+amount>WORK[k]
   if stage is not None:over=over or d['stages'][stage][k]+amount>STAGES[stage][k]
   if k!='worker_seconds':check(not over,'Work overrun '+k)
   d['counts'][k]+=amount
   if stage is not None:d['stages'][stage][k]+=amount
  check(not over,'Aggregate worker time overrun')
 def remaining(self):
  with self.locked() as d:return min(1800-d['counts']['worker_seconds'],(timestamp(self.p['approval']['expires_at'])-datetime.now(timezone.utc)).total_seconds())

def verify_resources(s):
 check(s['exit_code']==0 and s['resource_event'] is None and s['worker_exited'],'Worker incomplete/timeout/failure')
 check(s['wall_seconds']<=900 and s['worker_peak_rss_mib']<=1024 and s['worker_sampled_hwm_mib']<=1024 and s['parent_peak_mib']<=768 and s['combined_peak_mib']<=1792,'Memory/time resource gate')
 return True

def parent_guard():
 values={}
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.startswith(('VmRSS:','VmHWM:')):values[line.split(':')[0]]=int(line.split()[1])/1024
 check(values.get('VmRSS',769)<=768 and values.get('VmHWM',769)<=768,'Parent RSS/HWM resource gate')
 return values
