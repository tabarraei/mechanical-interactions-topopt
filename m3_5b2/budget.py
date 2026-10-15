"""Durable, locked authorization work accounting; charge BEFORE numerical work."""
import fcntl,time
from datetime import datetime,timezone
from contextlib import contextmanager
from .common import *
class Budget:
 def __init__(self,root,authorization,authorization_sha):self.root=Path(root);self.auth=authorization;self.sha=authorization_sha;self.file=self.root/'work_ledger.json'
 @contextmanager
 def locked(self):
  with (self.root/'work_ledger.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   state=read(self.file) if self.file.exists() else dict(authorizations={},lifetime=dict(FEM=0,OC=0,workers=0,worker_seconds=0.))
   state['authorizations'].setdefault(self.sha,dict(FEM=0,OC=0,workers=0,worker_seconds=0.))
   yield state
   atomic(self.file,state);fcntl.flock(lock,fcntl.LOCK_UN)
 def charge(self,kind,amount=1):
  check(amount>=0,'Negative work charge')
  if kind in ('FEM','OC') and 'approval' in self.auth:check(datetime.now(timezone.utc)<datetime.fromisoformat(self.auth['approval']['expires_at'].replace('Z','+00:00')),'Authorization time window exhausted before numerical entry')
  key={'FEM':'max_FEM_calls','OC':'max_OC_calls','workers':'max_worker_jobs','worker_seconds':'max_total_worker_seconds'}[kind]
  overrun=False
  with self.locked() as s:
   use=s['authorizations'][self.sha];overrun=use[kind]+amount>self.auth['maximum_work'][key]
   if kind!='worker_seconds':check(not overrun,'Authorization work exhausted '+kind)
   use[kind]+=amount;s['lifetime'][kind]+=amount
  check(not overrun,'Authorization work exhausted '+kind)
 def remaining_seconds(self):
  with self.locked() as s:
   remaining=self.auth['maximum_work']['max_total_worker_seconds']-s['authorizations'][self.sha]['worker_seconds']
   if 'approval' in self.auth:remaining=min(remaining,(datetime.fromisoformat(self.auth['approval']['expires_at'].replace('Z','+00:00'))-datetime.now(timezone.utc)).total_seconds())
   return remaining
