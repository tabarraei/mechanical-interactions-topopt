"""Separate protected human issuance boundary; no issuer or bypass is provided."""
from datetime import datetime,timezone
from .common import *
from .policy import POLICY,scope
FIELDS={'schema_version','authorization_id','kind','binding','output_root','environment','limits','work','actions','disk_bytes','approval'}
STOP='stop_and_quarantine_no_retry_no_release'
def timestamp(t):
 v=datetime.fromisoformat(t.replace('Z','+00:00'));check(v.tzinfo is not None,'Timezone required');return v
def validate_candidate(p,b,output,env,now):
 exact(p,FIELDS,'authorization');scope(p['binding']);check(p['binding']==b,'Source/input/target identity mismatch')
 check(p['schema_version']=='m3.5b2d.protocolA.authorization.v1' and p['kind']=='historical_saved_validation_only','Authorization kind/schema; no generation/adoption')
 check(isinstance(p['authorization_id'],str) and p['authorization_id'].isalnum() and len(p['authorization_id'])>=8,'Authorization ID')
 root=Path(output).resolve();check(p['output_root']==str(root) and root.is_relative_to(ROOT/'m3_5b2d/validation_runs'),'Output root mismatch')
 check(p['environment']==env and p['limits']==LIMITS and {k:type(v) for k,v in p['limits'].items()}=={k:type(v) for k,v in LIMITS.items()},'Environment/resource limits')
 check(p['work']==WORK and all(type(v) is int for v in p['work'].values()),'Work ceilings mismatch')
 check(p['actions']==['replay','supplemental',STOP],'Actions/stop rules')
 check(type(p['disk_bytes']) is int and p['disk_bytes']>=4*1024**3,'Disk floor')
 exact(p['approval'],{'identity','approved_at','expires_at','decision_reference'},'approval')
 check(all(isinstance(v,str) and v for v in p['approval'].values()),'Approval metadata')
 check(timestamp(p['approval']['approved_at'])<=now<timestamp(p['approval']['expires_at']),'Approval expired/not yet valid')
 return p

def review_book(book,*,kind,object_hash,binding_hash):
 exact(book,{'schema_version','records'},'protected authority');check(book['schema_version']=='m3.5b2d.reviewed_approvals.v1','Authority schema')
 found=[r for r in book['records'] if r.get('kind')==kind and r.get('sha256')==object_hash and r.get('binding_sha256')==binding_hash]
 check(len(found)==1,'Missing deliberately issued '+kind+' approval')
 r=found[0];exact(r,{'kind','sha256','binding_sha256','decision','identity','approved_at','expires_at','reference'},'approval receipt')
 check(r['identity'] and r['reference'] and timestamp(r['approved_at'])<=datetime.now(timezone.utc)<timestamp(r['expires_at']),'Approval receipt date/identity')
 if kind=='policy':check(r['decision']=='D2','D1/no decision blocks historical validation pathway')
 else:check(r['decision']=='approve','Approval decision missing')
 return r

def approval_record(kind,object_hash,binding):
 check(AUTHORITY.is_file(),'No protected human '+kind+' approval issued')
 return review_book(read(AUTHORITY),kind=kind,object_hash=object_hash,binding_hash=canonical(binding))

def live_guard(auth_path,output):
 # Missing approval fails before any numerical import, directory mutation or work.
 check(auth_path is not None and Path(auth_path).is_file(),'Missing authorization')
 p=read(auth_path);b=target_binding();validate_candidate(p,b,output,environment(),datetime.now(timezone.utc))
 check({k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}==dict(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='16',OMP_NUM_THREADS='16'),'Thread binding')
 approval_record('policy',digest(Path(__file__).with_name('policy.py')),b)
 receipt=approval_record('protocolA_execution',digest(auth_path),b)
 check(receipt['identity']==p['approval']['identity'] and receipt['reference']==p['approval']['decision_reference'] and receipt['approved_at']==p['approval']['approved_at'] and receipt['expires_at']==p['approval']['expires_at'],'Authorization issuance metadata binding')
 verify_binding(b);return p
import os
