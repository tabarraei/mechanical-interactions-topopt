"""Clean read-only acceptance process, after numerical workers have exited."""
import sys
from datetime import datetime,timezone
from .common import *
from .authorization import issued_authorization

def main(auth_path,kind,stage,pid,root):
 s=Selection();a=issued_authorization(auth_path,s,kind=kind,stage=int(stage),ids=[pid],output_root=root)
 # Import numerical-evidence readers only after a real issued scope is verified.
 from .validation import verify_acceptance
 root=Path(root);folder=root/'native/samples'/pid;evidence=root/'validation'/pid;binding=a['resume_bindings'].get(pid,{})
 resumed=binding.get('purpose')=='resume' or any((root/'reservations').glob(f'*_{pid}_resume.json.started'))
 execution=dict(authorization_sha256=digest(auth_path),resumed=resumed,restart_control=binding.get('restart_control') if resumed else None,executed_at=datetime.now(timezone.utc).isoformat())
 certificate=verify_acceptance(folder,evidence,s.specs[pid],s.manifest['source_binding']['original_execution_source_id'],execution)
 check(read(root/'checks'/f'{pid}.json')['passed'],'Supplemental feature/FD gate');meta=read(folder/'metadata.json')
 certificate_path=root/'acceptance'/f'{a["authorization_id"]}_{pid}.json';write(certificate_path,certificate)
 if kind=='resource_qualification':
  write(root/'qualification'/f'{pid}.json',dict(problem_id=pid,spec_sha256=s.jobs[pid]['spec_sha256'],passed=True,limits=LIMITS,completed_original_trajectory=True,full_cap_history_residency_qualified=meta['state_count']==s.specs[pid]['optimizer']['maxiter']+1,states=meta['state_count'],completed_updates=meta['completed_updates'],source_id=meta['source_id'],acceptance_path=str(certificate_path),acceptance_sha256=digest(certificate_path),production_release_allowed=False,authorization_sha256=digest(auth_path)))
 elif a['experiment']['release_allowed'] and 'adopt' in a['experiment']['actions']:
  entry=dict(problem_id=pid,family=s.jobs[pid]['family'],partition=s.jobs[pid]['partition'],folder=str(folder),states=meta['state_count'],transitions=meta['completed_updates'],termination_reason=meta['termination_reason'],source_id=meta['source_id'],metadata_sha256=digest(folder/'metadata.json'),array_hashes={k:v['sha256'] for k,v in meta['arrays'].items()},mechanically_validated=True,reference_label_eligible=False,validation_status='complete',authorization=dict(path=str(Path(auth_path).resolve()),sha256=digest(auth_path)),validation=dict(evidence=str(evidence),acceptance_path=str(certificate_path),acceptance_sha256=digest(certificate_path),execution=execution))
  write(root/'accepted_entries'/f'{pid}.json',entry)
if __name__=='__main__':main(*sys.argv[1:])
