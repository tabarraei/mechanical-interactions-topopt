"""Read-only repository-wide identity inventory; no historical ledger is reset."""
import subprocess
from .common import *
class Registry:
 def __init__(self,root,selection):
  self.root=Path(root).resolve();self.selection=selection;self.records={};self.coverage=[]
  result=subprocess.run(['rg','--files','-uuu'],cwd=self.root,capture_output=True,text=True)
  check(result.returncode==0 and not result.stderr.strip(),'Registry coverage cannot be established')
  files=result.stdout.splitlines();self.inventory_digest=canonical(sorted(files));self.paths=files
  for name in files:
   p=self.root/name
   if p.name not in ('metadata.json','problem.json','journal.json') and not (p.parent.name=='execution_locks' and p.suffix=='.json'):continue
   pid=p.parent.name if p.parent.name.startswith('m3_') else p.stem if p.parent.name=='execution_locks' else None
   if pid is None:continue
   record=read(p);self.coverage.append(dict(path=str(p),sha256=digest(p)))
   if p.name=='problem.json':
    check('mechanical' in record and record.get('problem_id')==pid,'Unknown problem inventory schema');eq=equivalent(record);status='incomplete' if 'incomplete' in p.parts else 'previously_attempted'
   elif p.name=='metadata.json':
    specfile=p.parent/'problem.json';check(specfile.is_file(),'Missing historical specification');spec=read(specfile);check(spec.get('problem_id')==pid and record.get('problem_id')==pid,'Historical ID mismatch');eq=equivalent(spec);status='previously_attempted'
    if 'problem_sha256' in record:check(digest(specfile)==record['problem_sha256'],'Historical spec hash')
   elif p.name=='journal.json':
    specfile=p.parent/'problem.json';check(specfile.is_file(),'Missing checkpoint spec');eq=equivalent(read(specfile));status='incomplete' if 'incomplete' in p.parts else 'previously_attempted'
   else:eq=record['configuration_equivalence_sha256'];status='previously_attempted'
   if 'quarantine' in name:status='quarantined'
   self.records.setdefault(pid,[]).append(dict(path=str(p),equivalence_sha256=eq,status=status,owner_output_root=record.get('output_root') if p.parent.name=='execution_locks' else None))
  for name in files:
   if 'quarantine' not in Path(name).name or not name.endswith('.json'):continue
   p=self.root/name
   try:r=read(p)
   except ValueError:continue
   folder=r.get('folder');pid=Path(folder).name if isinstance(folder,str) else r.get('problem_id')
   if pid in self.records:
    self.coverage.append(dict(path=str(p),sha256=digest(p)))
    for record in self.records[pid]:record['status']='quarantined'
  for e in selection.original:
   check(e['problem_id'] in self.records and digest(Path(e['folder'])/'metadata.json')==e['metadata_sha256'],'Original released registry coverage')
   for r in self.records[e['problem_id']]:
    r['historical_inventory_status']=r['status'];r['is_released_location']=Path(r['path']).is_relative_to(Path(e['folder']))
    if r['is_released_location']:r['status']='already_released'
  # Even an unjournaled orphan under a selected incomplete root blocks dispatch.
  for name in files:
   parts=Path(name).parts
   for i,part in enumerate(parts[:-1]):
    if part=='incomplete' and parts[i+1] in selection.jobs:
     pid=parts[i+1]
     if pid not in self.records:self.records[pid]=[dict(path=str(self.root/Path(*parts[:i+2])),equivalence_sha256=equivalent(selection.specs[pid]),status='incomplete')]
  self.coverage_sha256=canonical(self.coverage)
 def eligible(self,pid,*,own_root=None,allow_owned=False):
  check(pid in self.selection.jobs and pid not in self.selection.original_ids,'Unknown/original12 ID');eq=equivalent(self.selection.specs[pid]);conflicts=[]
  for other,records in self.records.items():
   for r in records:
    if other!=pid and r['equivalence_sha256']!=eq:continue
    if allow_owned and other==pid and own_root is not None and (Path(r['path']).is_relative_to(Path(own_root)) or r.get('owner_output_root')==str(Path(own_root))):continue
    conflicts.append(dict(problem_id=other,**r))
  check(not conflicts,'Duplicate/previously attempted/configuration-equivalent history: '+str(conflicts[:3]));return True
 def report(self):
  rows=[]
  for pid in sorted(set(self.records)|set(self.selection.jobs)):
   records=self.records.get(pid,[]);status='selected_but_unexecuted' if not records else 'already_released' if pid in self.selection.original_ids else 'quarantined' if any(r['status']=='quarantined' for r in records) else 'incomplete' if any(r['status']=='incomplete' for r in records) else 'previously_attempted'
   eq=equivalent(self.selection.specs[pid]) if pid in self.selection.specs else None
   equivalent_existing=any(r['equivalence_sha256']==eq for rs in self.records.values() for r in rs) if eq else bool(records)
   rows.append(dict(problem_id=pid,status=status,records=records,selected=pid in self.selection.jobs,identity_eligible_for_future_authorization=pid in self.selection.jobs and not records and not equivalent_existing,execution_authorized=False))
  return dict(schema_version='m3.5b2.registry.v1',root=str(self.root),coverage_complete=True,coverage_sha256=self.coverage_sha256,inventory_digest=self.inventory_digest,files=self.coverage,rows=rows)
