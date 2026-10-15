"""Read-only source integrity and field inventory; writes only new evidence."""
import json,hashlib,shutil,sys
from pathlib import Path
from .data import AcceptedData
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'m3_6a/evidence/20261015_architecture_v1'
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def write(name,value):(E/name).write_text(json.dumps(value,indent=2)+'\n')
def run():
 data=AcceptedData();fields={};examples=[]
 try:
  for e in data.entries:
   pid=e['problem_id'];a=data.release.open(pid)
   fields[pid]={k:dict(shape=list(v.shape),dtype=str(v.dtype),status='directly saved') for k,v in a.items()}
   examples.append(dict(problem_id=pid,mesh=[data.release.specs[pid]['mechanical']['nelx'],data.release.specs[pid]['mechanical']['nely']],states=e['states'],transitions=e['transitions'],termination=e['termination_reason'],partition=e['partition']))
  write('data_field_audit.json',dict(fields=fields,trajectories=examples,reconstructed=['zero-constraint component indicators from fixed DOFs','nodal force view from saved vector','full rectangular domain mask from complete connectivity','cell quadrature area','causal design-delta targets'],unavailable=['accepted nonrectangular geometry masks','nonzero prescribed displacement values','body-force/traction semantics','accepted held-out structural families']))
  write('verified_acceptance_files.json',data.release.verified_files)
 finally:data.close()
 before=json.loads((E/'protected_before.json').read_text());changed=[];missing=[]
 for n,h in before.items():
  p=ROOT/n
  if not p.is_file():missing.append(n)
  elif digest(p)!=h:changed.append(n)
 inspected=json.loads((E/'inspected_sources_before.json').read_text())
 inspected_changed=[n for n,h in inspected.items() if digest(ROOT/n)!=h]
 write('integrity_final.json',dict(protected_paths=len(before),changed=changed,missing=missing,inspected_source_changes=inspected_changed,passed=not(changed or missing or inspected_changed),scope='all inherited protected paths plus inspected sources; no claim for files outside this explicit baseline'))
 # Preserve/report concurrent workspace edits; never restore an old notebook.
 write('integrity_exceptions.json',dict(entries=[dict(path=n,baseline_sha256=before[n],observed_sha256=digest(ROOT/n),mtime_ns=(ROOT/n).stat().st_mtime_ns) for n in changed],attribution='Not established by hashes; old viewer notebook contains executed outputs. No notebook execution or writes to that old path were performed by this task.',dataset_and_inspected_source_exception=any(n!='m3_5v/visualizations/20261015_accepted12_v1/trajectory_viewer.ipynb' for n in changed) or bool(missing or inspected_changed)))
 forbidden=[n for n in sys.modules if n=='topopt_utils' or n.startswith('topopt_utils.') or n=='pypardiso']
 write('execution_scope.json',dict(FEM=0,OC=0,optimizer_steps=0,SLURM=0,numerical_replay=0,new_trajectories=0,policy_publication=0,alias_admission=0,dataset_publication=0,forbidden_imports=forbidden))
 assert not forbidden
 for src,name in [('/tmp/m3_6a_model_initial.log','model_initial.log'),('/tmp/m3_6a_full_tests.log','tests_initial_full.log'),('/tmp/m3_6a_final_tests.log','tests_final.log'),('/tmp/m3_6a_controls.log','viewer_controls.log')]:
  if Path(src).exists():shutil.copyfile(src,E/name)
 print(json.dumps(dict(protected_paths=len(before),changed=changed,accepted=len(examples),states=sum(e['states'] for e in examples),transitions=sum(e['transitions'] for e in examples))))
if __name__=='__main__':run()
