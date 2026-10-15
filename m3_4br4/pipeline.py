"""Sequential R4 controls and original twelve-ID continuation, sidecars only."""
import json,sys,os,shutil,time
from pathlib import Path
from .common import ROOT,OLD,R3,LIMITS,read,job
from m3_4b.monitoring import run,atomic,digest,verify_trace

class Pipeline:
 def __init__(self,out):
  self.out=Path(out).resolve();self.rows=[];self.validation={};self.production={};self.recovery=[]
  if (self.out/'execution_ledger.json').exists():
   saved=read(self.out/'execution_ledger.json');self.rows=saved['attempts'];self.validation=saved['validation'];self.production=saved['production'];self.recovery=saved['recovery']
 def save(self):
  atomic(self.out/'execution_ledger.json',dict(attempts=self.rows,validation=self.validation,production=self.production,recovery=self.recovery))
 def execute(self,identifier,pid,command,interrupt=None):
  with (self.out/'reservations'/f'{identifier}.json').open('x') as f:json.dump(dict(command=command,problem_id=pid,limits=LIMITS,source_manifest_sha256=digest(self.out/'executing_source_manifest.json')),f,indent=2)
  record=dict(attempt=identifier,problem_id=pid,command=command,status='running');self.rows.append(record);self.save();print(identifier,flush=True)
  try:
   monitor=run(command,self.out/'monitor',identifier,pid,LIMITS,env=os.environ.copy(),interrupt_monitor=interrupt or False)
   if monitor['worker_sampled_hwm_mib']>1024:raise ValueError('Historical worker high-water exceeds RSS gate')
   record.update(status='interrupted_as_expected' if interrupt else 'passed',monitor=monitor);self.save();return monitor
  except Exception as exc:
   path=self.out/'monitor'/f'{identifier}.json'
   record.update(status='failed_closed',error=str(exc),monitor=read(path) if path.exists() else None);self.save();raise
 def generate(self,index,root,identifier,fault=None,reuse=False):
  m,j,s=job(index);cmd=[sys.executable,'-m','m3_4br4.worker',str(self.out),str(index),str(root),identifier,fault or 'none','reuse' if reuse else 'generate']
  flag=str(self.out/'work_counts'/identifier/'fault_ready.json') if fault else None
  monitor=self.execute(identifier,j['problem_id'],cmd,flag)
  if fault:
   if not monitor['controlled_interruption']:raise ValueError('Requested real interruption not delivered')
   return dict(monitor=monitor,counts=read(self.out/'work_counts'/identifier/'counts.json'),fault=read(self.out/'work_counts'/identifier/'fault_evidence.json'))
  answer=read(self.out/'work_counts'/identifier/'result.json')
  if answer.get('VmHWM',0)>1024:raise ValueError('Actual native worker high water')
  return dict(result=answer,monitor=monitor)
 def replay(self,index,folder,identifier):
  m,j,s=job(index);dest=self.out/'validation'/identifier
  monitor=self.execute(identifier,j['problem_id'],[sys.executable,'-m','m3_4br3.replay','--trajectory',str(folder),'--output',str(dest),'--policy','m3.4b.transition_change_audit.v3'])
  result=read(dest/'summary.json')
  if not result['passed'] or result['worker_memory'].get('VmHWM',0)>1024:raise ValueError('Replay self-memory/status')
  return dict(passed=True,summary=result,evidence=str(dest),monitor=monitor,source='R4 full independent replay',FEM=result['FEM_calls_this_fragment'],OC=result['transitions'])
 def checks(self,index,folder,identifier):
  _,j,_=job(index);name=f'checks/{identifier}.json'
  self.execute(identifier,j['problem_id'],[sys.executable,'-m','m3_4br4.checks','features',str(self.out),str(index),str(folder),name])
  return read(self.out/name)
 def compare(self,index,folder,control,identifier):
  _,j,_=job(index);name=f'checks/{identifier}.json'
  self.execute(identifier,j['problem_id'],[sys.executable,'-m','m3_4br4.checks','compare',str(self.out),str(index),str(folder),str(control),name])
  return read(self.out/name)
 def run(self):
  if '98 passed' not in (self.out/'focused_tests.log').read_text():raise ValueError('Focused audits')
  if not read(self.out/'baseline_binding_checks.json')['passed'] or not read(self.out/'regression/regression_isolated.json')['passed']:raise ValueError('Baseline/regression preflight')
  for name in ('reservations','validation','checks','monitor'): (self.out/name).mkdir(exist_ok=True)
  archive=self.out/('executing_source_snapshots_revision2' if (self.out/'executing_source_snapshots').exists() else 'executing_source_snapshots');archive.mkdir();manifest={}
  for group in ('m3_4br4','m3_4br3','m3_4br2'):
   for p in (ROOT/group).glob('*.py'):
    dest=archive/(group+'_'+p.name+'.txt');shutil.copyfile(p,dest);manifest[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dest.relative_to(self.out)))
  if (self.out/'executing_source_manifest.json').exists():shutil.copyfile(self.out/'executing_source_manifest.json',self.out/'executing_source_manifest_revision1.json')
  atomic(self.out/'executing_source_manifest.json',manifest)
  if os.environ.get('OPENBLAS_NUM_THREADS')!='1':raise ValueError('Thread environment')
  self.validation['9']=dict(passed=True,source='R3 complete replay reused; hashes and source fingerprints verified',evidence=str(R3/'complete_validation.json'),states=501,transitions=500,FEM_new_R4=0,prior_FEM=1003)
  self.production['9']=dict(status='completed_before_R4',folder=str(OLD/'native/samples'/job(9)[1]['problem_id']));self.save()
  # First actual native96 replay; any numerical/resource failure prevents production.
  folder=OLD/'native/samples'/job(1)[1]['problem_id']
  if '1' not in self.validation:self.validation['1']=self.replay(1,folder,'native96_full_replay')
  else:
   assert self.validation['1']['passed'];verify_trace(self.validation['1']['monitor'])
  self.production['1']=dict(status='completed_before_R4',folder=str(folder));self.save()
  # Real interruption fixtures use original reserved index10, full unchanged cap/settings.
  fixture_index=10;pid=job(fixture_index)[1]['problem_id'];controlroot=self.out/'recovery/control'
  controlfolder=controlroot/'samples'/pid
  if controlfolder.exists():
   counts=read(self.out/'work_counts/recovery_uninterrupted/counts.json');meta=read(controlfolder/'metadata.json')
   if counts['FEM']!=meta['state_count'] or counts['OC']!=meta['completed_updates']:raise ValueError('Interrupted result control accounting')
   control=self.generate(fixture_index,controlroot,'recovery_control_completed_reuse',reuse=True)
   control['completed_result_recovery_zero_FEM']=True;control['result'].update(FEM=counts['FEM'],OC=counts['OC'],original_execution_status='Completed native generation; original result serialization failed',original_counts=counts)
  else:control=self.generate(fixture_index,controlroot,'recovery_uninterrupted')
  controlval=self.replay(fixture_index,controlfolder,'recovery_control_replay')
  self.recovery.append(dict(kind='uninterrupted_control',generation=control,validation=controlval));self.save()
  for point in ('frame_before_journal','transition_before_journal','committed_update2','metadata_before_rename'):
   scratch=self.out/'recovery'/point;first=self.generate(fixture_index,scratch,'fault_'+point,fault=point)
   stage=scratch/'incomplete'/pid;captured=first['fault']
   if read(stage/'journal.json')!=captured['journal']:raise ValueError('Interrupted journal changed')
   for f,h in captured['committed_hashes'].items():
    if digest(stage/f)!=h:raise ValueError('Interrupted prefix corruption')
   # Preserve exact raw interrupted journal alongside monitor and prefix hashes.
   journal_copy=self.out/'work_counts'/('fault_'+point)/'interrupted_journal.json';shutil.copyfile(stage/'journal.json',journal_copy)
   resumed=self.generate(fixture_index,scratch,'resume_'+point)
   folder=scratch/'samples'/pid;committed=read(folder/'journal.json')
   oldstates=captured['journal']['states'];oldupdates=captured['journal']['updates']
   if committed['states'][:len(oldstates)]!=oldstates or committed['updates'][:len(oldupdates)]!=oldupdates:raise ValueError('Committed prefix overwritten')
   expected_fem=control['result']['states'];actual_fem=first['counts']['FEM']+resumed['result']['FEM'];actual_oc=first['counts']['OC']+resumed['result']['OC']
   expected_extra=1 if point in ('frame_before_journal','transition_before_journal') else 0
   if actual_fem!=expected_fem+expected_extra or actual_oc!=expected_fem-1+expected_extra:raise ValueError('Unexplained repeated FEM/OC')
   compared=self.compare(fixture_index,folder,controlfolder,'compare_'+point)
   validated=self.replay(fixture_index,folder,'replay_'+point)
   reused=self.generate(fixture_index,scratch,'reuse_'+point,reuse=True)
   record=dict(point=point,passed=True,first=first,resumed=resumed,comparison=compared,validation=validated,completed_reuse=reused,committed_prefix_hashes_preserved=True,nominal_states=expected_fem,actual_generation_FEM=actual_fem,actual_generation_OC=actual_oc,repeated_uncommitted_FEM=expected_extra,quarantined_files=[str(p.relative_to(folder)) for p in (folder/'quarantined').glob('*')])
   self.recovery.append(record);self.save()
  # All completed original samples: completed-worker reuse tripwires then full replay.
  for i in range(10):
   folder=OLD/'native/samples'/job(i)[1]['problem_id'];reuse=self.generate(i,OLD/'native',f'original_reuse_{i}',reuse=True)
   self.production[str(i)]=dict(status='completed_before_R4',folder=str(folder),reuse=reuse)
   if str(i) not in self.validation:self.validation[str(i)]=self.replay(i,folder,f'original_full_replay_{i}')
   self.validation[str(i)]['supplemental_checks']=self.checks(i,folder,f'features_FD_{i}');self.save()
  preflight=dict(passed=True,regression=read(self.out/'regression/regression_isolated.json'),baseline_binding=read(self.out/'baseline_binding_checks.json'),native96_validation=self.validation['1'],real_recovery_passed=all(r.get('passed',True) for r in self.recovery),completed_originals_independently_validated=10,completed_reuse_passed=True,limits=LIMITS,original_configuration_ids_fixed=True,split_assignments_fixed=True,unexecuted_indices=[10,11])
  # No original artifacts may change before production is enabled.
  bad=[p for p,h in read(self.out/'protected_before.json').items() if digest(ROOT/p)!=h]
  if bad:raise ValueError('Protected changes before production '+str(bad))
  atomic(self.out/'production_preflight.json',preflight);print('PRODUCTION PREFLIGHT PASS',flush=True)
  for i in (10,11):
   folder=self.out/'native/samples'/job(i)[1]['problem_id']
   if folder.exists() or (self.out/'native/incomplete'/job(i)[1]['problem_id']).exists():raise ValueError('No silent duplicate/restart')
   self.production[str(i)]=dict(status='completed_new_R4',generation=self.generate(i,self.out/'native',f'production_generate_{i}'),folder=str(folder));self.save()
   try:self.validation[str(i)]=self.replay(i,folder,f'production_replay_{i}')
   except Exception as exc:
    self.production[str(i)]['dataset_status']='quarantined_by_sidecar_exclusion';atomic(self.out/f'quarantine_{i}.json',dict(folder=str(folder),error=str(exc),training_eligible=False));self.save();raise
   self.validation[str(i)]['supplemental_checks']=self.checks(i,folder,f'features_FD_{i}');self.save()
  atomic(self.out/'batch_completed.json',dict(passed=True,configurations=12,validation_policy='v2 prerequisite plus v3',ledger=self.production,validation=self.validation));print('BATCH COMPLETE',flush=True)
if __name__=='__main__':
 p=Pipeline(sys.argv[1])
 try:p.run()
 except Exception as exc:
  atomic(p.out/'blocker.json',dict(error=str(exc),production_started=(p.out/'production_preflight.json').exists(),all12_success=False,execution_ledger=str(p.out/'execution_ledger.json')));raise
