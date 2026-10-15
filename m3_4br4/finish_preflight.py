"""Complete authorized validation/recovery evidence after mandatory gate failed.
No production dispatch exists in this entry point.
"""
import sys,shutil,json
from pathlib import Path
from .common import OLD,R3,ROOT,job,read
from .pipeline import Pipeline
from m3_4b.monitoring import atomic,digest

def run(out):
 p=Pipeline(out);out=p.out
 archive=out/'preflight_continuation_source_snapshots';archive.mkdir();manifest={}
 for file in (ROOT/'m3_4br4').glob('*.py'):
  dest=archive/(file.name+'.txt');shutil.copyfile(file,dest);manifest[str(file.relative_to(ROOT))]=dict(sha256=digest(file),archive=str(dest.relative_to(out)))
 atomic(out/'preflight_continuation_source_manifest.json',manifest)
 pid=job(10)[1]['problem_id'];controlroot=out/'recovery/control';controlfolder=controlroot/'samples'/pid;N=read(controlfolder/'metadata.json')['state_count']
 points=('frame_before_journal','transition_before_journal','committed_update2','metadata_before_rename')
 for point in points:
  scratch=out/'recovery'/point;folder=scratch/'samples'/pid
  if point=='frame_before_journal':
   first=dict(monitor=read(out/'monitor'/('fault_'+point+'.json')),counts=read(out/'work_counts'/('fault_'+point)/'counts.json'),fault=read(out/'work_counts'/('fault_'+point)/'fault_evidence.json'))
   resumed=dict(result=read(out/'work_counts'/('resume_'+point)/'result.json'),monitor=read(out/'monitor'/('resume_'+point+'.json')))
  else:
   first=p.generate(10,scratch,'fault_'+point,fault=point)
   stage=scratch/'incomplete'/pid;captured=first['fault']
   if read(stage/'journal.json')!=captured['journal']:raise ValueError('Interrupted journal')
   for f,h in captured['committed_hashes'].items():
    if digest(stage/f)!=h:raise ValueError('Committed prefix')
   shutil.copyfile(stage/'journal.json',out/'work_counts'/('fault_'+point)/'interrupted_journal.json')
   resumed=p.generate(10,scratch,'resume_'+point)
  captured=first['fault'];journal=read(folder/'journal.json');old=captured['journal'];assert journal['states'][:len(old['states'])]==old['states'] and journal['updates'][:len(old['updates'])]==old['updates']
  extra=1 if point in points[:2] else 0;actualF=first['counts']['FEM']+resumed['result']['FEM'];actualOC=first['counts']['OC']+resumed['result']['OC']
  assert actualF==N+extra and actualOC==N-1+extra
  # Exhaust comparisons in a read-only worker; failing fields never get overridden.
  p.execute('diagnose_'+point,pid,[sys.executable,'-m','m3_4br4.recovery_diagnosis',str(out),point])
  diagnosed=read(out/'checks'/('recovery_diagnosis_'+point+'.json'))
  validated=p.replay(10,folder,'replay_'+point)
  reused=p.generate(10,scratch,'reuse_'+point,reuse=True)
  p.recovery.append(dict(point=point,passed=diagnosed['production_gate_pass'],checkpoint_recovery_passed=True,fixed_comparison_passed=diagnosed['production_gate_pass'],first=first,resumed=resumed,comparison=diagnosed,validation=validated,completed_reuse=reused,committed_prefix_hashes_preserved=True,actual_generation_FEM=actualF,actual_generation_OC=actualOC,repeated_uncommitted_FEM=extra,quarantined_files=[str(f.relative_to(folder)) for f in (folder/'quarantined').glob('*')]))
  p.save()
 # Complete independent validation and zero-work reuse of existing original outputs.
 for i in range(10):
  folder=OLD/'native/samples'/job(i)[1]['problem_id'];reuse=p.generate(i,OLD/'native',f'original_reuse_{i}',reuse=True)
  p.production[str(i)]=dict(status='completed_before_R4',folder=str(folder),reuse=reuse)
  if str(i) not in p.validation:p.validation[str(i)]=p.replay(i,folder,f'original_full_replay_{i}')
  p.validation[str(i)]['supplemental_checks']=p.checks(i,folder,f'features_FD_{i}');p.save()
 atomic(out/'preflight_failed_final.json',dict(passed=False,reason='Fixed real restart/control native comparison failures retained; no revised gate authorized',real_recovery_cases={r['point']:r['passed'] for r in p.recovery if 'point' in r},existing_independent_validation=10,unexecuted_original_indices=[10,11],new_production=0,production_gate_changed=False))
 print('PREFLIGHT FAILED; original10 validated,2 unexecuted, zero new production',flush=True)
if __name__=='__main__':run(sys.argv[1])
