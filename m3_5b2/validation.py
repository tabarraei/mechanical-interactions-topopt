"""Read-only full evidence acceptance using frozen independent/correctness adapters."""
import io,json
import numpy as np
from .common import *
POLICY_HASHES=read(PLAN)['approved_restart_policy']['files']
def verify_policy():
 for name,h in POLICY_HASHES.items():check(digest(ROOT/name)==h,'Approved restart policy/evidence hash')

def exact_final_checkpoint(side,before=None):
 """Exact native-to-original journal NPZ/JSON serialization; never runs physics."""
 from m3_4br4r1.policy import array_digest
 folder=Path(side['folder']);journal=read(folder/'journal.json');spec=side['spec'];a=side['arrays'];T=side['metadata']['state_count'];j=before['journal'] if before else journal
 check(j['binding']==dict(problem_id=spec['problem_id'],spec_sha=canonical(spec),source_id=side['metadata']['source_id']),'Checkpoint source/configuration')
 check(len(journal['states'])==T and len(journal['updates'])==T-1,'Final checkpoint counts');check(journal['states'][:len(j['states'])]==j['states'] and journal['updates'][:len(j['updates'])]==j['updates'],'Committed prefix ordering')
 if before:check(canonical(j)==before['journal_canonical_sha256'],'Interrupted checkpoint receipt')
 fields=[k for k in side['metadata']['arrays'] if k!='update_change'];hashes=[]
 for t,e in enumerate(j['states']):
  check(e['file']==f'frames/{t:05d}.npz','Checkpoint frame order');buf=io.BytesIO();np.savez_compressed(buf,**{k:a[k][t] for k in fields});h=__import__('hashlib').sha256(buf.getbuffer()).hexdigest();check(h==e['sha256'],'Exact committed NPZ bytes');hashes.append(h)
 for t,e in enumerate(j['updates']):
  check(e['file']==f'updates/{t:05d}.json','Checkpoint transition order');h=__import__('hashlib').sha256((json.dumps(side['transitions'][t],indent=2,allow_nan=False)+'\n').encode()).hexdigest();check(h==e['sha256'],'Exact committed transition bytes');hashes.append(h)
 return dict(status='verified_exact',source_identity=True,configuration_identity=True,source_id=side['metadata']['source_id'],configuration_digest=canonical(spec),committed_states=len(j['states']),committed_transitions=len(j['updates']),committed_native_digests={k:array_digest(v[:len(j['updates']) if k=='update_change' else len(j['states'])]) for k,v in a.items()},journal_sha256=digest(folder/'journal.json'),hash_checks=len(hashes))

def independent(folder,evidence,spec):
 verify_policy()
 from m3_4br4r1.evidence import load_native,independent_certificate
 side=load_native(folder);check(side['spec']==spec,'Independent exact original spec');side['certificate']=independent_certificate(side,evidence)
 return side

def restart_equivalence(control,control_evidence,restart,restart_evidence,spec,before):
 verify_policy()
 from m3_4br4r1.policy import compare
 a=independent(control,control_evidence,spec);b=independent(restart,restart_evidence,spec);proof=exact_final_checkpoint(b,before);result=compare(a,b,proof);check(result['status']=='Accepted','Approved category-C restart equivalence failed/indeterminate');return result

def verify_acceptance(folder,evidence,spec,source_id,execution):
 """No caller-supplied pass boolean can replace raw independent evidence."""
 side=independent(folder,evidence,spec);m=side['metadata'];T=m['state_count'];a=side['arrays'];check(m['source_id']==source_id and m['generation_status']=='completed','Native source/completion')
 check(not m['reference_label_eligible'] and not m['optimality_certified'],'Capped/reference optimum label')
 check(np.array_equal(a['x'][0],np.asarray(spec['initial_x'],dtype=np.float64)),'Specified initial design')
 check(m['optimizer_option']==spec['optimizer']['option'] and m['optimizer_version']==spec['optimizer']['version'] and m['tolerance']==spec['optimizer']['tolerance'],'Optimizer metadata binding')
 for t,tr in enumerate(side['transitions']):
  from m3_4br4r1.policy import array_digest
  check(tr['iteration']==t and tr['next_iteration']==t+1 and tr['valid'] and tr['optimizer']==spec['optimizer'] and tr['delta_digest']==array_digest(a['x'][t+1]-a['x'][t]),'Exact native OC action metadata')
 check(np.array_equal(a['iteration'],np.arange(T)) and T-1==m['completed_updates'],'State/transition timing')
 check(not bool(a['has_incoming_update'][0]) and np.all(a['has_incoming_update'][1:]) and a['previous_change'][0]==0,'Incoming timing')
 check(np.all(a['update_change'][:-1]>=spec['optimizer']['tolerance']),'Missed first stopping point')
 reason='change_tolerance' if a['update_change'][-1]<spec['optimizer']['tolerance'] else 'maxiter';check(m['termination_reason']==reason,'Stopping classification');check(reason!='maxiter' or T-1==spec['optimizer']['maxiter'],'Incomplete capped history')
 for t in range(T):
  x=a['x'][t];check(np.isfinite(x).all() and min(x)>=.001 and max(x)<=1 and x.mean()<=spec['mechanical']['volume_fraction'] and sum(x)<=len(x)*spec['mechanical']['volume_fraction'],'Strict design feasibility')
 checkpoint=exact_final_checkpoint(side)
 check(set(execution) in ({'authorization_sha256','resumed','restart_control'},{'authorization_sha256','resumed','restart_control','executed_at'}),'Execution provenance')
 if execution['resumed']:
  c=execution['restart_control'];check(isinstance(c,dict) and set(c)=={'control_folder','control_evidence','interrupted_checkpoint'},'Missing real restart-control evidence')
  category_C=restart_equivalence(c['control_folder'],c['control_evidence'],folder,evidence,spec,c['interrupted_checkpoint'])
 else:check(execution['restart_control'] is None,'Unexpected restart declaration');category_C=dict(status='not_applicable_no_restart',policy='m3.4b.restart_acceptance.v2')
 return dict(passed=True,execution=execution,problem_id=spec['problem_id'],spec_sha256=canonical(spec),source_id=source_id,states=T,transitions=T-1,termination_reason=reason,reference_label_eligible=False,checkpoint=checkpoint,category_B=side['certificate'],category_C=category_C,native_hashes={k:v['sha256'] for k,v in m['arrays'].items()},metadata_sha256=digest(Path(folder)/'metadata.json'),evidence_summary_sha256=digest(Path(evidence)/'summary.json'))
