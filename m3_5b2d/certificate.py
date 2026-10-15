"""Schema and future raw-evidence assembler; no current acceptance certificate."""
from .common import *
UNCERTIFIED='uncertified_missing_original_receipt_and_control'
FIELDS={'schema_version','fixture','binding','states','transitions','counts','resources','gates','v2','v3','supplemental','termination','reference_optimum','historical_interruption_status','missing_evidence','raw_evidence','approvals','raw_validation_complete','numerical_validation_accepted','release_eligible','certificate_hash'}
def seal(c):return {**c,'certificate_hash':canonical({k:v for k,v in c.items() if k!='certificate_hash'})}
def template(b):
 return seal(dict(schema_version='m3.5b2d.validation_certificate.v1',fixture=False,binding=b,states=0,transitions=0,counts={k:0 for k in WORK},resources=[],gates={k:False for k in sorted(GATES)},v2=False,v3=False,supplemental=False,termination='maxiter',reference_optimum=False,historical_interruption_status=UNCERTIFIED,missing_evidence=['original_interruption_receipt','original_target_specific_restart_control'],raw_evidence={},approvals=dict(policy=None,numerical_acceptance=None,alias=None,family_crosswalk=None),raw_validation_complete=False,numerical_validation_accepted=False,release_eligible=False))
def validate(c,b,complete=False):
 exact(c,FIELDS,'certificate');check(c['schema_version']=='m3.5b2d.validation_certificate.v1' and c['fixture'] is False,'Synthetic fixture cannot be real evidence')
 check(c['binding']==b and c['certificate_hash']==seal(c)['certificate_hash'],'Certificate target/source/input/hash')
 check(c['historical_interruption_status']==UNCERTIFIED and c['missing_evidence']==['original_interruption_receipt','original_target_specific_restart_control'],'Cannot certify missing historical interruption')
 check(c['termination']=='maxiter' and c['reference_optimum'] is False and c['release_eligible'] is False and c['numerical_validation_accepted'] is False,'No automatic reference/release/human numerical acceptance')
 exact(c['counts'],WORK,'work');check(all(type(c['counts'][k]) in (int,float) and 0<=c['counts'][k]<=v for k,v in WORK.items()),'Work overrun')
 exact(c['gates'],GATES,'gates');check(all(type(v) is bool for v in c['gates'].values()),'Gate types')
 exact(c['approvals'],{'policy','numerical_acceptance','alias','family_crosswalk'},'approvals')
 if complete:
  check(c['raw_validation_complete'] is True and c['states']==501 and c['transitions']==500 and all(c['gates'].values()) and c['v2'] is True and c['v3'] is True and c['supplemental'] is True,'Incomplete v2/v3/supplemental/current gates')
  check(c['counts']['FEM']==1005 and c['counts']['OC']==500 and c['counts']['selected_trials']==500 and c['counts']['workers']==2,'Incomplete work ledger')
  check(isinstance(c['raw_evidence'],dict) and set(c['raw_evidence'])=={'root','replay_summary_sha256','supplemental_sha256','ledger_sha256','parent_memory_sha256','checkpoint'},'Missing raw evidence')
  check(len(c['resources'])==2,'Partial monitored output')
  from .accounting import verify_resources
  for r in c['resources']:verify_resources(r)
 return c

def verify_saved_correctness(root,b):
 # All primary FEM/OC formulas and audits are delegated to frozen implementations.
 # This function separately verifies universal native bookkeeping; it does not
 # call the old resumed execution path with a false resumed=False assertion.
 from .accounting import parent_guard
 parent_guard()
 import numpy as np
 from m3_5b2.validation import independent,exact_final_checkpoint
 from m3_4br4r1.policy import array_digest
 side=independent(INPUT,Path(root)/'replay',read(INPUT/'problem.json'));a=side['arrays'];m=side['metadata'];spec=side['spec'];T=501
 check(m['source_id']==b['historical_source_id'] and m['generation_status']=='completed','Native identity/completion')
 check(np.array_equal(a['x'][0],np.asarray(spec['initial_x'])) and np.array_equal(a['iteration'],np.arange(T)),'Initial/order')
 check(m['optimizer_option']==spec['optimizer']['option'] and m['optimizer_version']==spec['optimizer']['version'] and m['tolerance']==spec['optimizer']['tolerance'],'Optimizer binding')
 check(len(side['transitions'])==500 and m['state_count']==T and m['completed_updates']==500 and m['termination_reason']=='maxiter' and not m['reference_label_eligible'] and not m['optimality_certified'],'Counts/cap/reference')
 check(not bool(a['has_incoming_update'][0]) and np.all(a['has_incoming_update'][1:]) and a['previous_change'][0]==0 and np.all(a['update_change']>=spec['optimizer']['tolerance']),'Original first stopping/timing')
 for t,x in enumerate(a['x']):
  if t%16==0:parent_guard()
  check(np.isfinite(x).all() and min(x)>=.001 and max(x)<=1 and x.mean()<=.5 and sum(x)<=len(x)*.5,'Strict native feasibility')
  if t<500:
   tr=side['transitions'][t];delta=a['x'][t+1]-x;change=float(np.max(abs(delta)))
   check(tr['iteration']==t and tr['next_iteration']==t+1 and tr['valid'] and tr['optimizer']==spec['optimizer'] and array_digest(delta)==tr['delta_digest'],'Transition identity')
   check(change==tr['maximum_change']==a['update_change'][t]==a['previous_change'][t+1],'Exact action/change bookkeeping')
 proof=exact_final_checkpoint(side);parent_guard();return proof

def assemble(root,p):
 root=Path(root).resolve();check(root.is_relative_to(ROOT/'m3_5b2d/validation_runs'),'Actual evidence root');b=p['binding'];verify_binding(b);checkpoint=verify_saved_correctness(root,b)
 sup=read(root/'supplemental.json');check(sup['passed'] and sup['problem_id']==HISTORICAL and sup['extra_FEM_calls']==3 and sup['extra_OC_calls']==0,'Actual supplemental gate')
 from m3_4b.monitoring import verify_trace
 from .accounting import verify_resources,parent_guard
 resources=[read(root/'monitor'/f'{s}.json') for s in ('replay','supplemental')]
 for r in resources:verify_trace(r);verify_resources(r)
 observation=parent_guard();parent_file=root/'parent_acceptance_memory.json'
 if not parent_file.exists():write(parent_file,observation)
 recorded=read(parent_file);check(recorded['VmRSS']<=768 and recorded['VmHWM']<=768,'Recorded parent acceptance memory')
 c=template(b);c.update(states=501,transitions=500,counts=read(root/'work_ledger.json')['counts'],resources=resources,gates={k:True for k in GATES},v2=True,v3=True,supplemental=True,raw_validation_complete=True,raw_evidence=dict(root=str(root),replay_summary_sha256=digest(root/'replay/summary.json'),supplemental_sha256=digest(root/'supplemental.json'),ledger_sha256=digest(root/'work_ledger.json'),parent_memory_sha256=digest(parent_file),checkpoint=checkpoint))
 from .authorization import approval_record
 c['approvals']['policy']=approval_record('policy',digest(Path(__file__).with_name('policy.py')),b)
 c=seal(c);validate(c,b,complete=True);return c
