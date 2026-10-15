"""Disabled alias/crosswalk and registry/release staging, no release publisher."""
from .common import *
from .certificate import UNCERTIFIED,validate,verify_saved_correctness
from .authorization import approval_record
from .policy import decision_state
ALIAS_FIELDS={'schema_version','binding','historical_spec','selected_spec','seeds','families','partitions','physical_history_key','transitions','termination','reference_optimum','historical_interruption_status','certificate_reference','alias_approved','family_crosswalk_approved','approval_metadata'}
def alias_template(b):
 old=read(ROOT/'m3_5b2s/reports/20261015_protocol_v1/alias_bindings.json')
 return dict(schema_version='m3.5b2d.provenance_alias.v1',binding=b,historical_spec=old['historical']['spec'],selected_spec=old['selected']['spec'],seeds={'historical':703,'selected':0},families={'historical':'right_vertical','selected':'left_clamp:vertical:loc0.5:aspect2','scope':'this_member_only'},partitions={'historical':'train','selected':'train'},physical_history_key=b['physical_history_key'],transitions=500,termination='maxiter',reference_optimum=False,historical_interruption_status=UNCERTIFIED,certificate_reference=None,alias_approved=False,family_crosswalk_approved=False,approval_metadata={'alias':None,'crosswalk':None})
def validate_alias(a,b,approved=False):
 exact(a,ALIAS_FIELDS,'alias');old=read(ROOT/'m3_5b2s/reports/20261015_protocol_v1/alias_bindings.json')
 check(a['schema_version']=='m3.5b2d.provenance_alias.v1' and a['binding']==b,'Alias target/source binding')
 check(a['historical_spec']==old['historical']['spec'] and a['selected_spec']==old['selected']['spec'] and a['seeds']=={'historical':703,'selected':0},'Both immutable specs/seeds')
 check(a['families']=={'historical':'right_vertical','selected':'left_clamp:vertical:loc0.5:aspect2','scope':'this_member_only'} and a['partitions']=={'historical':'train','selected':'train'},'Family/global merge/cross-partition alias')
 check(a['physical_history_key']==b['physical_history_key'] and a['transitions']==500 and a['termination']=='maxiter' and a['reference_optimum'] is False,'Duplicate/count/capped reference')
 check(a['historical_interruption_status']==UNCERTIFIED,'Historical interruption cannot be certified')
 exact(a['approval_metadata'],{'alias','crosswalk'},'independent approvals')
 check(type(a['alias_approved']) is bool and type(a['family_crosswalk_approved']) is bool,'Approval flag types')
 if approved:check(a['alias_approved'] and a['family_crosswalk_approved'] and all(a['approval_metadata'].values()),'Unapproved alias/crosswalk')
 return a

def ensure_single_history(entries,b):
 for e in entries:
  ids={e.get('problem_id'),e.get('historical_id'),e.get('selected_id')}
  check(not ids & {HISTORICAL,SELECTED} and e.get('physical_history_key')!=b['physical_history_key'] and e.get('equivalence_sha256')!=EQUIVALENCE,'Duplicate history/transition/family contribution')
 return dict(histories_added=1,transitions_added=500,family_contributions_added=1,partition='train')

class HistoricalRegistryAdapter:
 @staticmethod
 def inspect(b,existing_entries):
  actual=registry_check(b);actual.update(ensure_single_history(existing_entries,b));return actual

class HistoricalReleaseAdapter:
 """Only return a reviewed staging record; never write or activate an ML index.
 Existing M3.5A feature extraction/mmap/state0 train-only normalization untouched.
 """
 @staticmethod
 def stage_entry(alias_path,certificate_path,existing_entries):
  b=target_binding();a=validate_alias(read(alias_path),b,approved=True);c=read(certificate_path);validate(c,b,complete=True)
  check(a['certificate_reference']=={'path':str(Path(certificate_path).resolve()),'sha256':digest(certificate_path)},'Alias certificate reference')
  approvals={kind:approval_record(kind,h,b) for kind,h in [('policy',digest(Path(__file__).with_name('policy.py'))),('alias',digest(alias_path)),('family_crosswalk',canonical(a['families'])),('numerical_acceptance',digest(certificate_path))]}
  metadata=lambda r:{k:r[k] for k in ('identity','approved_at','reference')}
  check(a['approval_metadata']['alias']==metadata(approvals['alias']) and a['approval_metadata']['crosswalk']==metadata(approvals['family_crosswalk']),'Independent alias/crosswalk approval binding')
  HistoricalRegistryAdapter.inspect(b,existing_entries)
  from .certificate import assemble
  check((Path(c['raw_evidence']['root'])/'parent_acceptance_memory.json').is_file(),'Missing parent acceptance observation')
  actual=assemble(Path(c['raw_evidence']['root']),{'binding':b});check(actual==c,'Retained raw certificate differs from full independent evidence')
  state=decision_state('D2',policy_approved=True,numerical=True,provenance=True,alias=True,crosswalk=True);check(state['release_eligible'],'All candidate admission gates')
  return dict(schema_version='m3.5b2d.reviewed_admission_staging.v1',problem_id=SELECTED,historical_id=HISTORICAL,selected_id=SELECTED,physical_history_key=b['physical_history_key'],equivalence_sha256=EQUIVALENCE,folder=str(INPUT),states=501,transitions=500,partition='train',family=a['families']['selected'],historical_family=a['families']['historical'],source_id=b['historical_source_id'],historical_spec=a['historical_spec'],selected_spec=a['selected_spec'],termination_reason='maxiter',reference_label_eligible=False,historical_interruption_status=UNCERTIFIED,certificate_sha256=digest(certificate_path),approvals=approvals,published=False)
