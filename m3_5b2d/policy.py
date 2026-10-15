"""Separate candidate scope. Never changes restart policy or issues authority."""
from .common import *
POLICY='historical_saved_trajectory_admission.v1_candidate'
DEFAULT=dict(policy_approved=False,historical_admission_enabled=False,numerical_validation_accepted=False,alias_approved=False,family_crosswalk_approved=False,release_eligible=False)
def decision_state(decision,*,policy_approved=False,numerical=False,provenance=False,alias=False,crosswalk=False):
 check(decision in (None,'D1','D2'),'Unknown policy decision')
 eligible=decision=='D2' and all(type(x) is bool and x for x in (policy_approved,numerical,provenance,alias,crosswalk))
 return dict(policy=POLICY,decision=decision,policy_approved=policy_approved,historical_admission_enabled=decision=='D2' and policy_approved,numerical_validation_accepted=numerical,alias_approved=alias,family_crosswalk_approved=crosswalk,release_eligible=eligible,historical_interruption_status='uncertified_missing_original_receipt',production_restart_policy_changed=False,reference_optimum=False,reason='D1 blocks original-event admission' if decision=='D1' else 'D2 is a separate saved-state/provenance category; approval and every gate mandatory' if decision=='D2' else 'No decision approved')
def scope(binding):
 check(binding['historical_id']==HISTORICAL and binding['selected_id']==SELECTED and binding['manifest_sha256']==MANIFEST and binding['equivalence_sha256']==EQUIVALENCE and binding['input_root']==str(INPUT),'Candidate applies only to exact historical target, never new production')
 return True
