"""Correct planning labels/folds only; retain native identity and immutable specs."""
import json,copy,collections
from pathlib import Path
from .memory import digest
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'m3_4b/generations/20261014T013017Z'
REVIEW=ROOT/'m3_4br/reviews/20261014T024140Z'

def build(out):
 from m3_4b.design import audit
 from topopt_utils.fea.trajectory_dataset import load_problem,canonical_hash
 m=json.loads((OLD/'proposed_manifest.json').read_text());new=copy.deepcopy(m);changes=[]
 for e in new['jobs']:
  cfg=e['configuration'];oldpart=e['partition'];axis=e['ood_axis']
  if axis=='ID_interpolated_location_family':e['ood_axis']='held_location_extrapolation_within_support_geometry_stratum'
  # All directions in a held support/location cluster stay in the same fold.
  if cfg['location']==.75 and cfg.get('aspect',cfg['domain'][0]/cfg['domain'][1])==2 and cfg['support'] in ('left_clamp','bottom_clamp','pin_roller') and oldpart in ('validation','test_id'):
   e['partition']='validation' if cfg['support']=='left_clamp' else 'test_id'
   e['ood_axis']='held_location_extrapolation_seen_resolution_unseen_family'
  if e['partition']!=oldpart or e['ood_axis']!=axis:changes.append(dict(problem_id=e['problem_id'],family=e['family'],old_partition=oldpart,new_partition=e['partition'],old_axis=axis,new_axis=e['ood_axis']))
  # Copy specs unchanged into isolated manifest package for compatibility.
  src=OLD/e['spec'];dst=out/e['spec'];dst.parent.mkdir(exist_ok=True,parents=True);dst.write_bytes(src.read_bytes())
  if canonical_hash(json.loads(dst.read_text()))!=e['spec_sha256']:raise ValueError('Spec mismatch')
  p,x=load_problem(json.loads(dst.read_text()))
  if x.min()<.001 or x.max()>1 or x.mean()>p.volume_fraction or sum(x)>len(x)*p.volume_fraction:raise ValueError('Invalid initial spec')
 new['status']='planning_only_not_authorized';new['revision']='m3.4br2.planning.v2';new['parent_manifest_sha256']=digest(OLD/'proposed_manifest.json')
 new['scientific_claims']=dict(location_interpolation=False,held_075='location extrapolation; no independent-ID claim',resolution96='seen in train; held families at seen resolution',resolution64='held spacing; same-family 48/64 volume .5 pairs only isolate refinement',domain_growth='distinct from fixed-domain refinement',normalization='fit training problems only; never offline termination/targets')
 summary=audit(new['jobs']);assert len(new['jobs'])==164 and len({x['problem_id'] for x in new['jobs']})==164
 # Record every original candidate with complete original descriptors/variants.
 pairs=json.loads((REVIEW/'near_neighbor_classification.json').read_text())['pairs'];assignment={e['family']:e['partition'] for e in new['jobs']}
 for pair in pairs:
  pair['new_partitions']=[assignment[pair['a']],assignment[pair['b']]]
  cls=pair['classification']
  if 'Potentially' in cls:decision='Retain across folds as explicit held-location extrapolation; no interpolation/independent-ID claim.'
  elif 'Intentional' in cls:decision='Retain controlled direction OOD; match actual geometry, position, volume and discretization before attributing one factor.'
  elif pair['index'] in (3,6,9):decision='Group all same-support/location held directions into one fold; prevents validation/test selection on correlated angle variants.'
  elif pair['index'] in (13,17):decision='Retain and label composite direction plus location OOD; exact-position baseline controls provide one-factor tests separately.'
  else:decision='Retain composite family/position/direction/spacing OOD; only same-family volume-.5 coarse/fine pairs isolate refinement.'
  pair.update(disposition=decision,resolved=True,same_related_family=pair['a']==pair['b'])
 counts=collections.Counter(e['partition'] for e in new['jobs']);coverage={}
 for part in counts:
  es=[e for e in new['jobs'] if e['partition']==part]
  coverage[part]={k:sorted({json.dumps(e['configuration'].get(k)) for e in es}) for k in ('direction','support','volume','aspect','location','mesh','dx','dy','regime','initial')}
 (out/'corrected_planning_manifest.json').write_text(json.dumps(new,indent=2));(out/'split_comparison.json').write_text(json.dumps(dict(changes=changes,counts=counts,coverage=coverage,audit=summary,pairs=pairs,executed_assignments_changed=False,planning_only=True),indent=2))
 return counts
