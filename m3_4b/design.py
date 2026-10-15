"""Deterministic164 specification design; no FEM or changes to native equations."""
import math,json
from collections import Counter,defaultdict
from dataclasses import replace
import numpy as np
from topopt_utils.fea.production_dataset import specification as old_spec
from topopt_utils.fea.trajectory_dataset import canonical_hash,load_problem

def specification(**cfg):
 cfg=dict(cfg);direction=cfg.get('direction','vertical');sign=cfg.pop('force_sign',1)
 args=dict(cfg,direction='oblique' if direction=='oblique_alt' else direction)
 spec=old_spec(**args)
 if direction=='oblique_alt' or sign!=1:
  p,x=load_problem(spec);loads=[]
  for node,fx,fy in p.loads:
   if direction=='oblique_alt':fx,fy=.8*cfg.get('magnitude',1),-.6*cfg.get('magnitude',1)
   loads.append((node,fx*sign,fy*sign))
  p=replace(p,loads=tuple(loads));spec['mechanical']=p.to_dict();spec['force_scale']=float(np.linalg.norm(p.forces))
  body={k:v for k,v in spec.items() if k not in ('problem_id','label','related_group','support_pattern','configuration')};spec['problem_id']='m3_3b_'+canonical_hash(body)[:24]
 c=spec['configuration'];c['direction']=direction;c['force_sign']=sign;c['aspect']=c['domain'][0]/c['domain'][1]
 support='left_clamp' if c['support']=='right_clamp' else c['support']
 spec['related_group']=f"{support}:{direction}:loc{c['location']:g}:aspect{c['aspect']:g}"
 spec['label']=spec['related_group']+f":v{c['volume']}:mesh{c['mesh']}:F{c['magnitude']}:start{c['initial']}"
 p,x=load_problem(spec)
 if p.n>4608 or p.nelx>96 or p.nely>48 or p.maxiter!=500 or p.tolerance!=.001:raise ValueError('Prohibited size/control')
 if np.sum(x)>p.n*p.volume_fraction or x.mean()>p.volume_fraction:raise ValueError('Initial infeasible under either reduction')
 if c['regime']=='refinement' and not np.allclose(spec['filter_physical_semi_axes'],[2,2],rtol=0,atol=1e-12):raise ValueError('Refinement physical filter mismatch')
 return spec

def configurations():
 families=[]
 for support in ('left_clamp','bottom_clamp','pin_roller'):
  for direction in ('horizontal','vertical','oblique'):
   families.append((dict(support=support,direction=direction,location=.5),'train','ID'))
   families.append((dict(support=support,direction=direction,location=.25,mesh=(48,32),domain=(48.,32.)),'train','ID'))
 for support in ('left_clamp','bottom_clamp','pin_roller'):
  families.append((dict(support=support,direction='vertical',location=.75),'validation','ID_interpolated_location_family'))
  families.append((dict(support=support,direction='oblique',location=.75),'test_id','ID_interpolated_location_family'))
 for direction in ('horizontal','vertical','oblique'):families.append((dict(support='dual_pin',direction=direction,location=.5),'test_ood','support_only_matched_pin_roller'))
 for support in ('left_clamp','bottom_clamp','pin_roller'):families.append((dict(support=support,direction='vertical',location=.125),'test_ood','load_location_only'))
 for support in ('left_clamp','bottom_clamp'):families.append((dict(support=support,direction='oblique_alt',location=.5),'test_ood','force_direction_only'))
 for direction in ('horizontal','vertical'):families.append((dict(direction=direction,location=.5,mesh=(48,16),domain=(48.,16.)),'test_ood','aspect_geometry_boundary_relative_load'))
 for direction in ('horizontal','vertical'):families.append((dict(direction=direction,location=.75 if direction=='horizontal' else .875,mesh=(64,32),regime='refinement'),'test_ood','held_spacing_0.75_same_physical_geometry'))
 result=[]
 for cfg,part,axis in families:
  for v in (.3,.4,.5,.6):result.append((dict(cfg,volume=v),part,axis))
 for cfg,part,axis in families:
  if cfg.get('mesh',(48,24))==(48,24) and part in ('train','validation','test_id'):
   result.append((dict(cfg,volume=.5,mesh=(96,48),regime='refinement'),part,'seen96_unseen_family' if part!='train' else 'fixed_domain_refinement'))
 controls=[dict(direction='horizontal',magnitude=.1),dict(direction='vertical',magnitude=10.),dict(direction='oblique',initial='perturbed',seed=1021),dict(support='right_clamp',direction='horizontal',force_sign=-1,location=.25,mesh=(48,32),domain=(48.,32.)),dict(support='bottom_clamp',direction='horizontal',initial='perturbed',seed=1101)]
 # The fifth control also supplies a perturbed production case; growth replaces one refinement control below.
 for c in controls:result.append((dict(dict(location=.5,volume=.5),**c),'train','declared_force_start_mirror_control'))
 # Keep20 variants total and add domain growth within an already-declared family.
 index=next(i for i,(c,p,a) in enumerate(result) if c.get('support')=='pin_roller' and c.get('direction')=='horizontal' and c.get('mesh')==(96,48) and p=='train')
 c,p,a=result[index];result[index]=(dict(c,domain=(96.,48.),regime='growth'),p,'domain_growth_relative')
 # Held64 physical families remain wholly OOD; coarse matches stay with them.
 for support,direction in [('bottom_clamp','vertical'),('bottom_clamp','oblique')]:
  ix=next(i for i,(c,p,a) in enumerate(result) if c.get('support')==support and c.get('direction')==direction and c.get('mesh')==(96,48) and p=='train')
  result.pop(ix)
 for direction,loc in [('horizontal',.75),('vertical',.875)]:result.append((dict(direction=direction,location=loc,volume=.5),'test_ood','held_spacing_paired_coarse_same_partition'))
 assert len(result)==164
 return result

def entry(spec,path,part,axis):
 c=spec['configuration'];return dict(problem_id=spec['problem_id'],spec=path,spec_sha256=canonical_hash(spec),family=spec['related_group'],partition=part,ood_axis=axis,configuration=c)

def batch_indices(entries):
 def find(**kw):
  for i,e in enumerate(entries):
   c=e['configuration'];defaults=dict(mesh=[48,24],domain=[48.,24.],initial='uniform',magnitude=1.,support='left_clamp',location=.5,volume=.5,regime='baseline',direction='vertical',force_sign=1)
   defaults.update(kw)
   if all(c.get(k)==v for k,v in defaults.items()):return i
  raise ValueError('Required controlled batch spec absent: '+str(kw))
 return [find(direction='horizontal'),find(direction='horizontal',mesh=[96,48],regime='refinement'),find(direction='vertical',volume=.3),find(direction='oblique',volume=.4),find(support='bottom_clamp',direction='horizontal',initial='perturbed'),find(support='pin_roller',direction='vertical'),find(direction='vertical',location=.25,volume=.3,mesh=[48,32],domain=[48.,32.]),find(direction='oblique',location=.25,volume=.4,mesh=[48,32],domain=[48.,32.]),find(support='bottom_clamp',direction='vertical',volume=.6),find(support='pin_roller',direction='horizontal',location=.25,volume=.4,mesh=[48,32],domain=[48.,32.]),find(support='bottom_clamp',direction='oblique',location=.25,volume=.3,mesh=[48,32],domain=[48.,32.]),find(direction='horizontal',location=.25,volume=.6,mesh=[48,32],domain=[48.,32.])]

def audit(entries):
 groups=defaultdict(list);ids=set();coverage=defaultdict(lambda:defaultdict(set))
 for e in entries:
  if e['problem_id'] in ids:raise ValueError('Exact complete duplicate')
  ids.add(e['problem_id']);groups[e['family']].append(e)
  for k in ('support','direction','volume','aspect','mesh','initial','regime'):coverage[e['partition']][k].add(str(e['configuration'][k]))
 for f,es in groups.items():
  if len({e['partition'] for e in es})!=1:raise ValueError('Declared family leakage')
 sizes=[len(es) for es in groups.values()]
 if max(sizes)>6:raise ValueError('Quota exceeded: four volumes plus at most two distributed controls')
 train=coverage['train']
 if not {'horizontal','vertical','oblique'}<=train['direction'] or len(train['aspect'])<2 or '[96, 48]' not in train['mesh']:raise ValueError('Training coverage missing')
 representatives=[es[0] for es in groups.values()];neighbors=[]
 for i,a in enumerate(representatives):
  ca=a['configuration']
  for b in representatives[:i]:
   cb=b['configuration'];sa='left_clamp' if ca['support']=='right_clamp' else ca['support'];sb='left_clamp' if cb['support']=='right_clamp' else cb['support']
   if sa!=sb:continue
   angle={'horizontal':0.,'vertical':math.pi/2,'oblique':math.atan2(.8,.6),'oblique_alt':math.atan2(.6,.8)}
   delta=abs(angle[ca['direction']]-angle[cb['direction']])/math.pi;d=math.sqrt(delta**2+(ca['location']-cb['location'])**2+math.log(ca['aspect']/cb['aspect'])**2)
   if d<=.30:neighbors.append(dict(a=a['family'],b=b['family'],distance=d,cross_partition=a['partition']!=b['partition'],status='review_candidate_not_automatic_union',axes=[a['ood_axis'],b['ood_axis']]))
 return dict(configurations=len(entries),families=len(groups),exact_duplicates=[],declared_cross_partition_leakage=False,family_sizes={f:len(es) for f,es in groups.items()},max_per_family=max(sizes),largest_family_share=max(sizes)/len(entries),effective_configuration_weighted_families=len(entries)**2/sum(x*x for x in sizes),partitions=dict(Counter(e['partition'] for e in entries)),coverage={p:{k:sorted(v) for k,v in d.items()} for p,d in coverage.items()},near_neighbor_candidates=neighbors,cross_partition_candidates=sum(n['cross_partition'] for n in neighbors),descriptor='Same canonical support required. Euclidean(normalized load location, force-line angle/pi modulo sign, log physical aspect), threshold0.30. Different support is a categorical veto. Excludes volume/start/force/resolution declared relatives. No automatic union of distinct mechanics.',resolution_interpretation='96 is seen in train: held-family96 tests are not true unseen-resolution tests.64 mesh/dx=.75 held out with physical domain/filter fixed.192 future only. All physical refinement relatives share partition.',geometry='Aspect1.5 and2 training; aspect3 OOD changes absolute point height with preserved relative boundary position; not a support or direction change.')
