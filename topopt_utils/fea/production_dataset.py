"""M3.4 wrappers: explicit ownership, immutable plans and bounded analyses."""
import json,hashlib,gc,math
from pathlib import Path
from collections import Counter,defaultdict
from contextlib import contextmanager
import numpy as np
from .solver import FeaSolver
from .mechanical_problem import MechanicalProblem,DensityMap,array_digest
from .trajectory_dataset import canonical_hash,load_problem,Trajectory,features,PROFILES
from .robust_oc import OPTIONS,update,trial
from .dataset_storage import verify_state,atomic_json
from .consistent_trajectory import evaluate_state,independent_stiffness,symmetric_csr,stiffness_probes
from .trajectory_validation import comparison
from .dataset_readiness import endpoint_analysis,POLICY,quality_flags

SCHEMA='m3.4.production.v1'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

class OwnedSolver(FeaSolver):
    """Resource ownership only; all assembly/solve/sensitivity methods inherited."""
    def __enter__(self):return self
    def close(self):
        for ctx in list(self._symbolic_cache.values()):ctx.free()
        self._symbolic_cache.clear()
    def __exit__(self,*exc):self.close()


def specification(support='left_clamp',direction='vertical',location=.5,volume=.5,mesh=(48,24),domain=(48.,24.),regime='baseline',initial='uniform',magnitude=1.,seed=0,cap=500):
    nx,ny=mesh;w,h=domain
    if nx*ny>4608 or nx>96 or ny>48 or cap>500:raise ValueError('M3.4 excludes192/384 FEM and caps above500')
    node=lambda i,j:j*(nx+1)+i
    if support in ('left_clamp','right_clamp'):
        fixed=tuple(2*node(0 if support=='left_clamp' else nx,j)+a for j in range(ny+1) for a in (0,1));pos=(nx if support=='left_clamp' else 0,location*ny)
    elif support=='bottom_clamp':fixed=tuple(2*node(i,0)+a for i in range(nx+1) for a in (0,1));pos=(location*nx,ny)
    elif support in ('pin_roller','dual_pin'):
        fixed=(0,1,2*nx+1) if support=='pin_roller' else (0,1,2*nx,2*nx+1);pos=(location*nx,ny)
    else:raise ValueError('Unsupported stable support')
    if not np.allclose(pos,np.rint(pos),rtol=0,atol=1e-12):raise ValueError('Off-grid point load; no snapping')
    fx,fy={'horizontal':(1.,0.),'vertical':(0.,-1.),'oblique':(.6,-.8)}[direction]
    if support=='right_clamp':fx=-fx
    radius=2/(w/nx) if regime=='refinement' else 2.
    p=MechanicalProblem(nx,ny,float(w),float(h),fixed,((node(*map(int,pos)),fx*magnitude,fy*magnitude),),volume_fraction=volume,filter_radius=radius,maxiter=cap,seed=seed)
    x=np.full(p.n,volume-1e-8)
    if initial=='perturbed':
        noise=np.random.default_rng(seed).uniform(-1,1,p.n);noise-=noise.mean();noise*=.02/max(abs(noise));x+=noise-1e-5
    elif initial!='uniform':raise ValueError('Unknown start')
    body=dict(schema_version='m3.3b.problem.v1',mechanical=p.to_dict(),initial_x=x.tolist(),initial_digest=array_digest(x),optimizer=dict(option='feasible_upper_v1',version='m3.3a.oc.v1',**OPTIONS['feasible_upper_v1'],maxiter=cap,tolerance=.001,move=.1,x_min=.001,x_max=1.,volume_variable='x'),force_scale=float(np.linalg.norm(p.forces)),filter_physical_semi_axes=[radius*p.dx,radius*p.dy])
    canonical_support='left_clamp' if support=='right_clamp' else support
    family=f'{canonical_support}_{direction}_{location:g}_aspect{w/h:g}'
    config=dict(support=support,direction=direction,location=location,volume=volume,mesh=list(mesh),domain=list(domain),regime=regime,initial=initial,magnitude=magnitude,seed=seed,cap=cap,dx=p.dx,dy=p.dy,discrete_filter='unchanged DensityMap: normalized cone, boundary row division',physical_filter_semi_axes=body['filter_physical_semi_axes'],point_force='exact nodal, no traction or interpolation')
    spec=dict(body,problem_id='m3_3b_'+canonical_hash(body)[:24],label=family+f'_v{volume:g}_{nx}_{initial}_F{magnitude:g}',related_group=family,support_pattern=support,configuration=config)
    load_problem(spec);return spec


def proposed_cases():
    cases=[]
    for si,support in enumerate(('left_clamp','bottom_clamp','pin_roller')):
        for di,direction in enumerate(('horizontal','vertical','oblique')):
            for location in (.25,.5,.75):
                partition='train' if location!=.5 else ('validation' if (si+di)%2==0 else 'test_id')
                for volume in (.3,.4,.5,.6):cases.append((dict(support=support,direction=direction,location=location,volume=volume),partition,'ID'))
    for direction in ('horizontal','vertical','oblique'):
        for volume in (.3,.4,.5,.6):cases.append((dict(support='dual_pin',direction=direction,location=.25,volume=volume),'test_ood','support_only_vs_pin_roller'))
    for support in ('left_clamp','bottom_clamp','pin_roller'):
        for volume in (.3,.4,.5,.6):cases.append((dict(support=support,direction='vertical',location=.125,volume=volume),'test_ood','location_only'))
    for direction in ('horizontal','vertical','oblique'):
        for volume in (.3,.4,.5,.6):cases.append((dict(direction=direction,location=.25,volume=volume,mesh=(48,12),domain=(48.,12.)),'test_ood','geometry:boundary_relative_load_preserved'))
    base=[c for c in cases if c[1]=='train' and c[0]['location']==.25][:6]
    for cfg,part,axis in base:
        cases.append((dict(cfg,mesh=(96,48),regime='refinement'),part,'paired_refinement'))
        cases.append((dict(cfg,mesh=(96,48),domain=(96.,48.),regime='growth'),part,'paired_domain_growth'))
    for magnitude in (.1,10.):cases.append((dict(base[0][0],magnitude=magnitude),'train','force_control'))
    for seed in (1021,1022):cases.append((dict(base[0][0],initial='perturbed',seed=seed),'train','start_control'))
    for cfg,part,axis in base[:4]:cases.append((dict(cfg,support='right_clamp'),part,'mirror_control'))
    return cases


def pilot_cases(cap=500):
    return [dict(mesh=(12,6),domain=(12.,6.),direction='horizontal',seed=701,cap=cap),dict(direction='horizontal',cap=cap),dict(mesh=(96,48),regime='refinement',direction='horizontal',cap=cap),dict(direction='vertical',volume=.3,cap=cap),dict(direction='oblique',volume=.4,cap=cap),dict(support='bottom_clamp',direction='horizontal',volume=.6,initial='perturbed',mesh=(24,12),domain=(24.,12.),seed=1101,cap=cap),dict(support='pin_roller',direction='vertical',volume=.5,cap=cap)]


def audit_families(entries):
    families={};coverage=defaultdict(lambda:defaultdict(set));ids=set()
    for e in entries:
        if e['problem_id'] in ids:raise ValueError('Duplicate complete problem')
        ids.add(e['problem_id']);fam=e['family'];part=e['partition']
        if fam in families and families[fam]!=part:raise ValueError('Related family crosses partitions')
        families[fam]=part
        for key in ('direction','support','volume','regime'):coverage[part][key].add(str(e[key]))
    train=coverage.get('train',{})
    if not {'horizontal','vertical','oblique'}<=train.get('direction',set()):raise ValueError('Training lacks representative directions')
    if not {'.3','.4','.5','.6'}<=set(v.removeprefix('0') for v in train.get('volume',set())):raise ValueError('Training lacks volume categories')
    if len(train.get('support',set()))<3:raise ValueError('Training lacks support variation')
    return dict(unique_configurations=len(entries),mechanical_families=len(families),family_leakage=False,coverage={p:{k:sorted(v) for k,v in d.items()} for p,d in coverage.items()},OOD_axes=dict(Counter(e['ood_axis'] for e in entries if e['partition']=='test_ood')),future_resolution='192 planning only, no executable192 jobs',limitations='Geometric OOD changes physical boundary-relative load coordinates as stated; family keys conservative mirrors/force/start/resolution grouping, not proof against every continuous near-duplicate')


def stream_replay(tr):
    """One problem at a time, one reused owner, no retained comparison-row list."""
    p=tr.problem;a=tr.arrays;T=tr.metadata['state_count'];rows=0;max_error=0.
    comparison(a['x'][0],tr.initial,1e-12,'initial');np.testing.assert_array_equal(a['iteration'],np.arange(T))
    if bool(a['has_incoming_update'][0]) or not np.all(a['has_incoming_update'][1:]):raise ValueError('State timing mask')
    if np.any(a['update_change'][:-1]<p.tolerance):raise ValueError('Missed first stopping point')
    expected='change_tolerance' if a['update_change'][-1]<p.tolerance else 'maxiter'
    if tr.metadata['termination_reason']!=expected or (expected=='maxiter' and T-1!=p.maxiter):raise ValueError('Terminal timing')
    with OwnedSolver() as solver:
        for t in range(T):
            state={k:v[t] for k,v in a.items() if k!='update_change'};verify_state(p,state)
            truth=evaluate_state(p,a['x'][t],solver)
            for key in ('rho','u','compliance','g_rho','g_x','k_diagonal','k_probe_actions'):
                r=comparison(truth[key],a[key][t],1e-9,key);rows+=1;max_error=max(max_error,r.get('relative_error',0.))
            C,g=solver.compliance(a['rho'][t],*p.solver_args,sensitivities=True)
            comparison(C,a['compliance'][t],1e-9,'API C');comparison(g,a['g_rho'][t],1e-9,'API gradient');comparison(DensityMap(p).transpose(g),a['g_x'][t],1e-9,'transpose gradient');rows+=3
            physical=independent_stiffness(p,a['rho'][t],solver);free=np.ones(p.ndof);free[list(p.fixed_dofs)]=0
            from scipy.sparse import diags
            expectedK=diags(free)@physical@diags(free)+diags(1-free)
            conn=solver.get_connectivity(p.nelx,p.nely);pat=solver.get_mesh_pattern(p.nelx,p.nely)
            actual=symmetric_csr(solver.assemble_dirichlet(a['rho'][t],pat,solver.get_KE(p.dx,p.dy),conn,np.asarray(p.fixed_dofs)))
            comparison(expectedK@stiffness_probes(p),actual@stiffness_probes(p),1e-12,'COO actions');rows+=1
            if t<T-1:
                saved=tr.transitions[t];nextx,d=update(a['x'][t],DensityMap(p).transpose(g),p.volume_fraction,'feasible_upper_v1')
                comparison(nextx,a['x'][t+1],1e-8,'OC');comparison(np.max(abs(nextx-a['x'][t])),a['previous_change'][t+1],1e-12,'incoming change');rows+=2
                if array_digest(a['x'][t+1]-a['x'][t])!=saved['delta_digest']:raise ValueError('Delta fingerprint')
                for key,value in d.items():
                    if key=='seconds':continue
                    if isinstance(value,(str,bool)):
                        if value!=saved['diagnostic'][key]:raise ValueError('OC diagnostic '+key)
                    else:comparison(value,saved['diagnostic'][key],1e-12,'OC '+key)
                comparison(trial(a['x'][t],a['g_x'][t],saved['selected_multiplier']),a['x'][t+1],1e-12,'selected upper')
    return dict(passed=True,states=T,updates=T-1,comparison_rows=rows,independent_FEM_evaluations=2*T,comparison_max_relative_error=max_error)


def window_ranges(entries,k=(1,3),h=(1,5,10)):
    """Compact index: six ranges per problem, rather than materialized windows."""
    for e in entries:
        for history in k:
            for horizon in h:
                count=max(0,e['states']-horizon-history+1)
                yield dict(problem_id=e['problem_id'],family=e['family'],partition=e['partition'],k=history,h=horizon,first_anchor=history-1,count=count)


def draw_training(ranges,count,seed,mode='family_first',stage_balance=False):
    valid=[r for r in ranges if r['partition']=='train' and r['count']>0]
    if not valid:raise ValueError('No training ranges')
    if mode not in ('uniform_windows','uniform_problems','uniform_families','family_first'):raise ValueError('Sampler mode')
    problems=sorted({r['problem_id'] for r in valid});families=sorted({r['family'] for r in valid});rng=np.random.default_rng(seed)
    for _ in range(count):
        eligible=valid;prob=1.
        if mode in ('uniform_families','family_first'):
            fam=families[int(rng.integers(len(families)))];eligible=[r for r in valid if r['family']==fam];prob/=len(families)
        if mode in ('uniform_problems','family_first'):
            pids=sorted({r['problem_id'] for r in eligible});pid=pids[int(rng.integers(len(pids)))];eligible=[r for r in eligible if r['problem_id']==pid];prob/=len(pids)
        strata=[]
        for r in eligible:
            begin=r['first_anchor'];end=begin+r['count']
            for name,lo,hi in (('early',0,50),('middle',50,300),('late',300,501)):
                a=max(begin,lo);b=min(end,hi)
                if b>a:strata.append((r,name,a,b))
        if stage_balance:
            stages=sorted({s[1] for s in strata});chosen=stages[int(rng.integers(len(stages)))];strata=[s for s in strata if s[1]==chosen];prob/=len(stages)
        weights=np.array([s[3]-s[2] for s in strata],float);total=weights.sum();index=int(rng.choice(len(strata),p=weights/total));r,name,a,b=strata[index];anchor=int(rng.integers(a,b));prob/=total
        yield dict(problem_id=r['problem_id'],family=r['family'],partition='train',k=r['k'],h=r['h'],anchor=anchor,target=anchor+r['h'],stage=name,probability=float(prob),mode=mode)


class ProductionWindowLoader:
    """Bounded mmap cache and native-grid feature windows; addresses are explicit."""
    def __init__(self,root,entries,profile='no_fem_v1',cache_size=1):
        from collections import OrderedDict
        if profile not in PROFILES or not 1<=cache_size<=2:raise ValueError('Bounded profile/cache required')
        self.root=Path(root);self.entries={e['problem_id']:e for e in entries};self.profile=profile;self.cache_size=cache_size;self.cache=OrderedDict()
    def load(self,address):
        pid=address['problem_id'];entry=self.entries[pid]
        if address['partition']!=entry['partition']:raise ValueError('Partition mismatch')
        if pid not in self.cache:
            self.cache[pid]=Trajectory(self.root/'samples'/pid)
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        self.cache.move_to_end(pid);tr=self.cache[pid];t=address['anchor'];k=address['k'];h=address['h']
        if k<1 or h<1 or t<k-1 or t+h>=tr.metadata['state_count']:raise ValueError('Window timing')
        fs=[features(tr.problem,tr.arrays['x'][s],s,self.profile,tr.state(s) if PROFILES[self.profile]['fresh_fem'] else None) for s in range(t-k+1,t+1)]
        return dict(element=np.stack([f['element'] for f in fs]),node=np.stack([f['node'] for f in fs]),target_x=np.array(tr.arrays['x'][t+h]).reshape(tr.problem.nely,tr.problem.nelx),address=address)
