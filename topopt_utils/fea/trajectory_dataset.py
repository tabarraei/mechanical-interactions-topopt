"""M3.3B problem sampling, faithful features, splits and mmap window loading."""
from collections import OrderedDict
from pathlib import Path
import hashlib,json
import numpy as np
from .mechanical_problem import MechanicalProblem,DensityMap,array_digest
from .consistent_trajectory import evaluate_state
from .robust_oc import update,OPTIONS

SCHEMA='m3.3b.dataset.v1'

def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_problem(config,seed,controls):
    """Exact nodal coordinates only: never round a physical point into a node."""
    allowed={'label','mesh','domain','support','fixed_dofs','loads','volume_fraction','filter_radius','penal','initial','related_group'}
    if set(config)-allowed:raise ValueError('Unsupported problem settings')
    nx,ny=config['mesh'];width,height=config['domain']
    if any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in (nx,ny)) or nx*ny>1152 or not np.isfinite([width,height]).all() or min(width,height)<=0:raise ValueError('Unsupported mesh/domain')
    dx,dy=width/nx,height/ny
    if config.get('penal',3.)!=3.:raise ValueError('Pilot retains p=3')
    support=config['support'];node=lambda i,j:j*(nx+1)+i
    if support=='left_clamp':fixed=[2*node(0,j)+a for j in range(ny+1) for a in (0,1)]
    elif support=='right_clamp':fixed=[2*node(nx,j)+a for j in range(ny+1) for a in (0,1)]
    elif support=='bottom_clamp':fixed=[2*node(i,0)+a for i in range(nx+1) for a in (0,1)]
    elif support=='pin_roller':fixed=[0,1,2*nx+1]
    elif support=='custom':fixed=config['fixed_dofs']
    else:raise ValueError('Unknown support pattern')
    loads=[]
    for item in config['loads']:
        position=np.asarray(item['position'],float);direction=np.asarray(item['direction'],float);magnitude=item['magnitude']
        if position.shape!=(2,) or direction.shape!=(2,) or not np.isfinite(position).all() or not np.isfinite(direction).all() or not np.isfinite(magnitude) or magnitude<=0 or np.linalg.norm(direction)==0:
            raise ValueError('Invalid physical load')
        ij=position/np.array([dx,dy]);rounded=np.rint(ij).astype(int)
        if np.any(rounded<0) or rounded[0]>nx or rounded[1]>ny or not np.allclose(ij,rounded,rtol=0,atol=1e-10):
            raise ValueError('Physical point load is not on a supported mesh node; no interpolation or snapping')
        force=direction/np.linalg.norm(direction)*magnitude
        loads.append((node(*rounded),float(force[0]),float(force[1])))
    p=MechanicalProblem(nx,ny,float(width),float(height),tuple(sorted(fixed)),tuple(loads),volume_fraction=config['volume_fraction'],filter_radius=config.get('filter_radius',2.),
        maxiter=controls['maxiter'],tolerance=controls['tolerance'],seed=int(seed))
    initial=config.get('initial',{'kind':'uniform'});slack=initial.get('slack',0.)
    if not np.isfinite(slack) or not 0<=slack<p.volume_fraction-.001:raise ValueError('Invalid explicit initial slack')
    x=np.full(p.n,p.volume_fraction-slack)
    if initial['kind']=='perturbed':
        rng=np.random.default_rng(seed);noise=rng.uniform(-1,1,p.n);noise-=noise.mean();amplitude=initial.get('amplitude',.02)
        if not 0<amplitude<min(p.volume_fraction-.001,1-p.volume_fraction):raise ValueError('Perturbation crosses design bounds')
        noise*=amplitude/max(abs(noise));x=x+noise-1e-5
    elif initial['kind']!='uniform':raise ValueError('Unknown initial design')
    if not np.isfinite(x).all() or min(x)<.001 or max(x)>1 or x.mean()>p.volume_fraction:raise ValueError('Invalid or infeasible initial design')
    option=controls['option']
    if option not in OPTIONS:raise ValueError('Unknown optimizer option')
    body=dict(schema_version='m3.3b.problem.v1',mechanical=p.to_dict(),initial_x=x.tolist(),initial_digest=array_digest(x),
        optimizer=dict(option=option,version='m3.3a.oc.v1',**OPTIONS[option],maxiter=p.maxiter,tolerance=p.tolerance,move=.1,x_min=.001,x_max=1.,volume_variable='x'),
        force_scale=float(np.linalg.norm(p.forces)),filter_physical_semi_axes=[p.filter_radius*dx,p.filter_radius*dy])
    spec=dict(body,problem_id='m3_3b_'+canonical_hash(body)[:24],label=config['label'],related_group=config['related_group'],support_pattern=support,configuration=config)
    return p,x,spec


def load_problem(spec):
    body={k:v for k,v in spec.items() if k not in ('problem_id','label','related_group','support_pattern','configuration')}
    if spec['schema_version']!='m3.3b.problem.v1' or 'm3_3b_'+canonical_hash(body)[:24]!=spec['problem_id']:raise ValueError('Problem identity inconsistency')
    p=MechanicalProblem.from_dict(spec['mechanical']);x=np.asarray(spec['initial_x'],float)
    if x.shape!=(p.n,) or array_digest(x)!=spec['initial_digest'] or not np.isfinite(x).all() or x.mean()>p.volume_fraction or min(x)<.001 or max(x)>1:raise ValueError('Initial design inconsistency')
    if spec['force_scale']!=float(np.linalg.norm(p.forces)):raise ValueError('Force normalization inconsistency')
    return p,x


def split_problems(specs,seed,ood_supports,ratios=(.6,.2,.2),_validation_pass=False):
    """Union declared relatives and conservative coarse mechanics, then split groups.

    Scale/resolution/initial-condition variants, mirror-clamp variants and nearby
    locations/directions in a coarse mechanics bucket remain together.
    """
    if len(ratios)!=3 or min(ratios)<=0 or not np.isclose(sum(ratios),1):raise ValueError('Invalid split ratios')
    ids=[s['problem_id'] for s in specs]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate complete problem IDs')
    parent=list(range(len(specs)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[root(j)]=root(i)
    keys=[]
    for s in specs:
        p,_=load_problem(s);support=s['support_pattern'];loads=[]
        for n,fx,fy in p.loads:
            xy=p.nodes[n]/[p.width,p.height];direction=np.array([fx,fy]);direction/=np.linalg.norm(direction)
            if support=='right_clamp':xy[0]=1-xy[0];direction[0]*=-1
            first=np.flatnonzero(abs(direction)>1e-14)[0]
            if direction[first]<0:direction*=-1
            angle=np.arctan2(direction[1],direction[0]);loads.append([np.rint(xy*4).astype(int).tolist(),int(np.rint(angle/(np.pi/4)))])
        pattern='horizontal_clamp' if support in ('left_clamp','right_clamp') else support
        custom=s['mechanical']['fixed_dofs'] if support=='custom' else None
        keys.append(canonical_hash(dict(support=pattern,custom=custom,aspect_bin=int(round(p.width/p.height)),loads=loads)))
    for i,s in enumerate(specs):
        for j in range(i):
            if s['related_group']==specs[j]['related_group'] or keys[i]==keys[j]:union(i,j)
    groups={}
    for i,s in enumerate(specs):groups.setdefault(root(i),[]).append(s)
    components=list(groups.values());ood=[];ind=[]
    for group in components:
        if any(s['support_pattern'] in ood_supports for s in group):ood.append(group)
        else:ind.append(group)
    if len(ind)<3 or not ood:raise ValueError('Need >=3 independent ID groups and >=1 OOD group')
    rng=np.random.default_rng(seed);order=rng.permutation(len(ind));n=len(ind);ntrain=max(1,min(n-2,int(np.floor(ratios[0]*n))));nval=max(1,min(n-ntrain-1,int(np.floor(ratios[1]*n))))
    assigned={};group_manifest=[]
    for partition,chosen in [('train',[ind[i] for i in order[:ntrain]]),('validation',[ind[i] for i in order[ntrain:ntrain+nval]]),('test_id',[ind[i] for i in order[ntrain+nval:]]),('test_ood',ood)]:
        for group in chosen:
            group_id='group_'+canonical_hash(sorted(s['problem_id'] for s in group))[:16]
            group_manifest.append(dict(group_id=group_id,partition=partition,problem_ids=sorted(s['problem_id'] for s in group),declared_families=sorted({s['related_group'] for s in group})))
            for s in group:assigned[s['problem_id']]=dict(partition=partition,group_id=group_id)
    result=dict(schema_version='m3.3b.splits.v1',seed=seed,ratios=list(ratios),ood_supports=ood_supports,assignments=assigned,groups=group_manifest,
        ID_definition='Membership in declared sampler support domain; tiny demo is not an empirical distribution/generalization test',grouping='declared family union coarse normalized mechanics buckets; magnitude, seed and resolution variants grouped, horizontal mirror clamps canonicalized')
    if not _validation_pass:validate_splits(specs,result,check_grouping=False)
    return result


def validate_splits(specs,splits,check_grouping=True):
    ids={s['problem_id'] for s in specs}
    if set(splits['assignments'])!=ids:raise ValueError('Split coverage mismatch')
    group_partition={};family_partition={};seen=set()
    # Reconstruct the deterministic grouping itself to detect a manually
    # corrupted assignment that preserves labels but separates coarse relatives.
    if check_grouping:
        expected=split_problems(specs,splits['seed'],splits['ood_supports'],tuple(splits['ratios']),_validation_pass=True)
        if expected['assignments']!=splits['assignments'] or expected['groups']!=splits['groups']:raise ValueError('Split grouping or deterministic assignment changed')
    for group in splits['groups']:
        for pid in group['problem_ids']:
            if pid in seen or splits['assignments'][pid]!=dict(partition=group['partition'],group_id=group['group_id']):raise ValueError('Group leakage')
            seen.add(pid)
    if seen!=ids:raise ValueError('Group coverage mismatch')
    for s in specs:
        a=splits['assignments'][s['problem_id']];partition=a['partition'];group=a['group_id']
        if partition not in ('train','validation','test_id','test_ood'):raise ValueError('Unknown partition')
        if group in group_partition and group_partition[group]!=partition:raise ValueError('Group leakage')
        family=s['related_group']
        if family in family_partition and family_partition[family]!=partition:raise ValueError('Related family leakage')
        group_partition[group]=partition;family_partition[family]=partition
        if s['support_pattern'] in splits['ood_supports'] and partition!='test_ood':raise ValueError('OOD support leaks into ID')
    return True


ELEMENT_BASE=['x','rho','center_x','center_y','center_x_over_width','center_y_over_height','volume_fraction','iteration_over_budget','force_scale']
NODE_BASE=['force_x_over_scale','force_y_over_scale','fixed_ux','fixed_uy','node_x','node_y','node_x_over_width','node_y_over_height']
PROFILES={
 'no_fem_v1':dict(element=ELEMENT_BASE,node=NODE_BASE,fresh_fem=False),
 'fresh_gradient_v1':dict(element=ELEMENT_BASE+['g_x_over_force_scale_squared'],node=NODE_BASE,fresh_fem=True),
 'fresh_full_v1':dict(element=ELEMENT_BASE+['g_x_over_force_scale_squared','g_rho_over_force_scale_squared','elastic_energy_over_force_scale_squared'],node=NODE_BASE+['u_x_over_force_scale','u_y_over_force_scale'],fresh_fem=True),
}

def feature_schema():
    return dict(schema_version='m3.3b.features.v1',profiles=PROFILES,element_shape='(history,channels,nely,nelx)',node_shape='(history,channels,nely+1,nelx+1)',
        ordering='C-order elements and row-major nodes; components x,y; physical coordinates increase with array row; origin lower',
        units=dict(x='dimensionless',rho='dimensionless',g='compliance per dimensionless density; divided by known force_scale**2',u='domain displacement; divided by known force_scale',elastic_energy='0.5*s(rho)*u_e.T KE0 u_e / force_scale**2; sums to C/(2*force_scale**2)',force='exact nodal forces divided by L2 norm of complete force vector',coordinates='physical domain lengths; separately labeled normalized coordinates',iteration='absolute iteration/maxiter, never realized final trajectory length'),
        feature_cost=dict(no_fem_v1='x, local density filter, geometry, exact loads/supports and counters; zero FEM solves',fresh_gradient_v1='current g_x requires fresh FEM at current x',fresh_full_v1='current gradients, displacement and strain energy require fresh FEM at current x'),
        normalization='No empirical train/validation statistics or future-state scales; actual force magnitude retained as an element constant. Normalization changes model inputs only, never forces or optimizer gradients.',
        force_fidelity='Forces/support masks stay on native nodal grid; no point-load smearing into element channels. Element and node grids are returned separately.')


def features(p,x,iteration,profile,state=None):
    if profile not in PROFILES:raise ValueError('Unknown profile')
    if np.shape(x)!=(p.n,) or not np.isfinite(x).all() or np.min(x)<.001 or np.max(x)>1 or not 0<=iteration<=p.maxiter:raise ValueError('Invalid feature state')
    definition=PROFILES[profile];scale=float(np.linalg.norm(p.forces));shape=(p.nely,p.nelx);node_shape=(p.nely+1,p.nelx+1)
    rho=DensityMap(p).forward(x)
    if definition['fresh_fem']:
        if state is None or not np.array_equal(state['x'],x):raise ValueError('Fresh profile requires mechanics evaluated at exactly this x')
        if state.get('mechanical_id')!=p.problem_id:raise ValueError('Fresh mechanical problem mismatch')
    values=dict(x=x,rho=rho,center_x=p.centers[:,0],center_y=p.centers[:,1],center_x_over_width=p.centers[:,0]/p.width,center_y_over_height=p.centers[:,1]/p.height,
        volume_fraction=np.full(p.n,p.volume_fraction),iteration_over_budget=np.full(p.n,iteration/p.maxiter),force_scale=np.full(p.n,scale))
    bc=np.zeros(p.ndof);bc[list(p.fixed_dofs)]=1
    nodes=dict(force_x_over_scale=p.forces[0::2]/scale,force_y_over_scale=p.forces[1::2]/scale,fixed_ux=bc[0::2],fixed_uy=bc[1::2],node_x=p.nodes[:,0],node_y=p.nodes[:,1],node_x_over_width=p.nodes[:,0]/p.width,node_y_over_height=p.nodes[:,1]/p.height)
    if definition['fresh_fem']:
        values['g_x_over_force_scale_squared']=state['g_x']/scale**2
        if profile=='fresh_full_v1':
            from .solver import FeaSolver
            ue=state['u'][p.element_dofs];ke=FeaSolver().get_KE(p.dx,p.dy)
            values['g_rho_over_force_scale_squared']=state['g_rho']/scale**2
            from .q4 import __simp_stiffness__
            values['elastic_energy_over_force_scale_squared']=.5*__simp_stiffness__(rho,p.penal)*np.einsum('ei,ij,ej->e',ue,ke,ue)/scale**2
            nodes.update(u_x_over_force_scale=state['u'][0::2]/scale,u_y_over_force_scale=state['u'][1::2]/scale)
    element=np.stack([np.asarray(values[n]).reshape(shape) for n in definition['element']]);nodal=np.stack([np.asarray(nodes[n]).reshape(node_shape) for n in definition['node']])
    if not np.isfinite(element).all() or not np.isfinite(nodal).all():raise ValueError('Nonfinite channels')
    return dict(element=element,node=nodal,element_channel_names=definition['element'],node_channel_names=definition['node'],profile=profile,fresh_fem_required=definition['fresh_fem'])


def evaluate_prediction(p,x,correct=False,option='feasible_upper_v1'):
    if np.shape(x)!=(p.n,) or not np.isfinite(x).all() or min(x)<.001 or max(x)>1:raise ValueError('Predicted design outside supported box')
    state=evaluate_state(p,np.asarray(x));state['mechanical_id']=p.problem_id
    result=dict(state=state,equilibrium_solves=1,volume_excess=float(np.mean(x)-p.volume_fraction))
    if correct:result['corrected_x'],result['oc_diagnostics']=update(x,state['g_x'],p.volume_fraction,option)
    return result


class Trajectory:
    """Verified read-only NumPy memory maps; no trajectory-wide array copies."""
    def __init__(self,folder,verify=True):
        self.folder=Path(folder);self.spec=json.loads((self.folder/'problem.json').read_text());self.problem,self.initial=load_problem(self.spec)
        self.metadata=json.loads((self.folder/'metadata.json').read_text())
        if self.metadata['schema_version']!='m3.3b.trajectory.v1' or self.metadata['problem_id']!=self.spec['problem_id']:raise ValueError('Trajectory schema/problem mismatch')
        self.arrays={}
        for key,info in self.metadata['arrays'].items():
            file=self.folder/info['file']
            if verify and file_hash(file)!=info['sha256']:raise ValueError('Array checksum mismatch: '+key)
            a=np.load(file,mmap_mode='r',allow_pickle=False)
            if list(a.shape)!=info['shape'] or str(a.dtype)!=info['dtype']:raise ValueError('Array shape/dtype mismatch')
            self.arrays[key]=a
        if self.metadata['optimizer_option']!=self.spec['optimizer']['option'] or self.metadata['mechanical_id']!=self.problem.problem_id:raise ValueError('Effective optimizer/mechanics mismatch')
        if verify and file_hash(self.folder/'problem.json')!=self.metadata['problem_sha256']:raise ValueError('Problem file checksum mismatch')
        self.transitions=json.loads((self.folder/'transitions.json').read_text())
        if verify and file_hash(self.folder/'transitions.json')!=self.metadata['transitions_sha256']:raise ValueError('Transition integrity mismatch')
        if len(self.arrays['x'])!=self.metadata['state_count'] or len(self.transitions)!=self.metadata['completed_updates']:raise ValueError('State/transition count mismatch')
    def state(self,t):
        if not 0<=t<self.metadata['state_count']:raise IndexError(t)
        return dict({k:a[t] for k,a in self.arrays.items() if k!='update_change'},mechanical_id=self.problem.problem_id)
    def transition(self,t):
        if not 0<=t<self.metadata['completed_updates']:raise IndexError(t)
        x=self.arrays['x'];return dict(x_t=x[t],x_next=x[t+1],delta=x[t+1]-x[t],**self.transitions[t])


def window_index(entries,splits,history_lengths=(1,3),horizons=(1,5,10)):
    rows=[]
    for e in entries:
        assignment=splits['assignments'][e['problem_id']]
        for k in history_lengths:
            for h in horizons:
                if not isinstance(k,int) or k<1 or not isinstance(h,int) or h<1:raise ValueError('Positive integer history/horizon required')
                for t in range(k-1,e['state_count']-h):
                    rows.append(dict(problem_id=e['problem_id'],partition=assignment['partition'],group_id=assignment['group_id'],history_length=k,anchor=t,horizon=h,target=t+h))
    return rows


class WindowDataset:
    """NumPy samples with a bounded per-process mmap cache for future DataLoaders."""
    def __init__(self,generation,partition,profile='no_fem_v1',history_length=None,horizon=None,cache_size=2):
        self.generation=Path(generation);self.manifest=json.loads((self.generation/'manifest.json').read_text())
        if self.manifest['schema_version']!=SCHEMA:raise ValueError('Dataset schema mismatch')
        self.entries={e['problem_id']:e for e in self.manifest['samples']}
        for name in ('windows.json','splits.json','feature_schema.json'):
            if file_hash(self.generation/name)!=self.manifest['files'][name]:raise ValueError('Dataset index/schema checksum mismatch')
        rows=json.loads((self.generation/'windows.json').read_text());splits=json.loads((self.generation/'splits.json').read_text())
        for r in rows:
            if r['problem_id'] not in self.entries or r['target']!=r['anchor']+r['horizon'] or r['anchor']<r['history_length']-1 or r['target']>=self.entries[r['problem_id']]['state_count'] or r['partition']!=splits['assignments'][r['problem_id']]['partition']:raise ValueError('Invalid window or split leakage')
        self.rows=[r for r in rows if r['partition']==partition and (history_length is None or r['history_length']==history_length) and (horizon is None or r['horizon']==horizon)]
        if profile not in PROFILES or cache_size<1:raise ValueError('Invalid loader profile/cache')
        self.profile=profile;self.cache_size=cache_size;self.cache=OrderedDict()
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];pid=row['problem_id']
        if pid not in self.cache:
            folder=self.generation/self.entries[pid]['folder']
            if file_hash(folder/'metadata.json')!=self.entries[pid]['metadata_sha256']:raise ValueError('Trajectory metadata checksum mismatch')
            self.cache[pid]=Trajectory(folder)
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        self.cache.move_to_end(pid);trajectory=self.cache[pid];p=trajectory.problem;t=row['anchor'];k=row['history_length']
        inputs=[features(p,trajectory.arrays['x'][i],i,self.profile,trajectory.state(i) if PROFILES[self.profile]['fresh_fem'] else None) for i in range(t-k+1,t+1)]
        return dict(element=np.stack([s['element'] for s in inputs]),node=np.stack([s['node'] for s in inputs]),target_x=np.array(trajectory.arrays['x'][row['target']]).reshape(p.nely,p.nelx),
            actual_delta=np.array(trajectory.arrays['x'][row['target']]-trajectory.arrays['x'][t]).reshape(p.nely,p.nelx),
            index=row,feature_schema='m3.3b.features.v1',profile=self.profile,fresh_fem_required=PROFILES[self.profile]['fresh_fem'],
            target_termination=trajectory.metadata['termination_reason'] if row['target']==trajectory.metadata['completed_updates'] else 'intermediate')
    def __getstate__(self):return dict(self.__dict__,cache=OrderedDict())


def batch_indices(dataset,batch_size,seed=0):
    """Homogeneous mesh/history buckets; no hidden padding or interpolation."""
    if batch_size<1:raise ValueError('Positive batch size required')
    buckets={}
    for i,row in enumerate(dataset.rows):
        mesh=tuple(dataset.entries[row['problem_id']]['mesh']);key=(mesh,row['history_length'])
        buckets.setdefault(key,[]).append(i)
    rng=np.random.default_rng(seed);batches=[]
    for key in sorted(buckets):
        indices=np.asarray(buckets[key]);rng.shuffle(indices)
        batches.extend(indices[j:j+batch_size].tolist() for j in range(0,len(indices),batch_size))
    rng.shuffle(batches);return batches
