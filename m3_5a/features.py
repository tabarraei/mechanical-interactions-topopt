"""Versioned elemental/nodal features, with no future or mechanics calls."""
import numpy as np
from .io import check,canonical
STATIC_ELEMENT=['center_x/width','center_y/height','domain_mask','volume_fraction','move','x_min','x_max','dx','dy','width/48','height/48','log1p(force_scale)']
STATIC_NODE=['force_x/force_scale','force_y/force_scale','fixed_x','fixed_y']
DESCRIPTORS=['width','height','dx','dy','volume_fraction','move','x_min','x_max','filter_radius','penal','maxiter','tolerance','force_scale']
def schema(mode):
 check(mode in ('A','B'),'Feature mode')
 return dict(version='m3.5a.features.'+mode+'.v1',mode=mode,element_history=['x','previous_delta/move','iteration/maxiter']+(['rho','g_rho/F^2/rms','g_x/F^2/rms','compliance/F^2/rms'] if mode=='B' else []),node_history=['u_x/F/rms','u_y/F/rms'] if mode=='B' else [],static_element=STATIC_ELEMENT,static_node=STATIC_NODE,descriptor_order=DESCRIPTORS,fresh_FEM_at_inference=mode=='B',mechanical_history_schedule='B requires evaluated mechanical fields at each real history state; A has no mechanical input',target='design_delta_x',layout='K,C,H,W elements; K,C,H+1,W+1 nodes; C-order; increasing row y',padding='zero on left; index-1; validity and causal masks; no repeated initial state',forbidden=['future_x','future_rho','future_gradients','outgoing_delta','outgoing_change','selected_multiplier','termination_reason','terminal_compliance','final_state_count','reference_label'])
def history_indices(t,k):
 check(t>=0 and k>=1,'History request');n=min(t+1,k);return np.array([-1]*(k-n)+list(range(t-n+1,t+1)),dtype=np.int64)
def extract(release,pid,t,mode,k,normalization=None):
 sc=schema(mode);a=release.open(pid);spec=release.specs[pid];p=spec['mechanical'];H,W=p['nely'],p['nelx'];N=(H+1)*(W+1);F=spec['force_scale'];o=spec['optimizer'];idx=history_indices(t,k);mask=idx>=0
 if normalization is not None:normalization.validate(release)
 scale=normalization.scales if normalization is not None else dict(u=1.,g_rho=1.,g_x=1.,compliance=1.)
 el=np.zeros((k,len(sc['element_history']),H,W),np.float32);node=np.zeros((k,len(sc['node_history']),H+1,W+1),np.float32)
 for slot,s in enumerate(idx):
  if s<0:continue
  check(s<=t,'Future history state');x=a['x'][s].reshape(H,W);prev=np.zeros((H,W)) if s==0 else (a['x'][s]-a['x'][s-1]).reshape(H,W)
  vals=[x,prev/o['move'],np.full((H,W),s/o['maxiter'])]
  if mode=='B':
   vals += [a['rho'][s].reshape(H,W),a['g_rho'][s].reshape(H,W)/F**2/scale['g_rho'],a['g_x'][s].reshape(H,W)/F**2/scale['g_x'],np.full((H,W),a['compliance'][s]/F**2/scale['compliance'])]
   u=a['u'][s].reshape(N,2).T.reshape(2,H+1,W+1)/F/scale['u'];node[slot]=u
  el[slot]=np.asarray(vals)
 centers=np.asarray(p['element_centers']).reshape(H,W,2);xy=centers/np.array([p['width'],p['height']]);constant=lambda v:np.full((H,W),v)
 static_el=np.stack([xy[:,:,0],xy[:,:,1],constant(1),constant(p['volume_fraction']),constant(o['move']),constant(o['x_min']),constant(o['x_max']),constant(p['dx']),constant(p['dy']),constant(p['width']/48),constant(p['height']/48),constant(np.log1p(F))]).astype(np.float32)
 forces=np.asarray(p['force_vector']);check(forces.shape==(2*N,) and np.linalg.norm(forces)==F,'Load/force association')
 rebuilt=np.zeros_like(forces)
 for l in p['loads']:rebuilt[2*l['node_id']:2*l['node_id']+2]+=[l['fx'],l['fy']]
 check(np.array_equal(rebuilt,forces),'Nodal load definition')
 supports=np.zeros(2*N);supports[p['fixed_dofs']]=1
 static_node=np.concatenate([forces.reshape(N,2).T/F,supports.reshape(N,2).T],axis=0).reshape(4,H+1,W+1).astype(np.float32)
 desc=[p['width'],p['height'],p['dx'],p['dy'],p['volume_fraction'],o['move'],o['x_min'],o['x_max'],p['filter']['radius'],p['simp']['penal'],o['maxiter'],o['tolerance'],F]
 result=dict(element_history=el,node_history=node,static_element=static_el,static_node=static_node,descriptors=np.asarray(desc,np.float32),history_indices=idx,history_valid=mask,causal_attention=np.tril(np.ones((k,k),bool))&mask[:,None]&mask[None,:],element_valid=np.ones((H,W),bool),node_valid=np.ones((H+1,W+1),bool),schema=sc['version'],mode=mode,mesh=[W,H],normalization_id=normalization.identifier if normalization else 'analytical-only-v1',problem_id=pid,iteration=int(t))
 check(all(np.isfinite(v).all() for v in result.values() if isinstance(v,np.ndarray)),'Nonfinite feature')
 return result

def collate(samples):
 """Pad space only; element and node masks remain distinct. No mesh resampling."""
 check(bool(samples),'Empty batch');keys=[(s['proof']['problem_id'],s['proof']['transition']) for s in samples];check(len(keys)==len(set(keys)),'Duplicate batch examples');first=samples[0]['inputs'];check(all(s['inputs']['schema']==first['schema'] and s['inputs']['normalization_id']==first['normalization_id'] and s['inputs']['element_history'].shape[:2]==first['element_history'].shape[:2] for s in samples),'Mixed feature schema/history/normalization')
 B=len(samples);H=max(s['target'].shape[-2] for s in samples);W=max(s['target'].shape[-1] for s in samples);out={}
 for key,node,temporal in [('element_history',False,True),('node_history',True,True),('static_element',False,False),('static_node',True,False)]:
  head=first[key].shape[:-2];dst=np.zeros((B,*head,H+int(node),W+int(node)),np.float32)
  for b,s in enumerate(samples):
   x=s['inputs'][key];dst[b,...,:x.shape[-2],:x.shape[-1]]=x
  out[key]=dst
 out['target']=np.zeros((B,1,H,W),np.float32);out['element_valid']=np.zeros((B,H,W),bool);out['node_valid']=np.zeros((B,H+1,W+1),bool)
 for b,s in enumerate(samples):
  h,w=s['target'].shape[-2:];out['target'][b,:,:h,:w]=s['target'];out['element_valid'][b,:h,:w]=True;out['node_valid'][b,:h+1,:w+1]=True
 for key in ('history_valid','causal_attention','history_indices','descriptors'):out[key]=np.stack([s['inputs'][key] for s in samples])
 out['mesh']=[s['inputs']['mesh'] for s in samples];out['proof']=[s['proof'] for s in samples]
 return out

def crop_predictions(predictions,batch):
 return [predictions[b,...,:h,:w] for b,(w,h) in enumerate(batch['mesh'])]
