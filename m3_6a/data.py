"""Verified original-only causal multi-horizon interface. No normalization fit."""
from pathlib import Path
import numpy as np
from m3_5a.loader import Release
from m3_5a.normalization import Normalization
from m3_5a.io import ROOT,canonical,check,digest,read
from m3_5v.visualize import INDEX,INDEX_SHA
VIS=ROOT/'m3_6a/visualizations/20261015_complete_v1'
NORMALIZATION=ROOT/'m3_5a/results/20261015_initial_pilot/normalization.json'
GLOBAL_NAMES=['width/48','height/48','dx/48','dy/48','volume_target','E0','nu','thickness','penal','emin','filter_length_x/48','filter_length_y/48','OC_move','x_min','x_max','log1p_force_scale','log1p_lambda_upper','log10_OC_width','log10_stop_tolerance']

def mechanics_key(spec):
 m=dict(spec['mechanical']);m.pop('seed',None)
 return canonical(dict(mechanical=m,initial_x=spec['initial_x'],optimizer=spec['optimizer']))

class AcceptedData:
 def __init__(self,visualizations=VIS):
  self.release=Release(INDEX,INDEX_SHA);self.visualizations=Path(visualizations)
  from .visualization import verify_package
  verify_package(self.release,self.visualizations)
  self.entries=self.release.entries;self.by_id=self.release.by_id
  seen=set();families={}
  for e in self.entries:
   spec=self.release.specs[e['problem_id']];key=mechanics_key(spec);check(key not in seen,'Duplicate mechanically equivalent training history');seen.add(key)
   families.setdefault(spec['related_group'],set()).add(e['partition']);check(len(families[spec['related_group']])==1,'Family partition leakage')
  self.normalization=Normalization.load(NORMALIZATION,self.release)
 def close(self):self.release.close()
 def example(self,pid,t,k=3,h=1,mode='A'):
  check(pid in self.by_id and self.by_id[pid]['partition']=='train','Only accepted training histories; no historical/expansion admission')
  check(type(t) is int and type(k) is int and k>=1 and h in (1,5,10) and mode in ('A','B'),'History/horizon/mode')
  e=self.by_id[pid];check(0<=t and t+h<e['states'],'Horizon crosses termination; no interpolation')
  a=self.release.open(pid);s=self.release.specs[pid];m=s['mechanical'];o=s['optimizer'];H,W=m['nely'],m['nelx'];idx=list(range(max(0,t-k+1),t+1));K=len(idx)
  check(len(m['connectivity'])==H*W,'Only verified full rectangular accepted geometry supported by this source adapter')
  x=np.array(a['x'][idx],dtype=np.float32).reshape(K,1,H,W)
  fields=[x]
  if mode=='B':
   scales=self.normalization.scales;F=s['force_scale']
   fields.extend([np.array(a['rho'][idx],np.float32).reshape(K,1,H,W),np.array(a['g_rho'][idx]/F**2/scales['g_rho'],np.float32).reshape(K,1,H,W),np.array(a['g_x'][idx]/F**2/scales['g_x'],np.float32).reshape(K,1,H,W),np.broadcast_to(np.asarray(a['compliance'][idx]/F**2/scales['compliance'],np.float32)[:,None,None,None],(K,1,H,W)).copy()])
   u=np.array(a['u'][idx]/F/scales['u'],np.float32).reshape(K,H+1,W+1,2).transpose(0,3,1,2)
  else:u=np.zeros((K,0,H+1,W+1),np.float32)
  f=np.array(m['force_vector'],np.float32).reshape(H+1,W+1,2).transpose(2,0,1);fixed=np.zeros_like(f);flat=fixed.transpose(1,2,0).copy().reshape(-1);flat[m['fixed_dofs']]=1;fixed=flat.reshape(H+1,W+1,2).transpose(2,0,1)
  node=np.concatenate([f,fixed],axis=0)
  # Preserve exact raw nodal force definition; no cell-load reinterpretation here.
  coords=np.asarray(m['element_centers'],np.float32).reshape(H,W,2).transpose(2,0,1);nodes=np.asarray(m['node_coordinates'],np.float32).reshape(H+1,W+1,2).transpose(2,0,1)
  g=[m['width']/48,m['height']/48,m['dx']/48,m['dy']/48,m['volume_fraction'],m['material']['E0'],m['material']['nu'],m['material']['thickness'],m['simp']['penal'],m['simp']['emin'],m['filter']['radius']*m['dx']/48,m['filter']['radius']*m['dy']/48,o['move'],o['x_min'],o['x_max'],np.log1p(s['force_scale']),np.log1p(m['optimizer']['lambda_upper']),np.log10(o['width']),np.log10(o['tolerance'])]
  inputs=dict(element_history=np.concatenate(fields,axis=1),node_history=u,node_fields=node,element_coords=coords,node_coords=nodes,domain_mask=np.ones((H,W),bool),node_mask=np.ones((H+1,W+1),bool),area=np.full((H,W),m['dx']*m['dy'],np.float32),globals=np.asarray(g,np.float32),extent=np.array([m['width'],m['height']],np.float32),force_scale=np.float32(s['force_scale']),iterations=np.asarray(idx,np.float32),cap=np.float32(o['maxiter']),horizon=np.float32(h),mode=mode)
  target=np.array(a['x'][t+h]-a['x'][t],np.float32).reshape(1,H,W)
  check(all(np.isfinite(v).all() for v in inputs.values() if isinstance(v,np.ndarray)) and np.isfinite(target).all(),'Nonfinite input/target')
  return dict(inputs=inputs,target=target,proof=dict(problem_id=pid,partition=e['partition'],family=s['related_group'],mesh=[W,H],anchor=t,history=idx,target_iteration=t+h,horizon=h,source_id=e['source_id'],target='design_delta_x',diagnostic_scope='training-only architecture diagnostic',normalization='analytical A' if mode=='A' else digest(NORMALIZATION)))

def collate(samples):
 import torch
 check(samples,'Empty batch');modes={s['inputs']['mode'] for s in samples};check(len(modes)==1,'Mixed ModeA/ModeB forbidden')
 keys=[(s['proof']['problem_id'],s['proof']['anchor'],s['proof']['horizon']) for s in samples];check(len(keys)==len(set(keys)),'Duplicate batch examples')
 B=len(samples);K=max(len(s['inputs']['iterations']) for s in samples);H=max(s['target'].shape[-2] for s in samples);W=max(s['target'].shape[-1] for s in samples);C=samples[0]['inputs']['element_history'].shape[1];U=samples[0]['inputs']['node_history'].shape[1]
 out=dict(element_history=np.zeros((B,K,C,H,W),np.float32),node_history=np.zeros((B,K,U,H+1,W+1),np.float32),node_fields=np.zeros((B,4,H+1,W+1),np.float32),element_coords=np.zeros((B,2,H,W),np.float32),node_coords=np.zeros((B,2,H+1,W+1),np.float32),domain_mask=np.zeros((B,H,W),bool),node_mask=np.zeros((B,H+1,W+1),bool),area=np.zeros((B,H,W),np.float32),history_valid=np.zeros((B,K),bool),iterations=np.zeros((B,K),np.float32),target=np.zeros((B,1,H,W),np.float32))
 for b,s in enumerate(samples):
  x=s['inputs'];kk=len(x['iterations']);hh,ww=s['target'].shape[-2:]
  for name in ('element_history','node_history'):
   z=x[name];out[name][b,:kk,:,:z.shape[-2],:z.shape[-1]]=z
  for name in ('node_fields','element_coords','node_coords'):
   z=x[name];out[name][b,:,:z.shape[-2],:z.shape[-1]]=z
  for name in ('domain_mask','node_mask','area'):
   z=x[name];out[name][b,:z.shape[-2],:z.shape[-1]]=z
  out['history_valid'][b,:kk]=True;out['iterations'][b,:kk]=x['iterations'];out['target'][b,:,:hh,:ww]=s['target']
 for name in ('globals','extent','force_scale','cap','horizon'):out[name]=np.stack([s['inputs'][name] for s in samples])
 out={k:torch.from_numpy(v) for k,v in out.items()};out['mode']=next(iter(modes));out['proof']=[s['proof'] for s in samples];return out
