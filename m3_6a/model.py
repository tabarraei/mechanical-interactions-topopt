"""Two-scale coordinate-quadrature operator + per-anchor causal Transformer."""
import torch
from torch import nn

def corner_average(v):return .25*(v[...,:-1,:-1]+v[...,1:,:-1]+v[...,:-1,1:]+v[...,1:,1:])

def conservative_force_map(forces,mask):
 """Split each nodal point force over incident active cells; total force preserved."""
 count=torch.zeros_like(forces[:,:1]);m=mask[:,None].to(forces.dtype)
 for dy,dx in ((0,0),(0,1),(1,0),(1,1)):count[:,:,dy:dy+m.shape[-2],dx:dx+m.shape[-1]]+=m
 if torch.any((count==0)&(forces.abs().sum(1,keepdim=True)>0)):raise ValueError('Force on node outside active design domain')
 share=forces/count.clamp_min(1);result=torch.zeros_like(m).expand(-1,2,-1,-1).clone()
 for dy,dx in ((0,0),(0,1),(1,0),(1,1)):result+=share[:,:,dy:dy+m.shape[-2],dx:dx+m.shape[-1]]*m
 return result

class CoordinateKernel(nn.Module):
 def __init__(self,radius=None):
  super().__init__();self.radius=radius;self.net=nn.Sequential(nn.Linear(7,12),nn.Tanh(),nn.Linear(12,1),nn.Softplus())
 def weights(self,q,s):
  # q[B,Q,2], s[B,N,2], shared continuous kernel independent of grid size.
  d=q[:,:,None,:]-s[:,None,:,:];dist=(d.square().sum(-1)+1e-12).sqrt()
  z=torch.cat([d,dist[...,None],q[:,:,None,:].expand(-1,-1,s.shape[1],-1),s[:,None,:,:].expand(-1,q.shape[1],-1,-1)],-1)
  w=self.net(z).squeeze(-1)
  if self.radius is not None:w=w*(1-dist/self.radius).clamp_min(0)
  return w
 def forward(self,q,s,v,area):
  # Queries are chunked; no N×N element self-attention or dense operator.
  chunks=[];valid=[]
  for start in range(0,q.shape[1],64):
   w=self.weights(q[:,start:start+64],s)*area[:,None,:];mass=w.sum(-1,keepdim=True);chunks.append(w@v/mass.clamp_min(1e-12));valid.append(mass.squeeze(-1)>1e-12)
  return torch.cat(chunks,1),torch.cat(valid,1)

class MultiscaleTemporalOperator(nn.Module):
 def __init__(self,width=24,heads=4,mode='A',layers=1):
  super().__init__()
  if mode not in ('A','B') or width%heads:raise ValueError('Mode/head dimensions')
  self.mode=mode;self.width=width;self.scales=((4,2),(6,4))
  features=(1 if mode=='A' else 5)+4+(0 if mode=='A' else 2)+3
  self.lift=nn.Sequential(nn.Linear(features,width),nn.GELU(),nn.Linear(width,width));self.globals=nn.Linear(19,width)
  self.global_kernel=CoordinateKernel();self.local_kernel=CoordinateKernel(.3);self.node_kernel=CoordinateKernel();self.node_lift=nn.Linear(4,width)
  self.down=CoordinateKernel();self.up=CoordinateKernel();self.exchange=nn.Linear(2*width,width)
  self.time=nn.Linear(4,width);self.horizon=nn.Sequential(nn.Linear(2,width),nn.Tanh())
  layer=nn.TransformerEncoderLayer(width,heads,dim_feedforward=2*width,dropout=0.,batch_first=True,norm_first=True)
  self.temporal=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
  self.decode_global=CoordinateKernel();self.decode_local=CoordinateKernel(.35)
  self.head=nn.Sequential(nn.Linear(4*width,width),nn.GELU(),nn.Linear(width,1))
 def anchors(self,extent,nx,ny):
  x=(torch.arange(nx,device=extent.device,dtype=extent.dtype)+.5)/nx;y=(torch.arange(ny,device=extent.device,dtype=extent.dtype)+.5)/ny
  yy,xx=torch.meshgrid(y,x,indexing='ij');unit=torch.stack([xx.reshape(-1),yy.reshape(-1)],-1)
  return unit[None]*extent[:,None,:]
 def forward(self,b,return_details=False):
  if b['mode']!=self.mode:raise ValueError('Mode-specific model; no silent mixing')
  e=b['element_history'];B,K,C,H,W=e.shape;mask=b['domain_mask'];hv=b['history_valid'];device=e.device
  if not hv.any(1).all() or ((~hv[:,:-1])&hv[:,1:]).any():raise ValueError('Nonempty right-padded contiguous histories required')
  if C!=(1 if self.mode=='A' else 5) or b['node_fields'].shape!=(B,4,H+1,W+1):raise ValueError('Element/node schema')
  if b['globals'].shape!=(B,19) or b['element_coords'].shape!=(B,2,H,W) or b['node_coords'].shape!=(B,2,H+1,W+1) or b['area'].shape!=(B,H,W) or b['node_mask'].shape!=(B,H+1,W+1):raise ValueError('Geometry/global shape')
  if (b['cap']<=0).any() or (b['area'][mask]<=0).any():raise ValueError('Positive cap/quadrature area')
  if torch.any((b['iterations']<0)&hv) or torch.any((b['iterations']>b['cap'][:,None])&hv) or torch.any((b['iterations'][:,1:]<=b['iterations'][:,:-1])&hv[:,1:]):raise ValueError('Causal iteration ordering')
  if not torch.all((b['horizon']==1)|(b['horizon']==5)|(b['horizon']==10)):raise ValueError('Unsupported horizon')
  for key in ('element_history','node_history','node_fields','element_coords','node_coords','area','globals','extent','force_scale','iterations','cap','horizon'):
   v=b[key]
   if isinstance(v,torch.Tensor) and v.dtype.is_floating_point and not torch.isfinite(v).all():raise ValueError('Nonfinite '+key)
  if not mask.flatten(1).any(1).all() or (b['extent']<=0).any() or (b['force_scale']<=0).any():raise ValueError('Domain/extent/force scale')
  L=b['extent'].amax(1);extent=b['extent']/L[:,None];coords=(b['element_coords']/L[:,None,None,None]).flatten(2).transpose(1,2)
  nodes=(b['node_coords']/L[:,None,None,None]).flatten(2).transpose(1,2);area=(b['area']/L[:,None,None]**2*mask).flatten(1)
  nf=b['node_fields'];F=b['force_scale'][:,None,None,None];forces=nf[:,:2]/F
  if torch.any((~b['node_mask'])[:,None]&(nf.abs()>0)):raise ValueError('Condition on padded/inactive node')
  fm=conservative_force_map(forces,mask);bc=corner_average(nf[:,2:]);condition=torch.cat([fm,bc],1)*mask[:,None]
  inp=[e,condition[:,None].expand(-1,K,-1,-1,-1),coords.transpose(1,2).reshape(B,1,2,H,W).expand(-1,K,-1,-1,-1),mask[:,None,None].expand(-1,K,1,-1,-1).to(e.dtype)]
  if self.mode=='B':inp.append(corner_average(b['node_history']))
  z=torch.cat(inp,2).flatten(3).transpose(2,3);z=self.lift(z)+self.globals(b['globals'])[:,None,None,:];z=z*mask.flatten(1)[:,None,:,None]*hv[:,:,None,None]
  anchors=[self.anchors(extent,*s) for s in self.scales];repeated=lambda x:x[:,None].expand(-1,K,*x.shape[1:]).reshape(B*K,*x.shape[1:])
  cs=repeated(coords);vv=z.reshape(B*K,H*W,self.width);aa=repeated(area)
  encoded=[];valid=[];masses=[]
  for q,kernel in zip(anchors,(self.global_kernel,self.local_kernel)):
   a,good=kernel(repeated(q),cs,vv,aa)
   # Exact point force kernel sum at actual nodal coordinates; no area averaging.
   nw=self.node_kernel.weights(q,nodes)*b['node_mask'].flatten(1)[:,None,:]
   f=nw@forces.flatten(2).transpose(1,2)
   supports=nf[:,2:].flatten(2).transpose(1,2)
   bcmax=(nw[:,:,:,None]*supports[:,None,:,:]).amax(2)
   cond=self.node_lift(torch.cat([f,bcmax],-1))
   encoded.append(a.reshape(B,K,q.shape[1],self.width)+cond[:,None]);valid.append(good.reshape(B,K,q.shape[1]));masses.append(good.reshape(B,K,q.shape[1])[:,0].to(e.dtype))
  coarse,fine=encoded
  down,_=self.down(repeated(anchors[0]),repeated(anchors[1]),fine.reshape(B*K,-1,self.width),repeated(masses[1]))
  up,_=self.up(repeated(anchors[1]),repeated(anchors[0]),coarse.reshape(B*K,-1,self.width),repeated(masses[0]))
  coarse=self.exchange(torch.cat([coarse,down.reshape_as(coarse)],-1));fine=self.exchange(torch.cat([fine,up.reshape_as(fine)],-1))
  latent=torch.cat([coarse,fine],2);anchor_valid=torch.cat([valid[0],valid[1]],2);M=latent.shape[2]
  it=b['iterations']/b['cap'][:,None];elapsed=torch.cat([torch.zeros_like(it[:,:1]),it[:,1:]-it[:,:-1]],1)
  pos=torch.stack([it,elapsed,torch.sin(2*torch.pi*it),torch.cos(2*torch.pi*it)],-1)
  latent=latent+self.time(pos)[:,:,None,:]
  tokens=latent.permute(0,2,1,3).reshape(B*M,K,self.width)
  causal=torch.triu(torch.ones(K,K,dtype=torch.bool,device=device),diagonal=1);padding=(~hv)[:,None,:].expand(B,M,K).reshape(B*M,K)
  temporal=self.temporal(tokens,mask=causal,src_key_padding_mask=padding).reshape(B,M,K,self.width).permute(0,2,1,3)*hv[:,:,None,None]*anchor_valid[:,:,:,None]
  last=hv.sum(1)-1;which=torch.arange(B,device=device);now=temporal[which,last]
  c,_=self.decode_global(coords,anchors[0],now[:,:anchors[0].shape[1]],masses[0]);f,_=self.decode_local(coords,anchors[1],now[:,anchors[0].shape[1]:],masses[1])
  horizon=self.horizon(torch.stack([b['horizon']/10,torch.log1p(b['horizon'])],-1));skip=z[which,last]
  pred=self.head(torch.cat([c,f,skip,horizon[:,None].expand(-1,H*W,-1)],-1)).transpose(1,2).reshape(B,1,H,W)*.1*mask[:,None]
  return dict(delta=pred,temporal_tokens=temporal,anchor_coords=torch.cat(anchors,1)) if return_details else pred

def masked_loss(pred,target,mask):
 w=mask[:,None].to(pred.dtype)
 return ((pred-target).square()*w).sum()/w.sum().clamp_min(1)
