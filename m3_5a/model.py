"""Small spatial CNN; explicit nodal-to-element adapter used only by this baseline."""
import numpy as np
import torch
from torch import nn
from .io import check

def corner_average(node):return (node[...,:-1,:-1]+node[...,1:,:-1]+node[...,:-1,1:]+node[...,1:,1:])*.25

def encode(batch):
 """No target/proof/termination fields enter this function."""
 e=torch.from_numpy(batch['element_history']);n=torch.from_numpy(batch['node_history']);se=torch.from_numpy(batch['static_element']);sn=torch.from_numpy(batch['static_node']);B,K,_,H,W=e.shape
 valid=torch.from_numpy(batch['history_valid']).to(e.dtype)[:,:,None,None].expand(B,K,H,W)
 fields=[e.reshape(B,-1,H,W),se,corner_average(sn),valid]
 if n.shape[2]:fields.append(corner_average(n).reshape(B,-1,H,W))
 return torch.cat(fields,dim=1)
class UpdateCNN(nn.Module):
 def __init__(self,channels,width=24):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(channels,width,3,padding=1),nn.Tanh(),nn.Conv2d(width,width,3,padding=1),nn.Tanh(),nn.Conv2d(width,width,3,padding=1),nn.Tanh(),nn.Conv2d(width,1,1))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):return self.net(x)*.1

def metrics(prediction,target,mask):
 p=np.asarray(prediction,dtype=np.float64);y=np.asarray(target,dtype=np.float64);m=np.broadcast_to(mask[:,None],y.shape);err=(p-y)[m];truth=y[m];den=float(np.sqrt(np.mean(truth**2)));rmse=float(np.sqrt(np.mean(err**2)))
 return dict(RMSE=rmse,MAE=float(np.mean(abs(err))),target_RMS=den,relative_RMSE=rmse/max(den,1e-12),relative_denominator_floor=1e-12,masked_elements=int(m.sum()))
