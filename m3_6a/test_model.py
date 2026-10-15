"""Synthetic architecture diagnostics only; zero optimizer steps or FEM."""
import copy,sys
import numpy as np
import torch
import pytest
from m3_6a.data import collate
from m3_6a.model import MultiscaleTemporalOperator,CoordinateKernel,conservative_force_map,masked_loss

torch.set_num_threads(16)
def fixture_sample(H=6,W=10,K=3,h=1,mode='A',name='SYNTHETIC_ONLY'):
 rng=np.random.default_rng(3601);width,height=48.,24.;xx,yy=np.meshgrid((np.arange(W)+.5)*width/W,(np.arange(H)+.5)*height/H);nx,ny=np.meshgrid(np.arange(W+1)*width/W,np.arange(H+1)*height/H)
 nf=np.zeros((4,H+1,W+1),np.float32);nf[1,0,-1]=-2;nf[2:,:,0]=1
 x=rng.uniform(.2,.7,(K,1 if mode=='A' else 5,H,W)).astype(np.float32)
 inputs=dict(element_history=x,node_history=np.zeros((K,0 if mode=='A' else 2,H+1,W+1),np.float32),node_fields=nf,element_coords=np.stack([xx,yy]).astype(np.float32),node_coords=np.stack([nx,ny]).astype(np.float32),domain_mask=np.ones((H,W),bool),node_mask=np.ones((H+1,W+1),bool),area=np.full((H,W),width*height/H/W,np.float32),globals=np.zeros(19,np.float32),extent=np.array([width,height],np.float32),force_scale=np.float32(2),iterations=np.arange(K,dtype=np.float32),cap=np.float32(500),horizon=np.float32(h),mode=mode)
 return dict(inputs=inputs,target=np.zeros((1,H,W),np.float32),proof=dict(problem_id=name,anchor=K-1,horizon=h,fixture=True))

def clone(b):return {k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in b.items()}

@pytest.mark.parametrize('shape',[(24,48),(32,48),(48,96)])
def test_shared_cpu_native_resolutions(shape):
 model=MultiscaleTemporalOperator();b=collate([fixture_sample(*shape,K=1)]);before={n:tuple(p.shape) for n,p in model.named_parameters()}
 with torch.no_grad():out=model(b)
 assert out.shape==(1,1,*shape) and torch.isfinite(out).all();assert before=={n:tuple(p.shape) for n,p in model.named_parameters()}

@pytest.mark.parametrize('K,h',[(1,1),(2,5),(4,10)])
def test_history_horizon_and_backward(K,h):
 torch.manual_seed(3601);m=MultiscaleTemporalOperator();b=collate([fixture_sample(K=K,h=h)]);out=m(b);loss=masked_loss(out,b['target'],b['domain_mask']);loss.backward()
 assert torch.isfinite(loss)
 for module in (m.lift,m.global_kernel,m.local_kernel,m.temporal,m.head):
  grads=[p.grad for p in module.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0

def test_temporal_causality_future_target_ignored():
 torch.manual_seed(3601);m=MultiscaleTemporalOperator().eval();b=collate([fixture_sample(K=4)])
 with torch.no_grad():a=m(b,True);changed=clone(b);changed['element_history'][:,2:]+=.4;c=m(changed,True)
 assert torch.allclose(a['temporal_tokens'][:,:2],c['temporal_tokens'][:,:2],atol=1e-6,rtol=1e-6)
 b['target'].fill_(float('nan'));b['proof']=[{'future_termination':'forbidden'}]
 with torch.no_grad():other=m(b)
 assert torch.equal(a['delta'],other)

def test_padding_and_variable_mesh_batch():
 a=fixture_sample(4,6,K=1,name='SYNTHETIC_A');b=fixture_sample(6,9,K=3,name='SYNTHETIC_B');batch=collate([a,b]);m=MultiscaleTemporalOperator().eval()
 assert batch['history_valid'].tolist()==[[True,False,False],[True,True,True]]
 with torch.no_grad():p=m(batch);changed=clone(batch);changed['element_history'][0,1:]=1e4;q=m(changed)
 assert torch.equal(p[0],q[0]);assert torch.count_nonzero(p[0,:,4:])==0 and torch.count_nonzero(p[0,:,:4,6:])==0

def test_explicit_L_domain_mask_only_fixture():
 s=fixture_sample(8,10);s['inputs']['domain_mask'][4:,5:]=False;mask=s['inputs']['domain_mask'];nm=np.zeros((9,11),bool)
 for dy,dx in ((0,0),(0,1),(1,0),(1,1)):nm[dy:dy+8,dx:dx+10]|=mask
 s['inputs']['node_mask']=nm;b=collate([s]);m=MultiscaleTemporalOperator().eval()
 with torch.no_grad():p=m(b);z=clone(b);z['element_history'][:,:,:,4:,5:]=100;q=m(z)
 assert torch.count_nonzero(p[:,:,4:,5:])==0 and torch.allclose(p,q)

def test_force_conservation_and_conditioning():
 b=collate([fixture_sample()]);mapped=conservative_force_map(b['node_fields'][:,:2],b['domain_mask'])
 assert torch.allclose(mapped.sum((-1,-2)),b['node_fields'][:,:2].sum((-1,-2)))
 torch.manual_seed(3601);m=MultiscaleTemporalOperator().eval()
 with torch.no_grad():p=m(b);q=clone(b);q['node_fields'][:,2:]=0;out=m(q)
 assert not torch.allclose(p,out,atol=1e-8,rtol=1e-8)

def test_constant_quadrature_and_coordinate_scale():
 kernel=CoordinateKernel();q=torch.tensor([[[.2,.1],[.7,.4]]]);v=torch.ones(1,8,3);s=torch.rand(1,8,2);area=torch.ones(1,8)
 out,_=kernel(q,s,v,area);assert torch.allclose(out,torch.ones_like(out),atol=1e-6)
 model=MultiscaleTemporalOperator();b=collate([fixture_sample()]);d=model(b,True);assert d['anchor_coords'][...,0].max()<1 and d['anchor_coords'][...,1].max()<.5

def test_initialization_deterministic():
 torch.manual_seed(3601);a=MultiscaleTemporalOperator();torch.manual_seed(3601);b=MultiscaleTemporalOperator()
 assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))

def test_mode_B_explicit_separate_interface():
 b=collate([fixture_sample(mode='B')]);m=MultiscaleTemporalOperator(mode='B');assert torch.isfinite(m(b)).all()
 with pytest.raises(ValueError):MultiscaleTemporalOperator()(b)

def test_invalid_inputs_fail_closed():
 b=collate([fixture_sample()]);m=MultiscaleTemporalOperator();x=clone(b);x['history_valid'][0]=torch.tensor([False,True,True])
 with pytest.raises(ValueError):m(x)
 x=clone(b);x['horizon'].fill_(2)
 with pytest.raises(ValueError):m(x)
 x=clone(b);x['element_history'][0,0,0,0,0]=float('nan')
 with pytest.raises(ValueError):m(x)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable; CPU milestone only')
def test_small_GPU_forward():
 m=MultiscaleTemporalOperator().cuda();b=collate([fixture_sample(4,6,K=1)]);b={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in b.items()}
 with torch.no_grad():assert torch.isfinite(m(b)).all()

def test_no_FEM_OC_imports():
 assert not any(n.startswith(('topopt_utils','pypardiso','m3_4br3','m3_4b.')) for n in sys.modules)
