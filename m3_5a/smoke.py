"""Budgeted CPU tiny-subset overfit, separately for feature modes A and B."""
import sys,time,random,json
from pathlib import Path
import numpy as np
import torch
from .loader import Release,TrajectoryDataset
from .normalization import Normalization
from .features import collate,crop_predictions
from .model import UpdateCNN,encode,metrics
from .io import read,write,check

def run(config_path,out,mode):
 cfg=read(config_path);out=Path(out);out.mkdir(parents=True,exist_ok=False);check(mode in ('A','B'),'Smoke mode')
 seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
 start=time.perf_counter()
 with Release(cfg['release_index'],cfg['release_sha256'],cfg['release_manifest_sha256']) as release:
  norm=Normalization.load(cfg['normalization'],release);ds=TrajectoryDataset(release,mode,cfg['history'],norm)
  keys=cfg['tiny_subset'];check(len({(k['problem_id'],k['transition']) for k in keys})==len(keys),'Duplicate diagnostic examples')
  samples=[ds.sample(k['problem_id'],k['transition']) for k in keys]
  for s in samples:ds.validate_sample(s)
  batch=collate(samples);X=encode(batch);Y=torch.from_numpy(batch['target']);M=torch.from_numpy(batch['element_valid'])[:,None];model=UpdateCNN(X.shape[1],cfg['width']);opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);initial={k:v.detach().clone() for k,v in model.state_dict().items()}
  previous=np.stack([s['inputs']['element_history'][-1,1]*release.specs[s['proof']['problem_id']]['optimizer']['move'] for s in samples])[:,None]
  baseline=dict(zero_update=metrics(np.zeros_like(batch['target']),batch['target'],batch['element_valid']),previous_update=metrics(previous,batch['target'],batch['element_valid']))
  history=[];gradient_seen=False;finite=True
  for step in range(cfg['steps']):
   check(time.perf_counter()-start<cfg['wall_seconds'],'Smoke budget exhausted')
   opt.zero_grad(set_to_none=True);pred=model(X);loss=(((pred-Y)/.1)**2)[M].mean();check(torch.isfinite(loss).item(),'Nonfinite loss');loss.backward()
   grads=[p.grad for p in model.parameters() if p.grad is not None];check(grads and all(torch.isfinite(g).all().item() for g in grads),'Nonfinite gradients');gradient_seen|=any(torch.count_nonzero(g).item()>0 for g in grads)
   opt.step()
   if step%25==0 or step==cfg['steps']-1:history.append(dict(step=step,normalized_MSE=float(loss.detach())))
  model.eval()
  with torch.no_grad():prediction=model(X).numpy()
  result=metrics(prediction,batch['target'],batch['element_valid']);changed=any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items())
  mapped=crop_predictions(prediction,batch);check(all(p.shape==s['target'].shape for p,s in zip(mapped,samples)),'Map predictions to original mesh')
  # Exercise inference on every released mesh/family, without training on these.
  probes=[]
  with torch.no_grad():
   for e in release.entries:
    s=ds.sample(e['problem_id'],0);b=collate([s]);p=model(encode(b)).numpy();check(np.isfinite(p).all() and crop_predictions(p,b)[0].shape==s['target'].shape,'Full mesh inference')
    probes.append(dict(problem_id=e['problem_id'],mesh=s['inputs']['mesh'],prediction_shape=list(p.shape)))
  passed=gradient_seen and changed and result['relative_RMSE']<=cfg['overfit_relative_RMSE'] and result['RMSE']<baseline['zero_update']['RMSE']
  torch.save(dict(state_dict=model.state_dict(),channels=X.shape[1],width=cfg['width'],mode=mode,normalization_id=norm.identifier,release_sha256=release.index_sha256),out/'model.pt')
  np.savez_compressed(out/'tiny_predictions.npz',prediction=prediction,target=batch['target'],previous_update=previous,element_mask=batch['element_valid'])
  record=dict(passed=passed,mode=mode,feature_schema=samples[0]['inputs']['schema'],fresh_FEM_required_at_deployment=mode=='B',new_FEM_calls=0,new_OC_calls=0,scope='Training-only tiny-subset overfit; no held-out/OOD/generalization/acceleration claim',seed=seed,device='cpu',torch_threads=torch.get_num_threads(),steps=cfg['steps'],parameters=sum(p.numel() for p in model.parameters()),history=cfg['history'],tiny_examples=keys,baseline=baseline,model=result,gradient_nonzero=gradient_seen,parameters_changed=changed,loss_finite=True,all_mesh_family_inference=probes,training_curve=history,wall_seconds=time.perf_counter()-start,torch_version=torch.__version__,deterministic_algorithms=True,feasibility_projection=False,optimizer_rollout=False,normalization_id=norm.identifier)
  write(out/'result.json',record);print(json.dumps({k:record[k] for k in ('passed','mode','model','baseline','wall_seconds')}),flush=True)
  check(passed,'Tiny-subset overfit criterion failed; preserve failed evidence')
if __name__=='__main__':run(sys.argv[1],sys.argv[2],sys.argv[3])
