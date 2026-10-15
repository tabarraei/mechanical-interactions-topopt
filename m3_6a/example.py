"""Small accepted-data CPU forward diagnostic; no optimizer or training."""
import argparse,json,time,resource
from pathlib import Path
import torch
from .data import AcceptedData,collate
from .model import MultiscaleTemporalOperator

def run():
 torch.set_num_threads(16);torch.manual_seed(3601);model=MultiscaleTemporalOperator().eval();parameter_ids=[id(p) for p in model.parameters()];data=AcceptedData();rows=[];start=time.perf_counter()
 try:
  with torch.no_grad():
   for i,h,k in ((0,1,1),(6,5,3),(1,10,2)):
    sample=data.example(data.entries[i]['problem_id'],3,k=k,h=h);batch=collate([sample]);prediction=model(batch)
    assert parameter_ids==[id(p) for p in model.parameters()] and torch.isfinite(prediction).all()
    rows.append(dict(problem_id=sample['proof']['problem_id'],mesh=sample['proof']['mesh'],history=k,horizon=h,element_input=list(batch['element_history'].shape),node_input=list(batch['node_fields'].shape),output=list(prediction.shape),finite=True,model_instance_shared=True))
 finally:data.close()
 return dict(seed=3601,device='cpu',parameters=sum(p.numel() for p in model.parameters()),configuration=dict(width=24,heads=4,temporal_layers=1,latent_scales=[[4,2],[6,4]],mode='A'),examples=rows,wall_seconds=time.perf_counter()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,optimizer_steps=0,FEM=0,OC=0,untrained=True,generalization_claim=False,cuda_available=torch.cuda.is_available())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();r=run()
 if a.output:
  with Path(a.output).open('x') as f:json.dump(r,f,indent=2)
 print(json.dumps(r),flush=True)
