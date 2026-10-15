"""Small native-handle lifecycle probe, using unchanged verified solver methods."""
import argparse,json,gc
from pathlib import Path
from time import perf_counter
import numpy as np
from topopt_utils.fea.solver import FeaSolver
from run_m3_3c import specification,memory_status

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fresh','reuse','released'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
problem,x,_=specification(48,'A');trace=[];results=[];base=memory_status();shared=FeaSolver() if a.mode=='reuse' else None
for i in range(20):
 solver=shared if shared is not None else FeaSolver();t=perf_counter();c=solver.compliance(x,*problem.solver_args);seconds=perf_counter()-t;results.append(c)
 if a.mode=='released':
  for context in list(solver._symbolic_cache.values()):context.free()
  solver._symbolic_cache.clear()
 if shared is None:del solver;gc.collect()
 trace.append(dict(state=i,compliance=c,seconds=seconds,**memory_status()))
if shared is not None:
 for context in list(shared._symbolic_cache.values()):context.free()
 shared._symbolic_cache.clear()
result=dict(mode=a.mode,initial_memory=base,trace=trace,final_memory=memory_status(),FEM_evaluations=20,mesh=[48,24],scope='Same uniform density;20 FEM evaluations, zero OC/optimization; unchanged solver; released invokes existing context.free after numerical use')
a.output.write_text(json.dumps(result,indent=2)+'\n')
