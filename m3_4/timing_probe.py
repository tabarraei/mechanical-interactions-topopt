"""Isolated inclusive storage timing on the existing small pilot specification."""
import json,sys,time,hashlib,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from topopt_utils.fea import dataset_storage as storage
from topopt_utils.fea.trajectory_dataset import Trajectory
from run_m3_4 import binding,save
from run_m3_3c import memory_status

def main(out):
    out=Path(out).resolve();m=binding(out);work=out/'timing_probe'
    if work.exists():raise ValueError('Timing evidence requires a new output directory')
    work.mkdir();source=Path(__file__);archive=work/'timing_probe.py.txt';shutil.copyfile(source,archive)
    save(work/'source_binding.json',dict(helper_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),archive='timing_probe.py.txt',numerical_sources_id=m['source_id'],spec_sha256=m['jobs'][0]['spec_sha256']))
    spec=json.loads((out/m['jobs'][0]['spec']).read_text());times=defaultdict(float);calls=defaultdict(int)
    def wrap(name,label=None):
        old=getattr(storage,name);label=label or name
        def measured(*args,**kwargs):
            tick=time.perf_counter()
            try:return old(*args,**kwargs)
            finally:times[label]+=time.perf_counter()-tick;calls[label]+=1
        setattr(storage,name,measured)
    for name in ('enrich_state','update','atomic_frame','atomic_json','file_hash'):wrap(name)
    started=time.perf_counter();before=memory_status()
    a=storage.generate_sample(work,spec,m['source_id'],max_new_updates=20);prefix_seconds=time.perf_counter()-started
    snapshot=dict(times);prefix_calls=dict(calls);tick=time.perf_counter()
    b=storage.generate_sample(work,spec,m['source_id']);resume_seconds=time.perf_counter()-tick
    tr=Trajectory(b['folder']);original=Trajectory(out/'pilot/samples'/spec['problem_id'])
    for key,value in tr.arrays.items():np.testing.assert_allclose(value,original.arrays[key],rtol=1e-10,atol=1e-12)
    save(work/'timings.json',dict(prefix_seconds=prefix_seconds,resume_and_finalization_seconds=resume_seconds,total_seconds=time.perf_counter()-started,inclusive_function_seconds=dict(times),calls=dict(calls),prefix_function_seconds=snapshot,prefix_calls=prefix_calls,FEM_evaluations=calls['enrich_state'],OC_updates=calls['update'],state_count=tr.metadata['state_count'],history_equivalent=True,before=before,after=memory_status(),scope='Existing small specification, native20-update checkpoint then continuation; wrapper timings are inclusive and overlap. atomic_frame includes NPZ compression/fsync; atomic_json includes journal/transition/metadata serialization/fsync. file_hash includes checkpoint, final-array and integrity checks. Finalization is included in resumed wall. No protected numerical method changed.'))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=json.loads((work/'source_binding.json').read_text())['helper_sha256']:raise ValueError('Executing timing source changed')
    print('Timing probe passed',flush=True)

if __name__=='__main__':main(sys.argv[1])
