#!/usr/bin/env python
"""Read-only matched-state timing after the M3.3A generation; no FEM solves."""
import argparse,csv,json,hashlib,shutil
from pathlib import Path
from time import perf_counter
import numpy as np
from topopt_utils.fea.consistent_trajectory import load_trajectory
from topopt_utils.fea.robust_oc import update
from topopt_utils.topopt_2d import __oc_update_design__


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--generation',type=Path,required=True);parser.add_argument('--repeats',type=int,default=9);args=parser.parse_args()
    out=args.generation;rows=[];replicates=[]
    for entry in json.loads((out/'trajectory_manifest.json').read_text()):
        if entry['option']!='baseline_v1':continue
        p,a,m=load_trajectory(entry['folder'])
        for t in sorted(set([0,len(a['x'])//2,len(a['x'])-1])):
            x=a['x'][t];g=a['g_x'][t];functions=[('historical_raw',lambda:__oc_update_design__(x,g,p.volume_fraction)),('tight_midpoint_v1',lambda:update(x,g,p.volume_fraction,'tight_midpoint_v1')[0]),('feasible_upper_v1',lambda:update(x,g,p.volume_fraction,'feasible_upper_v1')[0])]
            for name,fn in functions:fn()  # warm each on identical input
            measures={name:[] for name,fn in functions}
            for rep in range(args.repeats):
                order=functions[rep%3:]+functions[:rep%3]
                for name,fn in order:
                    start=perf_counter();y=fn();elapsed=perf_counter()-start;measures[name].append(elapsed)
                    replicates.append(dict(label=entry['label'],iteration=t,option=name,replicate=rep,seconds=elapsed,mean_x=float(y.mean())))
            baseline=np.median(measures['historical_raw'])
            for name,values in measures.items():rows.append(dict(label=entry['label'],iteration=t,option=name,median_seconds=float(np.median(values)),p90_seconds=float(np.percentile(values,90)),median_ratio_to_raw=float(np.median(values)/baseline),repeats=args.repeats,saved_evaluation_seconds=float(a['evaluation_seconds'][t]),estimated_FEM_and_update_ratio=float((a['evaluation_seconds'][t]+np.median(values))/(a['evaluation_seconds'][t]+baseline))))
    for name,data in [('matched_state_timing',rows),('matched_state_timing_replicates',replicates)]:
        with (out/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summary={name:dict(states=sum(r['option']==name for r in rows),median_ratio_to_raw=float(np.median([r['median_ratio_to_raw'] for r in rows if r['option']==name])),median_seconds=float(np.median([r['median_seconds'] for r in rows if r['option']==name])),median_estimated_FEM_and_update_ratio=float(np.median([r['estimated_FEM_and_update_ratio'] for r in rows if r['option']==name]))) for name in ('historical_raw','tight_midpoint_v1','feasible_upper_v1')}
    summary['convention']='Same x/g/target at three saved baseline states per case; rotating call order, one warmup and nine repetitions; baseline is one actual untouched helper call, new options include their production guards. No diagnostic baseline replica or FEM in measured call. Timings are machine/thread dependent.'
    summary['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'matched_state_timing_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    shutil.copyfile(__file__,out/'source_snapshots/benchmark_oc.py')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
