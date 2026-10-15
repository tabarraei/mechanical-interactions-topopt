#!/usr/bin/env python
"""Isolated M3.3C scaling extension; production runs stop at 20 updates and 96x48."""
import argparse,json,os,platform,resource,subprocess,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
import numpy as np
from scipy import ndimage
from topopt_utils.fea.mechanical_problem import MechanicalProblem,DensityMap,array_digest
from topopt_utils.fea.trajectory_dataset import canonical_hash,file_hash,Trajectory,features,PROFILES,load_problem
from topopt_utils.fea.dataset_storage import atomic_json,generate_sample,validate_sample
from topopt_utils.fea.robust_oc import OPTIONS,update
from topopt_utils.fea.solver import FeaSolver
from topopt_utils.fea.consistent_trajectory import evaluate_state
from topopt_utils.topopt_2d import __create_filter_kernel_nd__
ROOT=Path(__file__).resolve().parent

def specification(nx,experiment):
    if nx not in (48,96,192,384) or experiment not in ('A','B'):raise ValueError('Unsupported scaling configuration')
    ny=nx//2;w,h=(48.,24.) if experiment=='A' else (float(nx),float(ny))
    fixed=tuple(2*j*(nx+1)+a for j in range(ny+1) for a in (0,1))
    p=MechanicalProblem(nx,ny,w,h,fixed,((ny//2*(nx+1)+nx,1.,0.),),filter_radius=2/(w/nx),maxiter=20)
    x=np.full(p.n,.5)
    body=dict(schema_version='m3.3b.problem.v1',mechanical=p.to_dict(),initial_x=x.tolist(),initial_digest=array_digest(x),optimizer=dict(option='feasible_upper_v1',version='m3.3a.oc.v1',**OPTIONS['feasible_upper_v1'],maxiter=20,tolerance=.001,move=.1,x_min=.001,x_max=1.,volume_variable='x'),force_scale=1.,filter_physical_semi_axes=[2.,2.])
    spec=dict(body,problem_id='m3_3b_'+canonical_hash(body)[:24],label=f'{experiment}_{nx}x{ny}',related_group='axial_scaling_family',support_pattern='left_clamp',configuration=dict(adapter='m3.3c.scaling.v1',experiment=experiment,mesh=[nx,ny],domain=[w,h],physical_radius=2.))
    load_problem(spec)
    return p,x,spec

def restrict(field,coarse_shape):
    a=np.asarray(field);cy,cx=coarse_shape
    if a.ndim!=2 or a.shape[0]%cy or a.shape[1]%cx:raise ValueError('Require aligned nested cells on the same physical domain')
    return a.reshape(cy,a.shape[0]//cy,cx,a.shape[1]//cx).mean(axis=(1,3))

def repeat(fn,n=5):
    times=[];result=None
    for _ in range(n):
        t=perf_counter();result=fn();times.append(perf_counter()-t)
    return result,dict(samples_seconds=times,median_seconds=float(np.median(times)))

def filter_audit(p):
    r=p.filter_radius;k=__create_filter_kernel_nd__(r,r);c=k.shape[0]//2;y,x=np.indices(k.shape)
    independent=np.maximum(0,1-np.hypot((x-c)*p.dx,(y-c)*p.dy)/2);independent/=independent.sum()
    np.testing.assert_allclose(k,independent,rtol=0,atol=1e-15)
    f=DensityMap(p);rng=np.random.default_rng(320);a=rng.normal(size=p.n);b=rng.normal(size=p.n)
    adjoint=abs(f.forward(a)@b-a@f.transpose(b));np.testing.assert_allclose(f.forward(np.ones(p.n)),1,atol=1e-14)
    return dict(index_radius=r,physical_radius=2.,kernel_shape=list(k.shape),positive_weights=int(np.count_nonzero(k)),kernel_max_error=float(abs(k-independent).max()),adjoint_absolute_error=float(adjoint),boundary_constant_preserved=True,cutoff='positive weight only at physical distance < 2; zero at equality',discretization='cell-center cone quadrature, row boundary normalization; not identical discrete operators across meshes')

def memory_status():
    values={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:')):values[line.split(':')[0]+'_mib']=int(line.split()[1])/1024
    values['ru_maxrss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    return values

def benchmark(nx,experiment,out):
    if nx>96:raise ValueError('192/384 execution requires separate explicit approval; this entry point performs estimates only')
    memory_before=memory_status();p,x,spec=specification(nx,experiment);solver=FeaSolver();mapping=DensityMap(p)
    rho,filtertime=repeat(lambda:mapping.forward(x));_,transpose=repeat(lambda:mapping.transpose(np.ones(p.n)))
    c,g,first=solver.compliance(rho,*p.solver_args,sensitivities=True,profile=True)
    profiles=[]
    for _ in range(5):c,g,t=solver.compliance(rho,*p.solver_args,sensitivities=True,profile=True);profiles.append(t)
    med={key:float(np.median([t[key] for t in profiles])) for key in profiles[0]}
    state=evaluate_state(p,x,solver);state['mechanical_id']=p.problem_id
    np.testing.assert_allclose(state['g_rho'],g,rtol=1e-10,atol=1e-11)
    _,oc=repeat(lambda:update(x,state['g_x'],.5,'feasible_upper_v1'))
    featuretimes={};shapes={}
    for profile in PROFILES:
        feat,t=repeat(lambda:features(p,x,0,profile,state));featuretimes[profile]=t
        shapes[profile]={k:list(feat[k].shape) for k in ('element','node')}
        assert np.count_nonzero(feat['node'][0])==1 and np.sum(feat['node'][0])==1
        if profile=='fresh_full_v1':np.testing.assert_allclose(feat['element'][-1].sum(),state['compliance']/2,rtol=1e-10)
    fd=[];direction=np.sin(p.centers[:,0]/48*2*np.pi)*(.5+.5*np.cos(p.centers[:,1]/24*2*np.pi))
    exact=float(state['g_x']@direction)
    for step in (1e-3,3e-4,1e-4):
        plus=solver.compliance(mapping.forward(x+step*direction),*p.solver_args)
        minus=solver.compliance(mapping.forward(x-step*direction),*p.solver_args)
        numerical=(plus-minus)/(2*step);fd.append(dict(step=step,relative_error=abs(numerical-exact)/max(abs(exact),1e-12)))
    assert min(d['relative_error'] for d in fd)<1e-5
    pat=solver.get_mesh_pattern(p.nelx,p.nely)
    record=dict(mesh=[p.nelx,p.nely],experiment=experiment,ndof=p.ndof,upper_nnz=pat.nnz,full_structural_nnz=2*pat.nnz-p.ndof,first_profile_seconds=first,warm_profile_samples=profiles,warm_median_seconds=med,filter=filtertime,filter_transpose=transpose,oc=oc,features=featuretimes,feature_shapes=shapes,finite_differences=fd,filter_audit=filter_audit(p),peak_rss_mib=memory_status()['VmHWM_mib'],memory_before=memory_before,memory_after=memory_status(),solver='MKL PARDISO mtype=2 direct SPD; cached phase11 symbolic, per-state phase22 numeric factorization and phase33 solve',threads={k:os.environ.get(k) for k in ('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    atomic_json(out,record)

def fresh_validate(out):
    rows=[]
    for folder in sorted((out/'samples').iterdir()):
        tr=Trajectory(folder);t=perf_counter();r=validate_sample(tr);rows.append(dict(problem_id=tr.spec['problem_id'],rows=len(r),seconds=perf_counter()-t))
    atomic_json(out/('fresh_validation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),dict(samples=rows,passed=True))

def fingerprints():
    old=json.loads((ROOT/'m3_3c/protected_before.json').read_text());changed=[n for n,h in old.items() if not (ROOT/n).is_file() or file_hash(ROOT/n)!=h]
    if changed:raise ValueError('Protected files changed: '+str(changed))
    return len(old)

def all_study(out):
    out.mkdir(parents=True,exist_ok=False);(out/'configs').mkdir();(out/'test_logs').mkdir()
    from run_m3_3b import source_manifest,assert_sources
    sources=source_manifest(out);sid=canonical_hash(sources)
    atomic_json(out/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,cpu_count=os.cpu_count(),execution_max_elements=4608,execution_max_updates=20,prior_scope='No existing M3.3C implementation found; isolated extension to M3.3B',physical_load=[48.,12.,1.,0.],point_load_caveat='2D concentrated nodal point force compliance may retain mesh-dependent load singularity; no continuum mesh-convergence inference'))
    for exp in ('A','B'):
        for nx in (48,96,192,384):
            p,x,s=specification(nx,exp);atomic_json(out/'configs'/f'{exp}_{nx}.json',s)
    benches=[]
    for nx,exp in ((48,'A'),(96,'A'),(96,'B')):
        file=out/f'benchmark_{exp}_{nx}.json'
        subprocess.run([sys.executable,str(ROOT/'run_m3_3c.py'),'benchmark','--mesh',str(nx),'--experiment',exp,'--output',str(file)],check=True)
        b=json.loads(file.read_text());benches.append(b)
        if b['peak_rss_mib']>2048 or b['warm_median_seconds']['total']>10:raise RuntimeError('Resource gate failed; no bounded trajectory authorized by measured safety gate')
        print(f"Resource gate passed {exp} {nx}: {b['peak_rss_mib']:.1f} MiB",flush=True)
    rows=[];trajectories=[]
    for nx,exp in ((48,'A'),(96,'A'),(96,'B')):
        p,x,s=specification(nx,exp);t=perf_counter();result=generate_sample(out,s,sid);generation=perf_counter()-t
        tr=Trajectory(result['folder']);t=perf_counter();validation=validate_sample(tr);validation_seconds=perf_counter()-t
        indices=iter(np.random.default_rng(4).integers(tr.metadata['state_count'],size=20));_,loading=repeat(lambda:np.array(tr.arrays['rho'][next(indices)]),20)
        # Measure independent full-history serialization; native generation total also includes checkpoint I/O and hashing.
        t=perf_counter();serial=out/f'serialization_{exp}_{nx}.npz'
        np.savez_compressed(serial,**tr.arrays);serial_seconds=perf_counter()-t
        size=sum(f.stat().st_size for f in tr.folder.rglob('*') if f.is_file())
        record=dict(experiment=exp,mesh=[nx,nx//2],folder=str(tr.folder.relative_to(out)),generation_including_checkpoint_seconds=generation,validation_seconds=validation_seconds,validation_rows=len(validation),serialization_seconds=serial_seconds,serialized_bytes=serial.stat().st_size,native_storage_bytes=size,random_access=loading,states=tr.metadata['state_count'],updates=tr.metadata['completed_updates'],stop=tr.metadata['termination_reason'],initial_compliance=float(tr.arrays['compliance'][0]),final_compliance=float(tr.arrays['compliance'][-1]),final_pg=float(tr.arrays['projected_gradient_inf'][-1]),max_signed_volume_excess=float(tr.arrays['volume_excess'].max()),final_change=float(tr.arrays['previous_change'][-1]),oc_multipliers=[t['selected_multiplier'] for t in tr.transitions],peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        trajectories.append(tr);rows.append(record);atomic_json(out/f'validation_{exp}_{nx}.json',json.loads(json.dumps(validation,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))))
        print(f'Completed and replayed {exp} {nx}: {record["states"]} states',flush=True)
    coarse,fine=trajectories[:2];comparisons=[]
    for t in range(min(coarse.metadata['state_count'],fine.metadata['state_count'])):
        f=fine.arrays['rho'][t].reshape(48,96);c=coarse.arrays['rho'][t].reshape(24,48);r=restrict(f,c.shape)
        comparisons.append(dict(iteration=t,rho_mae=float(abs(r-c).mean()),rho_max=float(abs(r-c).max()),restriction_integral_error=float(abs(r.sum()-f.sum()*.25)),fine_unresolved_rms=float(np.sqrt(np.mean((f-np.repeat(np.repeat(r,2,0),2,1))**2)))))
    atomic_json(out/'bounded_trajectories.json',rows);atomic_json(out/'physical_field_comparison.json',comparisons)
    # Smooth-field quadrature diagnostic is independent of optimization.
    filtered=[]
    for nx in (48,96):
        p,_,_=specification(nx,'A');v=.5+.1*np.sin(p.centers[:,0]/48*2*np.pi)*(.5+.5*np.cos(p.centers[:,1]/24*2*np.pi));filtered.append(DensityMap(p).forward(v).reshape(p.nely,p.nelx))
    atomic_json(out/'filter_discretization.json',dict(smooth_field_coarse_fine_restricted_mae=float(abs(filtered[0]-restrict(filtered[1],(24,48))).mean()),interpretation='Different center quadrature and boundary normalization, plus cell restriction; not exact filter invariance'))
    estimates=[]
    for exp in ('A','B'):
        base=benches[1 if exp=='A' else 2];r=rows[1 if exp=='A' else 2];solve=base['warm_median_seconds']['factorize']+base['warm_median_seconds']['solve']
        for nx in (192,384):
            ratio=(nx/96)**2;ndof=2*(nx+1)*(nx//2+1)
            # Full Q4 block structure has 4 scalar entries per node adjacency; upper=(full+ndof)/2.
            full=4*(3*(nx+1)-2)*(3*(nx//2+1)-2);upper=(full+ndof)//2
            lo,hi=solve*ratio**1.5,solve*ratio**2
            measured_perstate=r['generation_including_checkpoint_seconds']/r['states'];vl=r['validation_seconds']/r['states']
            storage=r['native_storage_bytes']*(501/r['states'])*ratio
            estimates.append(dict(experiment=exp,mesh=[nx,nx//2],elements=nx*nx//2,ndof=ndof,upper_structural_nnz=upper,csr_upper_bytes=12*upper+4*(ndof+1),connectivity_scatter_bytes=544*nx*nx//2,solve_scenario_seconds=[lo,hi],process_peak_rss_conservative_scenario_mib=[base['peak_rss_mib']*ratio,base['peak_rss_mib']*ratio**1.5],trajectory_500updates_storage_bytes=storage,generation_501state_scenario_seconds=[501*measured_perstate*ratio**1.5,501*measured_perstate*ratio**2],validation_501state_scenario_seconds=[501*vl*ratio**1.5,501*vl*ratio**2],dataset_100_200_generation_scenario_hours=[100*501*measured_perstate*ratio**1.5/3600,200*501*measured_perstate*ratio**2/3600],dataset_100_200_storage_bytes=[100*storage,200*storage],qualification='Unvalidated planning scenarios N^1.5..N^2 anchored at 96; not measured bounds or fitted performance law. Memory scenarios include repeatedly scaled import overhead, uncertain native fill/workspace; point load conditioning and physical-radius stencil growth may change cost. No large solves performed.'))
    atomic_json(out/'larger_mesh_estimates.json',estimates)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for j,(tr,r) in enumerate(zip(trajectories,rows)):
        axes[0,j].imshow(tr.arrays['rho'][-1].reshape(tr.problem.nely,tr.problem.nelx),origin='lower',extent=[0,tr.problem.width,0,tr.problem.height],vmin=0,vmax=1);axes[0,j].set_title(f"{r['experiment']} {r['mesh']} terminal rho")
        axes[1,j].plot(tr.arrays['compliance']);axes[1,j].set_xlabel('Completed updates');axes[1,j].set_ylabel('Compliance')
    fig.tight_layout();fig.savefig(out/'bounded_comparison.png');plt.close(fig)
    tests={}
    for name,extra in [('focused',['tests/test_mesh_scaling.py']),('full',[])]:
        with (out/'test_logs'/f'{name}.log').open('w') as log:
            t=perf_counter();ret=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);tests[name]=dict(exit_code=ret.returncode,seconds=perf_counter()-t)
        if ret.returncode:raise RuntimeError(name+' tests failed')
    subprocess.run([sys.executable,str(ROOT/'run_m3_3c.py'),'validate','--output',str(out)],check=True)
    assert_sources(sources);audit=dict(tests=tests,protected_files_unchanged=fingerprints(),source_bytes_unchanged=True,states=sum(r['states'] for r in rows),updates=sum(r['updates'] for r in rows),validation_rows=sum(r['validation_rows'] for r in rows),generation_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    atomic_json(out/'final_validation.json',audit);report(out,benches,rows,audit)
    atomic_json(ROOT/'m3_3c/latest_generation.json',dict(generation=str(out.relative_to(ROOT))))

def report(out,benches,rows,audit):
    lines=['# M3.3C — Mesh scaling and cross-resolution readiness','', 'This isolated extension preserves the verified numerical core and every prior artifact. No earlier M3.3C implementation was present. Source inspection included M3.1 state timing/physical identity, M3.2 stopping/stationarity, M3.3A guarded OC and M3.3B storage/features. Prior interaction locality and hybrid gains motivate hypotheses, not learned generalization claims.','', 'A fixes the physical domain at 48×24, unit +x nodal force at (48,12), left-edge clamp, volume .5, uniform x=.5, p=3, E0=1, nu=.3, thickness1, stiffness .001+.999 rho³. B keeps spacing1, grows the domain, moves the force to its right-middle point, and keeps physical radius2. The 48 baseline is shared, evaluated once. Complete configurations and coordinates for both experiments at all four target sizes are archived; 192/384 are configuration and estimates only.','', '## Measured costs','', '| Case | DOFs | Upper CSR nnz | Assembly ms | Numeric factor ms | Solve ms | C/gradient ms | Worker peak MiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for b in benches:
        m=b['warm_median_seconds'];lines.append(f"| {b['experiment']} {b['mesh']} | {b['ndof']} | {b['upper_nnz']} | {1000*m['assemble']:.4g} | {1000*m['factorize']:.4g} | {1000*m['solve']:.4g} | {1000*m['compliance']:.4g} | {b['peak_rss_mib']:.4g} |")
    lines+=['','PARDISO is direct SPD (mtype2): mesh connectivity/scatter and symbolic phase11 cached, numeric phase22 repeated per density, phase33 triangular solve. Five warm repetitions and a separately recorded first call distinguish setup. OpenBLAS1 and unchanged MKL/OMP16. Profile total excludes KE lookup and some API bookkeeping; native phase times are measured independently. Worker Linux /proc/self/status VmHWM includes imports, independent assembly, features and derivative checks; it is a post-exec process high-water mark, not isolated factor memory. Import baseline, current VmRSS and ru_maxrss are recorded separately; ru_maxrss can carry a larger pre-exec watermark from the config-building parent and is not used as the worker peak. Full-process study peak is in final_validation.json. No full-suite/trajectory timings overlap. Feature construction timings reuse an already evaluated state; fresh-mechanics FEM work is additional. Filter/transpose, guarded OC and all feature profiles have repeated samples in benchmark JSON. Serialization is a separate compressed full-history export; native generation includes atomic checkpoints, hashing, diagnostics and publication. Random-access times copy one seeded mmap slice and represent twenty seeded random warm accesses, not cold disk throughput.','', '## Bounded trajectories','', '| Case | Updates/stop | Initial C | Terminal C | Terminal PG inf | Maximum signed x excess | Generation s | Replay s | Native MiB |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['experiment']} {r['mesh']} | {r['updates']}/{r['stop']} | {r['initial_compliance']:.7g} | {r['final_compliance']:.7g} | {r['final_pg']:.4g} | {r['max_signed_volume_excess']:.3g} | {r['generation_including_checkpoint_seconds']:.4g} | {r['validation_seconds']:.4g} | {r['native_storage_bytes']/2**20:.4g} |")
    lines+=['','Every saved state and transition underwent independent FEM/COO equilibrium, energy, reaction force/moment, compliance/gradient API and guarded-OC replay. Benchmark initial states also have three-step centered objective derivatives in x using F.T. No model, dense interactions, extended trajectories or optimum labels. .001 change is provisional and PG scaling uses max|g|; multiplier sequences and changes remain in saved transitions/arrays. Spatial plots and full C, mean x/rho, PG and change histories are retained. Raw OC lambda acts on sum(x), so its magnitude across N is not directly a common mean-constraint multiplier; multiply by N for that convention.','', '## Filter, field transfer and loading','', 'The exact kernel is max(0,1-sqrt(sum((offset/r_index)^2))), globally normalized on a 2ceil(r)+1 stencil. Density mapping divides constant-zero convolution by convolution of ones. Global kernel normalization cancels this row normalization. Positive weight requires distance<radius; equality has zero weight. Square-cell index radii2/4/8/16 in A sample the same physical radius2 cone, with differing quadrature and boundary rows. This is a consistent nominal physical-length discretization, not identical filters or proven convergence. Constant preservation, adjoint identity and independent physical-distance weights are checked. filter_discretization.json separates an analytic smooth-field comparison from OC results. No core change is needed for this square-cell pilot. Rectangular cells with scalar r have elliptical physical semi-axes r*dx,r*dy; a circular physical radius needs a separate versioned anisotropic DensityMap accepting (R/dy,R/dx), independently validated before use. It is not silently supplied here.','', 'Fine-to-coarse transfer is aligned 2×2 cell averaging on the same 48×24 domain, preserving density integral exactly up to roundoff. physical_field_comparison.json reports discrepancy and unresolved subcell RMS; block prolongation is diagnostic only. No FEM array or feature is resized. Differences combine filter/FEM discretization and different 20-step optimization paths; transfer loss is separately quantified, but those sources cannot be fully identified from this pilot. The nodal concentrated force is physically corresponding and has unit resultant; 2D point-force compliance may have a mesh-dependent singularity, so identical loading does not guarantee a convergent continuum compliance. Future traction tests require separately integrated nodal forces. B is increasing size, never a mesh-convergence comparison.','', '## Dataset and M4/M5 design','', 'Native M3.3B mmap/problem loaders accept these explicit larger specifications and exact element/node arrays. The historical sampler <=1152-element guard remains unchanged; this versioned adapter supplies controlled scaling configurations only. Features are (channels,ny,nx) and (channels,ny+1,nx+1), C ordering, increasing Cartesian row-y; exact unit force stays at one node and clamps stay on boundary nodes. Coordinates, sensitivities, energies and optional displacements are checked; energy sums to C/2. Full problem metadata is lossless; the spatial feature tensors alone do not encode all variable filter/material controls. Before learning add explicit known conditioning (dx,dy,width,height,physical/index filter radius,material,SIMP,optimizer version); keep native tensors and homogeneous mesh batches. Any node-to-element learned interface must document shape-function/incident-node aggregation and retain force/support channels on native nodes as the reference; never reinterpret aggregated loads as FEM forces. Fresh-gradient/full profiles cost FEM and that work must count in acceleration comparisons.','', 'M4 first trains on independent mechanical families at48 and tests held-out loads/supports at48. Next mix48/96 with whole physical families, mirrored/near-related loads, seeds and every resolution grouped in one partition; windows never cross partitions. Hold out independent192 problems for frozen-weight extrapolation. Separate domain-growth evaluations condition on physical dimensions and force location. Compare multiscale U-Net, temporal U-Net, bottleneck attention and feasible full Transformer on matched data, comparable parameters and measured compute. Fixed-depth pixel receptive fields shrink in physical size under refinement; encode spacing and test physical receptive fields. Use continuous physical positional encodings or a validated interpolation scheme; learned flattened absolute embeddings do not automatically generalize. Full element attention score storage grows N²: one float32 score matrix is ~5.1/81/1296/20736 MiB at48/96/192/384, before heads/batches/gradients. A history adds memory and must be accounted for.','', 'Success requires x/rho error, independently recomputed C, strict volume, PG/KKT diagnostics, stable multi-step rollouts, actual FEM counts, end-to-end wall time and peak memory, held-out loads/supports/resolutions. Compare equal feasibility/stationarity/design-quality policies to explicit feasible-upper OC; low MSE alone is insufficient. All current samples are dynamics examples, not certified optimum labels.','', '## Larger meshes and go/no-go','', 'larger_mesh_estimates.json provides actual analytic DOF/structural nnz and sparse-array bytes, storage extrapolations to501 states and100–200 trajectories, plus unvalidated N^1.5..N^2 time scenarios anchored at96. These are planning scenarios, not measured bounds or an inferred universal scaling law. Factor fill, conditioning, import overhead, independent COO temporaries, serialization, validation replay and growing physical filter stencils prevent linear time/memory assumptions. Peak RSS is not known at192/384. Do not use scenarios as hardware guarantees.','', 'Go:48 for an initial controlled M4 pilot;96 for reviewed mixed-resolution follow-up, given successful bounded measurements. No-go:192/384 trajectory generation or a100–200-family rollout without a separate resource/quality-policy review. A192 single-state benchmark requires explicit approval and a separately reviewed execution command; the current CLI intentionally rejects execution above96. Prepared shell template (do not execute without approval): `OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -c "from run_m3_3c import specification; from topopt_utils.fea.consistent_trajectory import evaluate_state; import resource,time; p,x,s=specification(192,\'A\'); t=time.perf_counter(); z=evaluate_state(p,x); print(time.perf_counter()-t,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,z[\'compliance\'])"` . This runs one state only and does not launch OC.','', '## Direct answers','', '1. Yes, square-cell fixed-domain refinement is represented with explicit complete mechanics; no protected numerical changes.','2. Radius adjustment preserves nominal physical cone support; discrete quadrature/boundary differences remain.','3. The same physical point/resultant and continuous clamp boundary correspond; nodal approximation and point-load singularity remain.','4. See measured cost/storage tables and benchmark JSON; no linear-scaling claim.','5. Unmeasured factor fill/conditioning, validation and data I/O, stencil and attention growth are principal barriers.','6. Native grids and metadata retain information at variable resolutions; tensors need additional known conditioning before mixed-control learning.','7. Add conditioning, grouped splits, variable-grid architecture/position handling, resource gates and a quality/stopping policy.','8. A fixes geometry/physical radius while h changes; B fixes h/radius while geometry grows.','9. Start48; include96 only as a controlled, resource-reviewed follow-up.','10. Frozen models must meet independent mechanical quality and rollout criteria on held-out physical families/resolutions while reducing measured FEM/work.','', '## Validation and reproduction','',f"{audit['states']} states, {audit['updates']} updates, {audit['validation_rows']} replay comparison rows; {audit['protected_files_unchanged']} protected fingerprints unchanged. Focused/full logs retain stdout and exit codes; both exit zero. Source snapshots preserve executing bytes, with archived tests named .py.txt to prevent collection. No old artifacts overwritten, commit/push, HPC, training or large numerical runs.",'', 'Run from repository root: `OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3-3c-mpl /users/atabarra/.venv/bin/python run_m3_3c.py all`. A new timestamped generation is created; an explicit new `--output` may be supplied. Benchmark only: same environment, `run_m3_3c.py benchmark --mesh 96 --experiment A --output /tmp/m3-3c-benchmark.json`. Tests: same environment, `python -m pytest -q -p no:cacheprovider tests/test_mesh_scaling.py`, and omit the filename for full regression. Fresh replay: same environment, `run_m3_3c.py validate --output m3_3c/generations/GENERATION_ID`; timestamped validation files preserve the original audit. Exact archived execution: `python m3_3c/replay_snapshot.py --generation m3_3c/generations/GENERATION_ID validate`. All required sources are in source_manifest.json.']
    (out/'M3_3C_MESH_SCALING_REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['all','benchmark','validate']);ap.add_argument('--output',type=Path);ap.add_argument('--mesh',type=int,default=48);ap.add_argument('--experiment',choices=['A','B'],default='A');args=ap.parse_args()
    if args.command=='validate':fresh_validate(args.output)
    elif args.command=='benchmark':benchmark(args.mesh,args.experiment,args.output)
    else:
        fingerprints();out=args.output or ROOT/'m3_3c/generations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');all_study(out)
