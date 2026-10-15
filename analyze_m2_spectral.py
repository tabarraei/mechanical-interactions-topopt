#!/usr/bin/env python
"""M2.3 saved-matrix analysis; no exact-matrix generation or FEM calls.

Run from any directory. Existing m2_spectral outputs are archived, never discarded.
All M1/M2 artifacts and the numerical core are fingerprinted before and after work.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh
from scipy.spatial.distance import cdist

# Reuse generation logic verbatim; importing it does not invoke generate/main.
from generate_large_mesh_scaling import (
    _vector_batch_random, _vector_batch_smooth, _vector_batch_localized,
)
from topopt_utils.fea.interactions import normalize_mechanical_interactions
from topopt_utils.fea.spectral_interactions import (
    ENERGY_TARGETS, SELECTED_RANKS, active_principal_submatrix,
    symmetric_psd_eigensystem, spectral_statistics, cumulative_squared_energy,
    rank_for_energy, relative_frobenius_error, action_error_summary,
    spatial_truncation, approximation_costs, leading_eigenspace_comparison,
)

ROOT=Path(__file__).resolve().parent
FAMILIES=('random','smooth','localized')
SEEDS=dict(random=10,smooth=14,localized=16)  # matches M2.1 48x24 n_states=3 seeds


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def serial(value):
    if isinstance(value,dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serial(v) for v in value]
    if isinstance(value,np.ndarray): return serial(value.tolist())
    if isinstance(value,np.generic): return serial(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def write_csv(path,rows):
    if not rows: raise ValueError(f'No rows for {path}')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def fingerprint_protected():
    roots=['dataset','results/m1_interactions','results/m1_interactions_large_48x24',
           'm2_spatial','m2_spatial_large','m2_topology']
    paths=[p for name in roots for p in (ROOT/name).rglob('*') if p.is_file()]
    paths += [ROOT/name for name in ['topopt_utils/fea/solver.py','topopt_utils/fea/q4.py',
                                     'topopt_utils/topopt_2d.py','topo_transformer/physics_loss.py']]
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(paths))}


def discover_inputs():
    domains={}
    for domain in ['12x6','24x12','48x24']:
        folder=ROOT/'results'/('m1_interactions_large_48x24' if domain=='48x24' else 'm1_interactions')
        if domain!='48x24': folder/=domain
        paths=sorted(folder.glob('*.npz'))
        required={'uniform_state','middle_state_10' if domain=='12x6' else 'middle_state_15',
                  'late_state_19' if domain=='12x6' else 'late_state_29'}
        missing=required-{p.stem for p in paths}
        if missing: raise FileNotFoundError(f'Essential saved inputs missing in {folder}: {sorted(missing)}; no regeneration attempted')
        for p in paths:
            if not p.with_suffix('.json').exists(): raise FileNotFoundError(f'Missing metadata {p.with_suffix(".json")}')
        domains[domain]=paths
    return domains


def load_state(path):
    required={'P','R','rho','active_mask','element_x','element_y'}
    with np.load(path,allow_pickle=False) as data:
        missing=required-set(data.files)
        if missing: raise ValueError(f'{path}: essential arrays missing {missing}; no regeneration attempted')
        state={key:data[key] for key in required}
        if 'g_rho' in data: state['g_rho']=data['g_rho']
    state['mesh']=json.loads(path.with_suffix('.json').read_text())['mesh']
    state['name']=path.stem; state['path']=path
    p=state['P']; n=state['rho'].size
    if p.shape!=(n,n) or not np.all(np.isfinite(p)): raise ValueError(f'{path}: invalid P')
    if state['R'].shape!=(n,n) or not np.all(np.isfinite(state['R'])): raise ValueError(f'{path}: invalid R')
    for key in ['rho','element_x','element_y','active_mask']:
        if state[key].shape!=(n,): raise ValueError(f'{path}: invalid {key} shape')
    if state['active_mask'].dtype!=bool: raise ValueError(f'{path}: active_mask must be Boolean')
    r,active=normalize_mechanical_interactions(p)
    np.testing.assert_array_equal(active,state['active_mask'])
    np.testing.assert_allclose(r,state['R'],rtol=1e-10,atol=1e-12)
    state['normalization_max_abs_error']=float(np.max(abs(r-state['R'])))
    m=state['mesh']; nx,ny=int(m['nelx']),int(m['nely'])
    np.testing.assert_allclose(state['element_x'],np.tile((np.arange(nx)+.5)*m['dx'],ny),rtol=0,atol=1e-12)
    np.testing.assert_allclose(state['element_y'],np.repeat((np.arange(ny)+.5)*m['dy'],nx),rtol=0,atol=1e-12)
    if nx*ny!=n: raise ValueError('Metadata element count mismatch')
    return state


def baseline_vectors(nx,ny):
    return {
        'random':np.column_stack(_vector_batch_random(nx*ny,20,SEEDS['random'])),
        'smooth':np.column_stack(_vector_batch_smooth(nx,ny,20,SEEDS['smooth'])),
        'localized':np.column_stack(_vector_batch_localized(nx,ny,20,SEEDS['localized'])),
    }


def restricted_vectors(full_vectors,indices):
    result={}
    for family,vectors in full_vectors.items():
        restricted=vectors[indices]
        norms=np.linalg.norm(restricted,axis=0)
        keep=norms>1e-14
        if keep.any(): result[family]=restricted[:,keep]/norms[keep]
    return result


def benchmark_action(action,vector,repeats=7):
    """Warm action once; median single-vector application time, including temporaries."""
    action(vector)
    timings=[]
    for _ in range(repeats):
        start=perf_counter(); action(vector); timings.append(perf_counter()-start)
    return float(np.median(timings))


def evaluate_approximations(matrix,spectrum,indices,state,vectors,base):
    """No dense P_k constructed; spatial actions use full CSR representation."""
    n=matrix.shape[0]; xy=np.column_stack((state['element_x'][indices],state['element_y'][indices]))
    t0=perf_counter(); distances=cdist(xy,xy); distance_seconds=perf_counter()-t0
    reference={name:matrix@v for name,v in vectors.items()}
    probe=vectors['random'][:,0]
    dense_apply=benchmark_action(lambda v:matrix@v,probe)
    norm=float(np.linalg.norm(matrix,'fro'))
    spatial={}; rank_sources={}
    def add_rank(k,reason):
        if 0<=k<=n: rank_sources.setdefault(k,set()).add(reason)
    for k in SELECTED_RANKS:
        if k<=n: add_rank(k,'selected_rank')
    add_rank(n,'full_rank_validation')
    for target in ENERGY_TARGETS: add_rank(rank_for_energy(spectrum.eigenvalues,target),f'energy_{target:g}')
    comparisons=[]; action_rows=[]; detail_rows=[]
    for cutoff in (2.,4.,6.):
        t0=perf_counter()
        sparse=spatial_truncation(matrix,xy[:,0],xy[:,1],cutoff,distances)
        assembly_seconds=perf_counter()-t0
        costs=approximation_costs(n,sparse=sparse)
        discarded=np.where(distances>cutoff,matrix,0.)
        error=float(np.linalg.norm(discarded,'fro')/norm) if norm else 0.
        del discarded
        storage_rank=min(n,costs['storage_bytes']//(8*(n+1)))
        flop_rank=min(n,costs['application_flops']//(4*n+1))
        add_rank(int(storage_rank),f'storage_budget_r{cutoff:g}')
        add_rank(int(flop_rank),f'application_budget_r{cutoff:g}')
        apply_seconds=benchmark_action(lambda v:sparse@v,probe)
        min_eigenvalue=np.nan; psd_diagnostic_seconds=0.
        if base['domain']=='48x24' and base['variant']=='P_full':
            t0=perf_counter()
            dense=sparse.toarray(); dense=(dense+dense.T)*.5
            min_eigenvalue=float(eigvalsh(dense,subset_by_index=[0,0],check_finite=False)[0])
            psd_diagnostic_seconds=perf_counter()-t0
            del dense
        record={**base,'method':'spatial','rank':None,'cutoff':cutoff,
            'relative_frobenius_error':error,**costs,'dense_storage_bytes':8*n*n,
            'shared_distance_construction_seconds':distance_seconds,
            'construction_seconds':distance_seconds+assembly_seconds,
            'factor_extraction_seconds':0.,'eigendecomposition_seconds':0.,
            'single_vector_application_seconds':apply_seconds,'dense_application_seconds':dense_apply,
            'spatial_psd_guaranteed':False,'spatial_min_eigenvalue':min_eigenvalue,
            'spatial_psd_diagnostic_seconds':psd_diagnostic_seconds,
            'storage_matched_lowrank_rank':int(storage_rank),'flop_matched_lowrank_rank':int(flop_rank)}
        comparisons.append(record)
        spatial[cutoff]=(sparse,record)
        for family,v in vectors.items():
            summary,details=action_error_summary(reference[family],sparse@v)
            action_rows.append({**base,'method':'spatial','rank':None,'cutoff':cutoff,'family':family,**summary})
            for i in range(v.shape[1]):
                detail_rows.append({**base,'method':'spatial','rank':None,'cutoff':cutoff,'family':family,'vector_index':i,
                    **{key:values[i] for key,values in details.items()}})
    del spatial,distances  # CSR matrices no longer needed for spectral evaluation
    for k,reasons in sorted(rank_sources.items()):
        t0=perf_counter()
        q=np.ascontiguousarray(spectrum.eigenvectors[:,:k])
        eigenvalues=spectrum.eigenvalues[:k].copy()
        factor_seconds=perf_counter()-t0
        def apply(v):
            weights=eigenvalues if v.ndim==1 else eigenvalues[:,None]
            return q@(weights*(q.T@v))
        costs=approximation_costs(n,rank=k)
        apply_seconds=benchmark_action(apply,probe)
        comparisons.append({**base,'method':'lowrank','rank':k,'cutoff':None,
            'rank_selection_reasons':';'.join(sorted(reasons)),
            'relative_frobenius_error':relative_frobenius_error(spectrum.eigenvalues,k),
            **costs,'dense_storage_bytes':8*n*n,
            'eigendecomposition_seconds':spectrum.elapsed_seconds,'factor_extraction_seconds':factor_seconds,
            'construction_seconds':spectrum.elapsed_seconds+factor_seconds,
            'single_vector_application_seconds':apply_seconds,'dense_application_seconds':dense_apply,
            'spatial_psd_guaranteed':None,'spectral_psd_guaranteed':True})
        for family,v in vectors.items():
            summary,details=action_error_summary(reference[family],apply(v))
            action_rows.append({**base,'method':'lowrank','rank':k,'cutoff':None,'family':family,**summary})
            for i in range(v.shape[1]):
                detail_rows.append({**base,'method':'lowrank','rank':k,'cutoff':None,'family':family,'vector_index':i,
                    **{key:values[i] for key,values in details.items()}})
    return comparisons,action_rows,detail_rows


def save_figure(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight'); plt.close(fig)


def make_figures(summary_rows,rank_rows,action_rows,evolution_rows,spectra,out):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    selected=['uniform_state','middle_state_15','late_state_29']
    variants=['P_full','P_active','R_active','P_common','R_common']
    # Separate domain/variant panels keep active-space and full-space spectra distinct.
    for domain in ['12x6','24x12','48x24']:
        for variant in variants:
            fig,axes=plt.subplots(1,4,figsize=(18,4),layout='constrained')
            for data in spectra.values():
                if data['domain']!=domain or data['variant']!=variant:continue
                values=np.array(data['eigenvalues']); largest=values[0]; k=np.arange(1,len(values)+1)
                axes[0].plot(k,values/largest,label=data['state'])
                axes[1].semilogy(k,values/largest,label=data['state'])
                axes[2].plot(k,cumulative_squared_energy(values),label=data['state'])
                errors=[relative_frobenius_error(values,int(rank)) for rank in k]
                visible=np.asarray(errors)>0
                axes[3].semilogy(k[visible],np.asarray(errors)[visible],label=data['state'])
            for ax in axes: ax.set_xlabel('Mode index / rank'); ax.grid(alpha=.2)
            axes[0].set_ylabel('Eigenvalue / largest');axes[1].set_ylabel('Eigenvalue / largest (log)')
            axes[2].set_ylabel('Cumulative squared spectral energy');axes[3].set_ylabel('Relative Frobenius error')
            axes[0].legend(fontsize=8);fig.suptitle(domain+' '+variant)
            save_figure(fig,out,'spectra_'+domain+('' if variant=='P_full' else '_'+variant))
    for domain in ['12x6','24x12','48x24']:
        names=sorted({r['state'] for r in summary_rows if r['domain']==domain})
        # All available secondary states; chronological selected states in primary case.
        if domain=='48x24':names=selected
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for variant in variants:
            rows=[next(r for r in summary_rows if r['domain']==domain and r['state']==state and r['variant']==variant) for state in names]
            for ax,key,label in zip(axes,['entropy_effective_rank','stable_rank','participation_ratio'],['Entropy effective rank','Stable rank','Participation rank']):
                ax.plot(range(len(names)),[r[key] for r in rows],marker='o',label=variant)
                ax.set_ylabel(label);ax.set_xticks(range(len(names)),names,rotation=20,fontsize=8);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle(domain+' effective ranks')
        save_figure(fig,out,'effective_rank_evolution'+('' if domain=='48x24' else '_'+domain))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for state in selected:
        for variant in ['P_full','R_active']:
            values=np.array(spectra[f'48x24/{state}/{variant}']['eigenvalues'])
            k=np.arange(1,len(values)+1)
            axes[0].semilogy(k,values/values[0],label=state+' '+variant)
            errors=[relative_frobenius_error(values,int(rank)) for rank in k]
            visible=np.asarray(errors)>0
            axes[1].semilogy(k[visible],np.asarray(errors)[visible],label=state+' '+variant)
    axes[0].set(xlabel='Mode index',ylabel='Relative eigenvalue');axes[1].set(xlabel='Rank k',ylabel='Relative Frobenius error')
    axes[0].legend(fontsize=7);save_figure(fig,out,'P_vs_R_rank_error')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,state in zip(axes,selected):
        for family in FAMILIES:
            rows=[r for r in action_rows if r['domain']=='48x24' and r['state']==state and r['variant']=='P_full' and r['family']==family and r['method']=='lowrank' and r['rank']<r['dimension']]
            ax.loglog([max(r['rank'],.5) for r in rows],[max(r['median_relative_error'],1e-16) for r in rows],marker='.',label=family)
        ax.set(xlabel='Spectral rank k',ylabel='Median relative action error',title=state)
        ax.grid(alpha=.2)
    axes[0].legend();save_figure(fig,out,'lowrank_operator_actions')
    for cost_key,cost_label,name in [('storage_bytes','Representation bytes','storage_operator_comparison'),('application_flops','Estimated single-vector FLOPs','flops_operator_comparison')]:
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for ax,state in zip(axes,selected):
            for family in FAMILIES:
                for method,style in [('lowrank','-o'),('spatial','--s')]:
                    actions=[r for r in action_rows if r['domain']=='48x24' and r['state']==state and r['variant']=='P_full' and r['family']==family and r['method']==method and (method=='spatial' or r['rank']<r['dimension'])]
                    costs=[]
                    for r in actions:
                        costs.append(next(c[cost_key] for c in rank_rows if c['domain']==r['domain'] and c['state']==state and c['variant']=='P_full' and c['method']==method and c.get('rank')==r['rank'] and c.get('cutoff')==r['cutoff']))
                    color={'random':'#2878b5','smooth':'#e07a17','localized':'#39944b'}[family]
                    ax.loglog(costs,[max(r['median_relative_error'],1e-16) for r in actions],style,color=color,markersize=3,label=family+' '+method)
            ax.set(xlabel=cost_label,ylabel='Median relative action error',title=state);ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);save_figure(fig,out,name)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for variant in ['P_common','R_common']:
        for transition in ['uniform_to_middle','middle_to_late','uniform_to_late']:
            rows=[r for r in evolution_rows if r['domain']=='48x24' and r['variant']==variant and r['transition']==transition]
            axes[0].plot([r['rank'] for r in rows],[r['max_angle_degrees'] for r in rows],marker='o',label=variant+' '+transition)
            axes[1].plot([r['rank'] for r in rows],[r['normalized_projector_distance'] for r in rows],marker='o')
    axes[0].set(xlabel='Eigenspace dimension',ylabel='Maximum principal angle (degrees)');axes[1].set(xlabel='Eigenspace dimension',ylabel='Normalized projector distance')
    axes[0].legend(fontsize=7);save_figure(fig,out,'dominant_eigenspace_evolution')


def run_analysis(output_root=None):
    paths=discover_inputs()  # fail before writing output if essential input is absent
    protected=fingerprint_protected()
    out=Path(output_root) if output_root else ROOT/'m2_spectral'
    out.mkdir(parents=True,exist_ok=True)
    existing=[p for p in out.iterdir() if p.name!='archive']
    if existing:
        archive=out/'archive'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');archive.mkdir(parents=True)
        for p in existing: shutil.move(str(p),str(archive/p.name))
    rows=[];rank_rows=[];actions=[];costs=[];details=[];evolution=[];spectra={};input_records=[]
    elapsed_start=perf_counter()
    for domain,state_paths in paths.items():
        masks=[]; density_references={}; meshes=[]
        reference_forces=None; reference_fixed=None
        for path in state_paths:
            with np.load(path,allow_pickle=False) as data:
                masks.append(data['active_mask']);density_references[path.stem]=data['rho']
                if 'forces' not in data or 'fixed_dofs' not in data:
                    raise ValueError(f'{path}: cannot establish cross-state load/support identity; no regeneration attempted')
                if reference_forces is None:
                    reference_forces=data['forces'];reference_fixed=np.sort(data['fixed_dofs'])
                else:
                    np.testing.assert_array_equal(data['forces'],reference_forces)
                    np.testing.assert_array_equal(np.sort(data['fixed_dofs']),reference_fixed)
            meshes.append(json.loads(path.with_suffix('.json').read_text())['mesh'])
        if any(m!=meshes[0] for m in meshes):raise ValueError('Cross-state mesh geometry differs')
        common=np.logical_and.reduce(masks)
        m=meshes[0];nx,ny=int(m['nelx']),int(m['nely']);full_vectors=baseline_vectors(nx,ny)
        common_spectra={}
        for path in state_paths:
            state=load_state(path);name=state['name']
            print(f'{domain} {name}: full={nx*ny}, active={int(state["active_mask"].sum())}, common={int(common.sum())}',flush=True)
            input_records.append(dict(domain=domain,state=name,file=str(path.relative_to(ROOT)),
                sha256=protected[str(path.relative_to(ROOT))],mesh=m,active_count=int(state['active_mask'].sum()),
                common_count=int(common.sum()),common_selected_states=[p.stem for p in state_paths],
                normalization_max_abs_error=state['normalization_max_abs_error']))
            mechanical=[]
            for other,rho in density_references.items():
                if other in ('uniform_state','middle_state_10','middle_state_15','late_state_19','late_state_29'):
                    delta=rho-state['rho']
                    if np.linalg.norm(delta)>1e-14:mechanical.append(delta/np.linalg.norm(delta))
            full_state_vectors=dict(full_vectors)
            if mechanical:full_state_vectors['observed_density_evolution']=np.column_stack(mechanical)
            if 'g_rho' in state and np.linalg.norm(state['g_rho'])>1e-14:
                full_state_vectors['saved_compliance_gradient']=state['g_rho'][:,None]/np.linalg.norm(state['g_rho'])
            cache={}
            for variant,key,mask in [('P_full','P',np.ones(nx*ny,dtype=bool)),('P_active','P',state['active_mask']),
                                     ('R_active','R',state['active_mask']),('P_common','P',common),('R_common','R',common)]:
                matrix,indices=active_principal_submatrix(state[key],mask)
                cache_key=(key,indices.tobytes())
                reused=cache_key in cache
                spectrum=cache[cache_key] if reused else symmetric_psd_eigensystem(matrix)
                if not reused:cache[cache_key]=spectrum
                base=dict(domain=domain,state=name,variant=variant,dimension=len(indices))
                row={**base,**spectral_statistics(spectrum.eigenvalues),**spectrum.validation,
                    'active_count':int(state['active_mask'].sum()),'common_count':int(common.sum()),
                    'dense_storage_bytes':matrix.nbytes,'eigendecomposition_seconds':spectrum.elapsed_seconds,
                    'eigendecomposition_reused':reused,'normalization_max_abs_error':state['normalization_max_abs_error']}
                rows.append(row)
                spectra[f'{domain}/{name}/{variant}']={**base,'eigenvalues':spectrum.eigenvalues.tolist()}
                for target in ENERGY_TARGETS:
                    k=rank_for_energy(spectrum.eigenvalues,target)
                    rank_rows.append({**base,'selection':'energy_target','energy_target':target,'rank':k,
                        'rank_fraction':k/max(1,len(indices)),'relative_frobenius_error':relative_frobenius_error(spectrum.eigenvalues,k)})
                for k in SELECTED_RANKS:
                    if k<=len(indices):rank_rows.append({**base,'selection':'selected_rank','rank':k,'rank_fraction':k/max(1,len(indices)),
                        'squared_energy':float(cumulative_squared_energy(spectrum.eigenvalues)[k-1]),
                        'relative_frobenius_error':relative_frobenius_error(spectrum.eigenvalues,k)})
                if variant.endswith('common'):common_spectra[name,variant]=spectrum
                if key=='P':
                    vectors=restricted_vectors(full_state_vectors,indices)
                    c,a,d=evaluate_approximations(matrix,spectrum,indices,state,vectors,base)
                    costs.extend(c);actions.extend(a);details.extend(d)
            del cache,state
        names=['uniform_state','middle_state_10' if domain=='12x6' else 'middle_state_15',
               'late_state_19' if domain=='12x6' else 'late_state_29']
        for ia,ib,t in [(0,1,'uniform_to_middle'),(1,2,'middle_to_late'),(0,2,'uniform_to_late')]:
            for variant in ['P_common','R_common']:
                for k in [4,8,16,32]:
                    if k<=int(common.sum()):
                        info=leading_eigenspace_comparison(common_spectra[names[ia],variant],common_spectra[names[ib],variant],k)
                        angles=info.pop('angles_degrees')
                        evolution.append(dict(domain=domain,variant=variant,transition=t,dimension=int(common.sum()),
                            **info,angles_degrees=';'.join(f'{v:.12g}' for v in angles)))
        del common_spectra
    write_csv(out/'summary.csv',rows)
    write_csv(out/'spectral_rank_summary.csv',rank_rows)
    write_csv(out/'operator_action_summary.csv',actions)
    write_csv(out/'operator_action_vectors.csv',details)
    write_csv(out/'lowrank_vs_spatial.csv',costs)
    write_csv(out/'eigenspace_evolution.csv',evolution)
    (out/'spectra.json').write_text(json.dumps(serial(spectra),indent=2,allow_nan=False)+'\n')
    make_figures(rows,costs,actions,evolution,spectra,out)
    unchanged={path:sha256(ROOT/path)==digest for path,digest in protected.items()}
    if not all(unchanged.values()):raise RuntimeError('Protected M1/M2/core files changed')
    memory_n=1152;future_n=96*48
    summary=dict(schema_version=1,input_states=input_records,protected_sha256=protected,protected_files_unchanged=unchanged,
        spectra_file='spectra.json',spectral_metrics=rows,energy_ranks=rank_rows,eigenspace_evolution=evolution,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'analyze_m2_spectral.py',ROOT/'topopt_utils/fea/spectral_interactions.py',ROOT/'tests/test_spectral_interactions.py']},
        environment={name:version(name) for name in ['numpy','scipy','torch','matplotlib','pytest','pypardiso']},
        thread_environment={name:os.environ.get(name) for name in ['MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},
        seeds=SEEDS,vector_definitions='M2.1 exact helper functions; unit L2, 20 per standard family; restrict and renormalize on principal subspaces. Smooth probes differ only by tiny noise. Additional density directions/gradients come from saved arrays; no feasible-direction projection is implied.',
        definitions=dict(variants='P_full; per-state P_active/R_active; fixed all-selected-state-intersection P_common/R_common',
            stable_rank='sum(lambda^2)/lambda_max^2',entropy_rank='exp(-sum(p log p)), p=lambda/sum(lambda)',
            participation_rank='sum(lambda)^2/sum(lambda^2)',energy='cumulative squared eigenvalues, not action accuracy',
            psd_tolerance='100*N*float64_eps*max(abs(raw_eigenvalues)); only accepted tiny negative values become zero',
            symmetry_tolerance='relative Frobenius asymmetry <=1e-10, then symmetrize accepted roundoff',
            condition='full condition undefined when smallest nonnegative eigenvalue <= PSD tolerance; effective condition excludes eigenvalues <= selected relative threshold',
            action_denominator='relative error undefined for ||Pv||<1e-12 or zero; absolute norms are retained',
            lowrank_storage='8*(N*k+k) bytes, eigenvectors plus eigenvalues; estimated 4*N*k+k FLOPs/vector',
            spatial_storage='actual full CSR bytes, both triangles and diagonal retained; estimated 2*nnz FLOPs/vector',
            timing='median of 7 warm single-vector applications; construction includes dense full symmetric eigendecomposition or distances+CSR assembly; figures/I/O excluded from construction; PSD diagnostics charged separately',
            matched_budgets='largest feasible integer k under each spatial representation-byte or FLOP budget, evaluated on identical vectors',
            subspaces='principal angles on fixed common element indices; projector norm from basis overlap, no direct eigenvector-sign comparison'),
        scalability=dict(primary_dimension=memory_n,primary_dense_bytes=8*memory_n**2,
            primary_eigh_workspace_estimate_bytes=8*(4*memory_n**2+6*memory_n),
            future_96x48_dimension=future_n,future_dense_bytes=8*future_n**2,
            future_eigh_workspace_estimate_bytes=8*(4*future_n**2+6*future_n),
            future_dense_cubic_runtime_factor=(future_n/memory_n)**3,
            note='Workspace estimates cover symmetrization/eigenvectors/LAPACK, not all resident P/R inputs or BLAS-native allocations. No 96x48 solve/decomposition attempted.'),
        limitations=['Snapshots originate from a different optimization geometry/load than the evaluated interaction matrices; unchanged M2.2 provenance limitation.',
            'Few mechanical problems; cross-size cases are fixed-spacing domain scaling, not mesh refinement.',
            'Dense EVD is used for these verified matrices; partial matrix-free P action is a future path, not implemented.',
            'P is not the full Hessian H=2P-diag(D), nor the filtered design-space operator.',
            'After-state density directions are retrospective; random/smooth/localized probes are algebraic tests, not physical loading claims.',
            'Spatial truncation need not be PSD; actual minimum eigenvalue checked only for primary full-P spatial approximations.',
            'No neural networks, M2.4, HPC jobs, exact-P regeneration, commits or pushes.'],
        analysis_runtime_seconds=perf_counter()-elapsed_start,
        unique_eigendecomposition_count=sum(not r['eigendecomposition_reused'] for r in rows),
        total_unique_eigendecomposition_seconds=sum(r['eigendecomposition_seconds'] for r in rows if not r['eigendecomposition_reused']))
    write_report(summary,rows,rank_rows,actions,costs,evolution,out)
    (out/'summary.json').write_text(json.dumps(serial(summary),indent=2,allow_nan=False)+'\n')
    print(f'Completed {len(input_records)} states, {len(rows)} variants; protected {len(protected)} files unchanged.',flush=True)
    return summary

def matched_budget_comparisons(costs,actions):
    cost_index={(r['domain'],r['state'],r['variant'],r['method'],r.get('rank'),r.get('cutoff')):r for r in costs}
    action_index={(r['domain'],r['state'],r['variant'],r['method'],r.get('rank'),r.get('cutoff'),r['family']):r for r in actions}
    rows=[]
    for spatial in costs:
        if spatial['method']!='spatial':continue
        prefix=(spatial['domain'],spatial['state'],spatial['variant'])
        for budget,key in [('storage','storage_matched_lowrank_rank'),('application_flops','flop_matched_lowrank_rank')]:
            k=spatial[key];low=cost_index[(*prefix,'lowrank',k,None)]
            families=[r['family'] for r in actions if (r['domain'],r['state'],r['variant'])==prefix and r['method']=='spatial' and r['cutoff']==spatial['cutoff']]
            for family in families:
                la=action_index[(*prefix,'lowrank',k,None,family)]
                sa=action_index[(*prefix,'spatial',None,spatial['cutoff'],family)]
                rows.append(dict(domain=prefix[0],state=prefix[1],variant=prefix[2],budget=budget,
                    cutoff=spatial['cutoff'],rank=k,family=family,
                    lowrank_storage_bytes=low['storage_bytes'],spatial_storage_bytes=spatial['storage_bytes'],
                    storage_ratio=low['storage_bytes']/spatial['storage_bytes'],
                    lowrank_application_flops=low['application_flops'],spatial_application_flops=spatial['application_flops'],
                    application_flops_ratio=low['application_flops']/spatial['application_flops'],
                    lowrank_frobenius_error=low['relative_frobenius_error'],spatial_frobenius_error=spatial['relative_frobenius_error'],
                    lowrank_median_action_error=la['median_relative_error'],spatial_median_action_error=sa['median_relative_error'],
                    lowrank_mean_action_error=la['mean_relative_error'],spatial_mean_action_error=sa['mean_relative_error'],
                    lowrank_max_action_error=la['max_relative_error'],spatial_max_action_error=sa['max_relative_error'],
                    lowrank_application_seconds=low['single_vector_application_seconds'],spatial_application_seconds=spatial['single_vector_application_seconds'],
                    lowrank_construction_seconds=low['construction_seconds'],spatial_construction_seconds=spatial['construction_seconds']))
    return rows


def write_report(summary,metrics,ranks,actions,costs,evolution,out):
    states=['uniform_state','middle_state_15','late_state_29']
    def metric(state,variant):return next(r for r in metrics if r['domain']=='48x24' and r['state']==state and r['variant']==variant)
    def energy(state,variant,target):return next(r for r in ranks if r['domain']=='48x24' and r['state']==state and r['variant']==variant and r['selection']=='energy_target' and r['energy_target']==target)
    comparisons=matched_budget_comparisons(costs,actions)
    write_csv(out/'matched_budget_summary.csv',comparisons)
    primary=[r for r in comparisons if r['domain']=='48x24' and r['variant']=='P_full' and r['family'] in FAMILIES]
    summary['matched_budget_file']='matched_budget_summary.csv'
    summary['primary_budget_comparisons']=primary
    summary['scientific_report']='M2_3_SPECTRAL_REPORT.md'
    lines=['# M2.3 — Spectral structure and low-rank mechanical interactions','',
        'The analysis uses all 13 exact saved interaction states: five each on 12×6 and 24×12, and three on 48×24. Each state has full P, per-state active P/R, and fixed common-active P/R analyses (65 variants). R inactive placeholders are excluded. The fixed set is the intersection across all available states in each domain. Spectral conclusions apply to the saved evaluated operators, not automatically to the original optimizer, its filtered design variables, or the full SIMP Hessian.', '',
        '## Input validation and numerical conventions','',
        'Every P is square, finite, and symmetric within relative Frobenius tolerance 1e−10. After reporting that asymmetry, the accepted matrix is symmetrized and solved with SciPy symmetric `eigh`, driver `evd`. Each matrix variant is checked for PSD. The scale-aware tolerance is 100·N·float64_eps·max(abs(raw eigenvalues)). Substantial negative eigenvalues fail the analysis; only negative roundoff within tolerance is set to zero. Positive small eigenvalues are retained. Raw minimum/maximum eigenvalues, negative counts and magnitudes, tolerances, normalization errors and active counts are in `summary.csv`. Saved R and active masks are independently checked against the M1 normalization rule.', '',
        '| State (48×24 full P) | N | Active | Raw minimum eigenvalue | Maximum eigenvalue | PSD tolerance | Tiny negatives |','|---|---:|---:|---:|---:|---:|---:|']
    for state in states:
        r=metric(state,'P_full')
        lines.append(f"| {state} | {r['dimension']} | {r['active_count']} | {r['raw_smallest_eigenvalue']:.6g} | {r['largest_eigenvalue']:.6g} | {r['psd_tolerance']:.3g} | {r['roundoff_negative_count']} |")
    lines+=['','Numerical ranks use relative thresholds 1e−12,1e−10,1e−8,1e−6,1e−4. Full condition numbers are undefined where the smallest eigenvalue is below the PSD roundoff tolerance; thresholded effective condition numbers exclude small eigenvalues explicitly. Null JSON values represent undefined quantities, not zero.', '',
        '## 1–2. Is P approximately low-rank, and how many modes are needed?', '',
        'Compactness is judged by both mode counts and errors, rather than stable rank alone. Required ranks for the primary full operator:', '',
        '| State | Target squared energy | k | k/N | Relative Frobenius error |','|---|---:|---:|---:|---:|']
    for state in states:
        for target in ENERGY_TARGETS:
            r=energy(state,'P_full',target)
            lines.append(f"| {state} | {100*target:g}% | {r['rank']} | {r['rank_fraction']:.3%} | {r['relative_frobenius_error']:.6g} |")
    lines+=['','P has a strongly dominant spectral head, but a substantial tail. It is approximately low-rank for coarse Frobenius accuracy, especially in the uniform state; strict accuracy is not compact (about 29% of modes for 99.9% energy and 39–53% for 99.99%). The operator-action tests below contradict treating the dominant head as an accurate approximation for every perturbation family.', '', 'For example, 99% squared spectral energy allows a Frobenius error as large as 10%, and does **not** guarantee 1% action error. A vector concentrated in a discarded eigenspace can have 100% relative action error even when total squared energy retention is high. Spectral truncation is globally optimal among rank-k approximations in Frobenius norm for the accepted symmetric operator, not necessarily optimal for a particular vector family.', '',
        'All variants, all secondary states, energy targets 90/95/99/99.9/99.99%, and selected ranks 1,2,4,8,16,32,64,128,256 are in `spectral_rank_summary.csv`. Ranks above dimension are omitted; each exact spectrum is retained in `spectra.json`.', '',
        '## 3–4. Effective rank evolution and P versus R','',
        'Stable rank is sum(lambda²)/lambda_max² and emphasizes the strongest mode. Entropy rank uses trace-normalized eigenvalues and is sensitive to distributed weak modes. Participation rank is trace²/sum(lambda²). These measures quantify different concentration properties and are not interchangeable.', '',
        '| State | Variant | N | Stable rank | Entropy rank | Participation rank | k for 99% squared energy |','|---|---|---:|---:|---:|---:|---:|']
    for state in states:
        for variant in ['P_full','P_active','P_common','R_active','R_common']:
            r=metric(state,variant);e=energy(state,variant,.99)
            lines.append(f"| {state} | {variant} | {r['dimension']} | {r['stable_rank']:.5g} | {r['entropy_effective_rank']:.5g} | {r['participation_ratio']:.5g} | {e['rank']} |")
    uf=metric(states[0],'P_full');mi=metric(states[1],'P_full');la=metric(states[2],'P_full')
    summary['primary_spectral_evolution']={key:[metric(s,'P_common')[key] for s in states] for key in ['stable_rank','entropy_effective_rank','participation_ratio']}
    lines+=['',f"Full-P entropy rank changes {uf['entropy_effective_rank']:.5g} → {mi['entropy_effective_rank']:.5g} → {la['entropy_effective_rank']:.5g}; stable rank changes {uf['stable_rank']:.5g} → {mi['stable_rank']:.5g} → {la['stable_rank']:.5g}. The table's common-set rows distinguish actual spectral redistribution on fixed coordinates from the changing active dimension.", '',
        'R requires about 69% of active modes for 99% squared energy in middle/late, compared with about 20% of active P modes. The normalized operator is not compactly low-rank at that accuracy. P weights absolute mechanical sensitivity-response scale; R divides out self-coupling and can spread weight over many more directions. Removing low-self-coupling elements may barely change P energy while changing the dimension and relative-rank fraction; it is not legitimate to insert inactive R placeholders into a full-space spectrum. Compare mode counts, absolute effective ranks, and k/N together. Secondary-domain full/active/common metrics are included in the machine-readable tables; different domains at fixed element spacing are not mesh refinements.', '',
        '## 5. Dominant eigenspace stability','',
        'Principal angles compare leading common-set eigenspaces, using k=4,8,16,32. Projector distance is computed from basis overlaps: sqrt(2k−2||Q_a.T Q_b||_F²), normalized by sqrt(2k). No eigenvector signs are compared. The relative kth-to-(k+1)th eigengap, divided by lambda_max, is recorded; gaps ≤1e−6 flag boundaries where individual truncated subspaces may not be unique under perturbation.', '',
        '| Variant (48×24 common set) | Transition | k | Max angle (degrees) | Normalized projector distance | Near-degenerate boundary in either state? |','|---|---|---:|---:|---:|---|']
    for r in evolution:
        if r['domain']=='48x24':
            lines.append(f"| {r['variant']} | {r['transition']} | {r['rank']} | {r['max_angle_degrees']:.4f} | {r['normalized_projector_distance']:.5g} | {r['near_degenerate_boundary_a'] or r['near_degenerate_boundary_b']} |")
    lines+=['','The primary P leading subspaces reorganize strongly from uniform to middle. Middle-to-late is much more stable for k=4,8,32, but k=16 is an exception with a maximum angle near 45.6 degrees; it must not be hidden by an average stability claim. R has small middle-to-late principal angles across these dimensions.', '', 'Spectral complexity and spectral direction are different questions: a matrix can keep similar effective rank while its dominant subspace rotates substantially. Interpret large angles at a nearly repeated-eigenvalue boundary cautiously. Full angle lists, gaps, mean angles and projection distances for both secondary domains are in `eigenspace_evolution.csv`.', '',
        '## 6. Operator-action accuracy','',
        'Twenty unit-L2 random, smooth and localized probes per family are generated with the existing M2.1 functions and seeds 10,14,16 (the primary M2.1 seeds). Within each state/variant exactly the same vectors are used for spectral and spatial approximations. Principal-submatrix probes restrict the full-grid vectors and renormalize them. Smooth probes are almost the same analytic field plus tiny noise; localized centers vary slightly. Neither family is a broad independent physical ensemble.', '',
        'Additional probes use differences between saved density snapshots and saved compliance gradients where present. These are retrospective density/sensitivity directions from verified quantities, not claimed physical load cases or feasible projected design updates. Reference action norms below 1e−12 are flagged; absolute errors remain available and relative errors are left undefined.', '',
        'Primary full-P action errors at the minimum rank capturing 99% squared energy:', '',
        '| State | k | Family | Mean relative error | Median | p90 | Maximum |','|---|---:|---|---:|---:|---:|---:|']
    for state in states:
        k=energy(state,'P_full',.99)['rank']
        for family in FAMILIES:
            r=next(r for r in actions if r['domain']=='48x24' and r['state']==state and r['variant']=='P_full' and r['method']=='lowrank' and r['rank']==k and r['family']==family)
            lines.append(f"| {state} | {k} | {family} | {r['mean_relative_error']:.5g} | {r['median_relative_error']:.5g} | {r['p90_relative_error']:.5g} | {r['max_relative_error']:.5g} |")
    lines+=['','At the 99%-energy ranks, localized median errors are approximately 1.00, 0.846 and 0.846 for uniform/middle/late. Random and smooth probes have different errors, so there is no single action-accuracy interpretation of retained energy.', '', 'Selected-rank and energy-rank action summaries (including mean/median/p90/max absolute norms) are in `operator_action_summary.csv`; every individual probe error and denominator is retained in `operator_action_vectors.csv`. Vector-family action accuracy must be checked directly rather than inferred from energy retention.', '']
    action_rank_rows=[]
    lines+=['','Smallest **tested** rank attaining the indicated median action tolerance (the evaluated rank grid is not exhaustive):', '',
        '| State | Family | Median action tolerance | Smallest tested k | k/N |','|---|---|---:|---:|---:|']
    for state in states:
        for family in FAMILIES:
            tested=[r for r in actions if r['domain']=='48x24' and r['state']==state and r['variant']=='P_full' and r['method']=='lowrank' and r['family']==family]
            for tolerance in [.1,.01]:
                eligible=[r['rank'] for r in tested if np.isfinite(r['median_relative_error']) and r['median_relative_error']<=tolerance]
                k=min(eligible) if eligible else None
                action_rank_rows.append(dict(state=state,family=family,median_action_tolerance=tolerance,smallest_tested_rank=k,rank_fraction=k/1152 if k is not None else None))
                lines.append(f'| {state} | {family} | {tolerance:g} | {k} | '+(f'{k/1152:.3%}' if k is not None else 'not reached')+' |')
    summary['primary_action_rank_thresholds']=action_rank_rows
    lines+=['','Full rank is included as an EVD/action consistency control and retained in the CSV tables, but omitted from approximation-action plots so machine-precision identity errors do not obscure approximation differences. Exact zero Frobenius errors are omitted from log error curves. Full rank is not a compact approximation, and a large gap between tested ranks prevents claiming an exact minimal action rank.', '',
        '## 7. Low rank versus spatial truncation at matched budgets','',
        'Spatial cutoffs are 2,4,6 physical units, with diagonal terms retained. Actual nonzero CSR data/indices/indptr storage is counted, storing both symmetric entries rather than using symmetry compression. Low rank stores N×k eigenvector entries plus k eigenvalues (8k(N+1) bytes). Estimated applications cost 4Nk+k versus 2·nnz multiply/add FLOPs per vector. Storage-matched and FLOP-matched comparisons choose the largest integer k within the corresponding spatial budget. A storage match need not be an application-cost match.', '',
        'Localized-probe median relative errors and cost ratios for the primary full operator:', '',
        '| State | Budget matched | Radius | k | LR/spatial bytes | LR/spatial FLOPs | LR median error | Spatial median error |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in primary:
        if r['family']=='localized':
            lines.append(f"| {r['state']} | {r['budget']} | {r['cutoff']:g} | {r['rank']} | {r['storage_ratio']:.3f} | {r['application_flops_ratio']:.3f} | {r['lowrank_median_action_error']:.5g} | {r['spatial_median_action_error']:.5g} |")
    for budget in ['storage','application_flops']:
        valid=[r for r in primary if r['budget']==budget and np.isfinite(r['lowrank_median_action_error']) and np.isfinite(r['spatial_median_action_error'])]
        wins=sum(r['lowrank_median_action_error']<r['spatial_median_action_error'] for r in valid)
        lines.append(f'\nAt {budget} budgets, low rank has a lower median action error in {wins}/{len(valid)} primary state/cutoff/standard-family comparisons. This count describes only these probe families and budget choices, not a general superiority theorem.')
    lines+=['','Construction and measured application costs:', '',
        '| State | Full P eigendecomposition seconds | Dense single-vector application seconds | Radius-2 construction seconds | Radius-2 CSR application seconds | Radius-2 smallest eigenvalue |','|---|---:|---:|---:|---:|---:|']
    for state in states:
        r=metric(state,'P_full');s=next(r for r in costs if r['domain']=='48x24' and r['state']==state and r['variant']=='P_full' and r['method']=='spatial' and r['cutoff']==2)
        lines.append(f"| {state} | {r['eigendecomposition_seconds']:.5g} | {s['dense_application_seconds']:.5g} | {s['construction_seconds']:.5g} | {s['single_vector_application_seconds']:.5g} | {s['spatial_min_eigenvalue']:.5g} |")
    lines+=['','Eigendecomposition cost is shared across k values and charged once per matrix, with identical-index variants reusing it. Low-rank construction includes the measured EVD plus factor extraction; spatial construction includes pair distances plus CSR assembly. Warm single-vector application timings are medians of seven repeats and include action temporaries. PSD checks on spatial approximations are separate diagnostic costs, not hidden construction costs. These timings are machine/thread dependent microbenchmarks, not end-to-end optimizer timings. Dense O(N³) construction can dominate any application savings; repeated-use amortization and future partial decomposition must be assessed.', '',
        'Spatial truncation is not assumed PSD: primary full-P truncated minimum eigenvalues are recorded at all three radii; an analytically verified synthetic case is indefinite after truncation. Other spatial variants are labeled without a PSD guarantee and are not separately eigendecomposed.', '',
        '## Numerical scalability','',
        f"For N=1152, one dense float64 matrix occupies {8*1152**2/1024**2:.3f} MiB. The conservative EVD workspace/eigenvector estimate is {summary['scalability']['primary_eigh_workspace_estimate_bytes']/1024**2:.3f} MiB, excluding resident P/R and analysis arrays. The {summary['unique_eigendecomposition_count']} non-reused eigendecomposition calls took {summary['total_unique_eigendecomposition_seconds']:.3f} seconds in aggregate.", '',
        f"A 96×48 matrix would have N=4608, one dense matrix of {8*4608**2/1024**2:.3f} MiB, an EVD workspace estimate of {summary['scalability']['future_eigh_workspace_estimate_bytes']/1024**2:.3f} MiB, and an asymptotic cubic EVD cost about 64 times the 48×24 case (not a measured prediction). No such decomposition was attempted.", '',
        'A future scaling path is a partial symmetric eigensolver driven by verified P-vector actions, with per-state factorization reuse and numerical residual checks. Existing HVP supplies H v = 2P v − D v, so recovering P v requires the correct saved/computed D and consistent problem state. This phase neither changes the solver nor implements that matrix-free path. Full dense matrices, spectral factors, and native factorization memory must be included in any future end-to-end comparison.', '',
        '## 8. Does this support a local-plus-low-rank global architecture?', '',
        '**Verified observations** are the spectra, required ranks, fixed-set subspace changes, action errors, and matched-budget costs above. **Mechanically plausible interpretations** include concentration of absolute P energy in a few energetic response directions and different spectral concentration after R normalization. A global modal representation and an entrywise local representation approximate different structures.', '',
        '**Architectural hypothesis:** local plus low-rank global components are worth testing only if their measured joint errors and costs improve on alternatives. This analysis tests pure low-rank and pure spatial approximations; it does not fit a low-rank correction to a local residual, and therefore does not demonstrate that a hybrid is necessary or superior. Required next evidence is held-out-problem prospective prediction or optimization benchmarks at matched total storage, application and construction/amortization costs, including H/design-filter consistency. No neural network or M2.4 was implemented.', '',
        '## Provenance, limitations and reproducibility','',
        'Saved force vectors, constrained DOFs and physical meshes are checked identical within each domain before cross-state comparisons. M1/M2 datasets, figures, reports and numerical-core files were SHA-256 fingerprinted before/after analysis and are unchanged. Existing M2.3 outputs are archived on reruns. The known provenance mismatch remains: saved density trajectories originate in a different geometry/load representation from the evaluated exact matrices. Results describe the saved operators at those snapshots; they do not certify curvature along the original optimizer. There are few mechanical problems, and changes in geometry/state/active population complicate cross-size comparisons.', '',
        'Run with the existing environment:', '',
        '```bash',
        'PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m2-spectral-mpl /users/atabarra/.venv/bin/python analyze_m2_spectral.py',
        'PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m2-spectral-mpl /users/atabarra/.venv/bin/python -m pytest -q tests/test_spectral_interactions.py',
        'PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m2-spectral-mpl /users/atabarra/.venv/bin/python -m pytest -q',
        '```', '',
        'Focused/full test results are recorded in `validation.json` after execution. No exact P regeneration, numerical-core changes, HPC submission, commits or pushes occurred.']
    (out/'M2_3_SPECTRAL_REPORT.md').write_text('\n'.join(lines)+'\n')



if __name__=='__main__':
    run_analysis()
