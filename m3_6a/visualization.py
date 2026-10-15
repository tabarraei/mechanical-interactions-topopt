"""Complete visualization contract, distinct from scientific data acceptance."""
import gc,json,hashlib,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from m3_5a.io import ROOT,check,digest,read,write
from m3_5v.visualize import Case,INDEX_SHA,pil_frame,draw_state,save_snapshots,save_convergence,png_uri,create_notebook
VERSION='m3.6a.visualization.complete.v1'
OLD=ROOT/'m3_5v/visualizations/20261015_accepted12_v1'
PNG_ROLES={'initial_rho','middle_rho','final_rho','initial_x','middle_x','final_x','evolution','compliance','volumes','updates','diagnostics','annotated_final'}
def raster_hash(im):return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
def provenance(release,e):
 spec=release.specs[e['problem_id']]
 return dict(index_sha256=release.index_sha256,problem_id=e['problem_id'],source_id=e['source_id'],metadata_sha256=e['metadata_sha256'],array_hashes=e['array_hashes'],spec_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())
def rendering_sources():
 return {str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'m3_5v/visualize.py',ROOT/'m3_5v/viewer_template.html')}
def render_field(ax,field,extent,mask=None,label='physical density ρ'):
 import matplotlib.pyplot as plt
 arr=np.asarray(field);check(arr.ndim==2,'Element field shape')
 if mask is not None:check(np.shape(mask)==arr.shape,'Explicit domain mask shape');arr=np.ma.array(arr,mask=~np.asarray(mask,bool))
 cm=plt.get_cmap('gray_r').copy();cm.set_bad('#b7c8db')
 im=ax.imshow(arr,origin='lower',extent=extent,aspect='equal',interpolation='nearest',vmin=0,vmax=1,cmap=cm);ax.set_xlabel('physical x');ax.set_ylabel('physical y');ax.set_title(label);return im

def full_frame(c,t):
 im=pil_frame(c,t,canvas_size=(480,340));d=ImageDraw.Draw(im);d.rectangle((0,315,480,340),fill='white')
 import matplotlib
 font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),12)
 d.text((12,319),'Every saved state • chronological • fixed density [0,1]',fill='black',font=font)
 return im.quantize(colors=128)

def verify_package(release,output):
 """Blocks visualization-required bundles, never revokes scientific acceptance."""
 output=Path(output);top=read(output/'visualization_manifest.json');check(top['version']==VERSION and top['source_index_sha256']==release.index_sha256,'Stale visualization index/version')
 expected={e['problem_id'] for e in release.entries};check(set(top['trajectories'])==expected,'Missing/extra visualization trajectory')
 summary=[]
 for e in release.entries:
  pid=e['problem_id'];r=top['trajectories'][pid];path=(output/r['path']).resolve();check(path.is_relative_to(output.resolve()) and digest(path)==r['sha256'],'Manifest hash/path')
  m=read(path);check(m['problem_id']==pid and m['source']==provenance(release,e) and m['rendering_sources']==rendering_sources(),'Wrong trajectory/stale data or rendering source')
  check(m['scientifically_accepted'] and m['states']==e['states'] and m['transitions']==e['transitions'] and m['termination']==e['termination_reason'] and m['reference_optimum'] is False,'Counts/termination')
  check(set(m['pngs'])==PNG_ROLES,'Missing PNG roles')
  for p in set(m['pngs'].values()):
   f=output/p;check(f.resolve().is_relative_to(output.resolve()) and f.is_file() and digest(f)==m['artifact_hashes'][p],'Missing/corrupt PNG')
   with Image.open(f) as im:check(im.format=='PNG','PNG encoding');im.verify()
  gif=output/m['gif'];check(gif.is_file() and digest(gif)==m['artifact_hashes'][m['gif']],'Missing/corrupt GIF')
  check(m['frame_iterations']==list(range(e['states'])) and len(m['frame_raster_hashes'])==e['states'],'Incomplete/reordered frame records')
  with Image.open(gif) as im:
   check(im.n_frames==e['states'],'Decoded GIF state/frame mismatch')
   for t in range(im.n_frames):im.seek(t);check(raster_hash(im)==m['frame_raster_hashes'][t],'Decoded frame order/raster mismatch')
  summary.append(dict(problem_id=pid,states=e['states'],decoded_frames=e['states'],PNG_roles=len(PNG_ROLES),complete=True))
 return dict(passed=True,scientific_acceptance_unchanged=True,visualization_required_bundle_ready=True,trajectories=summary)

def dynamic_viewer(release,out):
 payload=[]
 for i,e in enumerate(release.entries):
  c=Case(e)
  try:
   r=dict(id=c.id,index=i,nx=c.nx,ny=c.ny,width=c.width,height=c.height,T=c.T,reason='maxiter capped' if e['termination_reason']=='maxiter' else 'change tolerance reached',target=c.m['volume_fraction'],support=c.spec['support_pattern'],nodes=c.m['node_coordinates'],fixed=c.m['fixed_dofs'],loads=c.m['loads'],metrics=[c.metrics(t) for t in range(c.T)],rho=[],x=[])
   for t in range(c.T):r['rho'].append(png_uri(c.frame(t)));r['x'].append(png_uri(c.frame(t,'x')))
   payload.append(r)
  finally:c.close()
 html=(ROOT/'m3_5v/viewer_template.html').read_text().replace('__DATASET__',json.dumps(payload,separators=(',',':')).replace('</','<\\/'))
 html=html.replace('12 accepted topology trajectories',f'{len(release.entries)} accepted topology trajectories').replace('3,709 states / 3,697 transitions',f'{sum(e["states"] for e in release.entries):,} states / {sum(e["transitions"] for e in release.entries):,} transitions')
 html=html.replace('All states are selectable, although standalone GIFs sample at most 64 states per case.','All states are selectable; each standalone GIF includes every saved state in chronological order.')
 (out/'trajectory_viewer.html').write_text(html);create_notebook(out)

def generate_package(release,output):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 output=Path(output).resolve()
 if output.exists():return verify_package(release,output) # Idempotent verification, never overwrite stale packages.
 output.mkdir(parents=True);prior=read(OLD/'artifact_manifest.json');old_index=read(ROOT/'m3_4br4approved/results/20261015_approved_continuation/accepted_dataset_index.json');old_entries={e['problem_id']:e for e in old_index['configurations']}
 top=dict(version=VERSION,source_index_sha256=release.index_sha256,source_index=str(release.index),trajectories={})
 for i,e in enumerate(release.entries):
  check(e['mechanically_validated'] and e['validation']['passed'],'Only scientifically accepted data may render as accepted')
  c=Case(e);folder=output/'trajectories'/c.id;folder.mkdir(parents=True);pngs={};reused={}
  try:
   for field in ('rho','x'):
    for phase,t in (('initial',0),('middle',(c.T-1)//2),('final',c.T-1)):
     p=folder/f'{phase}_{field}.png';fig,ax=plt.subplots(figsize=(6,4),layout='constrained');label='physical density ρ' if field=='rho' else 'raw design x'
     im=render_field(ax,c.frame(t,field),(0,c.width,0,c.height),label=f'{label} | t={t}');fig.colorbar(im,ax=ax);fig.suptitle(f'{c.id}\n{e["termination_reason"]}; no reference-optimum label',fontsize=9);fig.savefig(p,dpi=110);plt.close(fig);pngs[f'{phase}_{field}']=str(p.relative_to(output))
   p=folder/'annotated_final.png';fig,ax=plt.subplots(figsize=(7,4),layout='constrained');draw_state(ax,c,c.T-1);ax.set_title(f'{c.id}\nphysical density ρ; saved forces/supports',fontsize=9);fig.savefig(p,dpi=110);plt.close(fig);pngs['annotated_final']=str(p.relative_to(output))
   old=old_entries.get(c.id)
   for role,sub in (('evolution','trajectory_snapshots'),('convergence','convergence_plots')):
    p=folder/f'{role}.png';old_name=f'{sub}/{old["index"]:02d}_{c.id}.png' if old else None
    if old is not None and old['metadata_sha256']==e['metadata_sha256'] and old_name in prior and digest(OLD/old_name)==prior[old_name]:shutil.copyfile(OLD/old_name,p);reused[str(p.relative_to(output))]=dict(source=str(OLD/old_name),sha256=prior[old_name])
    elif role=='evolution':save_snapshots(c,p)
    else:save_convergence(c,p)
    if role=='evolution':pngs[role]=str(p.relative_to(output))
    else:
     for r in ('compliance','volumes','updates','diagnostics'):pngs[r]=str(p.relative_to(output))
   frames=[];hashes=[]
   for t in range(c.T):
    f=full_frame(c,t);hashes.append(raster_hash(f));frames.append(f)
   gif=folder/'all_saved_states.gif';frames[0].save(gif,save_all=True,append_images=frames[1:],duration=70,loop=0,optimize=False,disposal=2);frames.clear();gc.collect()
   paths=set(pngs.values())|{str(gif.relative_to(output))}
   manifest=dict(version=VERSION,problem_id=c.id,source=provenance(release,e),scientifically_accepted=True,states=c.T,transitions=c.T-1,mesh=[c.nx,c.ny],domain=[c.width,c.height],termination=e['termination_reason'],reference_optimum=False,pngs=pngs,gif=str(gif.relative_to(output)),frame_iterations=list(range(c.T)),frame_raster_hashes=hashes,fixed_density_scale=[0,1],orientation='C(nely,nelx); row-y upward; PIL flips rows once',rendering_sources=rendering_sources(),artifact_hashes={p:digest(output/p) for p in paths},reused_artifacts=reused)
   write(folder/'visualization.json',manifest);top['trajectories'][c.id]=dict(path=str((folder/'visualization.json').relative_to(output)),sha256=digest(folder/'visualization.json'))
   print(f'Full visualizations {i+1}/{len(release.entries)}: {c.id} ({c.T} frames)',flush=True)
  finally:c.close()
 write(output/'visualization_manifest.json',top);dynamic_viewer(release,output)
 result=verify_package(release,output);write(output/'completeness_verification.json',result);return result

if __name__=='__main__':
 from m3_5a.loader import Release
 from m3_5v.visualize import INDEX,INDEX_SHA
 import sys
 release=Release(INDEX,INDEX_SHA)
 try:generate_package(release,sys.argv[1])
 finally:release.close()
