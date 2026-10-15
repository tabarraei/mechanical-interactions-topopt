"""Read-only accepted12 visualization, bounded mmap access, no FEM/OC imports."""
import argparse,base64,gc,hashlib,io,json,math,sys,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
INDEX=ROOT/'m3_4br4approved/results/20261015_approved_continuation/accepted_dataset_index.json'
INDEX_SHA='303d5570d78ecbf1276459d99be6ed1420ccfb49b8f6c6224f38019bfe024a5a'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def check(x,msg):
 if not x:raise ValueError(msg)

def sampled_indices(T,count=10):
 check(T>0 and count>=2,'Positive length/count')
 early=[0,1,2,5,10]
 if T<=count:return list(range(T))
 values=set(i for i in early[:max(1,min(len(early),count-2))] if i<T)
 values.add(T-1)
 for v in np.linspace(0,T-1,count-len(values)+2,dtype=int):
  if len(values)>=count:break
  values.add(int(v))
 if len(values)<count:
  for v in np.linspace(0,T-1,count*2,dtype=int):
   values.add(int(v))
   if len(values)>=count:break
 return sorted(values)

def density_pixels(frame):
 """PIL top-left origin, so reverse only row-y; never transpose x/y."""
 return np.rint((1-np.clip(np.flipud(frame),0,1))*255).astype(np.uint8)

class Case:
 def __init__(self,entry):
  self.entry=entry;self.folder=Path(entry['folder']);self.spec=read(self.folder/'problem.json');self.meta=read(self.folder/'metadata.json');self.m=self.spec['mechanical'];self.id=entry['problem_id'];self.T=entry['states'];self.maps={}
  check(self.spec['problem_id']==self.meta['problem_id']==self.id and self.meta['source_id']==entry['source_id'],'Case identity')
  check(self.meta['state_count']==self.T and self.meta['completed_updates']==entry['transitions']==self.T-1,'Counts')
  check(self.meta['termination_reason']==entry['termination_reason'] and self.meta['termination_reason'] in ('maxiter','change_tolerance'),'Termination')
  check(not self.meta['reference_label_eligible'] and not self.meta['optimality_certified'],'No reference optima')
  self.nx=self.m['nelx'];self.ny=self.m['nely'];self.width=self.m['width'];self.height=self.m['height']
 def array(self,name):
  if name not in self.maps:
   e=self.meta['arrays'][name];p=(self.folder/e['file']).resolve();check(p.is_relative_to(self.folder.resolve()),'Array path traversal')
   a=np.load(p,mmap_mode='r',allow_pickle=False);check(list(a.shape)==e['shape'] and str(a.dtype)==e['dtype'],'Array shape/dtype');self.maps[name]=a
  return self.maps[name]
 def frame(self,t,field='rho'):
  check(field in ('rho','x') and type(t) in (int,np.int64) and 0<=t<self.T,'Field/state index')
  return self.array(field)[t].reshape(self.ny,self.nx,order='C')
 def close(self):
  for a in self.maps.values():
   if getattr(a,'_mmap',None) is not None:a._mmap.close()
  self.maps.clear()
 def validate(self):
  check(np.array_equal(self.array('iteration'),np.arange(self.T)),'Saved iteration order')
  check(self.array('rho').shape==(self.T,self.nx*self.ny) and self.array('x').shape==self.array('rho').shape,'Density layout')
  centers=np.asarray(self.m['element_centers']).reshape(self.ny,self.nx,2)
  check(np.allclose(centers[:,:,0],(np.arange(self.nx)+.5)*self.m['dx']) and np.allclose(centers[:,:,1],(np.arange(self.ny)[:,None]+.5)*self.m['dy']),'Physical C-order element-center orientation')
  nodes=np.asarray(self.m['node_coordinates']).reshape(self.ny+1,self.nx+1,2)
  check(np.allclose(nodes[:,:,0],np.arange(self.nx+1)*self.m['dx']) and np.allclose(nodes[:,:,1],np.arange(self.ny+1)[:,None]*self.m['dy']),'Node orientation')
  check(len(self.array('update_change'))==self.T-1 and not self.array('has_incoming_update')[0] and np.all(self.array('has_incoming_update')[1:]),'Change timing')
  check(np.array_equal(self.array('previous_change')[1:],self.array('update_change')),'Incoming alignment')
  check(np.isfinite(self.array('compliance')).all(),'Saved compliance')
  return True
 def metrics(self,t):
  return dict(iteration=int(self.array('iteration')[t]),compliance=float(self.array('compliance')[t]),physical_volume=float(self.array('physical_volume')[t]),design_volume=float(self.array('design_volume')[t]),target_design_volume=self.m['volume_fraction'],incoming_change=float(self.array('previous_change')[t]) if t else None,outgoing_update=float(self.array('update_change')[t]) if t<self.T-1 else None)

class Dataset:
 def __init__(self,index=INDEX):
  self.index=Path(index);check(self.index.resolve()==INDEX.resolve() and digest(self.index)==INDEX_SHA,'Only authoritative accepted index allowed')
  doc=read(self.index);check(doc['passed'] and doc['original_only'],'Accepted original-only index')
  self.entries=doc['configurations'];check((len(self.entries),sum(e['states'] for e in self.entries),sum(e['transitions'] for e in self.entries))==(12,3709,3697),'Accepted counts')
  check(len({e['problem_id'] for e in self.entries})==12 and all(e['mechanically_validated'] and e['validation']['passed'] for e in self.entries),'Accepted entries')
 def case(self,index):return Case(self.entries[index])
 def verify(self):
  files={str(self.index):digest(self.index)};rows=[]
  for i,e in enumerate(self.entries):
   c=self.case(i)
   try:
    check(digest(c.folder/'metadata.json')==e['metadata_sha256'],'Accepted metadata hash')
    for key,a in c.meta['arrays'].items():check(digest(c.folder/a['file'])==a['sha256']==e['array_hashes'][key],'Accepted native hash '+key)
    check(digest(c.folder/'problem.json')==c.meta['problem_sha256'] and digest(c.folder/'transitions.json')==c.meta['transitions_sha256'],'Problem/transition hash')
    c.validate()
    for p in c.folder.rglob('*'):
     if p.is_file():files[str(p)]=digest(p)
    rows.append(dict(index=i,problem_id=c.id,states=c.T,transitions=c.T-1,mesh=[c.nx,c.ny],domain=[c.width,c.height],termination=c.meta['termination_reason'],metrics_final=c.metrics(c.T-1),fields=list(c.meta['arrays'])))
   finally:c.close()
  return files,rows

def termination(c):return 'maxiter capped' if c.meta['termination_reason']=='maxiter' else 'change tolerance reached'
def overlays(ax,c):
 nodes=np.asarray(c.m['node_coordinates']);fixed=np.asarray(c.m['fixed_dofs'],int)
 for dof,color,marker in ((0,'#087f8c','|'),(1,'#e88900','_')):
  ids=fixed[fixed%2==dof]//2
  if len(ids):ax.scatter(nodes[ids,0],nodes[ids,1],c=color,marker=marker,s=28,linewidths=1,clip_on=False,zorder=4)
 for load in c.m['loads']:
  x,y=load['coordinates'];fx,fy=load['fx'],load['fy'];norm=math.hypot(fx,fy)
  if norm:
   L=.14*min(c.width,c.height);ax.annotate('',xy=(x,y),xytext=(x-fx/norm*L,y-fy/norm*L),arrowprops=dict(arrowstyle='-|>',color='#d62728',lw=1.7),zorder=5)
 ax.set_xlim(-.06*c.width,1.06*c.width);ax.set_ylim(-.08*c.height,1.08*c.height)

def draw_state(ax,c,t,field='rho',title=True):
 im=ax.imshow(c.frame(int(t),field),origin='lower',extent=(0,c.width,0,c.height),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest',aspect='equal');overlays(ax,c)
 ax.set_xlabel('physical x');ax.set_ylabel('physical y')
 if title:ax.set_title(f't={int(c.array("iteration")[t])} | C={c.array("compliance")[t]:.5g}',fontsize=9)
 return im

def save_gallery(ds,out):
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(4,3,figsize=(17,18),layout='constrained')
 for i,ax in enumerate(axes.flat):
  c=ds.case(i)
  try:
   im=draw_state(ax,c,c.T-1,title=False);loads='; '.join(f'F=({l["fx"]:g},{l["fy"]:g}) @ ({l["coordinates"][0]:g},{l["coordinates"][1]:g})' for l in c.m['loads'])
   ax.set_title(f'#{i} {c.id}\n{c.nx}×{c.ny} | {c.spec["support_pattern"]} | target x volume={c.m["volume_fraction"]:g}\n{loads}\n{c.T-1} updates | C={c.array("compliance")[-1]:.6g} | {termination(c)}',fontsize=8.5)
  finally:c.close()
 fig.suptitle('Accepted dataset • final saved physical density ρ\nSupports: teal vertical marks constrain x, orange horizontal marks constrain y; red arrows show load direction\nNeither termination class certifies an optimum. Density scale fixed [0,1].',fontsize=13)
 fig.colorbar(im,ax=list(axes.flat),label='physical density ρ',shrink=.5);fig.savefig(out/'dataset_gallery.png',dpi=160);plt.close(fig)

def save_snapshots(c,path):
 import matplotlib.pyplot as plt
 ids=sampled_indices(c.T,10);fig,axes=plt.subplots(2,5,figsize=(17,7),layout='constrained')
 for ax,t in zip(axes.flat,ids):draw_state(ax,c,t)
 fig.suptitle(f'{c.id} | saved physical density ρ | {termination(c)}\nActual saved iterations; no interpolation or FEM; fixed density [0,1]',fontsize=12)
 fig.savefig(path,dpi=140);plt.close(fig);return ids

def save_convergence(c,path):
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(3,2,figsize=(12,10),layout='constrained');t=np.asarray(c.array('iteration'))
 axs[0,0].plot(t,c.array('compliance'));axs[0,0].set_ylabel('saved compliance C')
 for k,label in [('physical_volume','physical mean ρ'),('design_volume','design mean x')]:axs[0,1].plot(t,c.array(k),label=label)
 axs[0,1].axhline(c.m['volume_fraction'],color='k',ls=':',label='design budget');axs[0,1].legend(fontsize=8);axs[0,1].set_ylabel('volume fraction (distinct fields)')
 axs[1,0].plot(t[1:],c.array('previous_change')[1:],label='incoming change at t')
 axs[1,0].plot(t[:-1],c.array('update_change'),ls='--',label='outgoing OC max |Δx| at t')
 axs[1,0].axhline(c.spec['optimizer']['tolerance'],color='k',ls=':',label='saved stopping tolerance');axs[1,0].legend(fontsize=8);axs[1,0].set_ylabel('same updates; shifted state indexing')
 for k in ('projected_gradient_inf','box_stationarity_inf'):
  if k in c.meta['arrays']:axs[1,1].plot(t,c.array(k),label=k)
 axs[1,1].legend(fontsize=8);axs[1,1].set_ylabel('stored diagnostics; not optimality proof')
 for k in ('residual_relative_force','energy_compliance_relative'):
  if k in c.meta['arrays']:axs[2,0].plot(t,c.array(k),label=k)
 axs[2,0].legend(fontsize=8);axs[2,0].set_ylabel('saved diagnostics (not recomputed)')
 for k in ('g_x_norm','g_rho_norm'):
  if k in c.meta['arrays']:axs[2,1].plot(t,c.array(k),label=k)
 axs[2,1].legend(fontsize=8);axs[2,1].set_ylabel('saved gradient norm')
 for ax in axs.flat:ax.set_xlabel('saved iteration (optimization step, not physical time)');ax.grid(alpha=.2)
 fig.suptitle(f'{c.id} | {termination(c)} | {c.T-1} updates',fontsize=12);fig.savefig(path,dpi=135);plt.close(fig)

def compare_cases(ds,left=0,right=1):
 import matplotlib.pyplot as plt
 a=ds.case(left);b=ds.case(right)
 try:
  fig,axs=plt.subplots(3,3,figsize=(14,11),layout='constrained')
  for row,c in enumerate((a,b)):
   for col,t in enumerate((0,(c.T-1)//2,c.T-1)):draw_state(axs[row,col],c,t);axs[row,col].set_title(f'#{c.entry["index"]} {c.nx}×{c.ny} • t={t}\nC={c.array("compliance")[t]:.5g}',fontsize=10)
  axs[2,1].remove();axs[2,2].remove()
  for c in (a,b):axs[2,0].plot(np.array(c.array('iteration')),np.array(c.array('compliance')),label=f'#{c.entry["index"]} {c.nx}×{c.ny}')
  axs[2,0].legend();axs[2,0].set_xlabel('saved iteration');axs[2,0].set_ylabel('saved C');axs[2,0].grid(alpha=.2)
  fig.suptitle('Two accepted cases • initial / own middle saved iteration / final\nPhysical axes preserved independently; no resampling, cell correspondence or normalization\nCompliance across different problems is not a quality ranking.',fontsize=12)
  return fig
 finally:a.close();b.close()

def pil_frame(c,t,canvas_size=(560,390)):
 from PIL import Image,ImageDraw,ImageFont
 import matplotlib
 font_path=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf';font=ImageFont.truetype(str(font_path),13)
 w,h=canvas_size;img=Image.new('RGB',(w,h),'white');d=ImageDraw.Draw(img);d.text((12,8),c.id,fill='black',font=font);d.text((12,29),f'Physical density rho | iteration {t}/{c.T-1} | C={c.array("compliance")[t]:.6g}',fill='black',font=font)
 scale=min((w-70)/c.width,(h-130)/c.height);iw=round(c.width*scale);ih=round(c.height*scale);x0=(w-iw)//2;y0=75
 rho=Image.fromarray(density_pixels(c.frame(t)),'L').resize((iw,ih),Image.Resampling.NEAREST).convert('RGB');img.paste(rho,(x0,y0));d.rectangle((x0,y0,x0+iw,y0+ih),outline='#666666')
 def xy(x,y):return x0+x/c.width*iw,y0+ih-y/c.height*ih
 nodes=c.m['node_coordinates']
 for dof in c.m['fixed_dofs']:
  x,y=xy(*nodes[dof//2]);d.line((x,y-3,x,y+3) if dof%2==0 else (x-3,y,x+3,y),fill='#087f8c' if dof%2==0 else '#e88900',width=2)
 for l in c.m['loads']:
  fx,fy=l['fx'],l['fy'];norm=math.hypot(fx,fy)
  if norm:
   px,py=xy(*l['coordinates']);L=27;tx,ty=px-L*fx/norm,py+L*fy/norm;d.line((tx,ty,px,py),fill='#d62728',width=3);vx,vy=(px-tx)/L,(py-ty)/L;d.polygon([(px,py),(px-9*vx+4*vy,py-9*vy-4*vx),(px-9*vx-4*vy,py-9*vy+4*vx)],fill='#d62728')
 d.text((12,h-44),f'{c.nx}x{c.ny} | {termination(c)} | density 0 white / 1 black',fill='black',font=font);d.text((12,h-24),'Physical aspect; selected saved frames; arrows = direction only',fill='black',font=font)
 return img

def save_gif(c,path):
 ids=sampled_indices(c.T,64);frames=[pil_frame(c,t).quantize(colors=128) for t in ids]
 frames[0].save(path,save_all=True,append_images=frames[1:],duration=110,loop=0,optimize=False,disposal=2)
 frames.clear();gc.collect();return ids

def png_uri(frame):
 from PIL import Image
 buf=io.BytesIO();Image.fromarray(density_pixels(frame),'L').save(buf,format='PNG');return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()

def build_html(ds,out):
 payload=[]
 for i in range(12):
  c=ds.case(i)
  try:
   record=dict(id=c.id,index=i,nx=c.nx,ny=c.ny,width=c.width,height=c.height,T=c.T,reason=termination(c),target=c.m['volume_fraction'],support=c.spec['support_pattern'],nodes=c.m['node_coordinates'],fixed=c.m['fixed_dofs'],loads=c.m['loads'],metrics=[c.metrics(t) for t in range(c.T)],rho=[],x=[])
   for t in range(c.T):record['rho'].append(png_uri(c.frame(t,'rho')));record['x'].append(png_uri(c.frame(t,'x')))
   payload.append(record)
  finally:c.close()
 template=(Path(__file__).with_name('viewer_template.html')).read_text();html=template.replace('__DATASET__',json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
 (out/'trajectory_viewer.html').write_text(html);return len(html.encode())

def create_notebook(out):
 cells=[]
 def cell(kind,text):cells.append(dict(cell_type=kind,id=f'cell-{len(cells):03d}',metadata={},source=text.splitlines(True),**({'execution_count':None,'outputs':[]} if kind=='code' else {})))
 cell('markdown','# Accepted topology trajectory viewer\n\nSaved physical densities and design fields only. No FEM/OC, normalization fitting or training. The HTML viewer has two trajectory dropdowns, independent saved-iteration sliders, play/pause, a rho/x selector, compliance histories and physical load/support overlays. All 3,709 states are selectable. Iteration is optimization time, not physical time. Maxiter is capped, not convergence.\n\nIf your VS Code renderer blocks scripted embedded HTML, open `trajectory_viewer.html` beside this notebook in a browser; it is self-contained and needs no server or widget package.')
 cell('code',"from pathlib import Path\nimport sys\n# Find this output folder regardless of whether kernel starts here or repo root.\nlocations = [Path.cwd(), *Path.cwd().parents]\nview_dir = next((p for p in locations if (p / 'trajectory_viewer.html').exists()), None)\nif view_dir is None:\n    view_dir = next((p / 'm3_5v/visualizations/20261015_accepted12_v1' for p in locations if (p / 'm3_5v/visualizations/20261015_accepted12_v1/trajectory_viewer.html').exists()), None)\nif view_dir is None:\n    raise FileNotFoundError('Open the notebook from its visualization folder or repository.')\nrepo = next(p for p in [view_dir, *view_dir.parents] if (p / 'm3_5v/visualize.py').exists())\nif str(repo) not in sys.path: sys.path.insert(0, str(repo))\nprint('Viewer:', view_dir / 'trajectory_viewer.html')\n")
 cell('code',"from IPython.display import HTML, display\nimport base64\n# Sandboxed document: controls do not execute Python or write source files.\nencoded = base64.b64encode((view_dir / 'trajectory_viewer.html').read_bytes()).decode()\ndisplay(HTML('<iframe sandbox=\"allow-scripts\" src=\"data:text/html;base64,' + encoded + '\" style=\"width:100%;height:1050px;border:0\"></iframe>'))\n")
 cell('markdown','## Python-side side-by-side inspection\n\nChange the two index values below (0–11). Each mesh uses its own physical coordinates without resampling; the middle frame is each case\'s own actual saved midpoint. Compliance depends on the problem and is not a universal quality ranking.')
 cell('code',"from m3_5v.visualize import Dataset, compare_cases\nimport matplotlib.pyplot as plt\ndataset = Dataset()\nfig = compare_cases(dataset, left=0, right=1)\nplt.show()\n")
 cell('markdown','## Direct access to any original float64 saved state\n\nThe interactive HTML uses fixed-scale 8-bit previews. This optional cell displays the original float64 memory-mapped rho field. Change `case_index`, `saved_iteration` or field (`rho`/`x`) below; no future-state features or fitted normalization are used.')
 cell('code',"from m3_5v.visualize import draw_state\ncase_index, saved_iteration, field = 0, 0, 'rho'\ncase = dataset.case(case_index)\ntry:\n    fig, ax = plt.subplots(figsize=(9, 5))\n    draw_state(ax, case, saved_iteration, field)\n    ax.set_title(f'{case.id} | {field} | {case.metrics(saved_iteration)}')\n    plt.show()\nfinally:\n    case.close()\n")
 # Final fallback binds the actual generated output root, not a fixed prior run.
 setup = "from pathlib import Path\nimport sys\nlocations = [Path.cwd(), *Path.cwd().parents]\nview_dir = next((p for p in locations if (p / 'trajectory_viewer.html').is_file()), None)\nif view_dir is None:\n    view_dir = Path(OUTPUT_PATH)\nif not (view_dir / 'trajectory_viewer.html').is_file():\n    raise FileNotFoundError('Cannot locate the generated HTML viewer.')\nrepo = Path(REPO_PATH)\nif str(repo) not in sys.path: sys.path.insert(0, str(repo))\nprint('Viewer:', view_dir / 'trajectory_viewer.html')\n"
 setup=setup.replace('OUTPUT_PATH',repr(str(out))).replace('REPO_PATH',repr(str(ROOT)))
 cells[1]['source']=setup.splitlines(True)
 write(out/'trajectory_viewer.ipynb',dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version=sys.version.split()[0])),nbformat=4,nbformat_minor=5))

def main(output):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 start=time.perf_counter();out=Path(output).resolve();check(not out.exists(),'New output directory required; no overwrite');out.mkdir(parents=True)
 ds=Dataset();baseline,rows=ds.verify();write(out/'source_dataset_hashes_before.json',baseline)
 for name in ('trajectory_snapshots','convergence_plots','animations'): (out/name).mkdir()
 save_gallery(ds,out);frames={};snapshots={}
 for i in range(12):
  c=ds.case(i)
  try:
   print(f'Visualizing #{i}: {c.id}',flush=True)
   snapshots[c.id]=save_snapshots(c,out/'trajectory_snapshots'/f'{i:02d}_{c.id}.png')
   save_convergence(c,out/'convergence_plots'/f'{i:02d}_{c.id}.png')
   frames[c.id]=save_gif(c,out/'animations'/f'{i:02d}_{c.id}.gif')
  finally:c.close()
  gc.collect()
 fig=compare_cases(ds);fig.savefig(out/'comparison_00_01.png',dpi=140);plt.close(fig)
 html_bytes=build_html(ds,out);create_notebook(out)
 changed=[p for p,h in baseline.items() if digest(p)!=h];check(not changed,'Dataset bytes changed')
 report=dict(passed=True,trajectories=12,states=3709,transitions=3697,index_sha256=INDEX_SHA,cases=rows,snapshot_iterations=snapshots,GIF_iterations=frames,GIF_frames_per_case_max=64,HTML_all_states_selectable=True,HTML_preview='Fixed [0,1] 8-bit grayscale PNG; original float64 fields accessible in notebook/Python',HTML_bytes=html_bytes,source_files_checked=len(baseline),source_files_changed=changed,worker_wall_seconds=time.perf_counter()-start,process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,FEM=0,OC=0,training=0)
 write(out/'visualization_validation.json',report)
 from .documentation import readme
 (out/'README.md').write_text(readme(report))
 write(out/'artifact_manifest.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()})
 print(json.dumps({k:v for k,v in report.items() if k not in ('cases','snapshot_iterations','GIF_iterations')}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();main(args.output)
