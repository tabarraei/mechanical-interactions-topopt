"""Completeness/corruption tests on copies, never modify real saved artifacts."""
import json,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from m3_5a.loader import Release
from m3_5a.io import digest
from m3_5v.visualize import INDEX,INDEX_SHA
from m3_6a.visualization import verify_package,generate_package,render_field,raster_hash,full_frame
from m3_6a.data import VIS

@pytest.fixture(scope='module')
def release():
 r=Release(INDEX,INDEX_SHA);yield r;r.close()

def test_all12_packages_and_complete_decoded_frames(release):
 r=verify_package(release,VIS);assert r['passed'] and len(r['trajectories'])==12 and sum(x['decoded_frames'] for x in r['trajectories'])==3709
 assert sum(e['termination_reason']=='maxiter' for e in release.entries)==4

def test_visualization_idempotence_no_mutation(release):
 files={str(p):digest(p) for p in VIS.rglob('*') if p.is_file()};assert generate_package(release,VIS)['passed']
 assert files=={str(p):digest(p) for p in VIS.rglob('*') if p.is_file()}

@pytest.fixture
def single_copy(release,tmp_path):
 e=release.entries[0];pid=e['problem_id'];top=json.loads((VIS/'visualization_manifest.json').read_text());row=top['trajectories'][pid];m=json.loads((VIS/row['path']).read_text())
 for name in set(m['pngs'].values())|{m['gif'],row['path']}:
  dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(VIS/name,dest)
 top['trajectories']={pid:row};(tmp_path/'visualization_manifest.json').write_text(json.dumps(top))
 view=SimpleNamespace(entries=[e],index_sha256=release.index_sha256,specs=release.specs)
 return tmp_path,view,top,m,row

@pytest.mark.parametrize('attack',['missing_png','missing_gif','corrupt_png','wrong_ID','stale_hash','frame_order','decoded_frame_count','cap_label'])
def test_gate_rejects_missing_corrupt_or_stale(single_copy,attack):
 out,view,top,m,row=single_copy;pid=view.entries[0]['problem_id']
 if attack=='missing_png':(out/m['pngs']['initial_rho']).unlink()
 elif attack=='missing_gif':(out/m['gif']).unlink()
 elif attack=='corrupt_png':(out/m['pngs']['initial_rho']).write_bytes(b'corrupt')
 elif attack=='wrong_ID':m['problem_id']='wrong'
 elif attack=='stale_hash':m['source']['metadata_sha256']='0'*64
 elif attack=='frame_order':m['frame_iterations'][0:2]=[1,0]
 elif attack=='decoded_frame_count':
  p=out/m['gif']
  with Image.open(p) as im:frame=im.convert('RGB')
  frame.save(p,format='GIF');m['artifact_hashes'][m['gif']]=digest(p)
 else:m['termination']='converged_reference_optimum'
 (out/row['path']).write_text(json.dumps(m));top['trajectories'][pid]['sha256']=digest(out/row['path']);(out/'visualization_manifest.json').write_text(json.dumps(top))
 with pytest.raises((ValueError,OSError)):verify_package(view,out)

def test_first_last_frames_match_actual_native_render(release):
 from m3_5v.visualize import Case
 e=release.entries[0];c=Case(e)
 try:
  m=json.loads((VIS/'trajectories'/c.id/'visualization.json').read_text())
  with Image.open(VIS/m['gif']) as im:
   assert raster_hash(im)==raster_hash(full_frame(c,0));im.seek(e['states']-1);assert raster_hash(im)==raster_hash(full_frame(c,e['states']-1))
 finally:c.close()

def test_future_mask_renderer_is_explicit_synthetic_fixture(tmp_path):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 arr=np.arange(24,dtype=float).reshape(4,6)/24;mask=np.ones((4,6),bool);mask[2:,3:]=False
 fig,ax=plt.subplots();im=render_field(ax,arr,(0,12,0,4),mask,label='SYNTHETIC L-shaped mask; not scientific data')
 assert im.origin=='lower' and list(im.get_extent())==[0,12,0,4] and np.ma.getmaskarray(im.get_array())[3,4] and im.get_clim()==(0,1)
 fig.savefig(tmp_path/'fixture.png');plt.close(fig)
