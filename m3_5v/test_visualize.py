import json,sys
from pathlib import Path
import numpy as np
import pytest
from m3_5v.visualize import Dataset,sampled_indices,density_pixels,Case

@pytest.mark.parametrize('i',range(12))
def test_accepted_layout_timing_and_mmap(i):
 ds=Dataset();c=ds.case(i)
 try:
  assert c.validate()
  assert isinstance(c.array('rho'),np.memmap) and not c.array('rho').flags.writeable
  assert c.frame(0).shape==(c.ny,c.nx)
  assert c.frame(0)[0,0]==c.array('rho')[0,0]
  assert c.frame(c.T-1)[-1,-1]==c.array('rho')[-1,-1]
  assert c.metrics(0)['incoming_change'] is None and c.metrics(c.T-1)['outgoing_update'] is None
  assert c.meta['termination_reason']==c.entry['termination_reason']
  assert c.meta['reference_label_eligible'] is False
  for bad in (-1,c.T):
   with pytest.raises(ValueError):c.frame(bad)
 finally:c.close()

@pytest.mark.parametrize('T',[3,103,192,193,501])
def test_snapshot_and_animation_indices(T):
 for n in (10,64):
  ids=sampled_indices(T,n);assert ids==sorted(set(ids)) and ids[0]==0 and ids[-1]==T-1 and len(ids)==min(T,n)
 assert set([0,1,2]).issubset(sampled_indices(T))

def test_orientation_raster_exact_corners():
 a=np.array([[0.,.2,.4],[.6,.8,1.]])
 px=density_pixels(a);assert px.shape==(2,3) and px[0,0]==102 and px[0,-1]==0 and px[-1,0]==255

def test_physical_plot_extent_no_transpose():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from m3_5v.visualize import draw_state
 c=Dataset().case(1)
 try:
  fig,ax=plt.subplots();im=draw_state(ax,c,7);assert im.origin=='lower';assert list(im.get_extent())==[0,c.width,0,c.height];assert im.get_array().shape==(48,96);assert im.get_clim()==(0,1);plt.close(fig)
 finally:c.close()

def test_reject_unapproved_index(tmp_path):
 p=tmp_path/'index.json';p.write_text('{}')
 with pytest.raises(ValueError):Dataset(p)

def test_no_scientific_solver_imports():
 assert not any(n.startswith(('topopt_utils','torch','pypardiso')) for n in sys.modules)

def test_notebook_schema_and_python_cells(tmp_path):
 from m3_5v.visualize import create_notebook
 import ast
 create_notebook(tmp_path);nb=json.loads((tmp_path/'trajectory_viewer.ipynb').read_text())
 assert nb['nbformat']==4 and nb['nbformat_minor']==5
 assert len({c['id'] for c in nb['cells']})==len(nb['cells'])
 for c in nb['cells']:
  if c['cell_type']=='code':ast.parse(''.join(c['source']))
 assert str(tmp_path) in ''.join(nb['cells'][1]['source'])
