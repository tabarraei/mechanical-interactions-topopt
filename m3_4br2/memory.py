"""Pointwise semantics only; never applied to global norm comparisons."""
import hashlib
from pathlib import Path
import numpy as np

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def frame_allclose(a,b,rtol=1e-10,atol=1e-12):
 if a.shape!=b.shape:raise AssertionError('Native full shape mismatch')
 if a.dtype!=b.dtype:raise AssertionError('Native dtype mismatch')
 if a.ndim==0:np.testing.assert_allclose(a,b,rtol=rtol,atol=atol);return
 for t in range(a.shape[0]):
  try:np.testing.assert_allclose(a[t],b[t],rtol=rtol,atol=atol)
  except AssertionError as e:raise AssertionError(f'Global frame {t}: {e}') from e

def load_trajectory(folder):
 """Same schema/shape/count checks, all original hashes verified in chunks.
 verify=False disables only duplicate full-file hashing inside original loader;
 every one of those hash checks is performed here before returning the object.
 """
 from topopt_utils.fea.trajectory_dataset import Trajectory
 tr=Trajectory(folder,verify=False)
 for key,e in tr.metadata['arrays'].items():
  if digest(tr.folder/e['file'])!=e['sha256']:raise ValueError('Array checksum mismatch: '+key)
 for file,key in [('problem.json','problem_sha256'),('transitions.json','transitions_sha256')]:
  if digest(tr.folder/file)!=tr.metadata[key]:raise ValueError('Checksum mismatch: '+file)
 return tr

def compare_native_trajectories(current,historical):
 """Same native problem and fields; exact original pointwise comparison kwargs."""
 if current.spec['problem_id']!=historical.spec['problem_id']:raise ValueError('Historical problem identity mismatch')
 if current.arrays.keys()!=historical.arrays.keys():raise AssertionError('Historical native field set mismatch')
 for name in current.arrays:
  try:frame_allclose(current.arrays[name],historical.arrays[name],rtol=1e-10,atol=1e-12)
  except AssertionError as e:raise AssertionError(f'Native field {name}: {e}') from e
 return dict(passed=True,fields=len(current.arrays),rtol=1e-10,atol=1e-12)
