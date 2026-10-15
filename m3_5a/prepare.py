"""Freeze ML inputs, training-prefix normalization and a tiny reproducible budget."""
import sys,shutil
from pathlib import Path
from datetime import datetime,timezone
from .io import ROOT,RELEASE,R4,R3,read,digest,write
from .loader import Release
from .normalization import Normalization

def run(out):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 protected=read(RELEASE/'protected_before.json').copy()
 for group in ('m3_4br4approved','m3_4br4r1'):
  for p in (ROOT/group).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:protected[str(p.relative_to(ROOT))]=digest(p)
 write(out/'protected_before.json',protected)
 with Release() as r:
  norm=Normalization.fit(r);write(out/'normalization.json',norm.record)
  cfg=dict(version='m3.5a.smoke.v1',created_utc=datetime.now(timezone.utc).isoformat(),release_index=str(r.index),release_sha256=r.index_sha256,release_manifest_sha256=digest(RELEASE/'artifact_manifest.json'),normalization=str(out/'normalization.json'),history=3,seed=3501,steps=400,width=24,learning_rate=.005,wall_seconds=600,overfit_relative_RMSE=.25,tiny_subset=[dict(problem_id=r.entries[0]['problem_id'],transition=t) for t in (0,1)],official_partitions='Unchanged:all12 train',development_split='none; tiny overfit is training-only',device='cpu',sampling='Uniform fixed two early transitions, no weighting; smoke only',loss='Element MSE on design_delta/move; masked spatial padding; no future target normalization',reference_labels=False,mechanical_work_authorized=False)
  write(out/'config.json',cfg);write(out/'input_verification.json',dict(passed=True,release_sha256=r.index_sha256,verified_file_count=len(r.verified_files),files=r.verified_files,artifact_manifest_bindings=r.bindings,new_FEM=0,new_OC=0))
 print(str(out/'config.json'),flush=True)
if __name__=='__main__':run(sys.argv[1])
