"""Re-evaluate final interface against saved tiny predictions, without training."""
import sys,numpy as np,torch
from pathlib import Path
from .io import read,write,check
from .loader import Release,TrajectoryDataset
from .normalization import Normalization
from .features import collate
from .model import UpdateCNN,encode

def main(out):
 out=Path(out);c=read(out/'config.json');rows=[]
 with Release(c['release_index'],c['release_sha256'],c['release_manifest_sha256']) as r:
  n=Normalization.load(c['normalization'],r)
  for mode in ('A','B'):
   checkpoint=torch.load(out/('smoke_'+mode)/'model.pt',map_location='cpu',weights_only=True)
   check(checkpoint['normalization_id']==n.identifier and checkpoint['release_sha256']==r.index_sha256,'Checkpoint provenance')
   d=TrajectoryDataset(r,mode,c['history'],n);b=collate([d.sample(x['problem_id'],x['transition']) for x in c['tiny_subset']]);m=UpdateCNN(checkpoint['channels'],checkpoint['width']);m.load_state_dict(checkpoint['state_dict']);m.eval()
   with torch.no_grad():p=m(encode(b)).numpy()
   with np.load(out/('smoke_'+mode)/'tiny_predictions.npz') as z:check(np.array_equal(p,z['prediction']),'Final interface prediction differs')
   rows.append(dict(mode=mode,bitwise_saved_prediction_match=True,new_training_steps=0,new_FEM=0,new_OC=0))
 write(out/'checkpoint_verification.json',dict(passed=True,checks=rows))
if __name__=='__main__':main(sys.argv[1])
