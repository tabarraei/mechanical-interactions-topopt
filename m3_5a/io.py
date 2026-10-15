"""Streaming hashes and explicit integrity checks; standard library only."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'m3_4br4approved/results/20261015_approved_continuation'
R4=ROOT/'m3_4br4/results/20261014T174028Z'
R3=ROOT/'m3_4br3/results'
R2=ROOT/'m3_4br2/generations/20261014T124054Z'
OLD=ROOT/'m3_4b/generations/20261014T013017Z'
def check(ok,message):
 if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def array_digest(a):
 import numpy as np
 a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def bound_path(base,name):
 p=(base/name).resolve();check(p.is_relative_to(base.resolve()),'Path outside artifact root');return p
