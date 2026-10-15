"""Restore preserved M3.3C sources and replay, without new optimization."""
import argparse,json,tempfile,subprocess,sys,os,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--generation',type=Path,required=True);p.add_argument('command',choices=['validate']);a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='m3-3c-sources-') as tmp:
 root=Path(tmp)
 for name,entry in json.loads((a.generation/'source_manifest.json').read_text()).items():
  data=(a.generation/entry['archive']).read_bytes()
  if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Archive corruption')
  target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 subprocess.run([sys.executable,str(root/'run_m3_3c.py'),a.command,'--output',str(a.generation.resolve())],env=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1'),check=True)
