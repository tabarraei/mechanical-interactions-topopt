"""Restore exact completion source bytes and independently replay saved trajectories."""
import argparse,json,hashlib,tempfile,sys,os,importlib.util
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--generation',type=Path,required=True);a=p.parse_args();generation=a.generation.resolve();repository=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='m3-3c-completion-snapshot-') as folder:
 root=Path(folder)
 for name,entry in json.loads((generation/'source_manifest.json').read_text()).items():
  data=(generation/entry['archive']).read_bytes()
  if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Archive corruption: '+name)
  target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 sys.path.insert(0,str(root));spec=importlib.util.spec_from_file_location('archived_completion',root/'run_m3_3c_completion.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 # Artifact fingerprint paths belong to the original workspace. Numerical imports
 # resolve to restored bytes; no source byte or completed sample is modified.
 module.ROOT=repository;result=module.validate(generation,fresh=True)
 print(json.dumps(dict(exact_archived_replay_passed=result['passed'],states=result['states'],updates=result['updates'])))
