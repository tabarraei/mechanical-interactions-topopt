#!/usr/bin/env python
"""Restore exact archived source bytes to /tmp and execute a generation command."""
import argparse,json,tempfile,shutil,subprocess,sys,hashlib,os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--generation',type=Path,required=True);parser.add_argument('command',choices=['resume','validate','report']);args=parser.parse_args()
    source=json.loads((args.generation/'source_manifest.json').read_text())
    with tempfile.TemporaryDirectory(prefix='m3-3b-snapshot-') as folder:
        root=Path(folder)
        for name,entry in source.items():
            data=(args.generation/entry['archive']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Archive corruption: '+name)
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        # The repository is still the working directory for protected artifacts;
        # numerical imports and source checks resolve to the restored source tree.
        env=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,str(root/'run_m3_3b.py'),args.command,'--output',str(args.generation.resolve())],env=env)
        if result.returncode:raise SystemExit(result.returncode)
    print('Exact archived source command completed; temporary restoration removed.')

if __name__=='__main__':main()
