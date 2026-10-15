"""Freeze actual ML worker revision before a new reproduction."""
import sys,shutil
from pathlib import Path
from .io import ROOT,write,digest
out=Path(sys.argv[1]).resolve();archive=out/'initial_source_snapshots';archive.mkdir();records={}
for p in (ROOT/'m3_5a').glob('*.py'):
 dst=archive/(p.name+'.txt');shutil.copyfile(p,dst);records[str(p.relative_to(ROOT))]=dict(sha256=digest(p),archive=str(dst.relative_to(out)))
write(out/'initial_source_manifest.json',records)
