"""Sequential ML-only QC, tiny smoke tests and automated leakage checks."""
import sys
from pathlib import Path
from .io import read,write
from .monitor import run

def main(out):
 out=Path(out).resolve();cfg=out/'config.json';mon=out/'monitor';rows=[]
 jobs=[('quality',[sys.executable,'-m','m3_5a.quality',str(cfg),str(out/'quality')],600),('smoke_A',[sys.executable,'-m','m3_5a.smoke',str(cfg),str(out/'smoke_A'),'A'],650),('smoke_B',[sys.executable,'-m','m3_5a.smoke',str(cfg),str(out/'smoke_B'),'B'],650),('tests',[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','m3_5a/test_pipeline.py'],180)]
 for label,cmd,seconds in jobs:
  print(label,flush=True)
  try:rows.append(dict(name=label,monitor=run(cmd,mon,label,seconds)));write(out/(label+'_completed.json'),dict(passed=True))
  except Exception as exc:write(out/'BLOCKER.json',dict(passed=False,phase=label,error=str(exc),completed=[r['name'] for r in rows],new_FEM=0,new_OC=0));raise
 write(out/'execution_ledger.json',dict(passed=True,phases=rows,new_FEM=0,new_OC=0))
if __name__=='__main__':main(sys.argv[1])
