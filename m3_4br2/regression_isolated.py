"""All regression modules, sequential fresh processes; unchanged tests/gates."""
import json,subprocess,sys,os,re
from pathlib import Path

def run(out):
 logs=out/'test_logs';logs.mkdir(exist_ok=True);rows=[]
 for file in sorted(Path('tests').glob('test_*.py'))+[Path('m3_4br2/test_audit.py'),Path('m3_4br2/test_infrastructure.py')]:
  log=logs/(file.stem+'.log');cmd=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(file)]
  with log.open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
  text=log.read_text();matches=re.findall(r'(\d+) passed',text);rows.append(dict(module=str(file),exit_code=r.returncode,passed_tests=int(matches[-1]) if matches else 0,log=str(log)))
  print(file,r.returncode,text.splitlines()[-1] if text else '',flush=True)
 (out/'regression_isolated.json').write_text(json.dumps(dict(passed=all(r['exit_code']==0 for r in rows),test_count=sum(r['passed_tests'] for r in rows),modules=rows,scope='Every original test module and both R2 modules in fresh sequential processes; temporary original fixtures authorized explicitly'),indent=2))
if __name__=='__main__':run(Path(sys.argv[1]))
