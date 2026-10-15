"""Restore byte-checked numerical sources and replay one saved pilot problem."""
import argparse,json,hashlib,shutil,subprocess,sys,tempfile,os,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generation',type=Path,required=True);p.add_argument('--index',type=int,default=0);a=p.parse_args();g=a.generation.resolve()
    m=json.loads((g/'execution_manifest.json').read_text());limits=m['limits']
    if not 0<=a.index<len(m['jobs']):raise ValueError('Unknown pilot index')
    records=json.loads((g/'source_manifest.json').read_text());stamp=str(time.time_ns());result=g/'worker_results'/('archived_replay_'+stamp+'.json');log=g/'worker_logs'/('archived_replay_'+stamp+'.log');samples=[];reason=None;parent_peak=0
    with tempfile.TemporaryDirectory(prefix='m3-4-source-') as directory:
        root=Path(directory)
        for name,entry in records.items():
            source=g/entry['archive']
            if hashlib.sha256(source.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Archive checksum mismatch')
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        environment=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1');started=time.monotonic()
        with log.open('w') as stream:
            child=subprocess.Popen([sys.executable,str(root/'run_m3_4.py'),'_worker','--output',str(g),'--task','validate','--index',str(a.index),'--result',str(result)],cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            while child.poll() is None:
                try:
                    data=Path(f'/proc/{child.pid}/status').read_text();rss=next(int(line.split()[1])/1024 for line in data.splitlines() if line.startswith('VmRSS:'))
                except (FileNotFoundError,StopIteration):rss=0
                parent_data=Path('/proc/self/status').read_text();parent_rss=next(int(line.split()[1])/1024 for line in parent_data.splitlines() if line.startswith('VmRSS:'));parent_peak=max(parent_peak,parent_rss)
                samples.append(dict(seconds=time.monotonic()-started,worker_rss_mib=rss,parent_rss_mib=parent_rss))
                if rss>limits['worker_rss_mib'] or parent_rss>limits['parent_rss_mib'] or rss+parent_rss>limits['combined_rss_mib']:reason='RSS gate exceeded'
                if time.monotonic()-started>limits['worker_timeout_seconds']:reason='Time gate exceeded'
                if reason:
                    child.terminate()
                    try:child.wait(timeout=2)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    break
                time.sleep(limits['poll_seconds'])
            code=child.wait()
        monitor=dict(restored_sources_verified=True,index=a.index,exit_code=code,reason=reason,samples=samples,worker_peak_mib=max(row['worker_rss_mib'] for row in samples),parent_peak_mib=parent_peak,wall_seconds=time.monotonic()-started,worker_exited=not Path(f'/proc/{child.pid}').exists(),log=str(log),result=str(result))
        result.with_name(result.stem+'_monitor.json').write_text(json.dumps(monitor,indent=2)+'\n')
        if reason or code:raise RuntimeError(f'Replay failed: {log}')
    print(result)

if __name__=='__main__':main()
