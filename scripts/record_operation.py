"""执行命令并保存真实控制台与墙钟计时；不记录环境变量。"""
import argparse,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--directory',required=True,type=Path);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
a.directory.mkdir(parents=True,exist_ok=True)
start=datetime.now(timezone.utc).isoformat();t=time.perf_counter()
path=a.directory/(a.name+'.log')
with path.open('x') as f:
    r=subprocess.run(a.command,stdout=f,stderr=subprocess.STDOUT)
record=dict(operation=a.name,started_at=start,finished_at=datetime.now(timezone.utc).isoformat(),
            seconds=time.perf_counter()-t,returncode=r.returncode,console=str(path))
(a.directory/(a.name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
raise SystemExit(r.returncode)
