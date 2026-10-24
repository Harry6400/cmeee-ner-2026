"""CPU观察GPU7连续空闲；只记录，不启动训练或干预进程。"""
import os,json,time,signal,subprocess
from datetime import datetime,timezone
from ner_core import ROOT
p=ROOT/'evidence/stage3/gpu-availability-wait.json';assert not p.exists()
u='GPU-f77a1700-de70-05d2-048f-18c8f9aa5974';start=time.perf_counter();idle_since=None;stop=[False]
signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
d=dict(status='waiting',started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),gpu_uuid=u,required_continuous_idle_seconds=300,observations=[],launches_GPU_work=False)
def save():
 t=p.with_suffix('.writing.json');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
while not stop[0] and time.perf_counter()-start<84600:
 gpu=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip();assert gpu.split(',')[0].strip()==u
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True);apps=[x for x in apps.splitlines() if u in x]
 busy=bool(apps) or int(gpu.split(',')[1].split()[0])>128
 if busy:idle_since=None
 elif idle_since is None:idle_since=time.perf_counter()
 idle=0 if idle_since is None else time.perf_counter()-idle_since
 d.update(status='ready' if idle>=300 else 'waiting' if busy else 'observing_idle',last_observed_at=datetime.now(timezone.utc).isoformat(),continuous_idle_seconds=idle)
 d['observations'].append(dict(at=d['last_observed_at'],gpu=gpu,compute_processes=apps,busy=busy,idle_seconds=idle));save();time.sleep(60)
d.update(status='stopped' if stop[0] else 'expired',finished_at=datetime.now(timezone.utc).isoformat());save()
