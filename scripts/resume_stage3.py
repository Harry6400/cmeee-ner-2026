"""等待GPU7后恢复原RoPE对照；原始失败队列和中断日志不覆盖。"""
import json,os,time,subprocess,signal,shutil
from datetime import datetime,timezone
from ner_core import ROOT,dump,digest

def now():return datetime.now(timezone.utc).isoformat()
def observe(uuid):
 gpu=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip()
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
 return gpu,[x for x in apps.splitlines() if uuid in x]

def main():
 cfg=json.loads((ROOT/'configs/stage3.json').read_text());uuid=cfg['gpu_uuid'];run=ROOT/'runs/stage3-20261019/N1-rope-off'
 prep=json.loads((ROOT/'evidence/stage3/resume-preparation.json').read_text());checkpoint=ROOT/prep['normalized_checkpoint'];assert digest(checkpoint)==prep['normalized_sha256']
 original=json.loads((ROOT/'evidence/stage3/queue-develop.json').read_text());assert original['status']=='failed'
 path=ROOT/'evidence/stage3/queue-resume.json';assert not path.exists()
 q=dict(status='waiting',started_at=now(),observations=[],runtime_observations=[],segments=[],source_original_queue='evidence/stage3/queue-develop.json',checkpoint=str(checkpoint.relative_to(ROOT)))
 dump(path,q);wait_start=time.perf_counter()
 while True:
  gpu,apps=observe(uuid);busy=bool(apps) or int(gpu.split(',')[1].split()[0])>128
  q['observations'].append(dict(at=now(),gpu=gpu,compute_processes=apps,busy=busy));dump(path,q)
  if not busy:break
  print(now()+' GPU7 busy; waiting60s',flush=True);time.sleep(60)
 q['prelaunch_polling_wall_seconds']=time.perf_counter()-wait_start
 assert shutil.disk_usage(ROOT).free>20*1024**3
 assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<50*1024**3
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,TMPDIR=str(ROOT/'.tmp'),XDG_CACHE_HOME=str(ROOT/'.cache'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/train_development.py'),'--method','N1','--encoder-lr','5e-05','--output',str(run),'--no-rope','--resume',str(checkpoint)]
 seg=dict(name='N1-rope-off-resume',started_at=now(),command=cmd,status='running');t=time.perf_counter()
 with (ROOT/'evidence/stage3/N1-rope-off-resume.log').open('x') as console:
  child=subprocess.Popen(cmd,env=env,stdout=console,stderr=subprocess.STDOUT,start_new_session=True);seg['pid']=child.pid;q.update(status='running',segments=[seg]);dump(path,q)
  while child.poll() is None:
   time.sleep(10);gpu,apps=observe(uuid);foreign=[x for x in apps if int(x.split(',')[0])!=child.pid]
   q['runtime_observations'].append(dict(at=now(),gpu=gpu,compute_processes=apps,foreign=foreign));dump(path,q)
   if foreign or time.perf_counter()-t>cfg['max_segment_seconds']-120:
    seg['interruption_reason']='foreign appeared' if foreign else 'segment time cap';child.send_signal(signal.SIGTERM)
    try:child.wait(timeout=120)
    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
    break
  code=child.wait()
 seg.update(finished_at=now(),seconds=time.perf_counter()-t,returncode=code,status='completed' if code==0 else 'failed')
 summary=json.loads((run/'summary.json').read_text());q.update(status='completed' if code==0 and summary['status'] in ('completed','early_stopped') else 'interrupted',finished_at=now());dump(path,q)
 if q['status']!='completed':raise RuntimeError('恢复未完成，记录保留；不得冒称完成')
 oldseg=original['segments'][-1];oldsummary=json.loads((run/'segments/segment-00/summary.json').read_text())
 lifecycle=dict(status='completed',segments=[oldseg,seg],original_queue_status=original['status'],original_stop='intentional safety wait after foreign process appeared',
  completed_epochs=summary['epochs'],discarded_partial_updates=prep['discarded_partial_updates'],discarded_partial_update_seconds=prep['discarded_partial_update_seconds'],
  total_runner_wall_seconds=oldsummary['seconds']+summary['seconds'],total_child_wall_seconds=oldseg['seconds']+seg['seconds'],
  prelaunch_polling_wall_seconds=q['prelaunch_polling_wall_seconds'],restored_epoch=8,restored_global_step=3472,source_preparation='evidence/stage3/resume-preparation.json')
 dump(run/'lifecycle.json',lifecycle)
 combined=dict(name='N1-rope-off',started_at=oldseg['started_at'],finished_at=seg['finished_at'],seconds=lifecycle['total_child_wall_seconds'],returncode=0,status='completed',source_segments=[oldseg,seg],resumed_after_external_occupancy=True)
 resolved=dict(status='completed',started_at=original['started_at'],finished_at=now(),gpu_uuid=uuid,segments=original['segments'][:-1]+[combined],observations=original['observations']+q['observations'],
  provenance='derived lifecycle reconciliation;original failed queue and interrupted summary retained unchanged',source_queues=['evidence/stage3/queue-develop.json','evidence/stage3/queue-resume.json'])
 dump(ROOT/'evidence/stage3/queue-develop-resolved.json',resolved);print(json.dumps(dict(status='completed',lifecycle=lifecycle)),flush=True)

if __name__=='__main__':main()
