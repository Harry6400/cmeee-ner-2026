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
 contract=json.loads((ROOT/'evidence/stage3/resume-02-contract.json').read_text())
 assert contract['authorization']=='explicit user resume on 2026-10-20'
 for name,want in contract['critical_source_sha256'].items():assert digest(ROOT/'scripts'/name)==want,name
 prep=json.loads((ROOT/'evidence/stage3/resume-preparation.json').read_text());checkpoint=ROOT/prep['normalized_checkpoint'];assert digest(checkpoint)==prep['normalized_sha256']
 original=json.loads((ROOT/'evidence/stage3/queue-develop.json').read_text());assert original['status']=='failed'
 path=ROOT/'evidence/stage3/queue-resume-02.json';assert not path.exists()
 q=dict(controller_pid=os.getpid(),status='waiting',started_at=now(),observations=[],runtime_observations=[],segments=[],source_original_queue='evidence/stage3/queue-develop.json',checkpoint=str(checkpoint.relative_to(ROOT)))
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
 seg=dict(name='N1-rope-off-resume-02',started_at=now(),command=cmd,status='running');t=time.perf_counter()
 with (ROOT/'evidence/stage3/N1-rope-off-resume-02.log').open('x') as console:
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
 oldseg=dict(original['segments'][-1]);first=json.loads((ROOT/'evidence/stage3/queue-resume.json').read_text());seg1=dict(first['segments'][0])
 archive=json.loads((ROOT/'evidence/stage3/resume-attempt-01-archive.json').read_text())
 oldseg.update(archive_dir='runs/stage3-20261019/N1-rope-off/segments/segment-00',console='evidence/stage3/N1-rope-off.log',runner_summary='runs/stage3-20261019/N1-rope-off/segments/segment-00/summary.json',discarded_partial_updates=prep['discarded_partial_updates'],discarded_partial_update_seconds=prep['discarded_partial_update_seconds'])
 seg1.update(archive_dir='runs/stage3-20261019/N1-rope-off/segments/segment-01',console='evidence/stage3/N1-rope-off-resume.log',runner_summary='runs/stage3-20261019/N1-rope-off/segments/segment-01/summary.json',discarded_partial_updates=archive['discarded_partial_updates'],discarded_partial_update_seconds=archive['discarded_partial_update_seconds'])
 seg.update(console='evidence/stage3/N1-rope-off-resume-02.log',runner_summary='runs/stage3-20261019/N1-rope-off/summary.json',discarded_partial_updates=0,discarded_partial_update_seconds=0)
 segments=[oldseg,seg1,seg];source_queues=['evidence/stage3/queue-develop.json','evidence/stage3/queue-resume.json','evidence/stage3/queue-resume-02.json']
 oldsummary=json.loads((run/'segments/segment-00/summary.json').read_text());summary1=json.loads((run/'segments/segment-01/summary.json').read_text())
 lifecycle=dict(status='completed',segments=segments,source_queues=source_queues,original_queue_status=original['status'],original_stop='intentional safety stop after foreign GPU process appeared in two prior segments',
  completed_epochs=summary['epochs'],discarded_partial_updates=sum(x['discarded_partial_updates'] for x in segments),discarded_partial_update_seconds=sum(x['discarded_partial_update_seconds'] for x in segments),
  total_runner_wall_seconds=oldsummary['seconds']+summary1['seconds']+summary['seconds'],total_child_wall_seconds=sum(x['seconds'] for x in segments),
  prelaunch_polling_wall_seconds=first['prelaunch_polling_wall_seconds']+q['prelaunch_polling_wall_seconds'],restored_epoch=8,restored_global_step=3472,source_preparation='evidence/stage3/resume-preparation.json',
  user_pause='evidence/stage3/user-requested-pause.json;excluded from GPU and queue waiting',resume_authorization='evidence/stage3/resume-02-contract.json')
 dump(run/'lifecycle.json',lifecycle)
 combined=dict(name='N1-rope-off',started_at=oldseg['started_at'],finished_at=seg['finished_at'],seconds=lifecycle['total_child_wall_seconds'],returncode=0,status='completed',source_segments=segments,resumed_after_external_occupancy=True)
 resolved=dict(status='completed',started_at=original['started_at'],finished_at=now(),gpu_uuid=uuid,segments=original['segments'][:-1]+[combined],observations=original['observations']+first['observations']+q['observations'],
  provenance='derived lifecycle reconciliation;original failed queue and both interrupted summaries retained unchanged',source_queues=source_queues)
 dump(ROOT/'evidence/stage3/queue-develop-resolved.json',resolved);print(json.dumps(dict(status='completed',lifecycle=lifecycle)),flush=True)

if __name__=='__main__':main()
