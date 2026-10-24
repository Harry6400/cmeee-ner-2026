"""轻量只读进度；区分原始中断、等待、续训和派生完成状态。"""
import json
from ner_core import ROOT
base=ROOT/'runs/stage3-20261019';e=ROOT/'evidence/stage3'
original=json.loads((e/'queue-develop.json').read_text());resolved=e/'queue-develop-resolved.json';resume=e/'queue-resume.json'
queue=json.loads(resolved.read_text()) if resolved.exists() else original
rq=json.loads(resume.read_text()) if resume.exists() else None
result=dict(status=queue['status'] if resolved.exists() or not rq else rq['status'],completed=[])
for s in queue['segments']:
 run=base/s['name'];p=run/'summary.json';d=json.loads(p.read_text()) if p.exists() else None
 if d and d['status'] in ('completed','early_stopped') and s['status']=='completed':
  life=run/'lifecycle.json';seconds=json.loads(life.read_text())['total_runner_wall_seconds'] if life.exists() else d['seconds']
  result['completed'].append(dict(name=s['name'],epochs=d['epochs'],best_epoch=d['best_epoch'],f1=d['best_f1'],seconds=seconds));continue
 active=dict(name=s['name'],pid=s.get('pid'));progress=run/'progress.json'
 if progress.exists():active.update(json.loads(progress.read_text()))
 if rq and s['name']=='N1-rope-off':
  active['pid']=rq['segments'][-1].get('pid') if rq['segments'] else None
  active['resume_status']=rq['status']
  if rq['status']=='waiting':active['status']='waiting_for_GPU7';active['last_gpu_observation']=rq['observations'][-1]
 metrics=run/'metrics.jsonl'
 if metrics.exists():
  with metrics.open('rb') as f:f.seek(max(0,metrics.stat().st_size-65536));lines=f.read().splitlines()
  for line in reversed(lines):
   try:d=json.loads(line)
   except Exception:continue
   if d.get('event')=='train':active.update(current_step=d['global_step'],current_epoch=d['epoch'],recent_loss=d['loss']);break
 result['active']=active
watch=e/'gpu-availability-wait.json'
if watch.exists() and not resolved.exists() and rq and rq['status']=='interrupted':
 w=json.loads(watch.read_text());result['status']='waiting_for_stable_GPU7'
 result.setdefault('active',{}).update(pid=None,status=w['status'],required_idle_seconds=w['required_continuous_idle_seconds'],observed_idle_seconds=w.get('continuous_idle_seconds',0),last_gpu_observation=w['observations'][-1])
if original['status']=='failed' and rq:result['original_interruption_preserved']=True
print(json.dumps(result,ensure_ascii=False))
