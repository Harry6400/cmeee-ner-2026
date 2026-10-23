"""核算已完成阶段三的分段耗时；不启动训练/API，不读取sealed。"""
import datetime,json,math
from pathlib import Path
from ner_core import ROOT,dump


def load(path):return json.loads((ROOT/path).read_text())


def source_summaries(directory,summary,life):
 if not life:return [(None,summary)]
 result=[]
 for index,seg in enumerate(life['segments']):
  relative=seg.get('runner_summary')
  if relative:path=ROOT/relative
  elif seg.get('archive_dir'):path=ROOT/seg['archive_dir']/'summary.json'
  elif index==len(life['segments'])-1:path=directory/'summary.json'
  else:path=directory/'segments'/f'segment-{index:02d}'/'summary.json'
  result.append((seg,json.loads(path.read_text())))
 return result


def main():
 path='evidence/stage3/queue-develop-resolved.json' if (ROOT/'evidence/stage3/queue-develop-resolved.json').exists() else 'evidence/stage3/queue-develop.json';q=load(path)
 if q['status']!='completed':raise ValueError('训练队列未完成')
 rows=[];excerpts=[]
 for seg in q['segments']:
  name=seg['name'];directory=ROOT/f'runs/stage3-20261019/{name}';s=json.loads((directory/'summary.json').read_text());records=s['records']
  life=json.loads((directory/'lifecycle.json').read_text()) if (directory/'lifecycle.json').exists() else None
  sources=source_summaries(directory,s,life)
  runner=sum(item['seconds'] for _,item in sources)
  if life and (not math.isclose(runner,life['total_runner_wall_seconds'],abs_tol=1e-6) or not math.isclose(sum(item['seconds'] for item,_ in sources),life['total_child_wall_seconds'],abs_tol=1e-6)):raise ValueError('生命周期墙钟不能从原始分段重算')
  train=sum(x['train_seconds'] for x in records);val=sum(x['validation']['seconds'] for x in records);save=sum(x['save_seconds'] for x in records)
  discarded=life['discarded_partial_updates'] if life else 0;discarded_seconds=life['discarded_partial_update_seconds'] if life else 0
  overhead=runner-train-val-save-discarded_seconds
  if overhead < -1e-5:raise ValueError('分段耗时相加超过实际runner墙钟')
  segment_rows=[]
  for index,(source,item) in enumerate(sources):
   source=source or seg
   segment_rows.append(dict(name=source['name'],started_at=source['started_at'],finished_at=source['finished_at'],child_wall_seconds=source['seconds'],runner_wall_seconds=item['seconds'],summary_status=item['status'],summary_epochs=item['epochs'],summary_global_step=item['global_step'],archive_dir=source.get('archive_dir'),console=source.get('console',f"evidence/stage3/{source['name']}.log")))
   console_path=source.get('console',f"evidence/stage3/{source['name']}.log");lines=(ROOT/console_path).read_text().splitlines()
   final=next((line for line in reversed(lines) if line.startswith('{') and 'best_checkpoint_sha256' in line),None)
   if final is None:raise ValueError('缺少真实子进程最终控制台行:'+console_path)
   excerpts.extend([f'# Source: {console_path}; verbatim final console line; terminal status remains original',final,''])
  rows.append(dict(name=name,started_at=seg['started_at'],finished_at=seg['finished_at'],child_wall_seconds=seg['seconds'],
    runner_wall_seconds=runner,training_loop_seconds=train,validation_seconds=val,checkpoint_save_seconds=save,
    remaining_runner_overhead_seconds=max(0,overhead),discarded_partial_updates=discarded,discarded_partial_update_seconds=discarded_seconds,
    optimizer_updates=s['global_step'],training_windows_including_logged_tail_repetitions=s['global_step']*32,
    training_loop_windows_per_second=s['global_step']*32/train,
    observed_update_windows_including_discarded=(s['global_step']+discarded)*32,
    observed_training_loop_plus_discarded_update_seconds=train+discarded_seconds,
    peak_tensor_memory_mib=max(item['peak_memory_mb'] for _,item in sources),source_segments=segment_rows))
 ops={}
 for name in ['add-jieba','add-jieba-remote','n1-independent-cpu','full-dictionary-boundary','api-development','api-development-grounded']:
  p=ROOT/f'evidence/stage3/{name}.json'
  if p.exists():
   d=json.loads(p.read_text());ops[name]={k:d[k] for k in ('operation','started_at','finished_at','seconds','returncode','console') if k in d}
 diagnose=load('evidence/stage3/queue-diagnose.json');cp=load('evidence/stage3/checkpoint-inspection.json')
 # 每个来源队列只算一次。用户主动暂停的整夜间隔不能充作GPU排队/计算时间。
 source_queues=q.get('source_queues',[path]);queue_rows=[];seen=set()
 for source in source_queues:
  if source in seen:continue
  seen.add(source);journal=load(source)
  queue_rows.append(dict(path=source,status=journal['status'],prelaunch_polling_wall_seconds=journal.get('prelaunch_polling_wall_seconds',0),busy_observations=sum(bool(o.get('busy')) for o in journal.get('observations',[]))))
 elapsed=(datetime.datetime.fromisoformat(q['finished_at'])-datetime.datetime.fromisoformat(q['started_at'])).total_seconds()
 result=dict(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='stage3 only;phase2 timings remain historical',
  gpu_development_jobs=rows,gpu_development_child_wall_seconds=sum(r['child_wall_seconds'] for r in rows),
  gpu_diagnostic_child_wall_seconds=sum(r['seconds'] for r in diagnose['segments']),
  gpu_child_wall_definition='single GPU process wall time including loading,training,validation,checkpointing;not isolated CUDA kernel time;all interrupted attempts included',
  queue_started_at=q['started_at'],queue_finished_at=q['finished_at'],queue_elapsed_including_user_pause_seconds=elapsed,
  external_busy_observations=sum(r['busy_observations'] for r in queue_rows),source_queue_timings=queue_rows,
  waiting_definition='busy observation counts are not durations;prelaunch polling wall is measured by source controllers;user pause and detached CPU-only observer are excluded from GPU compute and controller polling',
  completed_operations=ops,remote_cpu_checkpoint_verification_seconds=cp['seconds'],
  new_bert_weight_downloads=0,new_model_download_reason='reuse stage2 fixed official BERT;no new weight transfer to Mac',
  stage3_training_resume_events=sum(max(0,len(r['source_segments'])-1) for r in rows),
  interrupted_training_segments=sum(x['summary_status']=='segment_limit' for r in rows for x in r['source_segments']),
  discarded_partial_updates=sum(r['discarded_partial_updates'] for r in rows),discarded_partial_update_seconds=sum(r['discarded_partial_update_seconds'] for r in rows),
  resume_reason='external occupancy safety pause;raw interruption and CPU restore/replay proofs retained',
  prelaunch_resume_polling_wall_seconds=sum(r['prelaunch_polling_wall_seconds'] for r in queue_rows),
  user_requested_pause='evidence/stage3/user-requested-pause.json;interval not reclassified as GPU occupancy waiting',
  accounting='complete-epoch train/validation/save counted once from final cumulative records;discarded partial update seconds separately added;all source runner/child walls counted once;parallel CPU/API operations never summed as sequential wall time',
  throughput_definition='committed optimizer updates*32 divided by committed complete-epoch training_loop_seconds;discarded replay updates separately reported',
  evidence_copy_timing='evidence/stage3/final-evidence-transfer.json;generated by actual final transfer command')
 dump(ROOT/'evidence/stage3/runtime-summary.json',result)
 (ROOT/'evidence/stage3/training-console-excerpts.log').write_text('\n'.join(excerpts),encoding='utf-8')
 print(json.dumps(dict(status='PASS',gpu_development_child_wall_seconds=result['gpu_development_child_wall_seconds'],external_busy_observations=result['external_busy_observations'],interrupted_segments=result['interrupted_training_segments'])))

if __name__=='__main__':main()
