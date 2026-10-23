"""从原始开发预测、逐轮日志和预算记录核算阶段三；不调用API或访问sealed。"""
import argparse,json,math
from pathlib import Path
from decimal import Decimal
from ner_core import ROOT,read_rows,score,digest
from stage3_common import extra_metrics
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from historical_evidence import historical_source,historical_budget,historical_isolation
from stage4_metrics import dump_new


def inspect_training_segments(directory,summary,metadata,lifecycle,required,root=ROOT):
 """验证原始分段而不把恢复后的累计records重复当成训练。"""
 train_tags={'train/loss','train/lr','train/grad_norm','train/optimizer_step','train/throughput_samples_per_sec','system/gpu_memory_peak_mb'}
 sources=lifecycle['segments'] if lifecycle else [dict(name=directory.name)]
 details=[];all_tags=set();discarded_updates=0;discarded_seconds=0.;runner_seconds=0.;archived_metrics=[]
 for index,source in enumerate(sources):
  if source.get('runner_summary'):summary_path=root/source['runner_summary'];segment_dir=summary_path.parent
  elif source.get('archive_dir'):segment_dir=root/source['archive_dir'];summary_path=segment_dir/'summary.json'
  elif index==len(sources)-1:segment_dir=directory;summary_path=segment_dir/'summary.json'
  else:segment_dir=directory/'segments'/f'segment-{index:02d}';summary_path=segment_dir/'summary.json'
  original=json.loads(summary_path.read_text());meta=json.loads((segment_dir/'metadata.json').read_text())
  config=json.loads((segment_dir/'config.json').read_text())
  if config!=json.loads((directory/'config.json').read_text()):raise ValueError('恢复分段配置变化')
  for field in ('method','seed','precision','amp','tf32','train_samples','dev_samples','windows','vocab_size','unknown_characters','sources'):
   if meta[field]!=metadata[field]:raise ValueError('恢复分段元数据不一致:'+field)
  for file in ('train_development.py','stage3_common.py','ner_model.py','ner_core.py','ner.py'):
   historical_source(root,'scripts/'+file,meta['code'][file])
  runner_seconds+=original['seconds']
  if original['records']!=summary['records'][:len(original['records'])]:raise ValueError('原始已完成轮记录被恢复改写')
  rows=[json.loads(line) for line in (segment_dir/'metrics.jsonl').read_text().splitlines()]
  updates=[row for row in rows if row.get('event')=='train'];committed=original['records'][-1]['global_step'] if original['records'] else 0
  partial=[row for row in updates if row['global_step']>committed]
  if len(partial)!=original['global_step']-committed:raise ValueError('未完成轮回退计数与原始训练步不符')
  if index==len(sources)-1 and partial:raise ValueError('最后分段仍有未完成轮更新')
  discarded_updates+=len(partial);discarded_seconds+=sum(row['seconds'] for row in partial)
  if updates:
   expected_first=1 if index==0 else json.loads((root/'evidence/stage3/resume-preparation.json').read_text())['restored_global_step']+1
   if updates[0]['global_step']!=expected_first or updates[0]['window_cursor']!=32:raise ValueError('分段首次恢复更新不连续')
   for previous,current in zip(updates,updates[1:]):
    if current['global_step']!=previous['global_step']+1:raise ValueError('原始分段训练步不连续')
    cursor=previous['window_cursor']+32 if current['epoch']==previous['epoch'] else 32
    if current['window_cursor']!=cursor:raise ValueError('原始分段采样位置不连续')
   if any(not math.isfinite(row[key]) for row in updates for key in ('loss','grad_norm','seconds')):raise ValueError('原始更新非有限')
  ea=EventAccumulator(str(segment_dir/'tensorboard'),size_guidance={'scalars':0});ea.Reload();tags=set(ea.Tags()['scalars']);all_tags|=tags
  if updates and not train_tags<=tags:raise ValueError('分段TensorBoard训练标签缺失')
  for tag in tags:
   if not all(math.isfinite(row.value) for row in ea.Scalars(tag)):raise ValueError('分段TensorBoard非有限')
  details.append(dict(name=source['name'],directory=str(segment_dir.relative_to(root)),status=original['status'],runner_seconds=original['seconds'],first_global_step=updates[0]['global_step'] if updates else None,last_global_step=updates[-1]['global_step'] if updates else None,logged_updates=len(updates),discarded_partial_updates=len(partial),discarded_partial_update_seconds=sum(row['seconds'] for row in partial),tensorboard_tags=sorted(tags)))
  archived_metrics.append(updates)
 if not required<=all_tags:raise ValueError('全部分段合计TensorBoard标签缺失')
 if lifecycle:
  if lifecycle['status']!='completed' or lifecycle['completed_epochs']!=summary['epochs']:raise ValueError('生命周期未完成')
  if discarded_updates!=lifecycle['discarded_partial_updates'] or not math.isclose(discarded_seconds,lifecycle['discarded_partial_update_seconds'],abs_tol=1e-6):raise ValueError('回退更新及耗时无法从所有原始分段重算')
  if not math.isclose(runner_seconds,lifecycle['total_runner_wall_seconds'],abs_tol=1e-6) or not math.isclose(sum(source['seconds'] for source in sources),lifecycle['total_child_wall_seconds'],abs_tol=1e-6):raise ValueError('生命周期墙钟求和不符')
  prep=json.loads((root/'evidence/stage3/resume-preparation.json').read_text())
  if prep['status']!='PASS' or metadata['parent_checkpoint']!=prep['normalized_sha256']:raise ValueError('恢复父检查点不符')
  for filename,expected in prep['archived_files'].items():
   if digest(directory/'segments/segment-00'/filename)!=expected:raise ValueError('首次中断原始文件被改写')
  if len(sources)>=3:
   archive=json.loads((root/'evidence/stage3/resume-attempt-01-archive.json').read_text())
   proof=json.loads((root/'evidence/stage3/resume-replay-prefix-check.json').read_text())
   if not archive['original_checkpoint_unchanged'] or not proof['same_global_steps_epochs_cursors'] or not proof['loss_all_exactly_equal'] or not proof['gradient_norm_all_exactly_equal']:raise ValueError('第二中断归档或实际恢复前缀证据不通过')
   original={row['global_step']:row for row in archived_metrics[0]};replayed=archived_metrics[1]
   if len(replayed)!=proof['compared_updates']:raise ValueError('实际重放前缀数量不符')
   for row in replayed:
    if any(row[key]!=original[row['global_step']][key] for key in ('global_step','epoch','window_cursor','loss','grad_norm')):raise ValueError('实际恢复前缀不能复核')
 return sorted(all_tags),details,runner_seconds


def verify(root=ROOT):
 base=root/'runs/stage3-20261019';dev=read_rows(root/'data/processed/dev.jsonl');train=read_rows(root/'data/processed/train.jsonl')
 frozen=json.loads((root/'data/manifests/readiness.json').read_text())
 for split in ('train','dev'):
  if digest(root/f'data/processed/{split}.jsonl')!=frozen['files'][f'processed/{split}.jsonl']:raise ValueError('冻结训练/开发文件变化')
 train_terms={e['text'] for r in train for e in r['entities']};queue_path=root/'evidence/stage3/queue-develop-resolved.json';queue=json.loads((queue_path if queue_path.exists() else root/'evidence/stage3/queue-develop.json').read_text())
 if queue['status']!='completed':raise ValueError('开发队列尚未完成')
 selected=json.loads((base/'lr-selection.json').read_text());models={}
 required={'train/loss','train/lr','train/grad_norm','train/epoch_time_sec','train/optimizer_step','train/throughput_samples_per_sec','system/gpu_memory_peak_mb','val/precision','val/recall','val/f1','val/loss','val/nested_recall','val/character_f1'}
 for name in ('N0','N1-lr-1e-05','N1-lr-3e-05','N1-lr-5e-05','N1-rope-off'):
  directory=base/name;summary=json.loads((directory/'summary.json').read_text());metadata=json.loads((directory/'metadata.json').read_text())
  cfg=json.loads((directory/'config.json').read_text());records=summary['records']
  if summary['status'] not in ('completed','early_stopped') or summary['epochs']!=len(records):raise ValueError('训练未完成')
  if (metadata['train_samples'],metadata['dev_samples'],metadata['windows'])!=(13497,1503,13883):raise ValueError('样本数不符')
  for file in ('train_development.py','stage3_common.py','ner_model.py','ner_core.py','ner.py'):
   historical_source(root,'scripts/'+file,metadata['code'][file])
  for split in ('train','dev'):
   if metadata['sources'][split]!=digest(root/f'data/processed/{split}.jsonl'):raise ValueError('划分变化')
  vocabulary=json.loads((root/'evidence/stage3/character-vocabulary-audit.json').read_text())['methods'][metadata['method']]
  if metadata['vocab_size']!=vocabulary['vocab_size'] or any(metadata['unknown_characters'][s]!=vocabulary['splits'][s]['unknown_occurrences'] for s in ('train','dev')):raise ValueError('实际字符词表或UNK数不符')
  for rec in records:
   if rec['global_step']!=434*rec['epoch'] or rec['padded_training_windows']!=5 or rec['val']['samples']!=1503:
    raise ValueError('实际步数、尾批或dev分母不符')
  if cfg['micro_batch']*cfg['accumulation']!=32 or metadata['precision']!='fp32' or metadata['amp'] or metadata['tf32']:
   raise ValueError('批量或精度变化')
  best=max(records,key=lambda r:(r['val']['micro']['f1'],-r['epoch']))
  if (best['epoch'],best['val']['micro']['f1'])!=(summary['best_epoch'],summary['best_f1']):raise ValueError('best选择不符')
  predictions=read_rows(directory/f"dev-epoch-{summary['best_epoch']:02d}.jsonl")
  metrics=score(dev,predictions);metrics.update(extra_metrics(dev,predictions,train_terms))
  if metrics!=summary['metrics']:raise ValueError('原始预测指标无法重算')
  segment=next(s for s in queue['segments'] if s['name']==name)
  if any(seg['returncode'] or seg['seconds']>84600 for seg in segment.get('source_segments',[segment])):raise ValueError('运行失败或连续段超过23h30m')
  lifecycle=json.loads((directory/'lifecycle.json').read_text()) if (directory/'lifecycle.json').exists() else None
  tags,segment_checks,total_runner_seconds=inspect_training_segments(directory,summary,metadata,lifecycle,required,root=root)
  if lifecycle and segment['source_segments']!=lifecycle['segments']:raise ValueError('派生队列与生命周期分段不一致')
  models[name]=dict(best_epoch=summary['best_epoch'],epochs=summary['epochs'],strict=metrics['micro'],
      character=metrics['character_type_micro'],relaxed=metrics['relaxed_overlap_same_type_one_to_one'],
      nested_recall=metrics['nested_recall'],unseen=metrics['unseen_surface_strict'],seconds=total_runner_seconds,interrupted_lifecycle=lifecycle,
      child_wall_seconds=segment['seconds'],peak_memory_mb=max(r.get('peak_memory_mb',0) for r in summary['records']),
      segment_checks=segment_checks,tensorboard_tags=sorted(tags),best_checkpoint_sha256=summary['best_checkpoint_sha256'])
 candidates=selected['candidates'];expected=max(candidates,key=lambda c:(c['f1'],-c['lr']))
 if selected['selected']!=expected:raise ValueError('学习率选择不符')
 rope_cfg=json.loads((base/'N1-rope-off/config.json').read_text())
 if rope_cfg['rope'] or rope_cfg['n1_encoder_lr']!=expected['lr']:raise ValueError('RoPE实验配置不符')
 chosen_cfg=json.loads((base/expected['name']/'config.json').read_text())
 if {k:v for k,v in rope_cfg.items() if k!='rope'}!={k:v for k,v in chosen_cfg.items() if k!='rope'}:raise ValueError('RoPE不止一个因素变化')
 plan=json.loads((root/'configs/rope-comparison-plan.json').read_text())
 from datetime import datetime
 offmeta_path=base/'N1-rope-off/segments/segment-00/metadata.json'
 offmeta=json.loads((offmeta_path if offmeta_path.exists() else base/'N1-rope-off/metadata.json').read_text())
 if datetime.fromisoformat(plan['recorded_at'])>=datetime.fromisoformat(offmeta['started_at']):raise ValueError('固定预算方案未在off前记录')
 fixed={}
 for name in (expected['name'],'N1-rope-off'):
  summary=json.loads((base/name/'summary.json').read_text());rec=summary['records'][plan['fixed_epoch']-1]
  pred=read_rows(base/name/f"dev-epoch-{plan['fixed_epoch']:02d}.jsonl")
  m=score(dev,pred)
  if rec['global_step']!=plan['optimizer_steps'] or m['micro']!=rec['val']['micro']:raise ValueError('固定预算指标不符')
  fixed[name]=dict(epoch=plan['fixed_epoch'],optimizer_steps=rec['global_step'],strict=m['micro'])
 baseline=json.loads((base/'baselines/summary.json').read_text())
 for mode in ('character','jieba-boundary'):
  pred=read_rows(base/'baselines'/(mode+'-predictions.jsonl'));m=score(dev,pred);m.update(extra_metrics(dev,pred,train_terms))
  if m!=baseline['results'][mode]['metrics']:raise ValueError('D0指标不能重算')
 performance=json.loads((root/'evidence/stage3/performance-summary.json').read_text())
 if performance['status']!='completed' or not performance['selected_equivalence_passed'] or performance['tuning_budget_seconds']!=110 or performance['tuning_total_seconds']>110:raise ValueError('新增吞吐核查尚未完成或等价/预算未通过')
 api=json.loads((root/'evidence/stage3/api-audit.json').read_text())
 if api['status']!='PASS':raise ValueError('API核算未通过')
 ledger,original,budget=historical_budget(root,3)
 if ledger['requests'][:816]!=original['requests'] or ledger['api_enabled'] or Decimal(ledger['spent_cny'])>100:raise ValueError('费用历史或关闭状态不符')
 isolation=historical_isolation(root,3)
 result=dict(status='PASS_WITH_WARNINGS',models=models,fixed_budget_rope=fixed,lr_selection=selected,baseline=baseline,api_summary=api,
    performance_summary=performance,budget=budget,external_isolation=isolation,sealed_evaluation=False,
    diagnostics_not_formal_scores=True,main_scores_are_internal_dev=True,no_claim_final_generalization=True)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Historical verification output already exists')
 result=verify(a.root);dump_new(a.output,result)
 print(json.dumps(dict(status=result['status'],models=list(result['models']),selected_lr=result['lr_selection']['selected']['lr'],historical_cumulative_cny=result['budget']['spent_cny'])))

if __name__=='__main__':main()
