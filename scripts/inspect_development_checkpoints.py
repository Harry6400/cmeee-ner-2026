"""只在远端CPU验证本项目训练产物，不推理、不训练、不访问sealed。"""
import json,time,gc,datetime
from pathlib import Path
import torch
torch.set_num_threads(4)
from ner_core import ROOT,digest,dump
from ner_model import Extractor


def main():
 root=ROOT/'runs/stage3-20261019';result={};start=time.perf_counter()
 for name in ('N0','N1-lr-1e-05','N1-lr-3e-05','N1-lr-5e-05','N1-rope-off'):
  p=root/name;summary=json.loads((p/'summary.json').read_text());t=time.perf_counter()
  best=torch.load(p/'best.pt',map_location='cpu',weights_only=True)
  model=Extractor(best['method'],best['config'],best['vocab'],best['bert_config']);model.load_state_dict(best['model'],strict=True)
  if best['epoch']!=summary['best_epoch'] or not all(torch.isfinite(x).all() for x in best['model'].values()):raise ValueError('best不符')
  if best['method']=='N0':
   torch.manual_seed(best['config']['seed']);initial=Extractor('N0',best['config'],best['vocab'],best['bert_config'])
   delta=(best['model']['embedding.weight']-initial.state_dict()['embedding.weight']).abs().max().item();del initial
   update_basis='N0 embedding vs fixed-seed original CPU module initialization'
  else:
   from safetensors import safe_open
   with safe_open(str(ROOT/best['config']['bert_directory']/'model.safetensors'),framework='pt',device='cpu') as original:
    key=next(k for k in original.keys() if k.endswith('embeddings.word_embeddings.weight'))
    initial_embedding=original.get_tensor(key)
    delta=(best['model']['encoder.embeddings.word_embeddings.weight']-initial_embedding).abs().max().item();del initial_embedding
   update_basis='N1 encoder word embedding vs pinned original BERT weight'
  if not delta>0:raise ValueError('未检出参数实际更新')
  actual=digest(p/'best.pt')
  if actual!=summary['best_checkpoint_sha256']:raise ValueError('best文件内容变化')
  del model,best;gc.collect()
  last=torch.load(p/'last.pt',map_location='cpu',weights_only=False)
  model=Extractor(last['method'],last['config'],last['vocab'],last['bert_config']);model.load_state_dict(last['model'],strict=True)
  if not all(torch.isfinite(x).all() for x in last['model'].values()):raise ValueError('last权重非有限')
  del model;gc.collect()
  steps=sorted({int(v['step'].item()) for v in last['optimizer']['state'].values() if 'step' in v})
  if (last['epoch'],last['global_step'],last['best_epoch'])!=(summary['epochs'],summary['global_step'],summary['best_epoch']) or steps!=[summary['global_step']]:raise ValueError('last或优化器状态不符')
  groups=[{k:v for k,v in g.items() if k!='params'} for g in last['optimizer']['param_groups']]
  result[name]=dict(status='PASS',best_epoch=summary['best_epoch'],last_epoch=last['epoch'],global_step=last['global_step'],optimizer_steps=steps,
       optimizer_groups=groups,best_sha256=actual,last_bytes=(p/'last.pt').stat().st_size,best_bytes=(p/'best.pt').stat().st_size,
       cpu_strict_load=True,finite_weights=True,parameter_update_check=dict(basis=update_basis,max_absolute_delta=delta,scope='one named tensor;not every parameter'),seconds=time.perf_counter()-t)
  progress_path=p/'progress.json';progress=json.loads(progress_path.read_text())
  before=p/'progress-before-terminal-sync.json'
  if not before.exists():before.write_bytes(progress_path.read_bytes())
  previous=progress['status'];synced_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
  progress.update(status=summary['status'],finished_at=summary['finished_at'],terminal_status_synced_at=synced_at)
  dump(progress_path,progress)
  if json.loads(progress_path.read_text())['status']!=summary['status']:raise ValueError('终态显示复测失败')
  result[name]['progress_terminal_sync']=dict(previous_status=previous,status=summary['status'],at=synced_at,
     reason='runner wrote last epoch progress as running and terminal summary separately;late display status reconciliation;raw logs unchanged')
  del last;gc.collect()
 dump(ROOT/'evidence/stage3/checkpoint-inspection.json',dict(status='PASS',device='cpu',models=result,seconds=time.perf_counter()-start))
 print(json.dumps(dict(status='PASS',models=list(result),seconds=time.perf_counter()-start)))

if __name__=='__main__':main()
