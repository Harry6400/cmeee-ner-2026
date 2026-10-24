"""CPU归档真实中断，恢复已完成轮状态；不启动GPU/API，不访问sealed。"""
import json,os,shutil,time,datetime,gc
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4)
from ner_core import ROOT,digest,dump

def same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def main():
 t=time.perf_counter();run=ROOT/'runs/stage3-20261019/N1-rope-off';out=run/'segments/segment-00';out.mkdir(parents=True,exist_ok=False)
 q=json.loads((ROOT/'evidence/stage3/queue-develop.json').read_text());assert q['status']=='failed'
 s=json.loads((run/'summary.json').read_text());assert s['status']=='segment_limit' and s['epochs']==8
 for f in ['summary.json','metadata.json','progress.json','config.json']:
  shutil.copy2(run/f,out/f)
 for f in ['best.pt','last.pt']:os.link(run/f,out/f)
 shutil.move(str(run/'metrics.jsonl'),str(out/'metrics.jsonl'));shutil.move(str(run/'tensorboard'),str(out/'tensorboard'))
 last=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert last['epoch']==8 and last['global_step']==3472
 records=[json.loads((run/f'epoch-{i:02d}.json').read_text()) for i in range(1,9)]
 for a,b in zip(last['records'],records):
  assert {k:v for k,v in a.items() if k!='save_seconds'}=={k:v for k,v in b.items() if k!='save_seconds'}
 normalized=dict(last,records=records);target=out/'resume-normalized.pt';torch.save(normalized,target)
 loaded=torch.load(target,map_location='cpu',weights_only=False)
 for key,value in last.items():
  if key!='records':assert same(value,loaded[key]),key
 assert same(records,loaded['records']);assert not torch.cuda.is_initialized()
 metrics=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()];partial=[x for x in metrics if x.get('event')=='train' and x['global_step']>last['global_step']]
 result=dict(status='PASS',recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),device='cpu',seconds=time.perf_counter()-t,
  original_checkpoint=str((out/'last.pt').relative_to(ROOT)),original_sha256=digest(out/'last.pt'),
  normalized_checkpoint=str(target.relative_to(ROOT)),normalized_sha256=digest(target),
  normalization='only records.save_seconds recovered from original completed epoch JSON;all other values including model,optimizer,RNG checked recursively identical after CPU serialization',
  restored_epoch=8,restored_global_step=3472,interrupted_summary_global_step=s['global_step'],discarded_partial_updates=len(partial),
  discarded_partial_update_seconds=sum(x['seconds'] for x in partial),raw_segment_directory=str(out.relative_to(ROOT)),
  archived_files={f:digest(out/f) for f in ['summary.json','metadata.json','progress.json','config.json','metrics.jsonl']},
  original_tensorboard_preserved=True,original_console='evidence/stage3/N1-rope-off.log',original_queue_preserved=True)
 assert len(partial)==s['global_step']-3472
 dump(ROOT/'evidence/stage3/resume-preparation.json',result);print(json.dumps(result))

if __name__=='__main__':main()
