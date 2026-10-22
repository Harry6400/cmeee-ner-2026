"""Retrospective processed-dev threshold/latency experiments. No API, GPU or sealed reads."""
from __future__ import annotations
import argparse,json,math,platform,statistics,subprocess,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import torch
from ner_core import ROOT,read_rows,dump,write_rows,digest,windows,merge,keys,score
from ner_model import collate,decode
from cpu_inference import CPUModels
from stage3_common import extra_metrics
from stage5_repairs import Engine

def now():return datetime.now(timezone.utc).isoformat()
def stats(v):
 a=sorted(v)
 def q(x):
  at=(len(a)-1)*x;l=math.floor(at);h=math.ceil(at);return a[l]+(a[h]-a[l])*(at-l)
 return dict(n=len(a),total_seconds=sum(a),mean_seconds=statistics.mean(a),p50_seconds=q(.5),p95_seconds=q(.95),quantile='linear interpolation at(n-1)*q')
def data(study):
 for name in ('train','dev'):
  p=ROOT/f'data/processed/{name}.jsonl'
  if digest(p)!=study[f'{name}_sha256']:raise ValueError('processed split differs from registered supplement')
 train=read_rows(ROOT/'data/processed/train.jsonl');dev=read_rows(ROOT/'data/processed/dev.jsonl')
 if len(train)!=13497 or len(dev)!=1503:raise ValueError('wrong split sizes')
 return train,dev

def thresholds(study,output,progress):
 output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();at=now();train,gold=data(study);inputs=[dict(id=r['id'],text=r['text']) for r in gold];train_terms={e['text'] for r in train for e in r['entities']};models=CPUModels(ROOT)
 results=[]
 for method,run in [('N0','N0'),('N1','N1-lr-5e-05')]:
  model,state=models.load(method);cfg=state['config'];parts={t:defaultdict(list) for t in study['thresholds']};clock=time.perf_counter()
  ws=[w for r in inputs for w in windows(dict(r,entities=[]),cfg['window_chars'],cfg['overlap_chars'])]
  with torch.inference_mode():
   for i in range(0,len(ws),cfg['micro_batch']):
    batch=ws[i:i+cfg['micro_batch']];ids,attn,mask,_=collate(batch,state['vocab'],method,'cpu',labels=False);logits=model(ids,attn,mask)
    if not torch.isfinite(logits).all():raise ValueError('nonfinite logits')
    previous=None
    for threshold in study['thresholds']:
     ents=decode(logits,batch,method,threshold)
     current=[{(e['start'],e['end'],e['type']) for e in ee} for ee in ents]
     if previous is not None and any(not n<=p for n,p in zip(current,previous)):raise ValueError('threshold candidate nesting violated')
     previous=current
     for w,ee in zip(batch,ents):parts[threshold][w['id']].append((w,ee))
    if i%200==0:
     s=dict(status='THRESHOLD_RUNNING',method=method,completed_windows=min(i+cfg['micro_batch'],len(ws)),total_windows=len(ws),seconds=time.perf_counter()-clock,at=now());dump(progress,s);print(json.dumps(s),flush=True)
  for threshold in study['thresholds']:
   predictions=[merge(dict(r,entities=[]),parts[threshold][r['id']]) for r in inputs];metrics=score(gold,predictions);metrics.update(extra_metrics(gold,predictions,train_terms))
   tag=f'{method}/threshold-{threshold:g}';p=output/tag;write_rows(p/'predictions.jsonl',predictions);dump(p/'metrics.json',metrics)
   record=dict(method=method,threshold=threshold,seed=cfg['seed'],metrics=metrics,predictions=str((p/'predictions.jsonl').relative_to(ROOT)),prediction_sha256=digest(p/'predictions.jsonl'),model_sha256=next(x['sha256'] for x in models.manifest['items'] if x['method']==method),load_seconds=models.loads[method],shared_forward_all_thresholds_seconds=time.perf_counter()-clock,new_API_tokens=0,new_API_calls=0)
   if threshold==0:
    s=json.loads((ROOT/f'runs/stage3-20261019/{run}/summary.json').read_text());reference_path=ROOT/f'runs/stage3-20261019/{run}/dev-epoch-{s["best_epoch"]:02d}.jsonl';old=read_rows(reference_path)
    mismatches=[dict(id=a['id'],old_only=sorted(keys(a)-keys(b)),new_only=sorted(keys(b)-keys(a))) for a,b in zip(old,predictions) if keys(a)!=keys(b)]
    record['zero_threshold_reference']=str(reference_path.relative_to(ROOT));record['zero_threshold_exact_match']=len(old)==len(predictions) and not mismatches;record['zero_threshold_mismatches']=mismatches
    if not record['zero_threshold_exact_match']:dump(p/'mismatch.json',record);raise AssertionError('zero threshold differs from original dev predictions')
   results.append(record);print(json.dumps(dict(event='threshold_completed',method=method,threshold=threshold,micro=metrics['micro'])),flush=True)
 summary=dict(status='COMPLETED',started_at=at,finished_at=now(),seconds=time.perf_counter()-start,split='processed/dev',samples=1503,reference_entities=sum(len(keys(r)) for r in gold),threshold_grid=study['thresholds'],results=results,shared_forward=True,shared_forward_time_is_not_single_condition_latency=True,all_threshold_candidate_sets_nested=True,device='cpu',precision='fp32',cpu_threads=4,new_API_calls=0,sealed_read=False,default_threshold_unchanged=True,retrospective_development_analysis=True)
 dump(output/'summary.json',summary);return summary

def latency(study,output,progress):
 output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();at=now();train,gold=data(study);inputs=[dict(id=r['id'],text=r['text']) for r in gold];models=CPUModels(ROOT);clock=time.perf_counter();engine=Engine(train,output/'jieba-cache');dictionary_load=time.perf_counter()-clock;results={}
 for method in ['D0','N0','N1']:
  if method!='D0':models.load(method)
  predict=(lambda row:engine.predict(row,'baseline')[0]) if method=='D0' else (lambda row:models.predict_many([row],method)[0][0])
  w=time.perf_counter()
  for row in inputs[:study['latency_warmup_samples']]:predict(row)
  warmup_seconds=time.perf_counter()-w;rounds=[];all_times=[];reference=None
  for repeat in range(1,study['latency_repetitions']+1):
   timing=[];predictions=[]
   for index,row in enumerate(inputs):
    t=time.perf_counter();p=predict(row);seconds=time.perf_counter()-t;predictions.append(p);timing.append(dict(id=row['id'],repeat=repeat,index=index,seconds=seconds,characters=len(row['text'])))
    if index%150==0:
     s=dict(status='LATENCY_RUNNING',method=method,repeat=repeat,completed_samples=index+1,total_samples=len(inputs),at=now());dump(progress,s);print(json.dumps(s),flush=True)
   if reference is None:reference=predictions
   elif any(keys(a)!=keys(b) for a,b in zip(reference,predictions)):raise AssertionError('repeated CPU output changed')
   p=output/method/f'round-{repeat}';write_rows(p/'predictions.jsonl',predictions);write_rows(p/'per-sample-timing.jsonl',timing)
   vals=[x['seconds'] for x in timing];all_times+=vals;rounds.append(dict(repeat=repeat,latency=stats(vals),predictions=str((p/'predictions.jsonl').relative_to(ROOT)),timing=str((p/'per-sample-timing.jsonl').relative_to(ROOT))))
  metrics=score(gold,reference);metrics.update(extra_metrics(gold,reference,{e['text'] for r in train for e in r['entities']}))
  results[method]=dict(load_seconds=dictionary_load if method=='D0' else models.loads[method],warmup_samples=study['latency_warmup_samples'],warmup_seconds=warmup_seconds,repetitions=rounds,aggregate=stats(all_times),metrics=metrics,output_repeat_equal=True,new_API_tokens=0,new_API_calls=0)
  dump(output/method/'summary.json',results[method]);print(json.dumps(dict(event='latency_completed',method=method,aggregate=stats(all_times))),flush=True)
 hardware={}
 for key in ['hw.model','machdep.cpu.brand_string','hw.memsize']:
  p=subprocess.run(['sysctl','-n',key],capture_output=True,text=True);hardware[key]=p.stdout.strip() if not p.returncode else None
 summary=dict(status='COMPLETED',started_at=at,finished_at=now(),seconds=time.perf_counter()-start,split='processed/dev',samples=1503,platform=platform.platform(),machine=platform.machine(),hardware=hardware,python=platform.python_version(),torch=torch.__version__,device='cpu',precision='fp32',cpu_threads=4,results=results,scope='model preloaded;10 warmup samples excluded;all1503 dev texts in identical order each of3 repeats;one full text per request including window/collation/forward/validation/decode/merge;model and dictionary loading excluded',replicates_are_timing_repeats_not_independent_training=True,new_API_calls=0,sealed_read=False)
 dump(output/'summary.json',summary);return summary

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['threshold','latency','all']);a=p.parse_args();study=json.loads((ROOT/'configs/stage6-supplement.json').read_text());E=ROOT/study['evidence_root'];progress=E/'cpu-progress.json';base=ROOT/study['run_root']
 if study['api_enabled'] or study['sealed_read']:raise ValueError('no API/sealed access')
 started=now();clock=time.perf_counter()
 try:
  if a.mode in ['threshold','all']:thresholds(study,base/'cpu-threshold',progress)
  if a.mode in ['latency','all']:latency(study,base/'cpu-latency',progress)
  dump(E/'cpu-completion.json',dict(status='COMPLETED',started_at=started,finished_at=now(),seconds=time.perf_counter()-clock,mode=a.mode,new_API_calls=0,sealed_read=False));dump(progress,dict(status='COMPLETED',at=now()))
 except BaseException as ex:
  dump(E/'cpu-failure.json',dict(status='FAILED',started_at=started,at=now(),seconds=time.perf_counter()-clock,error_type=type(ex).__name__,error=str(ex)));raise
if __name__=='__main__':main()
