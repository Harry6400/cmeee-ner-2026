"""Authorized final holdout evaluation; training-scientific sources remain immutable.
Full5000 uses frozen batch4 evaluate; common500 main predictions are ID-aligned subset.
A separate no-gold single-sample pass over fixed common500 measures end-to-end latency.
"""
import argparse,collections,json,math,os,time,signal
from datetime import datetime,timezone
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from ner_core import ROOT,TYPES,read_rows,write_rows,dump,digest,validate,score,keys,windows,merge,coverage
from ner_model import Extractor,configure,collate,decode
from stage3_common import evaluate,extra_metrics
from ner import cuda_gate

ACTIVE_OUTPUT=None
STOP=[False]
def now():return datetime.now(timezone.utc).isoformat()
def stop_before_forward(*_):
    if STOP[0]:raise InterruptedError('authorized own evaluation stopped at forward boundary')
def authorized_rows(path,expected):
    path=Path(path).resolve()
    if path not in [(ROOT/'data/sealed/holdout.jsonl').resolve(),(ROOT/'data/sealed/common500.jsonl').resolve()]:raise ValueError('unregistered sealed path')
    readiness=json.loads((ROOT/'data/manifests/readiness.json').read_text())
    if digest(path)!=readiness['files'][str(path.relative_to(ROOT/'data'))]:raise ValueError('final data differs from frozen source')
    rows=[json.loads(x) for x in path.read_text().splitlines() if x]
    if len(rows)!=expected or len({r['id'] for r in rows})!=expected:raise ValueError('sealed sample count orIDs mismatch')
    for r in rows:validate(r)
    return rows

def latency_stats(values):
    a=sorted(values)
    def quantile(q):
        index=(len(a)-1)*q;lo=math.floor(index);hi=math.ceil(index);return a[lo]+(a[hi]-a[lo])*(index-lo)
    return dict(samples=len(a),total_seconds=sum(a),mean_seconds=sum(a)/len(a),p50_seconds=quantile(.5),p95_seconds=quantile(.95),quantile='linear interpolation of (n-1)*q')
def scalars(directory,metrics,split):
    with SummaryWriter(str(directory/'tensorboard')) as writer:
        for key in ['precision','recall','f1']:writer.add_scalar(split+'/'+key,metrics['micro'][key],1)
        writer.add_scalar(split+'/nested_recall',metrics['nested_recall'],1)
        for typ in TYPES:writer.add_scalar(split+'/class_f1/'+typ,metrics['by_type'][typ]['f1'],1)
        writer.flush()
def build_manifest(route,seed,split,directory,checkpoint_hash):
    return dict(route=route,seed=seed,split=split,training_run=str(checkpoint_hash[1]),training_summary_path=str(Path(checkpoint_hash[1])/'summary.json'),path=str((directory/'predictions.jsonl').relative_to(ROOT)),metrics_path=str((directory/'metrics.json').relative_to(ROOT)),timing_path=str((directory/'timing.json').relative_to(ROOT)),metadata_path=str((directory/'metadata.json').relative_to(ROOT)),checkpoint_sha256=checkpoint_hash[0])
def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,default=ROOT/'configs/qha-stage4-contract.json');p.add_argument('--protocol',type=Path,default=ROOT/'configs/stage4-frozen-protocol.json');p.add_argument('--method',choices=['N0','N1'],required=True);p.add_argument('--seed',type=int,choices=[20261019,20261020,20261021],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    # Contract schema is frozen by the parent before any final access. Same exact bytes checked by queue.
    execution=json.loads(a.contract.read_text());protocol=json.loads(a.protocol.read_text())
    if execution.get('stage')!=4 or execution.get('authorization')!='explicit_user_stage4_2026-10-20' or not execution.get('final_holdout_authorized'):raise ValueError('explicit stage4 final evaluation authorization required')
    if execution['frozen_protocol_sha256']!=digest(a.protocol):raise ValueError('frozen protocol changed')
    if ROOT.resolve()!=Path('/nfsdata_a40/cyf/codes/CT/cmeee_ner') or not a.output.resolve().is_relative_to(ROOT.resolve()):raise ValueError('remote output isolation failed')
    cfg_expected=dict(protocol['selected'][a.method]['config']);cfg_expected['seed']=a.seed
    if a.seed!=20261019:cfg_expected['stage']=4
    if a.checkpoint.resolve().parents[1] not in [(ROOT/'runs/stage3-20261019').resolve(),(ROOT/'runs/stage4-20261020').resolve()]:raise ValueError('checkpoint outside registered run roots')
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT=a.output
    signal.signal(signal.SIGTERM,lambda *_:STOP.__setitem__(0,True))
    gate=cuda_gate(cfg_expected);configure(a.seed,cfg_expected['cpu_threads']);start=time.perf_counter();started=now();a.output.mkdir(parents=True,exist_ok=False)
    full=authorized_rows(ROOT/'data/sealed/holdout.jsonl',5000);common=authorized_rows(ROOT/'data/sealed/common500.jsonl',500)
    source_rows={r['id']:r for r in full}
    if any(source_rows.get(r['id'])!=r for r in common):raise ValueError('common500 must be exact full5000 subset')
    train=read_rows(ROOT/'data/processed/train.jsonl');train_terms={e['text'] for r in train for e in r['entities']}
    load_start=time.perf_counter();checkpoint_hash=digest(a.checkpoint);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if state['method']!=a.method or state['config']!=cfg_expected:raise ValueError('checkpoint selection/config mismatch')
    if any(not torch.isfinite(t).all() for t in state['model'].values()):raise ValueError('non-finite checkpoint')
    if any(t.is_floating_point() and t.dtype!=torch.float32 for t in state['model'].values()):raise ValueError('actual checkpoint precision differs fromFP32')
    model=Extractor(a.method,state['config'],state['vocab'],state['bert_config']);model.load_state_dict(state['model'],strict=True);model.to('cuda');model.eval();model.register_forward_pre_hook(stop_before_forward);torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    if torch.cuda.device_count()!=1 or any(x.dtype!=torch.float32 for x in model.parameters()):raise ValueError('singleGPU FP32 required')
    torch.cuda.reset_peak_memory_stats()
    full_pred,full_metrics,batch=evaluate(model,full,state['vocab'],state['config'],a.method,device='cuda',train_terms=train_terms)
    pred_byid={r['id']:r for r in full_pred};common_pred=[pred_byid[r['id']] for r in common]
    common_metrics=score(common,common_pred);common_metrics.update(extra_metrics(common,common_pred,train_terms))
    full_dir=a.output/'holdout5000';common_dir=a.output/'common500';full_dir.mkdir();common_dir.mkdir()
    write_rows(full_dir/'predictions.jsonl',full_pred);write_rows(common_dir/'predictions.jsonl',common_pred)
    dump(full_dir/'metrics.json',full_metrics);dump(common_dir/'metrics.json',common_metrics)
    # Loading and batch-evaluation timing are distinct from single-sample response timing.
    # No labels enter this separate pass; original main predictions never change.
    times=[];single_predictions=[];mismatches=[]
    with torch.inference_mode(),(common_dir/'per-sample-timing.jsonl').open('x') as timinglog:
        for index,row in enumerate(common):
            stop_before_forward()
            torch.cuda.synchronize();t=time.perf_counter();ws=windows(dict(id=row['id'],text=row['text'],entities=[]),cfg_expected['window_chars'],cfg_expected['overlap_chars']);parts=[]
            for i in range(0,len(ws),cfg_expected['micro_batch']):
                sub=ws[i:i+cfg_expected['micro_batch']];ids,attn,mask,_=collate(sub,state['vocab'],a.method,'cuda',labels=False);logits=model(ids,attn,mask)
                if not torch.isfinite(logits).all():raise ValueError('non-finite single-sample logits')
                parts.extend(zip(sub,decode(logits,sub,a.method,cfg_expected['decode_threshold'])))
            result=merge(row,parts);torch.cuda.synchronize();seconds=time.perf_counter()-t;single_predictions.append(result);times.append(seconds)
            equal=keys(result)==keys(pred_byid[row['id']]);record=dict(id=row['id'],index=index,seconds=seconds,windows=len(ws),characters=len(row['text']),single_sample=True,first_sample=index==0,main_batch_prediction_equal=equal)
            timinglog.write(json.dumps(record,ensure_ascii=False)+'\n');timinglog.flush()
            if not equal:mismatches.append(dict(id=row['id'],single_minus_main=sorted(keys(result)-keys(pred_byid[row['id']])),main_minus_single=sorted(keys(pred_byid[row['id']])-keys(result))))
            if (index+1)%50==0:print(json.dumps(dict(event='latency',method=a.method,seed=a.seed,samples=index+1,mean_seconds=sum(times)/len(times))),flush=True)
    write_rows(common_dir/'single-sample-diagnostic-predictions.jsonl',single_predictions)
    dump(common_dir/'single-vs-batch-diagnostic.json',dict(status='PASS' if not mismatches else 'DIFFERENCE',mismatches=mismatches,main_outputs_unchanged=True,definition='Exact tuple equality; FP32 numerical batch-shape differences recorded, never used for selecting parameters'))
    common_timing=dict(load_seconds=load_seconds,single_sample_latency=latency_stats(times),first_single_sample_seconds=times[0],warmup='none; every fixed500 item included, first identified',scope='fixed common500; one sample with its windows per call; model already loaded; preprocessing,decode,merge andCUDA synchronization included',main_predictions='derived exact ID subset from full5000 batch4 outputs',batch_full5000=batch)
    dump(common_dir/'timing.json',common_timing);dump(full_dir/'timing.json',dict(load_seconds=load_seconds,batch_evaluation=batch,throughput_samples_per_second=5000/batch['seconds'],single_sample_latency='not inferred from batch throughput; see same common500 per-sample timings',common500_latency_reference=str((common_dir/'timing.json').relative_to(ROOT))))
    shared=dict(started_at=started,finished_at=now(),method=a.method,seed=a.seed,checkpoint=str(a.checkpoint.relative_to(ROOT)),checkpoint_sha256=checkpoint_hash,checkpoint_epoch=state['epoch'],checkpoint_global_step=state['global_step'],checkpoint_selected_on='internal dev only; never final holdout',device=torch.cuda.get_device_name(),gpu_uuid=cfg_expected['gpu_uuid'],gate=gate,precision='fp32',amp=False,tf32=False,load_seconds=load_seconds,total_wall_seconds=time.perf_counter()-start,peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,source_sha256={n:digest(ROOT/'scripts'/n) for n in ['local_evaluate_stage4.py','ner_core.py','ner_model.py','stage3_common.py']},contract_sha256=digest(a.contract),protocol_sha256=digest(a.protocol),train_source_sha256=digest(ROOT/'data/processed/train.jsonl'),full_source_sha256=digest(ROOT/'data/sealed/holdout.jsonl'),common_source_sha256=digest(ROOT/'data/sealed/common500.jsonl'),no_tuning_on_holdout=True,invalid_output_count=0)
    for split,rows,pred,metrics,directory in [('holdout5000',full,full_pred,full_metrics,full_dir),('common500',common,common_pred,common_metrics,common_dir)]:
        data=dict(shared,split=split,samples=len(rows),ordered_ids=[r['id'] for r in rows],predictions_sha256=digest(directory/'predictions.jsonl'),reference_coverage=coverage(rows,cfg_expected['window_chars'],cfg_expected['overlap_chars']),provenance='single full5000 frozen batch4 inference' if split=='holdout5000' else 'ID-ordered exact subset from full5000; not an additional independent test set')
        dump(directory/'metadata.json',data);scalars(directory,metrics,'test');dump(directory/'manifest-entry.json',build_manifest(a.method,a.seed,split,directory,(checkpoint_hash,str(a.checkpoint.parent.relative_to(ROOT)))))
    dump(a.output/'summary.json',dict(status='completed',method=a.method,seed=a.seed,full5000=full_metrics['micro'],common500=common_metrics['micro'],single_sample_latency=latency_stats(times),loading_seconds=load_seconds,seconds=time.perf_counter()-start,source_split='final holdout; no parameter selection',paid_api_calls=0));print(json.dumps(json.loads((a.output/'summary.json').read_text()),ensure_ascii=False),flush=True)
if __name__=='__main__':
    try:main()
    except InterruptedError as exc:
        if ACTIVE_OUTPUT is not None and ACTIVE_OUTPUT.exists():dump(ACTIVE_OUTPUT/'interruption.json',dict(at=now(),status='interrupted',error=repr(exc),final_outputs_are_incomplete=True))
        raise SystemExit(3)
    except Exception as exc:
        if ACTIVE_OUTPUT is not None and ACTIVE_OUTPUT.exists():dump(ACTIVE_OUTPUT/'failure.json',dict(at=now(),status='failed',error=repr(exc)))
        raise
