"""阶段二CLI：subset、baseline、train、predict、audit。禁止读取sealed与调用API。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from ner_core import ROOT,TYPES,dump,read_rows,write_rows,digest,validate,score,windows,merge,make_subset,dictionary,dictionary_predict


def now():
    return datetime.now(timezone.utc).isoformat()


def log(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def load_cfg(path):
    cfg = json.loads(Path(path).read_text())
    if cfg['stage'] != 2 or cfg['api_enabled'] or cfg['train_samples'] > 128 or cfg['dev_samples'] > 64:
        raise ValueError('超出阶段二范围')
    if cfg['micro_batch']*cfg['accumulation'] != 32:
        raise ValueError('有效批量必须为32')
    return cfg


def baseline(cfg, subset, output):
    output.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    table, ambiguity = dictionary(read_rows(subset/'train.jsonl'))
    build_seconds = time.perf_counter()-t0
    rows = read_rows(subset/'dev.jsonl')
    timings, predicted = [], []
    for row in rows:
        t = time.perf_counter()
        predicted.append(dictionary_predict(row,table))
        timings.append(time.perf_counter()-t)
    write_rows(output/'predictions.jsonl',predicted)
    dump(output/'dictionary.json',table)
    dump(output/'ambiguity.json',ambiguity)
    dump(output/'metrics.json',score(rows,predicted))
    dump(output/'timing.json',dict(build_seconds=build_seconds,total_seconds=time.perf_counter()-t0,
         per_sample_seconds=timings,scope='128 train dictionary;64 internal dev;smoke only'))
    print(json.dumps(dict(dictionary_terms=len(table),ambiguous_terms=len(ambiguity),
                         metrics=score(rows,predicted)),ensure_ascii=False),flush=True)


def make_vocab(method, train, cfg):
    if method=='N0':
        chars = sorted({ch for r in train for ch in r['text']})
        return {'[PAD]':0,'[UNK]':1,**{c:i+2 for i,c in enumerate(chars)}}
    return {ch:i for i,ch in enumerate((ROOT/cfg['bert_directory']/'vocab.txt').read_text().splitlines())}


def cuda_gate(cfg):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != cfg['gpu_uuid']:
        raise ValueError('CUDA_VISIBLE_DEVICES必须为授权UUID')
    q = subprocess.check_output(['nvidia-smi','-i',cfg['gpu_uuid'],
        '--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip()
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                                   '--format=csv,noheader'],text=True)
    if any(cfg['gpu_uuid'] in line for line in apps.splitlines()):
        raise RuntimeError('GPU7存在计算进程；等待，不抢卡')
    fields = [s.strip() for s in q.split(',')]
    if int(fields[1].split()[0])>128 or int(fields[2].split()[0])<12000:
        raise RuntimeError('GPU7显存状态不满足空闲启动门槛')
    if __import__('shutil').disk_usage(ROOT).free < 20*1024**3:
        raise RuntimeError('共享磁盘余量不足20GiB')
    return q


def state_digest(state):
    h = hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(cfg, method, subset, output, target, resume, max_seconds):
    started, t0 = now(), time.perf_counter()
    gate = cuda_gate(cfg)
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from ner_model import Extractor,configure,collate,loss_fn
    configure(cfg['seed'],cfg['cpu_threads'])
    if torch.cuda.device_count()!=1:
        raise RuntimeError('只允许单卡')
    output.mkdir(parents=True, exist_ok=False)
    segment = 'resume' if resume else 'initial'
    writer = SummaryWriter(str(output/'tensorboard'))
    history = output/'metrics.jsonl'
    interrupted = [False]
    signal.signal(signal.SIGTERM,lambda *args: interrupted.__setitem__(0,True))
    train_rows = read_rows(subset/'train.jsonl')
    dev_rows = read_rows(subset/'dev.jsonl')
    train_windows = [w for r in train_rows for w in windows(r,cfg['window_chars'],cfg['overlap_chars'])]
    vocab = make_vocab(method,train_rows,cfg)
    bert_cfg = None
    if method=='N1':
        bert_cfg = json.loads((ROOT/cfg['bert_directory']/'config.json').read_text())
    checkpoint = torch.load(resume,map_location='cpu',weights_only=False) if resume else None
    if checkpoint and (checkpoint['config']!=cfg or checkpoint['method']!=method or
                       checkpoint['subset_sha256']!=digest(subset/'manifest.json')):
        raise ValueError('恢复配置、方法或样本清单不一致')
    model = Extractor(method,cfg,vocab,bert_cfg,
                      str(ROOT/cfg['bert_directory']) if method=='N1' and not resume else None)
    if checkpoint:
        model.load_state_dict(checkpoint['model'],strict=True)
    model.to('cuda')
    if method=='N1':
        optimizer = torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':cfg['n1_encoder_lr']},
                                      {'params':model.head.parameters(),'lr':cfg['n1_head_lr']}])
    else:
        optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['n0_lr'])
    step,cursor = 0,0
    restore = None
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        step,cursor = checkpoint['global_step'],checkpoint['cursor']
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        restore = dict(parent=str(resume),parent_sha256=digest(resume),global_step=step,cursor=cursor,
                       model_equal=state_digest(model.state_dict())==checkpoint['model_digest'],
                       python_rng_equal=random.getstate()==checkpoint['python_rng'],
                       numpy_rng_equal=bool(np.array_equal(np.random.get_state()[1],checkpoint['numpy_rng'][1]) and np.random.get_state()[0]==checkpoint['numpy_rng'][0] and np.random.get_state()[2:]==checkpoint['numpy_rng'][2:]),
                       torch_rng_equal=bool(torch.equal(torch.get_rng_state(),checkpoint['torch_rng'])),
                       cuda_rng_equal=all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),checkpoint['cuda_rng'])),
                       optimizer_steps=sorted({int(v['step'].item()) for v in optimizer.state.values() if 'step' in v}))
        if not all(restore[k] for k in ('model_equal','python_rng_equal','numpy_rng_equal','torch_rng_equal','cuda_rng_equal')) or restore['optimizer_steps']!=[step]:
            raise RuntimeError('恢复状态核查失败')
        dump(output/'restore-check.json',restore)
    parameter_before = model.head.proj.weight.detach().clone()
    permutation = list(range(len(train_windows)))
    random.Random(cfg['seed']).shuffle(permutation)
    # 采样由固定排列与累计cursor决定；恢复不重新从头取样。
    torch.cuda.reset_peak_memory_stats()
    setup_seconds = time.perf_counter()-t0
    log(history,dict(event='start',at=started,method=method,segment=segment,device=torch.cuda.get_device_name(),
                    gpu_gate=gate,config=cfg,parameters=sum(p.numel() for p in model.parameters()),
                    train_windows=len(train_windows),setup_seconds=setup_seconds,restore=restore,
                    global_step=step,cursor=cursor,precision='float32',tf32=False,amp=False,
                    code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('ner*.py')}))
    losses=[]
    status='completed'
    training_t0=time.perf_counter()
    try:
        model.train()
        while step < target:
            if interrupted[0] or time.perf_counter()-t0>max_seconds-25:
                status='time_limit';break
            optimizer.zero_grad(set_to_none=True)
            ids_used=[]
            t=time.perf_counter()
            loss_total=0.0
            for _ in range(cfg['accumulation']):
                batch=[]
                for _ in range(cfg['micro_batch']):
                    idx=permutation[cursor%len(permutation)]
                    batch.append(train_windows[idx]);ids_used.append(dict(id=train_windows[idx]['id'],start=train_windows[idx]['start'],cursor=cursor))
                    cursor+=1
                ids,attn,chars,labels=collate(batch,vocab,method,'cuda')
                loss=loss_fn(model(ids,attn,chars),labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('非有限训练损失')
                (loss/cfg['accumulation']).backward()
                loss_total+=loss.item()/cfg['accumulation']
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step();step+=1
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
            values=dict(event='optimizer_step',global_step=step,cursor=cursor,loss=loss_total,
                        grad_norm=float(grad),learning_rates=[g['lr'] for g in optimizer.param_groups],
                        seconds=elapsed,throughput_windows_per_sec=32/elapsed,
                        gpu_memory_peak_mb=torch.cuda.max_memory_allocated()/2**20,samples=ids_used)
            log(history,values);losses.append(loss_total)
            for tag,value in {'train/loss':loss_total,'train/grad_norm':float(grad),
                              'train/lr':optimizer.param_groups[0]['lr'],'train/optimizer_step':step,
                              'train/throughput_samples_per_sec':32/elapsed,
                              'system/gpu_memory_peak_mb':values['gpu_memory_peak_mb']}.items():
                writer.add_scalar(tag,value,step)
            writer.flush()
        train_seconds=time.perf_counter()-training_t0
        changed=not torch.equal(parameter_before,model.head.proj.weight.detach())
        if step> (checkpoint['global_step'] if checkpoint else 0) and not changed:
            raise RuntimeError('参数未实际更新')
        torch.cuda.synchronize()
        save_t=time.perf_counter()
        model_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        state=dict(method=method,config=cfg,vocab=vocab,bert_config=bert_cfg,model=model_state,global_step=step)
        torch.save(state,output/'inference.pt')
        full=dict(**state,optimizer=optimizer.state_dict(),cursor=cursor,
                  model_digest=state_digest(model_state),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                  torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                  subset_sha256=digest(subset/'manifest.json'))
        torch.save(full,output/'last.pt')
        save_seconds=time.perf_counter()-save_t
        if step!=target:
            status='time_limit'
        # GPU段只训练/恢复；64条真实开发推理在本机CPU做，节约GPU计算预算。
        summary=dict(status=status,method=method,segment=segment,global_step=step,cursor=cursor,
                     expected_target=target,parameter_updated=changed,losses=losses,train_seconds=train_seconds,
                     setup_seconds=setup_seconds,save_seconds=save_seconds,total_seconds=time.perf_counter()-t0,
                     peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,
                     inference_sha256=digest(output/'inference.pt'),started_at=started,finished_at=now(),
                     checkpoint=str(output/'last.pt'),validation='local CPU after GPU process exits',
                     omitted_tensorboard_tags='image metrics and patient gates do not apply to NER; epoch_time unavailable in step-bounded smoke')
        dump(output/'summary.json',summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as exc:
        dump(output/'failure.json',dict(at=now(),error=repr(exc),global_step=step,cursor=cursor,
                                       seconds=time.perf_counter()-t0))
        raise
    finally:
        writer.flush();writer.close()


def predict(cfg, checkpoint, subset, output):
    import torch
    from ner_model import Extractor,configure,collate,decode
    configure(cfg['seed'],cfg['cpu_threads'])
    output.mkdir(parents=True,exist_ok=False)
    t0=time.perf_counter()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    method=state['method']
    if state['config']!=cfg:
        raise ValueError('推理配置与训练不一致')
    model=Extractor(method,cfg,state['vocab'],state['bert_config'])
    model.load_state_dict(state['model'],strict=True)
    model.eval()
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise RuntimeError('本次推理必须为CPU')
    load_seconds=time.perf_counter()-t0
    gold=read_rows(subset/'dev.jsonl')
    predicted=[];timings=[];tensor_records=[]
    with torch.inference_mode():
        for row in gold:
            start=time.perf_counter()
            ws=windows(row,cfg['window_chars'],cfg['overlap_chars'])
            pieces=[]
            for i in range(0,len(ws),cfg['micro_batch']):
                batch=ws[i:i+cfg['micro_batch']]
                ids,attn,chars,_=collate(batch,state['vocab'],method,labels=False)
                logits=model(ids,attn,chars)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('非有限推理分数')
                pieces.extend(zip(batch,decode(logits,batch,method,cfg['decode_threshold'])))
                tensor_records.append(dict(sample_id=row['id'],windows=len(batch),shape=list(logits.shape)))
            result=merge(row,pieces)
            predicted.append(result);timings.append(time.perf_counter()-start)
    write_rows(output/'predictions.jsonl',predicted)
    metrics=score(gold,predicted)
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(output/'tensorboard')) as tb:
        for key in ('precision','recall','f1'):
            tb.add_scalar('val/'+key,metrics['micro'][key],1)
        tb.add_scalar('val/nested_recall',metrics['nested_recall'],1)
        tb.flush()
    dump(output/'metrics.json',metrics)
    values=sorted(timings)
    dump(output/'summary.json',dict(method=method,device='cpu',platform=platform.platform(),
         torch_version=torch.__version__,checkpoint_sha256=digest(checkpoint),samples=len(predicted),
         load_seconds=load_seconds,inference_seconds=sum(timings),total_seconds=time.perf_counter()-t0,
         latency_mean_seconds=sum(timings)/len(timings),latency_p50_seconds=values[len(values)//2],
         latency_p95_seconds=values[min(len(values)-1,__import__('math').ceil(.95*len(values))-1)],
         per_sample_seconds=timings,unknown_characters=sum(ch not in state['vocab'] for r in gold for ch in r['text']),
         characters=sum(len(r['text']) for r in gold),sealed_read=False,
         scope='64 internal dev smoke; not final performance',metric=metrics['micro']))
    dump(output/'tensor-shapes.json',tensor_records)
    print(json.dumps(dict(method=method,device='cpu',samples=len(predicted),metrics=metrics['micro'],
                         load_seconds=load_seconds,inference_seconds=sum(timings)),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/stage2.json')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('subset');p.add_argument('--output',type=Path,required=True)
    for name in ('baseline','train','predict','audit'):
        p=sub.add_parser(name);p.add_argument('--subset',type=Path,required=True)
        if name!='audit':p.add_argument('--output',type=Path,required=True)
        if name=='train':
            p.add_argument('--method',choices=['N0','N1'],required=True)
            p.add_argument('--target-step',type=int,required=True)
            p.add_argument('--resume',type=Path)
            p.add_argument('--max-seconds',type=float,default=240)
        if name=='predict':p.add_argument('--checkpoint',type=Path,required=True)
        if name=='audit':p.add_argument('--predictions',type=Path,required=True)
    args=parser.parse_args();cfg=load_cfg(args.config)
    if args.command=='subset':
        print(json.dumps(make_subset(cfg,args.output),ensure_ascii=False))
    elif args.command=='baseline':baseline(cfg,args.subset,args.output)
    elif args.command=='train':
        if args.target_step not in (cfg['initial_steps'],cfg['initial_steps']+cfg['resume_steps']):
            raise ValueError('仅允许阶段二4步/6步目标')
        train(cfg,args.method,args.subset,args.output,args.target_step,args.resume,args.max_seconds)
    elif args.command=='predict':predict(cfg,args.checkpoint,args.subset,args.output)
    elif args.command=='audit':
        print(json.dumps(score(read_rows(args.subset/'dev.jsonl'),read_rows(args.predictions)),ensure_ascii=False))

if __name__=='__main__':
    main()
