"""模板补充：独立合同的四组开发RoPE消融；不读取sealed、不调用API。"""
import argparse,json,random,time,os,signal,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from ner_core import ROOT,TYPES,read_rows,windows,dump,write_rows,digest
from ner_model import Extractor,configure,collate,loss_fn
from ner import cuda_gate,make_vocab,log
from stage3_common import evaluate


def utc():return datetime.now(timezone.utc).isoformat()


def save_atomic(state,path):
    temp=path.with_suffix('.writing.pt');torch.save(state,temp);os.replace(temp,path)


def recursive_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(recursive_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(recursive_equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/stage6-supplement.json')
    p.add_argument('--contract',type=Path,default=ROOT/'configs/qha-stage6-supplement-contract.json')
    p.add_argument('--job',required=True);p.add_argument('--resume',type=Path);a=p.parse_args()
    study=json.loads(a.protocol.read_text());contract=json.loads(a.contract.read_text())
    if ROOT.resolve()!=Path('/nfsdata_a40/cyf/codes/CT/cmeee_ner'):raise ValueError('registered remote root required')
    if contract['stage']!=6 or contract['authorization']!='explicit_user_implementation_2026-10-20' or contract['study_sha256']!=digest(a.protocol):raise ValueError('supplement authorization/config mismatch')
    if study['api_enabled'] or study['sealed_read'] or contract['api_enabled']:raise ValueError('no API/sealed access')
    if digest(ROOT/study['base_config'])!=study['base_config_sha256']:raise ValueError('original base config changed')
    if digest(ROOT/'data/processed/train.jsonl')!=study['train_sha256'] or digest(ROOT/'data/processed/dev.jsonl')!=study['dev_sha256']:raise ValueError('processed split changed')
    job=next(j for j in study['training_jobs'] if j['name']==a.job)
    a.method='N1';a.seed=job['seed'];a.output=ROOT/study['run_root']/job['name']
    if not a.output.resolve().is_relative_to((ROOT/contract['run_root']).resolve()):raise ValueError('output outside study root')
    cfg=json.loads((ROOT/study['base_config']).read_text());cfg.update(stage=6,seed=job['seed'],n1_encoder_lr=job['encoder_lr'],rope=False)
    if cfg['precision']!='fp32' or cfg['micro_batch']!=4 or cfg['accumulation']!=8 or cfg['gpu_uuid']!=contract['gpu_uuid']:raise ValueError('fixed control changed')
    if cfg['n1_max_epochs']!=10 or cfg['patience']!=3 or cfg['window_chars']!=256 or cfg['overlap_chars']!=160:raise ValueError('training control changed')
    gate=cuda_gate(cfg);configure(cfg['seed'],cfg['cpu_threads']);start=time.perf_counter();started=utc()
    if torch.cuda.device_count()!=1:raise RuntimeError('只允许单卡')
    if a.resume:
        if not a.output.is_dir():raise ValueError('续跑仅可在原run')
    else:
        if a.output.exists():
            if {x.name for x in a.output.iterdir()}-{'segments'}:raise ValueError('初始重启目录含未归档输出')
        else:a.output.mkdir(parents=True,exist_ok=False)
    console=a.output/'metrics.jsonl';writer=SummaryWriter(str(a.output/'tensorboard'))
    stopping=[False];signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    train=read_rows(ROOT/'data/processed/train.jsonl');dev=read_rows(ROOT/'data/processed/dev.jsonl')
    train_terms={e['text'] for r in train for e in r['entities']}
    ws=[w for r in train for w in windows(r,cfg['window_chars'],cfg['overlap_chars'])]
    vocab=make_vocab(a.method,train,cfg)
    bc=json.loads((ROOT/cfg['bert_directory']/'config.json').read_text()) if a.method=='N1' else None
    parent=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
    model=Extractor(a.method,cfg,vocab,bc,str(ROOT/cfg['bert_directory']) if a.method=='N1' and not parent else None)
    if parent:model.load_state_dict(parent['model'],strict=True)
    model.to('cuda')
    groups=([{'params':model.encoder.parameters(),'lr':cfg['n1_encoder_lr']},{'params':model.head.parameters(),'lr':cfg['n1_head_lr']}]
            if a.method=='N1' else [{'params':model.parameters(),'lr':cfg['n0_lr']}])
    optimizer=torch.optim.AdamW(groups)
    epoch_start=1;global_step=0;best=-1.0;best_epoch=None;bad=0;records=[]
    if parent:
        if parent['config']!=cfg:raise ValueError('续跑配置不同')
        optimizer.load_state_dict(parent['optimizer']);epoch_start=parent['epoch']+1;global_step=parent['global_step']
        best=parent['best_f1'];best_epoch=parent['best_epoch'];bad=parent['bad_epochs'];records=__import__('copy').deepcopy(parent['records'])
        metadata_normalization=[]
        for rec in records:
            if 'save_seconds' not in rec:
                historical=json.loads((a.output/f"epoch-{rec['epoch']:02d}.json").read_text())
                if historical['global_step']!=rec['global_step']:raise ValueError('历史保存计时步数不一致')
                rec['save_seconds']=historical['save_seconds'];metadata_normalization.append(dict(epoch=rec['epoch'],field='save_seconds',source=f"epoch-{rec['epoch']:02d}.json"))
        random.setstate(parent['python_rng']);np.random.set_state(parent['numpy_rng']);torch.set_rng_state(parent['torch_rng']);torch.cuda.set_rng_state_all(parent['cuda_rng'])
        restore=dict(parent_sha256=digest(a.resume),epoch=parent['epoch'],global_step=global_step,
                     optimizer_equal=recursive_equal(parent['optimizer'],optimizer.state_dict()),optimizer_steps=sorted({int(v['step'].item()) for v in optimizer.state.values() if 'step' in v}),
                     model_equal=all(torch.equal(v.detach().cpu(),parent['model'][k]) for k,v in model.state_dict().items()),
                     python_rng_equal=random.getstate()==parent['python_rng'],numpy_rng_equal=bool(np.array_equal(np.random.get_state()[1],parent['numpy_rng'][1]) and np.random.get_state()[0]==parent['numpy_rng'][0] and np.random.get_state()[2:]==parent['numpy_rng'][2:]),
                     torch_rng_equal=torch.equal(torch.get_rng_state(),parent['torch_rng']),cuda_rng_equal=all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),parent['cuda_rng'])),
                     metadata_only_normalization=metadata_normalization,sampling='epoch seed+epoch; restart next complete epoch; discarded partial updates recorded by queue')
        if restore['optimizer_steps']!=[global_step] or not all(restore[k] for k in ['model_equal','optimizer_equal','python_rng_equal','numpy_rng_equal','torch_rng_equal','cuda_rng_equal']):raise RuntimeError('恢复状态核查失败')
        dump(a.output/'restore-check.json',restore)
    maximum=cfg['n0_max_epochs'] if a.method=='N0' else cfg['n1_max_epochs']
    dump(a.output/'config.json',cfg)
    dump(a.output/'metadata.json',dict(started_at=started,method=a.method,seed=cfg['seed'],device=torch.cuda.get_device_name(),gate=gate,
          precision='fp32',amp=False,tf32=False,parameters=sum(p.numel() for p in model.parameters()),train_samples=len(train),dev_samples=len(dev),
          windows=len(ws),vocab_size=len(vocab),scheduler=None,validation='every completed epoch; all1503 internal dev; no padding',
          batch=32,micro_batch=4,accumulation=8,workers=0,code={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
          sources={s:digest(ROOT/f'data/processed/{s}.jsonl') for s in ('train','dev')},
          contract_sha256=digest(a.contract),protocol_sha256=digest(a.protocol),parent_checkpoint=digest(a.resume) if parent else None,max_segment_seconds=cfg['max_segment_seconds'],
          unknown_characters={name:sum(ch not in vocab for r in rows for ch in r['text']) for name,rows in [('train',train),('dev',dev)]}))
    torch.cuda.reset_peak_memory_stats();status='completed'
    try:
        for epoch in range(epoch_start,maximum+1):
            if stopping[0] or time.perf_counter()-start>cfg['max_segment_seconds']-120:
                status='segment_limit';break
            order=list(range(len(ws)));random.Random(cfg['seed']+epoch).shuffle(order)
            extra=(-len(order))%32;order+=order[:extra]
            epoch_start_time=time.perf_counter();model.train();loss_sum=0.;steps_this=0
            for j in range(0,len(order),32):
                # 运行时间上限在更新边界检查；保留上一完整epoch的last，禁止自动超时继续。
                if stopping[0] or time.perf_counter()-start>cfg['max_segment_seconds']-120:
                    status='segment_limit';break
                optimizer.zero_grad(set_to_none=True);loss_value=0.;step_start=time.perf_counter()
                for k in range(0,32,4):
                    batch=[ws[i] for i in order[j+k:j+k+4]];ids,attn,chars,y=collate(batch,vocab,a.method,'cuda')
                    loss=loss_fn(model(ids,attn,chars),y)
                    if not torch.isfinite(loss):raise FloatingPointError('非有限训练损失')
                    (loss/8).backward();loss_value+=float(loss.detach())/8
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();global_step+=1;steps_this+=1;loss_sum+=loss_value;torch.cuda.synchronize()
                elapsed=time.perf_counter()-step_start
                values=dict(event='train',epoch=epoch,global_step=global_step,window_cursor=j+32,loss=loss_value,grad_norm=float(grad),
                            seconds=elapsed,throughput_windows_per_sec=32/elapsed,peak_mb=torch.cuda.max_memory_allocated()/2**20)
                # 每步均记录机器可读指标，控制台只每50步输出，避免吞吐被大日志影响。
                with console.open('a') as f:f.write(json.dumps(values)+'\n')
                if global_step%50==0:print(json.dumps(values),flush=True)
                for key,val in {'train/loss':loss_value,'train/grad_norm':float(grad),'train/lr':optimizer.param_groups[0]['lr'],
                                'train/optimizer_step':global_step,'train/throughput_samples_per_sec':32/elapsed,
                                'system/gpu_memory_peak_mb':values['peak_mb']}.items():writer.add_scalar(key,val,global_step)
            if status=='segment_limit':break
            train_seconds=time.perf_counter()-epoch_start_time
            predicted,metrics,validation=evaluate(model,dev,vocab,cfg,a.method,train_terms=train_terms)
            f1=metrics['micro']['f1'];improved=f1>best+1e-12
            if improved:best=f1;best_epoch=epoch;bad=0
            else:bad+=1
            record=dict(epoch=epoch,global_step=global_step,train_loss=loss_sum/steps_this,train_seconds=train_seconds,
                        padded_training_windows=extra,permutation_sha256=__import__('hashlib').sha256(json.dumps(order).encode()).hexdigest(),
                        val=metrics,validation=validation,best_f1=best,best_epoch=best_epoch,bad_epochs=bad,
                        peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,finished_at=utc())
            writer.add_scalar('train/epoch_time_sec',train_seconds,global_step)
            for key in ('precision','recall','f1'):writer.add_scalar('val/'+key,metrics['micro'][key],epoch)
            writer.add_scalar('val/loss',validation['loss'],epoch);writer.add_scalar('val/nested_recall',metrics['nested_recall'],epoch)
            writer.add_scalar('val/character_f1',metrics['character_type_micro']['f1'],epoch)
            for t in TYPES:writer.add_scalar('val/class_f1/'+t,metrics['by_type'][t]['f1'],epoch)
            writer.flush()
            write_rows(a.output/f'dev-epoch-{epoch:02d}.jsonl',predicted)
            save_start=time.perf_counter()
            inference=dict(method=a.method,config=cfg,vocab=vocab,bert_config=bc,model={k:v.detach().cpu() for k,v in model.state_dict().items()},global_step=global_step,epoch=epoch)
            if improved:save_atomic(inference,a.output/'best.pt')
            records.append(record)
            last=dict(**inference,optimizer=optimizer.state_dict(),best_f1=best,best_epoch=best_epoch,bad_epochs=bad,
                      records=records,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            save_atomic(last,a.output/'last.pt');record['save_seconds']=time.perf_counter()-save_start
            dump(a.output/f'epoch-{epoch:02d}.json',record);log(console,dict(event='epoch',**record))
            dump(a.output/'progress.json',dict(status='running',best_f1=best,best_epoch=best_epoch,epoch=epoch,global_step=global_step,
                 elapsed_seconds=time.perf_counter()-start,eta_max_remaining_seconds=(maximum-epoch)*(train_seconds+validation['seconds']+record['save_seconds'])))
            if bad>=cfg['patience']:status='early_stopped';break
        summary=dict(status=status,method=a.method,best_f1=best,best_epoch=best_epoch,epochs=len(records),global_step=global_step,
                     seconds=time.perf_counter()-start,peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,started_at=started,finished_at=utc(),
                     best_checkpoint_sha256=digest(a.output/'best.pt') if (a.output/'best.pt').exists() else None,
                     metrics=records[best_epoch-1]['val'] if best_epoch else None,records=records,sealed_read=False,paid_api_calls=0)
        dump(a.output/'summary.json',summary)
        dump(a.output/'progress.json',dict(status=status,best_f1=best,best_epoch=best_epoch,epoch=len(records),global_step=global_step,elapsed_seconds=time.perf_counter()-start,eta_max_remaining_seconds=0 if status in ('completed','early_stopped') else None))
        print(json.dumps({k:v for k,v in summary.items() if k not in ('records','metrics')}),flush=True)
    except Exception as e:
        dump(a.output/'failure.json',dict(error=repr(e),at=utc(),seconds=time.perf_counter()-start,global_step=global_step));raise
    finally:writer.flush();writer.close()

if __name__=='__main__':main()
