"""N1空预测诊断：独立标签/损失/解码校验＋仅训练资料的有限过拟合。"""
import argparse,json,time,math
from pathlib import Path
import torch
from ner_core import ROOT,TYPES,read_rows,windows,merge,score,dump,write_rows,digest
from ner_model import Extractor,configure,collate,loss_fn,decode
from ner import cuda_gate,log


def evaluate(model,rows,vocab,cfg,device):
    predicted=[];positives=[];positive_predictions=0;candidate_max=[]
    model.eval()
    with torch.inference_mode():
        for row in rows:
            ws=windows(row,cfg['window_chars'],cfg['overlap_chars']);pieces=[]
            for i in range(0,len(ws),4):
                batch=ws[i:i+4];ids,a,c,y=collate(batch,vocab,'N1',device)
                x=model(ids,a,c)
                positives+=x[y.bool()].cpu().tolist()
                positive_predictions+=int((x>0).sum())
                candidate_max.append(float(x.max()))
                pieces.extend(zip(batch,decode(x,batch,'N1',0)))
            predicted.append(merge(row,pieces))
    return predicted,dict(gold_logit_min=min(positives),gold_logit_max=max(positives),
        gold_logit_mean=sum(positives)/len(positives),all_logits_max=max(candidate_max),
        positive_logit_count=positive_predictions,gold_positions=len(positives))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','overfit']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((ROOT/'configs/stage3.json').read_text());configure(cfg['seed'],4)
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    state=torch.load(ROOT/'runs/stage2-smoke-20261019/N1-resume/inference.pt',map_location='cpu',weights_only=True)
    train=read_rows(ROOT/'runs/stage2-subset/train.jsonl');dev=read_rows(ROOT/'runs/stage2-subset/dev.jsonl')
    if a.mode=='cpu':
        oracle=[];label_counts={t:0 for t in TYPES};mismatches=[]
        for row in train+dev:
            ws=windows(row,256,160);pieces=[]
            for w in ws:
                ids,attn,mask,y=collate([w],state['vocab'],'N1')
                for t in TYPES:label_counts[t]+=int(y[0,TYPES.index(t)].sum())
                x=torch.full_like(y,-20);x[y.bool()]=20
                pieces.append((w,decode(x,[w],'N1',0)[0]))
            oracle.append(merge(row,pieces))
        oracle_score=score(train+dev,oracle)
        # 对手工小张量逐项构建正负集合，独立计算目标值与梯度方向。
        x=torch.tensor([[[[.2,-.5],[.8,1.1]]]],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([[[[1.,0.],[0.,1.]]]],dtype=torch.float64)
        expected=torch.log1p(torch.exp(torch.tensor(-.5,dtype=torch.float64))+torch.exp(torch.tensor(.8,dtype=torch.float64)))+torch.log1p(torch.exp(torch.tensor(-.2,dtype=torch.float64))+torch.exp(torch.tensor(-1.1,dtype=torch.float64)))
        actual=loss_fn(x,y);actual.backward()
        sign_ok=bool((x.grad[y.bool()]<0).all() and (x.grad[~y.bool()]>0).all())
        model=Extractor('N1',state['config'],state['vocab'],state['bert_config']);model.load_state_dict(state['model'])
        predicted,distribution=evaluate(model,dev,state['vocab'],cfg,'cpu')
        metrics=score(dev,predicted)
        result=dict(mode='CPU independent diagnosis',oracle=oracle_score,label_counts=label_counts,
                    loss_expected=float(expected),loss_actual=float(actual.detach()),loss_equal=bool(torch.allclose(actual,expected)),
                    gradient_sign_correct=sign_ok,stage2_reproduced=metrics,logits=distribution,
                    elapsed_seconds=time.perf_counter()-start,scope='stage2 train/dev only; no threshold tuning')
        if oracle_score['micro']['f1']!=1 or not result['loss_equal'] or not sign_ok:raise RuntimeError('独立实现检查失败')
        dump(a.output/'diagnosis.json',result);write_rows(a.output/'reproduced-predictions.jsonl',predicted)
        print(json.dumps(result,ensure_ascii=False),flush=True);return
    # 调用前尚未初始化CUDA；configure中的CUDA RNG操作为lazy，不建立训练上下文。
    cuda_gate(cfg)
    selected=[r for r in train if r['entities'] and 10<=len(r['text'])<=80][:8]
    dump(a.output/'samples.json',dict(ids=[r['id'] for r in selected],selection='first8 stage2 train nonempty length10..80',not_formal=True))
    ws=[w for r in selected for w in windows(r,256,160)]
    model=Extractor('N1',state['config'],state['vocab'],state['bert_config']);model.load_state_dict(state['model']);model.to('cuda')
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':cfg['n1_encoder_lr']},{'params':model.head.parameters(),'lr':cfg['n1_head_lr']}])
    from torch.utils.tensorboard import SummaryWriter
    tb=SummaryWriter(str(a.output/'tensorboard'));history=[];torch.cuda.reset_peak_memory_stats()
    for step in range(1,81):
        if time.perf_counter()-start>270:break
        t=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for i in range(0,len(ws),4):
            batch=ws[i:i+4];ids,attn,chars,y=collate(batch,state['vocab'],'N1','cuda')
            loss=loss_fn(model(ids,attn,chars),y);(loss*len(batch)/len(ws)).backward();losses.append(float(loss.detach()))
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
        record=dict(step=step,loss=sum(losses)/len(losses),grad_norm=float(grad),seconds=time.perf_counter()-t)
        tb.add_scalar('train/loss',record['loss'],step)
        if step==1 or step%10==0:
            pred,distribution=evaluate(model,selected,state['vocab'],cfg,'cuda');m=score(selected,pred)
            record.update(train_f1=m['micro']['f1'],distribution=distribution)
            tb.add_scalar('diagnostic/train_f1',m['micro']['f1'],step)
            log(a.output/'metrics.jsonl',record)
            if m['micro']['f1']>=.95:
                history.append(record);break
        history.append(record)
    tb.flush();tb.close()
    pred,distribution=evaluate(model,selected,state['vocab'],cfg,'cuda');m=score(selected,pred)
    result=dict(status='PASS' if m['micro']['f1']>=.9 else 'NEEDS_INVESTIGATION',steps=step,
                training_metrics=m,distribution=distribution,history=history,seconds=time.perf_counter()-start,
                peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,from_checkpoint=digest(ROOT/'runs/stage2-smoke-20261019/N1-resume/inference.pt'),
                conclusion='independent oracle/loss plus train-only overfit evidence; not formal dev score; no generalization claim')
    dump(a.output/'diagnosis.json',result);write_rows(a.output/'predictions.jsonl',pred)
    print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
    if result['status']!='PASS':raise SystemExit(2)

if __name__=='__main__':main()
