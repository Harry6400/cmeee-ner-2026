"""阶段三共享评估；始终仅使用processed train/dev。"""
import time
from collections import defaultdict
import torch
from ner_core import TYPES,windows,merge,score,keys,counts
from ner_model import collate,decode,loss_fn


def extra_metrics(gold,predicted,train_terms=None):
    g={r['id']:r for r in gold};tot=[0,0,0];relaxed=[0,0,0];oov=[0,0,0]
    for pred in predicted:
        ref=g[pred['id']];a,b=keys(ref),keys(pred)
        # 多标签字符覆盖，嵌套与同跨度异类不会压成单一BIO标签。
        aa={(i,t) for s,e,t in a for i in range(s,e)}
        bb={(i,t) for s,e,t in b for i in range(s,e)}
        for j,v in enumerate((len(aa&bb),len(bb),len(aa))):tot[j]+=v
        refs=sorted(a);ps=sorted(b);assigned={}
        def visit(i,seen):
            s,e,t=ps[i]
            for j,(ss,ee,tt) in enumerate(refs):
                if j in seen or t!=tt or max(s,ss)>=min(e,ee):continue
                seen.add(j)
                if j not in assigned or visit(assigned[j],seen):assigned[j]=i;return True
            return False
        matched=sum(visit(i,set()) for i in range(len(ps)))
        for j,v in enumerate((matched,len(ps),len(refs))):relaxed[j]+=v
        if train_terms is not None:
            aa={x for x in a if ref['text'][x[0]:x[1]] not in train_terms}
            bb={x for x in b if ref['text'][x[0]:x[1]] not in train_terms}
            for j,v in enumerate((len(aa&bb),len(bb),len(aa))):oov[j]+=v
    return dict(character_type_micro=counts(*tot),
                character_definition='micro over unique Unicode(character_position,type) pairs; non-O coverage; case-sensitive; nested multi-hot',
                relaxed_overlap_same_type_one_to_one=counts(*relaxed),
                relaxed_definition='diagnostic only; same type and positive overlap, maximum one-to-one bipartite matching',
                unseen_surface_strict=counts(*oov) if train_terms is not None else None)


def flat_upper_bound(rows):
    retained=total=0;affected=0
    for row in rows:
        entities=sorted(keys(row),key=lambda x:(x[1],x[0],x[2]));end=-1;n=0
        for s,e,t in entities:
            if s>=end:n+=1;end=e
        retained+=n;total+=len(entities);affected+=n<len(entities)
    return dict(gold=total,max_representable_without_overlapping_intervals=retained,
                max_recall=retained/total if total else 0,affected_samples=affected,
                definition='single-label flat BIO; choose maximum number of non-overlapping reference intervals; coextensive different types compete')


def evaluate(model,rows,vocab,cfg,method,device='cuda',train_terms=None):
    ws=[w for r in rows for w in windows(r,cfg['window_chars'],cfg['overlap_chars'])]
    parts=defaultdict(list);losses=[];start=time.perf_counter();model.eval()
    with torch.inference_mode():
        for i in range(0,len(ws),cfg['micro_batch']):
            batch=ws[i:i+cfg['micro_batch']];ids,attn,mask,y=collate(batch,vocab,method,device)
            logits=model(ids,attn,mask);loss=loss_fn(logits,y)
            if not torch.isfinite(loss):raise FloatingPointError('非有限验证损失')
            losses.append((float(loss),len(batch)))
            for w,entities in zip(batch,decode(logits,batch,method,cfg['decode_threshold'])):
                parts[w['id']].append((w,entities))
    if device=='cuda':torch.cuda.synchronize()
    predictions=[merge(r,parts[r['id']]) for r in rows]
    metrics=score(rows,predictions);metrics.update(extra_metrics(rows,predictions,train_terms))
    return predictions,metrics,dict(seconds=time.perf_counter()-start,windows=len(ws),samples=len(rows),
                                    loss=sum(a*b for a,b in losses)/len(ws))
