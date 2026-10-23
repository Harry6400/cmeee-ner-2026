"""从真实开发输出整理候选错误，不把启发式分类冒充已确认因果。"""
import json
from pathlib import Path
from ner_core import ROOT,read_rows,keys,dump


def main():
 root=ROOT/'runs/stage3-20261019';selection=json.loads((root/'lr-selection.json').read_text())['selected']
 choices=['N0',selection['name'],'N1-rope-off'];dev=read_rows(ROOT/'data/processed/dev.jsonl')
 terms={e['text'] for r in read_rows(ROOT/'data/processed/train.jsonl') for e in r['entities']}
 pools={}
 for name in choices:
  d=json.loads((root/name/'summary.json').read_text());preds=read_rows(root/name/f"dev-epoch-{d['best_epoch']:02d}.jsonl")
  candidates={k:[] for k in ('boundary_overlap','same_span_wrong_type','unseen_surface_missed','nested_missed')}
  for g,p in zip(dev,preds):
   if g['id']!=p['id']:raise ValueError('样本顺序不符')
   a,b=keys(g),keys(p)
   for e in sorted(a-b):
    s,t,kind=e;surface=g['text'][s:t]
    flags=[]
    if any(k==kind and max(x,s)<min(y,t) for x,y,k in b):flags.append('boundary_overlap')
    if any((x,y)==(s,t) and k!=kind for x,y,k in b):flags.append('same_span_wrong_type')
    if surface not in terms:flags.append('unseen_surface_missed')
    if any(e[:2]!=v[:2] and ((s<=v[0] and v[1]<=t) or (v[0]<=s and t<=v[1])) for v in a):flags.append('nested_missed')
    for flag in flags:
     if len(candidates[flag])<5:candidates[flag].append(dict(id=g['id'],text=g['text'],missed_reference=dict(start=s,end=t,type=kind,text=surface),
          reference=g['entities'],prediction=p['entities'],source_model=name,epoch=d['best_epoch'],status='candidate;heuristic category,causal attribution not confirmed'))
   if all(len(v)>=5 for v in candidates.values()):break
  pools[name]=candidates
 dump(root/'development-error-candidates.json',dict(source='internal dev only;already used for model selection',not_final_holdout=True,models=pools,
      warning='unseen surface is objective training-lexicon absence, not proof annotation wrong;overlap/type categories are observed discrepancies,not proven mechanism'))
 print(json.dumps({name:{k:len(v) for k,v in groups.items()} for name,groups in pools.items()}))

if __name__=='__main__':main()
