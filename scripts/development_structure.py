"""从已有预测统计结构分层召回；类别可交叠，绝不声称训练过平面BIO模型。"""
import json
from ner_core import ROOT,read_rows,keys,dump


def main():
 root=ROOT/'runs/stage3-20261019';dev=read_rows(ROOT/'data/processed/dev.jsonl');result={}
 selection=json.loads((root/'lr-selection.json').read_text())['selected']['name']
 for name in ('N0',selection,'N1-rope-off'):
  d=json.loads((root/name/'summary.json').read_text());pred=read_rows(root/name/f"dev-epoch-{d['best_epoch']:02d}.jsonl")
  count={k:[0,0] for k in ('strict_nesting','crossing','coextensive_different_type','isolated_no_overlap')}
  for g,p in zip(dev,pred):
   if g['id']!=p['id']:raise ValueError('ID顺序不符')
   a,b=keys(g),keys(p)
   for e in a:
    s,t,kind=e;others=a-{e};groups=[]
    if any(e[:2]!=v[:2] and ((s<=v[0] and v[1]<=t) or (v[0]<=s and t<=v[1])) for v in others):groups.append('strict_nesting')
    if any((s<v[0]<t<v[1]) or (v[0]<s<v[1]<t) for v in others):groups.append('crossing')
    if any((s,t)==v[:2] and kind!=v[2] for v in others):groups.append('coextensive_different_type')
    if not any(max(s,v[0])<min(t,v[1]) for v in others):groups.append('isolated_no_overlap')
    for group in groups:count[group][1]+=1;count[group][0]+=e in b
  result[name]={k:dict(tp=v[0],gold=v[1],recall=v[0]/v[1] if v[1] else 0) for k,v in count.items()}
  if abs(result[name]['strict_nesting']['recall']-d['metrics']['nested_recall'])>1e-12:raise ValueError('嵌套口径不一致')
 dump(root/'structural-recall.json',dict(groups_may_overlap=True,models=result,
      interpretation='BIO upper bound is representational ceiling,not performance of trained BIO;isolated and strictly nested recalls are observed model strata',
      source='all1503 internal dev;no holdout'))
 print(json.dumps(result))

if __name__=='__main__':main()
