"""完整训练词典；同一内部dev上的字符匹配与jieba词边界限制。"""
import json,time,os,math
from pathlib import Path
import jieba
from ner_core import ROOT,TYPES,read_rows,dictionary,score,keys,dump,write_rows
from stage3_common import extra_metrics,flat_upper_bound


def build_trie(table):
 trie={}
 for word,t in table.items():
  node=trie
  for ch in word:node=node.setdefault(ch,{})
  node['']=t
 return trie


def matches(row,trie):
 out=[]
 for start in range(len(row['text'])):
  node=trie
  for end in range(start,len(row['text'])):
   node=node.get(row['text'][end])
   if node is None:break
   if '' in node:out.append(dict(start=start,end=end+1,type=node[''],text=row['text'][start:end+1]))
 return out


def main():
 output=ROOT/'runs/stage3-20261019/baselines';output.mkdir(parents=True,exist_ok=False)
 train=read_rows(ROOT/'data/processed/train.jsonl');dev=read_rows(ROOT/'data/processed/dev.jsonl')
 start=time.perf_counter();table,ambiguity=dictionary(train);trie=build_trie(table);build_seconds=time.perf_counter()-start
 tok=jieba.Tokenizer();cache=ROOT/'.cache/jieba';cache.mkdir(parents=True,exist_ok=True);tok.tmp_dir=str(cache);tok.cache_file='jieba-stage3.cache'
 t=time.perf_counter();tok.initialize();jieba_seconds=time.perf_counter()-t
 results={};all_predictions={}
 for mode in ('character','jieba-boundary'):
  predictions=[];timings=[]
  for row in dev:
   t=time.perf_counter();entities=matches(row,trie)
   if mode=='jieba-boundary':
    tokens=list(tok.tokenize(row['text'],HMM=True));starts={s for _,s,e in tokens};ends={e for _,s,e in tokens}
    entities=[e for e in entities if e['start'] in starts and e['end'] in ends]
   predictions.append(dict(id=row['id'],text=row['text'],entities=entities));timings.append(time.perf_counter()-t)
  metrics=score(dev,predictions);metrics.update(extra_metrics(dev,predictions,set(table)))
  results[mode]=dict(metrics=metrics,inference_seconds=sum(timings),latency_mean_seconds=sum(timings)/len(timings),
                     latency_p50_seconds=sorted(timings)[len(timings)//2],latency_p95_seconds=sorted(timings)[math.ceil(.95*len(timings))-1],
                     per_sample_seconds=timings)
  write_rows(output/(mode+'-predictions.jsonl'),predictions);all_predictions[mode]=predictions
 cases=[]
 for g,char,word in zip(dev,all_predictions['character'],all_predictions['jieba-boundary']):
  dropped=(keys(g)&keys(char))-keys(word)
  if dropped:
   cases.append(dict(id=g['id'],text=g['text'],reference=g['entities'],character_prediction=char['entities'],
                     boundary_prediction=word['entities'],dropped_true_entities=sorted(dropped),tokens=list(tok.tokenize(g['text'],HMM=True))))
  if len(cases)==5:break
 dump(output/'dictionary.json',table);dump(output/'ambiguity.json',ambiguity)
 dump(output/'boundary-cases.json',cases)
 dump(output/'flat-bio-upper-bound.json',{s:flat_upper_bound(rows) for s,rows in [('train',train),('dev',dev)]})
 summary=dict(status='completed',train_samples=len(train),dev_samples=len(dev),dictionary_terms=len(table),ambiguous_terms=len(ambiguity),
              dictionary_build_seconds=build_seconds,jieba_init_seconds=jieba_seconds,jieba_version=jieba.__version__,
              jieba_settings=dict(HMM=True,custom_user_dictionary=False,normalization=False),results=results,sealed_read=False)
 dump(output/'summary.json',summary)
 print(json.dumps({k:v for k,v in summary.items() if k!='results'},ensure_ascii=False))
 for mode,r in results.items():print(mode,json.dumps(r['metrics']['micro']))

if __name__=='__main__':main()
