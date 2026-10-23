"""完整训练/开发字符词表审计，明确逐字符BERT映射的UNK；不访问sealed。"""
import json
from collections import Counter
from ner_core import ROOT,read_rows,dump
from ner import make_vocab

def main():
 cfg=json.loads((ROOT/'configs/stage3.json').read_text());rows={s:read_rows(ROOT/f'data/processed/{s}.jsonl') for s in ('train','dev')};out={}
 for method in ('N0','N1'):
  vocab=make_vocab(method,rows['train'],cfg);out[method]={'vocab_size':len(vocab),'splits':{}}
  for split,rr in rows.items():
   total=sum(len(r['text']) for r in rr);unknown=Counter(ch for r in rr for ch in r['text'] if ch not in vocab)
   out[method]['splits'][split]=dict(character_occurrences=total,unknown_occurrences=sum(unknown.values()),unknown_rate=sum(unknown.values())/total,unknown_distinct=len(unknown),examples=unknown.most_common(20))
 dump(ROOT/'evidence/stage3/character-vocabulary-audit.json',dict(scope='processed train/dev only;Unicode positions unchanged',methods=out,
  N1_definition='one original Unicode character per BERT vocab lookup;unknown->UNK;no WordPiece normalization or text rewriting',
  limitation='deliberate reproducible interface;differs from standard BERT tokenization,not a universal best tokenizer claim'))
 print(json.dumps({m:{s:v['unknown_rate'] for s,v in d['splits'].items()} for m,d in out.items()}))
if __name__=='__main__':main()
