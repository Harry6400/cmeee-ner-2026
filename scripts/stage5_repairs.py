"""Separate retrospective CPU repairs. No heldout labels, API calls, or model updates."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import jieba
from ner_core import TYPES, ROOT, dictionary, validate
from entity_wire import occurrences, from_wire

def load_policy(path=None):
    path = Path(path) if path else ROOT/'configs/review-runtime.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    policy = dict(data.get('repair_policy', data.get('policy', {})))
    required = {'min_term_count','min_term_length','min_type_purity','context_chars','min_context_similarity','min_context_margin','min_context_type_examples','api_repair'}
    if set(policy) != required: raise ValueError('repair policy keys differ from supported configuration')
    for key in ['min_term_count','min_term_length','context_chars','min_context_type_examples']:
        if type(policy[key]) is not int or policy[key] < 1: raise ValueError('invalid integer policy: '+key)
    for key in ['min_type_purity','min_context_similarity','min_context_margin']:
        if type(policy[key]) not in (int,float) or not 0 <= policy[key] <= 1: raise ValueError('invalid probability policy: '+key)
    if policy['api_repair'] != 'drop only overflow occurrence when all real occurrences of same surface/type already covered; otherwise fail closed': raise ValueError('unsupported API repair semantics')
    return policy

POLICY = load_policy()

def features(text, start, end, context_chars):
    left, right = text[max(0,start-context_chars):start], text[end:end+context_chars]
    return {f'L:{left[i:i+2]}' for i in range(len(left)-1)} | {f'R:{right[i:i+2]}' for i in range(len(right)-1)}

class Engine:
    def __init__(self, train, cache, config_path=None):
        self.policy = load_policy(config_path)
        self.table, self.ambiguity = dictionary(train)
        self.freq = defaultdict(Counter)
        self.contexts = defaultdict(list)
        self.trie = {}
        for row in train:
            for e in row['entities']:
                self.freq[e['text']][e['type']] += 1
                if e['text'] in self.ambiguity:
                    self.contexts[e['text']].append(dict(type=e['type'], id=row['id'], start=e['start'], end=e['end'], features=features(row['text'],e['start'],e['end'],self.policy['context_chars'])))
        for word, typ in self.table.items():
            node = self.trie
            for ch in word: node = node.setdefault(ch,{})
            node[''] = typ
        cache = Path(cache); cache.mkdir(parents=True,exist_ok=True)
        self.tokenizer = jieba.Tokenizer(); self.tokenizer.tmp_dir=str(cache); self.tokenizer.cache_file='jieba-stage5.cache'; self.tokenizer.initialize()

    def raw(self, text):
        found=[]
        for s in range(len(text)):
            node=self.trie
            for e in range(s,len(text)):
                node=node.get(text[e])
                if node is None: break
                if '' in node: found.append(dict(start=s,end=e+1,type=node[''],text=text[s:e+1]))
        return found

    def predict(self, row, mode='baseline'):
        # Only id/text cross this interface; any accidental gold attachment is rejected.
        if set(row) != {'id','text'}: raise ValueError('prediction interface accepts id/text only')
        if mode not in ('baseline','lexicon','context','combined'): raise ValueError('unknown repair mode')
        tokens=list(self.tokenizer.tokenize(row['text'],HMM=True))
        starts={s for _,s,e in tokens}; ends={e for _,s,e in tokens}
        out=[]; changes=[]
        for e in self.raw(row['text']):
            boundary=e['start'] in starts and e['end'] in ends
            f=self.freq[e['text']]; n=sum(f.values()); purity=f[e['type']]/n
            rescue=(len(e['text'])>=self.policy['min_term_length'] and n>=self.policy['min_term_count'] and purity>=self.policy['min_type_purity'])
            if not boundary:
                if mode not in ('lexicon','combined') or not rescue: continue
                changes.append(dict(action='rescue_boundary',entity=dict(e),train_count=n,purity=purity))
            if mode in ('context','combined') and e['text'] in self.ambiguity:
                q=features(row['text'],e['start'],e['end'],self.policy['context_chars']); best={}; evidence={}
                for c in self.contexts[e['text']]:
                    fs=c['features']; sim=len(q&fs)/len(q|fs) if q|fs else 0
                    if sim>best.get(c['type'],-1): best[c['type']]=sim; evidence[c['type']]=c
                ranking=sorted(best,key=lambda t:(-best[t],TYPES.index(t)))
                top=ranking[0]; margin=best[top]-(best[ranking[1]] if len(ranking)>1 else 0)
                if top!=e['type'] and f[top]>=self.policy['min_context_type_examples'] and best[top]>=self.policy['min_context_similarity'] and margin>=self.policy['min_context_margin']:
                    changes.append(dict(action='context_type',before=dict(e),new_type=top,similarity=best[top],margin=margin,training_example={k:v for k,v in evidence[top].items() if k!='features'}))
                    e=dict(e,type=top)
            out.append(e)
        result=dict(row,entities=out); validate(result)
        return result, changes, tokens

def repair_wire(row, parsed):
    """Retrospective repair with explicit deletions; never fabricate an occurrence."""
    if not isinstance(parsed,dict) or not isinstance(parsed.get('entities'),list): raise ValueError('missing entities')
    valid=[]; over=[]; issues=[]
    for index,e in enumerate(parsed['entities']):
        if not isinstance(e,dict): raise ValueError('entity must be object')
        surface=e.get('text'); typ=e.get('type'); occ=e.get('occurrence')
        pos=occurrences(row['text'],surface)
        if typ not in TYPES or type(occ) is not int or occ<0 or not pos:
            issues.append(dict(index=index,entity=e,reason='unknown type,absent exact surface,or invalid occurrence type')); continue
        if occ>=len(pos): over.append((index,e,len(pos)))
        else: valid.append(e)
    actions=[]
    for index,e,n in over:
        covered={x['occurrence'] for x in valid if x['text']==e['text'] and x['type']==e['type']}
        if covered==set(range(n)):
            actions.append(dict(action='drop_overflow_duplicate',index=index,entity=e,actual_occurrences=n))
        else: issues.append(dict(index=index,entity=e,reason='overflow occurrence with incomplete real occurrence coverage'))
    if issues: return dict(status='invalid',prediction=None,actions=actions,issues=issues)
    return dict(status='repaired' if actions else 'valid_unchanged',prediction=from_wire(row,dict(entities=valid)),actions=actions,issues=[])

def rows(path): return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]
