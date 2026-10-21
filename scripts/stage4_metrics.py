"""Independent CPU final-evaluation metrics. No network, GPU, text normalization or gold edits."""
from __future__ import annotations
import hashlib, json, math, random, statistics
from pathlib import Path
from itertools import combinations
TYPES = ('dis','sym','dru','equ','pro','bod','ite','mic','dep')
SEEDS = (20261019,20261020,20261021)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    # The caller must first check stage4 authorization and its immutable protocol.
    with Path(path).open(encoding='utf-8') as f:
        rows=[json.loads(line) for line in f if line.strip()]
    if not all(isinstance(row,dict) for row in rows):raise ValueError('JSONL rows must be objects')
    return rows


def dump_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def entity_keys(row):
    if not isinstance(row.get('id'),str) or not isinstance(row.get('text'),str):raise ValueError('Missing id/text')
    if not isinstance(row.get('entities'),list):raise ValueError(f"{row.get('id')}: missing entities; not empty prediction")
    found=set()
    for e in row['entities']:
        if not isinstance(e,dict):raise ValueError('Entity must be object')
        a,b,t=e.get('start'),e.get('end'),e.get('type')
        if type(a) is not int or type(b) is not int or not 0<=a<b<=len(row['text']):raise ValueError(f"{row['id']}: invalid span")
        if t not in TYPES or e.get('text')!=row['text'][a:b]:raise ValueError(f"{row['id']}: invalid type/surface")
        found.add((a,b,t))
    return found


def counts(tp,predicted,gold):
    p=tp/predicted if predicted else 0.;r=tp/gold if gold else 0.
    return dict(tp=tp,predicted=predicted,gold=gold,precision=p,recall=r,f1=2*tp/(predicted+gold) if predicted+gold else 0.)


def validate_gold(rows,expected):
    if len(rows)!=expected or len({r.get('id') for r in rows})!=len(rows):raise ValueError('Gold size or duplicate ID mismatch')
    for row in rows:entity_keys(row)


def validate_predictions(gold,predicted,allow_invalid=False):
    if len(gold)!=len(predicted) or [r['id'] for r in gold]!=[r.get('id') for r in predicted]:raise ValueError('Prediction IDs/order/count must exactly match frozen reference')
    valid=[];failures=[];statuses={}
    for g,p in zip(gold,predicted):
        if p.get('text')!=g['text']:raise ValueError(f"{g['id']}: prediction text changed")
        status=p.get('status',p.get('output_status','valid'))
        statuses[status]=statuses.get(status,0)+1
        if status=='valid':entity_keys(p);valid.append((g,p))
        elif status in ('invalid','error','failed','not_sent') and allow_invalid:
            failures.append(dict(id=g['id'],status=status,error=p.get('validation_error',p.get('error'))))
        else:raise ValueError(f"{g['id']}: invalid or forbidden status {status}")
    return valid,failures,statuses


def strata(ref):
    groups={k:set() for k in ('strict_nesting','crossing','coextensive_different_type','isolated_no_overlap')}
    for x in ref:
        s,e,t=x;others=ref-{x}
        if any(x[:2]!=y[:2] and ((s<=y[0] and y[1]<=e) or (y[0]<=s and e<=y[1])) for y in others):groups['strict_nesting'].add(x)
        if any(s<y[0]<e<y[1] or y[0]<s<y[1]<e for y in others):groups['crossing'].add(x)
        if any(x[:2]==y[:2] and t!=y[2] for y in others):groups['coextensive_different_type'].add(x)
        if not any(max(s,y[0])<min(e,y[1]) for y in others):groups['isolated_no_overlap'].add(x)
    return groups


def relaxed_count(ref,pred):
    refs=sorted(ref);ps=sorted(pred);assigned={}
    def augment(i,seen):
        s,e,t=ps[i]
        for j,(ss,ee,tt) in enumerate(refs):
            if j in seen or t!=tt or max(s,ss)>=min(e,ee):continue
            seen.add(j)
            if j not in assigned or augment(assigned[j],seen):assigned[j]=i;return True
        return False
    return sum(augment(i,set()) for i in range(len(ps)))


def length_bucket(length):
    return '<=2' if length<=2 else '3-5' if length<=5 else '6-10' if length<=10 else '11-20' if length<=20 else '>20'


def doc_bucket(length):return '<=128' if length<=128 else '129-256' if length<=256 else '>256'


def evaluate_pairs(pairs,train_terms):
    total=[0,0,0];types={t:[0,0,0] for t in TYPES};chars=[0,0,0];relaxed=[0,0,0];oov=[0,0,0]
    structure={k:[0,0] for k in ('strict_nesting','crossing','coextensive_different_type','isolated_no_overlap')}
    lengths={k:[0,0,0] for k in ('<=2','3-5','6-10','11-20','>20')};docs={k:[0,0,0] for k in ('<=128','129-256','>256')}
    duplicate_gold=duplicate_predictions=0;doc_counts=[];uncovered=[]
    for g,p in pairs:
        a,b=entity_keys(g),entity_keys(p);v=(len(a&b),len(b),len(a));doc_counts.append(dict(id=g['id'],tp=v[0],predicted=v[1],gold=v[2]))
        duplicate_gold+=len(g['entities'])-len(a);duplicate_predictions+=len(p['entities'])-len(b)
        for j,n in enumerate(v):total[j]+=n;docs[doc_bucket(len(g['text']))][j]+=n
        for t in TYPES:
            aa={e for e in a if e[2]==t};bb={e for e in b if e[2]==t}
            for j,n in enumerate((len(aa&bb),len(bb),len(aa))):types[t][j]+=n
        aa={(i,t) for s,e,t in a for i in range(s,e)};bb={(i,t) for s,e,t in b for i in range(s,e)}
        for j,n in enumerate((len(aa&bb),len(bb),len(aa))):chars[j]+=n
        for j,n in enumerate((relaxed_count(a,b),len(b),len(a))):relaxed[j]+=n
        aa={e for e in a if g['text'][e[0]:e[1]] not in train_terms};bb={e for e in b if g['text'][e[0]:e[1]] not in train_terms}
        for j,n in enumerate((len(aa&bb),len(bb),len(aa))):oov[j]+=n
        for k,ss in strata(a).items():structure[k][0]+=len(ss&b);structure[k][1]+=len(ss)
        for k in lengths:
            aa={e for e in a if length_bucket(e[1]-e[0])==k};bb={e for e in b if length_bucket(e[1]-e[0])==k}
            for j,n in enumerate((len(aa&bb),len(bb),len(aa))):lengths[k][j]+=n
        starts=range(0,max(len(g['text']),1),96)
        windows=[]
        for s in starts:
            end=min(s+256,len(g['text']));windows.append((s,end))
            if end==len(g['text']):break
        for s,e,t in a:
            if not any(ws<=s and e<=we for ws,we in windows):uncovered.append(dict(id=g['id'],start=s,end=e,type=t,text=g['text'][s:e]))
    return dict(samples=len(pairs),micro=counts(*total),by_type={k:counts(*v) for k,v in types.items()},
        nested_recall=structure['strict_nesting'][0]/structure['strict_nesting'][1] if structure['strict_nesting'][1] else 0.,
        structural_reference_recall={k:dict(tp=v[0],gold=v[1],recall=v[0]/v[1] if v[1] else 0.) for k,v in structure.items()},
        structural_groups_may_overlap=True,character_type_micro=counts(*chars),relaxed_overlap_same_type_one_to_one=counts(*relaxed),
        unseen_surface_strict=counts(*oov),entity_length_strict={k:counts(*v) for k,v in lengths.items()},document_length_strict={k:counts(*v) for k,v in docs.items()},
        duplicate_gold=duplicate_gold,duplicate_predictions=duplicate_predictions,window_uncovered_reference_entities=uncovered,
        window_uncovered_policy='all reference entities retained in denominator;256-character windows overlap160;not a reason to drop references',
        definitions=dict(strict='Unicode start+exclusive end+type;case-sensitive;exact tuples deduplicated;zero denominator=0',
            character='unique(character_position,type) multi-hot micro excluding O;no normalization',
            relaxed='diagnostic positive overlap and same type,maximum one-to-one bipartite cardinality',
            unseen='surface absent from all training gold entities;filter both reference and predictions',length='Unicode entity character length'),per_document_counts=doc_counts)


def evaluate_route(gold,predicted,train_terms,allow_invalid=False):
    pairs,failures,statuses=validate_predictions(gold,predicted,allow_invalid)
    conditional=evaluate_pairs(pairs,train_terms)
    all_valid=len(pairs)==len(gold)
    return dict(samples_expected=len(gold),valid=len(pairs),invalid=len(gold)-len(pairs),statuses=statuses,
        output_validity_rate=len(pairs)/len(gold) if gold else 0.,main_metrics=conditional if all_valid else None,
        conditional_valid_only_metrics=None if all_valid else conditional,conditional_valid_only_is_diagnostic=True,
        failures=failures,primary_availability='available_on_complete_frozen_set' if all_valid else 'N/A: incomplete or invalid outputs;not silently scored as empty')


def percentile(values,q):
    ordered=sorted(values)
    if not ordered:raise ValueError('Empty percentile')
    p=(len(ordered)-1)*q;a=int(p);b=min(a+1,len(ordered)-1)
    return ordered[a]+(ordered[b]-ordered[a])*(p-a)


def paired_bootstrap(routes,replicates=2000,seed=20261019):
    """One shared resampling-index sequence for every eligible route/pair."""
    all_names=sorted(routes);eligible={k:v for k,v in routes.items() if v['main_metrics'] is not None}
    ids=None
    for k,v in eligible.items():
        current=[r['id'] for r in v['main_metrics']['per_document_counts']]
        if ids is None:ids=current
        if ids!=current:raise ValueError('Bootstrap requires identical document IDs and order')
    rng=random.Random(seed);samples={k:[] for k in eligible};n=len(ids or [])
    index_digest=hashlib.sha256()
    arrays={k:[(d['tp'],d['predicted'],d['gold']) for d in v['main_metrics']['per_document_counts']] for k,v in eligible.items()}
    for _ in range(replicates):
        indices=[rng.randrange(n) for _ in range(n)] if n else []
        index_digest.update((','.join(map(str,indices))+'\n').encode())
        for name,rows in arrays.items():
            tp=pr=go=0
            for i in indices:a,b,c=rows[i];tp+=a;pr+=b;go+=c
            samples[name].append(2*tp/(pr+go) if pr+go else 0.)
    pairs={}
    for a,b in combinations(all_names,2):
        name=b+' minus '+a
        if a not in eligible or b not in eligible:
            pairs[name]=dict(status='N/A',reason='one or both routes lacks valid predictions on complete frozen set');continue
        diff=[y-x for x,y in zip(samples[a],samples[b])]
        pairs[name]=dict(status='available',difference=routes[b]['main_metrics']['micro']['f1']-routes[a]['main_metrics']['micro']['f1'],
            ci95=[percentile(diff,.025),percentile(diff,.975)],unit='F1 fraction; multiply by100 for percentage points',
            reference=a,comparison=b)
    return dict(method='paired document bootstrap;resample same complete documents with replacement;recompute entity-micro F1 each replicate',
        replicates=replicates,seed=seed,samples=n,confidence=.95,percentile_method='linear interpolation at(n-1)q',
        shared_resampling_indices_sha256=index_digest.hexdigest(),eligible_routes=sorted(eligible),differences=pairs,
        limitations='conditional on this corpus and fixed checkpoints;single-seed intervals do not measure training-seed variance;no multiple-comparison correction;no claim of causal or universal superiority')


def seed_summary(values):
    if set(values)!=set(SEEDS):raise ValueError('Require all3 frozen seeds;no best-holdout seed selection')
    scores=[values[s] for s in SEEDS]
    return dict(seeds=list(SEEDS),values=scores,mean=statistics.mean(scores),sample_sd=statistics.stdev(scores),ddof=1,
        primary_seed=20261019,primary_value=values[20261019],best_seed_selected_on_holdout=False)


def equivalent(a,b,tolerance=1e-12):
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equivalent(a[k],b[k],tolerance) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equivalent(x,y,tolerance) for x,y in zip(a,b))
    if type(a) in (float,int) and type(b) in (float,int):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tolerance
    return a==b
