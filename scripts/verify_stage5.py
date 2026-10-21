"""Independent stdlib-only check of frozen results, route identity, repaired cases and summaries."""
import argparse,json,hashlib,time
from collections import Counter
from decimal import Decimal
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1];RUN=R/'runs/stage5-20261020';E=R/'evidence/stage5'
TYPES={'dis','sym','dru','equ','pro','bod','ite','mic','dep'}
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l]
def require(ok,message):
    if not ok:raise ValueError(message)
def tuples(row):
    require(isinstance(row,dict) and isinstance(row.get('id'),str) and isinstance(row.get('text'),str) and isinstance(row.get('entities'),list),'invalid canonical row')
    out=set()
    for e in row['entities']:
        s,z,t=e.get('start'),e.get('end'),e.get('type')
        require(type(s)is int and type(z)is int and 0<=s<z<=len(row['text']),'invalid span')
        require(row['text'][s:z]==e.get('text') and t in TYPES,'surface/type mismatch');out.add((s,z,t))
    return out
def metric(gold,pred):
    require(len(gold)==len(pred) and len({p['id'] for p in pred})==len(pred),'missing/duplicate predictions')
    tp=pn=gn=0
    for g,p in zip(gold,pred):
        require(g['id']==p['id'] and g['text']==p['text'],'id/text mismatch');gg,pp=tuples(g),tuples(p);tp+=len(gg&pp);gn+=len(gg);pn+=len(pp)
    return dict(tp=tp,predicted=pn,gold=gn,precision=tp/pn if pn else 0,recall=tp/gn if gn else 0,f1=2*tp/(pn+gn) if pn+gn else 0)
def check_scores(a,b):
    require(a.keys()==b.keys(),'score fields differ')
    require(all(type(b[k]) in (int,float) and abs(a[k]-b[k])<1e-12 for k in a),'recomputed score differs')
def positions(text,surface):return [i for i in range(len(text)) if text.startswith(surface,i)] if isinstance(surface,str) and surface else []
def independent_wire(text,wire):
    """Reconstruct allowed removals without importing the repaired parser."""
    require(isinstance(wire,dict) and isinstance(wire.get('entities'),list),'wire schema invalid')
    valid=[];overflow=[];bad=False
    for index,e in enumerate(wire['entities']):
        if not isinstance(e,dict):bad=True;continue
        found=positions(text,e.get('text'));n=e.get('occurrence')
        if e.get('type') not in TYPES or type(n)is not int or n<0 or not found:bad=True;continue
        if n>=len(found):overflow.append((index,e,found))
        else:valid.append((e,found[n]))
    deletes=[]
    for index,e,found in overflow:
        covered={v['occurrence'] for v,s in valid if v['text']==e['text'] and v['type']==e['type']}
        if covered==set(range(len(found))):deletes.append(index)
        else:bad=True
    keys={(s,s+len(e['text']),e['type']) for e,s in valid}
    return dict(original_valid=not overflow and not bad,invalid=bad,deleted=deletes,keys=keys)
def validate_api_records(root,gold,records,summary):
    root=Path(root);base=root/'runs/stage4-20261020/api-final';calls=rows(base/'calls.jsonl');bycall={c['id']:c for c in calls};bygold={g['id']:g for g in gold}
    require(len(bycall)==len(calls) and len({r['call_id'] for r in records})==len(records) and {r['call_id'] for r in records}==set(bycall),'API call coverage mismatch')
    require(len({(r['route'],r['sample_id']) for r in records})==len(records),'duplicate route/sample')
    execution=json.loads((root/'configs/stage4-execution-continuation02.json').read_text())['audit']
    old={route:{p['id']:p for p in rows(root/path)} for route,path in execution['api_prediction_paths'].items()}
    unchanged=repaired=invalid=0
    for rec in records:
        call=bycall[rec['call_id']]
        require(rec['route']==call['route'] and rec['sample_id']==call['sample_id'] and rec['original_status']==call['status'],'API route/sample/status identity mismatch')
        expected=(base/call['raw_file']).resolve();require((root/rec['raw_file']).resolve()==expected,'raw response path mismatch')
        raw=json.loads(expected.read_text());require(raw['route']==call['route'] and raw['sample_id']==call['sample_id'] and raw['id']==call['id'],'raw response identity mismatch')
        ref=bygold[rec['sample_id']];wire=json.loads(raw['response']['choices'][0]['message']['content']);fresh=independent_wire(ref['text'],wire)
        require(fresh['original_valid']==(call['status']=='valid'),'original validity mismatch')
        require([a['index'] for a in rec['actions']]==fresh['deleted'],'repair action index mismatch')
        for action in rec['actions']:
            ent=wire['entities'][action['index']]
            require(action==dict(action='drop_overflow_duplicate',index=action['index'],entity=ent,actual_occurrences=len(positions(ref['text'],ent['text']))),'repair action content mismatch')
        if fresh['invalid']:
            require(rec['status']=='invalid' and rec['prediction'] is None and bool(rec['issues']),'invalid output was concealed');invalid+=1;continue
        status='repaired' if fresh['deleted'] else 'valid_unchanged'
        require(rec['status']==status and rec['issues']==[],'repaired validity/status mismatch')
        p=rec['prediction'];require(p['id']==ref['id'] and p['text']==ref['text'] and tuples(p)==fresh['keys'],'repaired prediction mismatch')
        if call['status']=='valid':
            require(tuples(old[rec['route']][ref['id']])==tuples(p),'original valid prediction changed');unchanged+=1
        else:repaired+=1
    for route in ['L0','L3']:
        selected=[r for r in records if r['route']==route];valid=[r for r in selected if r['prediction'] is not None];s=summary[route]
        expected=dict(responses=len(selected),repaired=sum(r['status']=='repaired' for r in selected),remaining_invalid=sum(r['status']=='invalid' for r in selected),conditional_valid_samples=len(valid),network_calls=0)
        require(all(s[k]==v for k,v in expected.items()),'route-specific API summary mismatch')
        check_scores(metric([bygold[r['sample_id']] for r in valid],[r['prediction'] for r in valid]),s['conditional_diagnostic_micro'])
        require(s['primary_F1'].startswith('N/A'),'retrospective API score promoted to main result')
    return dict(responses=len(records),original_valid_unchanged=unchanged,repaired=repaired,invalid=invalid)
def validate_cases(root,gold,records,original,reviewed):
    root=Path(root);bygold={r['id']:r for r in gold};require(len(reviewed)>=3 and len({c['case_id'] for c in reviewed})==len(reviewed),'case coverage invalid')
    require(len(original)==len(reviewed),'reviewed case count differs')
    d0={p['id']:p for p in rows(root/'runs/stage5-20261020/common500-diagnostic/combined-predictions.jsonl')}
    for old,c in zip(original,reviewed):
        require({k:v for k,v in old.items() if k!='limitations'}=={k:v for k,v in c.items() if k!='limitations'},'reviewed case changed scientific content')
        require(all(isinstance(c[k],str) and c[k].strip() for k in ['observed_facts','attribution','limitations']),'case narrative missing')
        g=bygold[c['id']];require(c['input']==g['text'] and c['reference']==g['entities'],'displayed input/reference changed')
        if c['route']=='L0':
            rec=next(r for r in records if r['route']==c['route'] and r['sample_id']==c['id']);require(c['source_prediction']==rec['raw_file'] and c['original_prediction'] is None,'API case source mismatch');p=rec['prediction']
        else:
            src=next(r for r in rows(root/c['source_prediction']) if r['id']==c['id']);require(c['original_prediction']==src['entities'],'original case prediction changed');p=d0[c['id']] if c['route']=='D0' else src
        require(c['repaired_prediction']==(p['entities'] if p else None),'displayed repaired prediction changed')
        if c['target']:
            e=c['target'];key=(e['start'],e['end'],e['type']);require(g['text'][e['start']:e['end']]==e['text'] and key in tuples(g),'case target is not reference')
            require(key not in tuples(dict(id=c['id'],text=c['input'],entities=c['original_prediction'])),'successful example disguised as failure')
            require(c['target_recovered']==(p is not None and key in tuples(p)),'displayed recovery claim differs from actual prediction')
    require(len({c['category'] for c in reviewed})>=3,'too few error categories')
    return len(reviewed)
def verify(root):
    root=Path(root);run=root/'runs/stage5-20261020';e=root/'evidence/stage5';checks=[];s=json.loads((run/'summary.json').read_text())
    for split,src in [('dev1503','data/processed/dev.jsonl'),('common500-diagnostic','data/sealed/common500.jsonl')]:
        gold=rows(root/src);summary=json.loads((run/split/'summary.json').read_text())
        for mode in ['baseline','lexicon','context','combined']:
            pred=rows(run/split/f'{mode}-predictions.jsonl');check_scores(metric(gold,pred),summary[mode]['metrics']['micro']);ts=rows(run/split/f'{mode}-timing-actions.jsonl')
            require([t['id'] for t in ts]==[g['id'] for g in gold] and all(t['seconds']>=0 for t in ts),'timing coverage invalid')
        checks.append(split+': four strict metrics independently recomputed')
    gold=rows(root/'data/sealed/common500.jsonl');records=rows(run/'api-final-diagnostic-replay.jsonl');summary=json.loads((run/'api-final-diagnostic-summary.json').read_text())
    api=validate_api_records(root,gold,records,summary);require(api==dict(responses=998,original_valid_unchanged=987,repaired=5,invalid=6),'unexpected actual API counts')
    count=validate_cases(root,gold,records,json.loads((run/'cases.json').read_text()),json.loads((run/'cases-reviewed.json').read_text()))
    checks+=['API original identity,all invalid reasons,all allowed deletions,route-specific conditional summaries independently verified','reviewed cases used by UI verified against original source,reference,and repaired predictions']
    before=json.loads((e/'frozen-before.json').read_text());require(before==json.loads((e/'frozen-after.json').read_text()),'frozen manifests differ')
    require(all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in before.items()),'frozen content changed')
    checks.append(str(len(before))+' frozen files and ledger unchanged')
    ledger=json.loads((root/'private/budget/ledger.json').read_text());known=Decimal(ledger['spent_cny']);reserved=sum((Decimal(r['reserved_cny']) for r in ledger['requests'] if r['status']!='settled'),Decimal(0))
    require(known+reserved<=100 and not ledger['api_enabled'] and Decimal(s['budget']['exposure_upper_bound_cny'])==known+reserved,'budget mismatch')
    return dict(status='PASS_WITH_WARNINGS',checks=checks,api=api,cases=count,budget_upper_cny=str(known+reserved),independence='separate stdlib implementation,same AI producer;not human review',not_verified=['Windows','medical expert review','new blind test'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=R);p.add_argument('--output',type=Path,default=E/'independent-verification.json');a=p.parse_args();start=time.perf_counter();result=verify(a.root);result.update(at=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
