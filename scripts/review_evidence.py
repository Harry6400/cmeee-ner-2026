"""Perform the concrete source/metric reviews assigned to the two teaching roles."""
import json,random,time,zipfile,hashlib,collections
from pathlib import Path
from datetime import datetime,timezone
from ner_core import ROOT,score,keys,validate
from stage5_repairs import Engine,rows

OUT=ROOT/'evidence/prestage6-review';RUN=ROOT/'runs/prestage6-review'
def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def independent(gold,pred):
    count=[0,0,0]
    for g,p in zip(gold,pred):
        assert g['id']==p['id'] and g['text']==p['text'];gk={(x['start'],x['end'],x['type']) for x in g['entities']};pk={(x['start'],x['end'],x['type']) for x in p['entities']}
        count=[a+b for a,b in zip(count,[len(gk&pk),len(pk),len(gk)])]
    tp,pn,gn=count;return dict(tp=tp,predicted=pn,gold=gn,precision=tp/pn if pn else 0,recall=tp/gn if gn else 0,f1=2*tp/(pn+gn) if pn+gn else 0)
def main():
    start=time.perf_counter();common=rows(ROOT/'data/sealed/common500.jsonl');byid={g['id']:g for g in common};train=rows(ROOT/'data/processed/train.jsonl')
    chosen=[]
    def add(row,reason):
        if row['id'] not in [x[0]['id'] for x in chosen] and len(chosen)<12:chosen.append((row,reason))
    for ident in ['cmeee-v2-dev-00296','cmeee-v2-dev-00202','cmeee-v2-dev-01574','cmeee-v2-dev-00004','cmeee-v2-dev-00022','cmeee-v2-dev-00109','cmeee-v2-dev-00121']:add(byid[ident],'preselected actual failure/repair or model-boundary review')
    conditions=[('empty reference',lambda r:not r['entities']),('nested structure',lambda r:any(a!=b and a['start']<=b['start'] and b['end']<=a['end'] for a in r['entities'] for b in r['entities'])),('multiwindow text',lambda r:len(r['text'])>256),('rare equipment class',lambda r:any(e['type']=='equ' for e in r['entities'])),('crossing intervals',lambda r:any(a['start']<b['start']<a['end']<b['end'] for a in r['entities'] for b in r['entities']))]
    for reason,condition in conditions:
        eligible=[r for r in common if condition(r) and r['id'] not in [x[0]['id'] for x in chosen]]
        if eligible:add(eligible[0],reason)
    for row in random.Random(20261020).sample(common,len(common)):add(row,'fixed seed coverage supplement')
    with zipfile.ZipFile(ROOT/'data/raw/CMeEE-V2.zip') as archive:
        member=next(n for n in archive.namelist() if n.endswith('_dev.json'));official=json.loads(archive.read(member))
    raw_checks=[]
    for row,reason in chosen:
        t=time.perf_counter();raw=official[int(row['id'].rsplit('-',1)[1])];assert raw['text']==row['text']
        original=collections.Counter((e['start_idx'],e['end_idx'],e['type'],raw['text'][e['start_idx']:e['end_idx']]) for e in raw['entities'])
        canonical=collections.Counter((e['start'],e['end'],e['type'],e['text']) for e in row['entities']);assert original==canonical;validate(row)
        raw_checks.append(dict(id=row['id'],selection_reason=reason,characters=len(row['text']),entities=len(row['entities']),source_member=member,source_text_identical=True,all_source_spans_types_surfaces_identical=True,normalization_applied=False,seconds=time.perf_counter()-t))
    selected=[x[0] for x in chosen];metrics={}
    for method in ['D0','N0','N1']:
        pred={r['id']:r for r in rows(ROOT/f'runs/prestage6-review/cpu-common500/{method}-predictions.jsonl')};values=[pred[r['id']] for r in selected]
        first=score(selected,values)['micro'];second=independent(selected,values);assert all(abs(first[k]-second[k])<1e-12 for k in first)
        metrics[method]=dict(first_implementation=first,separate_implementation=second,match=True,source='fresh Mac CPU predictions from fixed selected model')
    n1={r['id']:r for r in rows(ROOT/'runs/prestage6-review/cpu-common500/N1-predictions.jsonl')};observations=collections.Counter();extra=[]
    for g in common:
        gg,pp=keys(g),keys(n1[g['id']]);observations['gold']+=len(gg);observations['predicted']+=len(pp);observations['tp']+=len(gg&pp);observations['fp']+=len(pp-gg)
        for key in sorted(gg-pp):
            same=[k for k in pp if k[:2]==key[:2]];overlap=[k for k in pp if k[2]==key[2] and max(k[0],key[0])<min(k[1],key[1])]
            category='same_span_wrong_type' if same else 'same_type_overlapping_boundary' if overlap else 'other_missed_reference';observations[category]+=1
        if g['id'] in ['cmeee-v2-dev-00022','cmeee-v2-dev-00121','cmeee-v2-dev-00004']:
            target=next(e for e in g['entities'] if (g['id'].endswith('00022') and e['text']=='心脏损害') or (g['id'].endswith('00121') and e['text']=='发热超过38.5℃') or (g['id'].endswith('00004') and e['text']=='腋温'))
            assert (target['start'],target['end'],target['type']) not in pp
            observed={'cmeee-v2-dev-00022':'same span wrong type','cmeee-v2-dev-00121':'same type shortened boundary','cmeee-v2-dev-00004':'nested reference missed inside predicted clinical manifestation'}[g['id']]
            extra.append(dict(id=g['id'],input=g['text'],reference=g['entities'],target=target,prediction=n1[g['id']]['entities'],source='fresh CPU prediction equals frozen GPU prediction',observed=observed,cause_not_proven=True))
    dump(OUT/'source-and-metric-review.json',dict(at=datetime.now(timezone.utc).isoformat(),selected_documents=len(selected),review_scope='raw source and technical annotation/metric consistency;not clinical expert judgment',selection='coverage-focused and known cases,not random accuracy estimate',records=raw_checks,second_metric_check=metrics,teaching_assignment={'first_review':'周栩乐：原始数据和标注规范','cross_review':'吴皓睿：字符接口及另一实现计数'},actual_executor='Codex;teaching role experiences disclosed in final AppendixII',seconds=time.perf_counter()-start))
    dump(OUT/'n1-error-observations.json',dict(at=datetime.now(timezone.utc).isoformat(),counts=dict(observations),mutually_exclusive_FN_priority=['same span wrong type','same type overlapping boundary','remaining missing'],causal_claim=False,cases=extra,main_scores_unchanged=True))
    # Re-run every repair candidate with the now externalized default configuration.
    engine=Engine(train,RUN/'policy-parity-cache');parity={}
    for split,source in [('dev1503','data/processed/dev.jsonl'),('common500-diagnostic','data/sealed/common500.jsonl')]:
        data=rows(ROOT/source);parity[split]={}
        for mode in ['baseline','lexicon','context','combined']:
            t=time.perf_counter();fresh=[engine.predict(dict(id=g['id'],text=g['text']),mode)[0] for g in data];old=rows(ROOT/f'runs/stage5-20261020/{split}/{mode}-predictions.jsonl');different=[a['id'] for a,b in zip(fresh,old) if a['id']!=b['id'] or keys(a)!=keys(b)];assert not different
            parity[split][mode]=dict(samples=len(fresh),different_documents=different,micro=independent(data,fresh),seconds=time.perf_counter()-t)
    dump(OUT/'external-config-parity.json',dict(status='PASS',actual_policy=engine.policy,variants=parity,old_outputs_overwritten=False))
    guideline=(ROOT/'data/raw/annotation-guidelines.txt').read_text();pages=guideline.split('\f');norm=[]
    for section in ['3.1','6.1 疾病(dis)和临床表现(sym)','6.3 药物(dru)和身体(bod)']:
        candidates=[i+1 for i,page in enumerate(pages) if section in page];norm.append(dict(section=section,PDF_pages_candidates=candidates))
    dump(OUT/'guideline-review.json',dict(source='data/raw/annotation-guidelines.pdf',source_sha256=hashlib.sha256((ROOT/'data/raw/annotation-guidelines.pdf').read_bytes()).hexdigest(),sections=norm,findings=['Body includes body substances as well as body parts;UI label should not reduce bod to anatomical parts','Section6.3 distinguishes external administered substance from internal bodily state;this supports contextual reading of folate but does not prove every annotation consistent','Section6.1 discusses disease/clinical-manifestation context ambiguity;keep reference and do not silently relabel heart damage','Reference guideline nested-annotation wording is narrower than all representable interval structures;model can represent general spans without claiming every nested output should be annotated'],frozen_API_prompt_limit='bod was abbreviated as body part in frozen prompt;incomplete description is a plausible limitation,not proven single-case cause;do not change frozen scores'))
    print(json.dumps(dict(status='PASS',source_documents=len(selected),independent_metric_routes=list(metrics),N1_observations=dict(observations),external_config_parity='all8variants exact',seconds=time.perf_counter()-start),ensure_ascii=False))
if __name__=='__main__':main()
