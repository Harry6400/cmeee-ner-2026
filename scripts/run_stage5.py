"""Execute isolated retrospective repairs and preserve authentic per-step evidence."""
import sys,json,time,hashlib,platform,traceback
from datetime import datetime,timezone
from pathlib import Path
from decimal import Decimal
from collections import Counter
from ner_core import ROOT,score,keys,validate
from stage5_repairs import Engine,POLICY,repair_wire,rows

RUN=ROOT/'runs/stage5-20261020'; EVD=ROOT/'evidence/stage5'
def now():return datetime.now(timezone.utc).isoformat()
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def write(p,data):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x') as f:
        for d in data:f.write(json.dumps(d,ensure_ascii=False)+'\n')
def log(event,**kw):
    d=dict(at=now(),event=event,**kw)
    with (EVD/'steps.jsonl').open('a') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
    print(json.dumps(d,ensure_ascii=False),flush=True)
def frozen_files():
    files=[]
    for directory in ['evidence/stage4','runs/stage4-20261020','data/processed','data/sealed']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file())
    files.extend((ROOT/'configs').glob('*stage4*'))
    files.append(ROOT/'private/budget/ledger.json')
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files)) if p.is_file()}

def evaluate(engine,gold,name):
    directory=RUN/name;directory.mkdir(exist_ok=False)
    all_preds={};summary={}
    for mode in ['baseline','lexicon','context','combined']:
        start=time.perf_counter();pred=[];records=[]
        for g in gold:
            t=time.perf_counter(); p,changes,tokens=engine.predict(dict(id=g['id'],text=g['text']),mode)
            pred.append(p);records.append(dict(id=g['id'],seconds=time.perf_counter()-t,changes=changes))
        m=score(gold,pred);elapsed=time.perf_counter()-start
        summary[mode]=dict(seconds=elapsed,metrics=m,changes=sum(len(x['changes']) for x in records),samples=len(gold))
        write(directory/f'{mode}-predictions.jsonl',pred);write(directory/f'{mode}-timing-actions.jsonl',records)
        all_preds[mode]=pred
        log('cpu_prediction_complete',split=name,mode=mode,seconds=elapsed,micro=m['micro'],changes=summary[mode]['changes'])
    before=all_preds['baseline'];after=all_preds['combined']
    regressions=[];improvements=[]
    for g,b,a in zip(gold,before,after):
        kb,ka,kg=keys(b),keys(a),keys(g)
        # Strict count changes; examples can improve in one respect and regress in another.
        delta=dict(id=g['id'],gained_tp=sorted((ka-kb)&kg),lost_tp=sorted((kb-ka)&kg),new_fp=sorted((ka-kb)-kg),removed_fp=sorted((kb-ka)-kg))
        if delta['lost_tp'] or delta['new_fp']:regressions.append(delta)
        if delta['gained_tp'] or delta['removed_fp']:improvements.append(delta)
    summary['affected']=dict(improved_documents=len(improvements),regressed_documents=len(regressions),can_overlap=True)
    write(directory/'regressions.jsonl',regressions);write(directory/'improvements.jsonl',improvements)
    dump(directory/'summary.json',summary)
    return all_preds,summary

def api_replay(source,samples,label):
    start=time.perf_counter(); byid={r['id']:r for r in samples};records=[];seen=set()
    for f in sorted(source.glob('calls*.jsonl')):
        for c in rows(f):
            ident=c.get('sample_id')
            if ident not in byid:continue
            if c['id'] in seen:raise ValueError('duplicate API call')
            seen.add(c['id']);raw=json.loads((source/c['raw_file']).read_text())
            response=raw.get('response',raw)
            content=response['choices'][0]['message']['content']
            try: result=repair_wire(dict(id=ident,text=byid[ident]['text']),json.loads(content))
            except (ValueError,TypeError,KeyError) as exc:result=dict(status='invalid',prediction=None,actions=[],issues=[dict(reason=str(exc))])
            record=dict(route=c['route'],sample_id=ident,call_id=c['id'],original_status=c['status'],raw_file=str((source/c['raw_file']).relative_to(ROOT)),**result)
            records.append(record)
    write(RUN/f'api-{label}-replay.jsonl',records)
    summary={}
    for route in ['L0','L3']:
        group=[r for r in records if r['route']==route];valid=[r for r in group if r['prediction'] is not None];g=[byid[r['sample_id']] for r in valid]
        summary[route]=dict(responses=len(group),repaired=sum(r['status']=='repaired' for r in group),remaining_invalid=sum(r['status']=='invalid' for r in group),conditional_valid_samples=len(valid),conditional_diagnostic_micro=score(g,[r['prediction'] for r in valid])['micro'] if valid else None,primary_F1='N/A: retrospective changed parser; not new model/API evaluation',network_calls=0)
    dump(RUN/f'api-{label}-summary.json',summary)
    log('offline_raw_response_reparse_complete',split=label,seconds=time.perf_counter()-start,summary=summary)
    return records,summary

def make_cases(engine,common,pred,api_records):
    byid={r['id']:r for r in common};base={r['id']:r for r in pred['baseline']};after={r['id']:r for r in pred['combined']}
    n1={r['id']:r for r in rows(ROOT/'runs/stage4-20261020/local-evaluation/N1-seed20261019/common500/predictions.jsonl')}
    specs=[
      ('F01','cmeee-v2-dev-00296','分词边界过滤','D0',dict(start=23,end=28,type='mic',text='革兰阳性菌')),
      ('F02','cmeee-v2-dev-00202','词表多数类丢失语境','D0',dict(start=6,end=8,type='bod',text='叶酸')),
      ('F03','cmeee-v2-dev-01574','生成输出的原文定位契约','L0',None),
      ('F04','cmeee-v2-dev-00004','跨度边界过度延长','N1',dict(start=46,end=48,type='sym',text='发热')),
      ('F05','cmeee-v2-dev-00022','训练词表覆盖不足','D0',dict(start=0,end=4,type='dis',text='SARS')),
    ];cases=[]
    for cid,ident,category,route,target in specs:
        g=byid[ident];inputs=dict(id=ident,text=g['text']);b,changes,tokens=engine.predict(inputs,'combined')
        if route=='D0': original=base[ident]; corrected=after[ident]
        elif route=='N1':original=n1[ident];corrected=n1[ident]
        else:
            rec=next(r for r in api_records if r['route']==route and r['sample_id']==ident)
            raw=json.loads((ROOT/rec['raw_file']).read_text());wire=json.loads(raw['response']['choices'][0]['message']['content'])
            original=None;corrected=rec['prediction']
        if target:
            kt=(target['start'],target['end'],target['type']); assert kt in keys(g) and kt not in keys(original)
        evidence=dict(tokens=tokens,train_type_counts=dict(engine.freq.get(target['text'],{})) if target else None,repair_actions=changes if route=='D0' else [])
        if cid=='F01':
            evidence.update(raw_dictionary_match=target in engine.raw(g['text']),target_start_token_boundary=23 in {x[1] for x in tokens},target_end_token_boundary=28 in {x[2] for x in tokens})
            facts='训练词典含该精确词；原始字符枚举能产生该跨度，jieba边界过滤将其删除。'
            cause='预处理过滤是本例漏检的直接可复核原因；不能外推为全部漏检原因。'
            limitations='该词在训练中仅1次，低于已冻结的3次门槛，本例仍未恢复。其他训练高置信词可能恢复，但总体开发F1下降，补丁不进入正式默认。'
        elif cid=='F02':
            facts='同跨度预测dru，参考bod；训练中该词有dru、bod、ite三类，原D0按全局多数类固定为dru。'
            cause='词典按表面词唯一多数类，无法表示语境类别差异，这是确定的结构限制；参考类别的临床含义/一致性仍未由专家核验。'
            limitations='训练上下文近邻覆盖不足可能不触发或选错；不以改参考或硬编码本例类别掩盖问题。'
        elif cid=='F03':
            evidence.update(raw_file=rec['raw_file'],wire=wire,api_repair_actions=rec['actions'],exact_GH_positions=[i for i in range(len(g['text'])) if g['text'].startswith('GH',i)])
            facts='原文GH只出现两次，原响应额外给occurrence=2，原严格解析因此整条无效。'
            cause='生成模型违反原文出现次数契约；不是Unicode右开坐标实现错误。删除冗余越界项只在同词同类的所有真实位置已显式覆盖时允许。'
            limitations='仅离线重解析既有付费响应，未再次调用API；仍可能类别错误。无精确原文或未覆盖位置的输出继续无效。'
        elif cid=='F04':
            evidence.update(same_type_overlapping_predictions=[e for e in original['entities'] if e['type']==target['type'] and max(e['start'],target['start'])<min(e['end'],target['end'])],window_chars=256,text_chars=len(g['text']),all_text_within_one_window=True)
            facts='参考为[46,48)发热；N1输出[46,58)发热持续时间平均超过一周，严格边界不符。全文短于256字符，排除窗口未覆盖。'
            cause='可确认跨度决策失配；长修饰语导致过度延展只是机制假设，无法由单例确定。'
            limitations='本轮词典与API补丁不修改N1；未修复。需新的训练/解码开发实验，不把宽松指标算成成功。'
        else:
            evidence.update(in_training_dictionary=target['text'] in engine.table)
            facts='训练实体词表不含SARS；D0只精确枚举训练实体，无法产生该候选。'
            cause='训练实体词表覆盖不足是该词典路线的确定性召回上限；不代表N1或预训练模型未见过该词。'
            limitations='本轮补丁不注入留出实体或参考，仍漏检；N1历史预测检出只作路线能力对照，不冒充本轮新推理。'
        fixed=target and (target['start'],target['end'],target['type']) in keys(corrected)
        cases.append(dict(case_id=cid,id=ident,category=category,route=route,input=g['text'],reference=g['entities'],original_prediction=original['entities'] if original else None,repaired_prediction=corrected['entities'] if corrected else None,target=target,observed_facts=facts,attribution=cause,limitations=limitations,evidence=evidence,target_recovered=bool(fixed) if target else None,source_prediction=f'runs/stage4-20261020/local-evaluation/{"D0" if route=="D0" else "N1-seed20261019"}/common500/predictions.jsonl' if route!='L0' else rec['raw_file'],retrospective_known_case=True,original_is_real_failure=True))
    dump(RUN/'cases.json',cases);log('real_cases_verified',cases=len(cases),categories=len({c['category'] for c in cases}))
    return cases

def main():
    t0=time.perf_counter();RUN.mkdir(exist_ok=False);EVD.mkdir(exist_ok=True)
    log('start',platform=platform.platform(),python=sys.version,mode='local_cpu_only',api_calls=0,gpu_started=False)
    opening=frozen_files();dump(EVD/'frozen-before.json',opening)
    cfg=dict(stage=5,authorization='explicit user request 2026-10-20',runtime='local CPU only',policy=POLICY,selection='fixed prospective patch parameters before replay; motivated by observed final failures; retrospective, not unbiased test',train_source='data/processed/train.jsonl',train_sha256=digest(ROOT/'data/processed/train.jsonl'),dev_source='data/processed/dev.jsonl',diagnostic_final_source='data/sealed/common500.jsonl',no_training=True,no_API=True,no_gpu=True,main_results_immutable=True,run_root=str(RUN.relative_to(ROOT)),source_sha256={p.name:digest(p) for p in [ROOT/'scripts/stage5_repairs.py',ROOT/'scripts/run_stage5.py']})
    dump(ROOT/'configs/stage5-repair-protocol.json',cfg);dump(RUN/'protocol-at-run.json',cfg)
    train=rows(ROOT/'data/processed/train.jsonl');dev=rows(ROOT/'data/processed/dev.jsonl')
    t=time.perf_counter();engine=Engine(train,RUN/'cache');log('train_only_engine_built',seconds=time.perf_counter()-t,terms=len(engine.table),ambiguous_terms=len(engine.ambiguity))
    _,ds=evaluate(engine,dev,'dev1503')
    frozen_dev=rows(ROOT/'runs/stage3-20261019/baselines/jieba-boundary-predictions.jsonl')
    fresh=rows(RUN/'dev1503/baseline-predictions.jsonl');assert all(keys(a)==keys(b) and a['id']==b['id'] and a['text']==b['text'] for a,b in zip(frozen_dev,fresh)) and len(frozen_dev)==len(fresh)
    log('development_baseline_reproduced',samples=len(fresh))
    api_dev=ROOT/'runs/stage3-20261019/api-development-grounded'; dev_api_samples=rows(api_dev/'dev.jsonl')
    # Development log has its own raw file convention.
    dev_records=[]
    for c in rows(api_dev/'calls.jsonl'):
        row=next(r for r in dev_api_samples if r['id']==c['sample_id'])
        raw=json.loads((api_dev/f"raw-{c['id']}.json").read_text());response=raw.get('response',raw)
        result=repair_wire(dict(id=row['id'],text=row['text']),json.loads(response['choices'][0]['message']['content']))
        dev_records.append(dict(id=row['id'],route=c['route'],original_status=c['output_status'],**result))
    write(RUN/'api-development-replay.jsonl',dev_records);log('api_development_replay_complete',responses=len(dev_records),changed=sum(bool(r['actions']) for r in dev_records),invalid=sum(r['status']=='invalid' for r in dev_records))
    common=rows(ROOT/'data/sealed/common500.jsonl');pred,cs=evaluate(engine,common,'common500-diagnostic')
    frozen=rows(ROOT/'runs/stage4-20261020/local-evaluation/D0/common500/predictions.jsonl')
    assert len(frozen)==len(common) and all(keys(a)==keys(b) and a['id']==b['id'] and a['text']==b['text'] for a,b in zip(frozen,pred['baseline']))
    log('frozen_final_baseline_reproduced',samples=500)
    apis,api_summary=api_replay(ROOT/'runs/stage4-20261020/api-final',common,'final-diagnostic')
    cases=make_cases(engine,common,pred,apis)
    ledger=json.loads((ROOT/'private/budget/ledger.json').read_text());unknown=[x for x in ledger['requests'] if x['status']!='settled'];known=Decimal(ledger['spent_cny']);reserved=sum((Decimal(x['reserved_cny']) for x in unknown),Decimal(0))
    budget=dict(known_conservative_cny=str(known),unknown_reserve_cny=str(reserved),exposure_upper_bound_cny=str(known+reserved),limit_cny='100',new_calls=0,new_cost_cny='0',api_enabled=ledger['api_enabled'],ledger_unchanged=True)
    assert known+reserved<=100 and not ledger['api_enabled']
    closing=frozen_files();assert opening==closing,'frozen source changed';dump(EVD/'frozen-after.json',closing)
    summary=dict(status='PASS_WITH_WARNINGS',started_source='evidence/stage5/steps.jsonl',finished_at=now(),seconds=time.perf_counter()-t0,dev=ds,final_diagnostic=cs,api_offline_diagnostic=api_summary,api_development_responses=len(dev_records),cases=len(cases),budget=budget,frozen_files_verified=len(opening),main_five_route_scores_unchanged=True,warnings=['retrospective known-case repairs; no new unbiased test','repaired parser does not give new official five-route ranking','context repair may not fix selected example; regressions retained','N1 boundary and D0 OOV remain unresolved','no new N0/N1 inference or training','UI verification still separate'])
    dump(RUN/'summary.json',summary);log('complete',seconds=summary['seconds'],budget=budget,frozen_files=len(opening))

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();log('failure',traceback=traceback.format_exc());raise
