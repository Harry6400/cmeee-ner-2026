"""Frozen train-only jieba-boundary dictionary on final5000 and its fixed common500; CPU only."""
import argparse,json,time
from pathlib import Path
import jieba
from ner_core import ROOT,read_rows,dictionary,score,dump,write_rows,digest
from stage3_common import extra_metrics
from development_baselines import build_trie,matches
from local_evaluate_stage4 import authorized_rows,latency_stats,now,scalars

def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,default=ROOT/'configs/qha-stage4-contract.json');p.add_argument('--protocol',type=Path,default=ROOT/'configs/stage4-frozen-protocol.json');p.add_argument('--output',type=Path,default=ROOT/'runs/stage4-20261020/local-evaluation/D0');a=p.parse_args()
    contract=json.loads(a.contract.read_text());protocol=json.loads(a.protocol.read_text())
    if contract['stage']!=4 or contract['authorization']!='explicit_user_stage4_2026-10-20' or not contract['final_holdout_authorized'] or contract['frozen_protocol_sha256']!=digest(a.protocol):raise ValueError('stage4 contract authorization required')
    if protocol['selected']['D0']['variant']!='jieba-boundary':raise ValueError('frozen D0selection differs')
    start=time.perf_counter();started=now();a.output.mkdir(parents=True,exist_ok=False)
    train=read_rows(ROOT/'data/processed/train.jsonl');t=time.perf_counter();table,ambiguity=dictionary(train);trie=build_trie(table);build_seconds=time.perf_counter()-t
    tok=jieba.Tokenizer();cache=ROOT/'.cache/jieba';cache.mkdir(parents=True,exist_ok=True);tok.tmp_dir=str(cache);tok.cache_file='jieba-stage4.cache';t=time.perf_counter();tok.initialize();init_seconds=time.perf_counter()-t
    full=authorized_rows(ROOT/'data/sealed/holdout.jsonl',5000);common=authorized_rows(ROOT/'data/sealed/common500.jsonl',500);byid={r['id']:r for r in full}
    if any(byid.get(r['id'])!=r for r in common):raise ValueError('common subset mismatch')
    predicted=[];timings=[]
    for index,row in enumerate(full):
        t=time.perf_counter();entities=matches(row,trie);tokens=list(tok.tokenize(row['text'],HMM=True));starts={s for _,s,e in tokens};ends={e for _,s,e in tokens};entities=[e for e in entities if e['start'] in starts and e['end'] in ends];result=dict(id=row['id'],text=row['text'],entities=entities);seconds=time.perf_counter()-t;predicted.append(result);timings.append(dict(id=row['id'],index=index,seconds=seconds,characters=len(row['text']),single_sample=True,first_sample=index==0))
    pred_byid={r['id']:r for r in predicted};timing_byid={r['id']:r for r in timings};entries=[]
    dump(a.output/'dictionary.json',table);dump(a.output/'ambiguity.json',ambiguity)
    original_dictionary=ROOT/'runs/stage3-20261019/baselines/dictionary.json'
    if original_dictionary.exists() and json.loads(original_dictionary.read_text())!=table:raise ValueError('D0train dictionary differs from development')
    for split,rows,preds,ts in [('holdout5000',full,predicted,timings),('common500',common,[pred_byid[r['id']] for r in common],[dict(timing_byid[r['id']],index=i) for i,r in enumerate(common)])]:
        directory=a.output/split;directory.mkdir();write_rows(directory/'predictions.jsonl',preds);write_rows(directory/'per-sample-timing.jsonl',ts);metrics=score(rows,preds);metrics.update(extra_metrics(rows,preds,set(table)));dump(directory/'metrics.json',metrics);scalars(directory,metrics,'test');dump(directory/'timing.json',dict(dictionary_build_seconds=build_seconds,jieba_initialize_seconds=init_seconds,load_seconds=build_seconds+init_seconds,single_sample_latency=latency_stats([r['seconds'] for r in ts]),scope='CPU single-sample text matching, jiebaHMM tokenization, boundaryfilter; model/dictionary loading excluded; common500 timing subset comes from actual5000 single sample run'))
        dump(directory/'metadata.json',dict(started_at=started,finished_at=now(),route='D0',seed=None,variant='jieba-boundary',split=split,samples=len(rows),ordered_ids=[r['id'] for r in rows],device='cpu',dictionary_terms=len(table),ambiguous_terms=len(ambiguity),dictionary_build_seconds=build_seconds,jieba_initialize_seconds=init_seconds,jieba_version=jieba.__version__,jieba_settings=dict(HMM=True,custom_user_dictionary=False,normalization=False),train_source_sha256=digest(ROOT/'data/processed/train.jsonl'),dictionary_sha256=digest(a.output/'dictionary.json'),full_source_sha256=digest(ROOT/'data/sealed/holdout.jsonl'),common_source_sha256=digest(ROOT/'data/sealed/common500.jsonl'),contract_sha256=digest(a.contract),protocol_sha256=digest(a.protocol),source_sha256={n:digest(ROOT/'scripts'/n) for n in ['dictionary_evaluate_stage4.py','development_baselines.py','ner_core.py']},predictions_sha256=digest(directory/'predictions.jsonl'),no_tuning_on_holdout=True,invalid_output_count=0,provenance='single CPU prediction over full5000' if split=='holdout5000' else 'exact ID-ordered subset from full5000; not independent test set'))
        entry=dict(route='D0',seed=None,split=split,path=str((directory/'predictions.jsonl').relative_to(ROOT)),metrics_path=str((directory/'metrics.json').relative_to(ROOT)),timing_path=str((directory/'timing.json').relative_to(ROOT)),metadata_path=str((directory/'metadata.json').relative_to(ROOT)),checkpoint_sha256=None,training_run='train-only dictionary; no gradient training',training_summary_path='runs/stage3-20261019/baselines/summary.json');dump(directory/'manifest-entry.json',entry);entries.append(entry)
    dump(a.output/'summary.json',dict(status='completed',started_at=started,finished_at=now(),seconds=time.perf_counter()-start,full5000=score(full,predicted)['micro'],common500=score(common,[pred_byid[r['id']] for r in common])['micro'],dictionary_terms=len(table),entries=entries));print(json.dumps(json.loads((a.output/'summary.json').read_text()),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
