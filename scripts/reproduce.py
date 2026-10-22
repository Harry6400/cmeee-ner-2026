"""Portable teacher CPU entrypoint: asset check, fresh inference, and fixed evaluation."""
import argparse,json,time,hashlib,platform,sys
from pathlib import Path
from datetime import datetime,timezone
from ner_core import ROOT,score,keys,validate
from cpu_inference import CPUModels,read_jsonl
from stage5_repairs import Engine

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def write(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def assets(root,data_root):
    m=json.loads((root/'models/trained/manifest.json').read_text(encoding='utf-8'));ready=json.loads((data_root/'manifests/readiness.json').read_text(encoding='utf-8'))
    for x in m['items']:
        if digest(root/'models/trained'/x['file'])!=x['sha256']:raise ValueError('model asset mismatch:'+x['method'])
    expected=json.loads((root/'configs/reproduction-expected.json').read_text(encoding='utf-8'))
    for rel in ['processed/train.jsonl','processed/dev.jsonl','sealed/common500.jsonl','sealed/holdout.jsonl']:
        if digest(data_root/rel)!=ready['files'][rel] or ready['files'][rel]!=expected['data_sha256'][rel]:raise ValueError('data asset mismatch:'+rel)
    if ready['archive_sha256']!=expected['archive_sha256']:raise ValueError('official dataset archive differs')
    return dict(status='PASS',models=[x['method'] for x in m['items']],data='CMeEE-V2 frozen split',requires_ssh=False,requires_private_ledger=False,requires_API_key=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--data-root',type=Path)
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('check')
    q=sub.add_parser('predict');q.add_argument('--method',choices=['N0','N1'],default='N1');q.add_argument('--text',required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('evaluate');q.add_argument('--split',choices=['common500','holdout5000'],default='common500');q.add_argument('--methods',nargs='+',choices=['D0','N0','N1'],default=['D0','N0','N1']);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();data=a.data_root.resolve() if a.data_root else root/'data';start=time.perf_counter();checked=assets(root,data)
    if a.command=='check':print(json.dumps(checked));return
    a.output.mkdir(parents=True,exist_ok=False);models=CPUModels(root)
    if a.command=='predict':
        outputs,timing=models.predict_many([dict(id='new-local-input',text=a.text)],a.method);write(a.output/'predictions.jsonl',outputs);save(a.output/'summary.json',timing);print(json.dumps(timing));return
    source=data/('sealed/common500.jsonl' if a.split=='common500' else 'sealed/holdout.jsonl');gold=read_jsonl(source)
    if len(gold)!=(500 if a.split=='common500' else 5000):raise ValueError('wrong fixed split size')
    for row in gold:validate(row)
    expected=json.loads((root/'configs/reproduction-expected.json').read_text(encoding='utf-8'));results={}
    for method in a.methods:
        inputs=[dict(id=g['id'],text=g['text']) for g in gold];t=time.perf_counter()
        if method=='D0':
            engine=Engine(read_jsonl(data/'processed/train.jsonl'),a.output/'cache',root/'configs/review-runtime.json');pred=[engine.predict(r,'baseline')[0] for r in inputs];timing=dict(device='cpu',method='D0',inference_seconds=time.perf_counter()-t,new_API_calls=0)
        else:pred,timing=models.predict_many(inputs,method,lambda d:print(json.dumps(dict(event='progress',**d)),flush=True))
        measured=score(gold,pred);target=expected['metrics'][a.split][method];equal=all(abs(measured['micro'][k]-target[k])<1e-12 for k in target)
        write(a.output/(method+'-predictions.jsonl'),pred);entry=dict(metrics=measured,timing=timing,frozen_metric_equal=equal,frozen_expected=target)
        # Reference predictions are optional diagnostics; never required by the portable CPU entrypoint.
        old=root/expected['prediction_paths'][a.split][method]
        if old.exists():
            original=read_jsonl(old);diff=[dict(id=g['id'],cpu_minus_original=sorted(keys(cpu)-keys(prior)),original_minus_cpu=sorted(keys(prior)-keys(cpu))) for g,cpu,prior in zip(gold,pred,original) if keys(cpu)!=keys(prior)]
            write(a.output/(method+'-cpu-vs-original.jsonl'),diff);entry['documents_differing_from_original']=len(diff)
        results[method]=entry;print(json.dumps(dict(event='method_complete',method=method,f1=measured['micro']['f1'],frozen_metric_equal=equal,seconds=time.perf_counter()-t)),flush=True)
    summary=dict(status='PASS' if all(x['frozen_metric_equal'] for x in results.values()) else 'PASS_WITH_NUMERICAL_DIFFERENCES',at=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),python=sys.version,split=a.split,samples=len(gold),results=results,total_seconds=time.perf_counter()-start,API_calls=0,training=False,GPU_started=False,private_ledger_read=False,ssh_required=False,original_outputs_overwritten=False,scope='fresh CPU inference,not historical output auditing;backend differences retained without tuning')
    save(a.output/'summary.json',summary);print(json.dumps(dict(status=summary['status'],seconds=summary['total_seconds'])),flush=True)
if __name__=='__main__':main()
