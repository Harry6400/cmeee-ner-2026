"""Read frozen final outputs and independently aggregate;never train/tune/call API."""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
from stage4_metrics import SEEDS,digest,read_jsonl,dump_new,validate_gold,evaluate_route,paired_bootstrap,seed_summary,equivalent


def utc():return datetime.now(timezone.utc).isoformat()


def load_manifest(root,execution):
    path=root/execution;plan=json.loads(path.read_text());a=plan['audit']
    if a.get('final_evaluation_authorized') is not True:raise PermissionError('Explicit stage4 final-evaluation authorization required')
    protocol_path=root/a.get('protocol','configs/stage4-frozen-protocol.json')
    if a.get('protocol_sha256')!=digest(protocol_path):raise ValueError('Frozen development protocol differs from authorized execution snapshot')
    protocol=json.loads(protocol_path.read_text())
    if protocol['training']['seeds']!=list(SEEDS) or a.get('primary_seed')!=20261019:raise ValueError('Frozen seeds or primary seed changed')
    if a.get('bootstrap')!={'replicates':2000,'seed':20261019,'confidence':.95}:raise ValueError('Bootstrap execution specification changed')
    for path,sha in a.get('pre_evaluation_files',{}).items():
        if digest(root/path)!=sha:raise ValueError('Pre-evaluation frozen file changed:'+path)
    entries=list(a.get('predictions',[]))
    if a.get('local_manifest'):
        entries+=json.loads((root/a['local_manifest']).read_text())['entries']
    if a.get('api_run_dir'):
        for route in ('L0','L3'):entries.append(dict(route=route,seed=None,split='common500',path=a.get('api_prediction_paths',{}).get(route,str(Path(a['api_run_dir'])/(route+'-predictions.jsonl')))))
    expected={('D0',None,s) for s in ('common500','holdout5000')}
    expected|={(r,s,t) for r in ('N0','N1') for s in SEEDS for t in ('common500','holdout5000')}
    expected|={(r,None,'common500') for r in ('L0','L3')}
    actual=[(e['route'],e.get('seed'),e['split']) for e in entries]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Require exactly frozen 16 route/seed/split entries;no best-seed shortcut')
    readiness=json.loads((root/'data/manifests/readiness.json').read_text())
    paths=dict(common500=a.get('gold',{}).get('common500','data/sealed/common500.jsonl'),holdout5000=a.get('gold',{}).get('holdout5000','data/sealed/holdout.jsonl'))
    paths['train']=a.get('train','data/processed/train.jsonl')
    for path in paths.values():
        relative=str(Path(path).relative_to('data'))
        if digest(root/path)!=readiness['files'][relative]:raise ValueError('Frozen source data changed:'+path)
    return plan,a,protocol,entries,paths


def key(e):return e['route']+(('-seed'+str(e['seed'])) if e.get('seed') is not None else '')+'/'+e['split']


def evaluate_predictions_with_transport(gold,pred,train_terms,allow_invalid=False):
    # Operational adapter only;the frozen strict/auxiliary/bootstrap mathematics is unchanged.
    for row in pred:
        if row.get('status')=='transport_error' and 'entities' in row:raise ValueError('Unavailable transport response cannot have empty/fabricated entities')
    adapted=[dict(row,status='error') if row.get('status')=='transport_error' else row for row in pred]
    result=evaluate_route(gold,adapted,train_terms,allow_invalid)
    if any(row.get('status')=='transport_error' for row in pred):
        result['statuses']={status:sum(row.get('status','valid')==status for row in pred) for status in {row.get('status','valid') for row in pred}}
        for failure in result['failures']:
            if next(row for row in pred if row['id']==failure['id']).get('status')=='transport_error':failure['status']='transport_error'
        result['transport_failure_is_unavailable_not_empty']=True
    return result


def calculate(root,execution):
    started_at=utc();start=time.perf_counter();plan,a,protocol,entries,paths=load_manifest(root,execution)
    # Only after authorization, protocol and data-file integrity checks does this reader access final labels.
    gold={split:read_jsonl(root/path) for split,path in paths.items() if split!='train'}
    validate_gold(gold['common500'],500);validate_gold(gold['holdout5000'],5000)
    full={r['id']:r for r in gold['holdout5000']}
    if any(full.get(r['id'])!=r for r in gold['common500']):raise ValueError('Common500 is not exact frozen subset of5000')
    train=read_jsonl(root/paths['train']);validate_gold(train,13497)
    train_terms={e['text'] for row in train for e in row['entities']}
    results={};inputs={str(execution):digest(root/execution)}
    for path in paths.values():inputs[path]=digest(root/path)
    for e in entries:
        pred=read_jsonl(root/e['path']);result=evaluate_predictions_with_transport(gold[e['split']],pred,train_terms,e['route'] in ('L0','L3'))
        result.update(route=e['route'],seed=e.get('seed'),split=e['split'],prediction_path=e['path'],prediction_sha256=digest(root/e['path']))
        results[key(e)]=result;inputs[e['path']]=result['prediction_sha256']
        if e.get('metrics_path'):
            recorded=json.loads((root/e['metrics_path']).read_text())
            for metric in ('micro','character_type_micro','relaxed_overlap_same_type_one_to_one','unseen_surface_strict'):
                if metric in recorded and not equivalent(recorded[metric],result['main_metrics'][metric]):raise ValueError('Inference recorded metric differs from independent calculation:'+key(e)+' '+metric)
            inputs[e['metrics_path']]=digest(root/e['metrics_path'])
    # Common predictions are a documented subset of the same5000 main run,never separate optimized runs.
    for route in ('D0','N0','N1'):
        for seed in ((None,) if route=='D0' else SEEDS):
            prefix=route+(('-seed'+str(seed)) if seed is not None else '')
            pp={r['id']:r for r in read_jsonl(root/results[prefix+'/holdout5000']['prediction_path'])}
            common=read_jsonl(root/results[prefix+'/common500']['prediction_path'])
            for row in common:
                if row!=pp[row['id']]:raise ValueError('Common500 main prediction differs from full5000 source:'+prefix)
    primary={r:results[r+('-seed20261019' if r in ('N0','N1') else '')+'/common500'] for r in ('D0','N0','N1','L0','L3')}
    bootstrap=paired_bootstrap(primary,**{k:v for k,v in a['bootstrap'].items() if k in ('replicates','seed')})
    seeds={}
    for route in ('N0','N1'):
        seeds[route]={}
        for split in ('common500','holdout5000'):
            seeds[route][split]={metric:seed_summary({s:results[f'{route}-seed{s}/{split}']['main_metrics']['micro'][metric] for s in SEEDS}) for metric in ('precision','recall','f1')}
    incomplete=any(r['statuses'].get('not_sent',0) for r in primary.values())
    invalid=any(r['invalid'] for r in primary.values())
    return dict(schema_version=1,status='INCOMPLETE' if incomplete else 'COMPLETED_WITH_INVALID_API_OUTPUTS' if invalid else 'COMPLETED',
        started_at=started_at,finished_at=utc(),seconds=time.perf_counter()-start,
        execution_manifest=str(execution),execution_manifest_sha256=digest(root/execution),frozen_development_protocol_sha256=a['protocol_sha256'],
        primary_seed=20261019,bootstrap=bootstrap,results=results,multi_seed=seeds,input_files_sha256=inputs,
        same_common500=True,common500_subset_of_holdout5000=True,common_and_full_are_not_independent_tests=True,
        final_labels_used_for_selection=False,all_reference_entities_retained=True,independent_of_inference_metric_implementation=True,
        no_claim_official_leaderboard=True,raw_outputs_unmodified=True,
        limitations=['Only final inference and analysis are authorized on sealed labels;no parameter/prompt/seed tuning.','Three-seed SD is sample SD(ddof=1),not proof of significance.','Invalid API outputs give N/A primary entity metrics and paired intervals;conditional diagnostics use a different denominator and are not comparable primary scores.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--execution',type=Path,default=Path('configs/stage4-execution.json'));p.add_argument('--output',type=Path,default=Path('evidence/stage4/final-aggregation.json'));args=p.parse_args()
    result=calculate(args.root,args.execution);dump_new(args.root/args.output,result)
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],output=str(args.output),api_valid={r:result['results'][r+'/common500']['valid'] for r in ('L0','L3')}),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
