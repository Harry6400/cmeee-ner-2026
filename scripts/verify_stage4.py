"""Independent final-output, training, latency, cost and freeze verification. Read-only inputs."""
from __future__ import annotations
import argparse,hashlib,json,math,time,traceback
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path
from aggregate_stage4 import calculate,load_manifest
from stage4_metrics import SEEDS,digest,read_jsonl,dump_new,equivalent,percentile
from historical_evidence import historical_source,checked_file


def check_training_segments(root,run,summary,metadata,required_tags,queue_entry=None,stage3_segment=None,queue_sources=None):
    """Reconstruct every original segment;discarded partial updates never counted as committed work."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    sources=queue_entry['segments'] if queue_entry else [dict(segment=0,status='completed',wall_seconds=stage3_segment['seconds'] if stage3_segment else summary['seconds'])]
    details=[];all_tags=set();logged_updates=discarded_updates=0;discarded_seconds=runner_seconds=child_wall_seconds=0.;committed_steps=set();next_start=1;previous_checkpoint=None
    for index,source in enumerate(sources):
        wall=source.get('wall_seconds',source.get('seconds'))
        if not isinstance(wall,(int,float)) or not math.isfinite(wall) or wall<0 or wall>84600:raise ValueError('Training child segment wall time missing/over cap')
        child_wall_seconds+=wall
        if source.get('console') and not (root/source['console']).is_file():raise ValueError('Missing original training segment console')
        if source['status']=='prelaunch_busy_race':
            if source.get('archive_dir') or source.get('discarded_partial_updates',0):raise ValueError('Busy-race segment cannot hide training updates')
            details.append(dict(segment=source['segment'],status=source['status'],child_wall_seconds=wall,logged_updates=0,discarded_partial_updates=0));continue
        directory=root/source['archive_dir'] if source.get('archive_dir') else run
        raw=json.loads((directory/'summary.json').read_text());meta=json.loads((directory/'metadata.json').read_text());cfg=json.loads((directory/'config.json').read_text())
        if cfg!=json.loads((run/'config.json').read_text()):raise ValueError('Archived segment config changed')
        for field in ('method','seed','precision','amp','tf32','train_samples','dev_samples','windows','vocab_size','sources'):
            if meta[field]!=metadata[field]:raise ValueError('Archived segment metadata changed:'+field)
        for filename in ('ner_core.py','ner_model.py','ner.py','stage3_common.py','train_stage4.py' if queue_entry else 'train_development.py'):
            historical_source(root,'scripts/'+filename,meta['code'][filename])
        if queue_sources:
            for filename,sha in queue_sources.items():
                historical_source(root,'scripts/'+filename,sha)
        if raw['records']!=summary['records'][:len(raw['records'])]:raise ValueError('Archived completed epochs were changed by recovery')
        runner_seconds+=raw['seconds'];rows=read_jsonl(directory/'metrics.jsonl');updates=[r for r in rows if r.get('event')=='train'];logged_updates+=len(updates)
        committed=raw['records'][-1]['global_step'] if raw['records'] else 0;partial=[r for r in updates if r['global_step']>committed]
        if len(partial)!=raw['global_step']-committed:raise ValueError('Partial-update count differs from original summary')
        partial_seconds=sum(r['seconds'] for r in partial);discarded_updates+=len(partial);discarded_seconds+=partial_seconds
        if source.get('archive_dir'):
            if source.get('last_complete_global_step')!=committed or source.get('discarded_partial_updates')!=len(partial) or not math.isclose(source.get('discarded_partial_update_seconds',-1),partial_seconds,abs_tol=1e-8):raise ValueError('Queue discarded-update evidence does not match archived logs')
            archived_info=json.loads((directory/'archive.json').read_text())
            for field in ('last_complete_global_step','discarded_partial_updates','discarded_partial_update_seconds','last_checkpoint_sha256'):
                if not equivalent(source[field],archived_info[field]):raise ValueError('Queue/archive evidence mismatch:'+field)
            if source.get('last_checkpoint_sha256') is not None and digest(directory/'last.pt')!=source['last_checkpoint_sha256']:raise ValueError('Archived parent checkpoint changed')
        elif partial:raise ValueError('Final segment still contains discarded partial epoch')
        if previous_checkpoint is not None:
            if meta.get('parent_checkpoint')!=previous_checkpoint:raise ValueError('Recovered segment parent checkpoint hash differs')
            restore=json.loads((directory/'restore-check.json').read_text())
            if restore['parent_sha256']!=previous_checkpoint or restore['global_step']!=next_start-1 or any(not restore[k] for k in ('model_equal','optimizer_equal','python_rng_equal','numpy_rng_equal','torch_rng_equal','cuda_rng_equal')):raise ValueError('Recovered model/optimizer/RNG/cursor verification failed')
        if updates:
            if updates[0]['global_step']!=next_start or updates[0]['window_cursor']!=32:raise ValueError('Restored optimizer step/sample cursor is discontinuous')
            for before,after in zip(updates,updates[1:]):
                expected_cursor=before['window_cursor']+32 if before['epoch']==after['epoch'] else 32
                if after['global_step']!=before['global_step']+1 or after['window_cursor']!=expected_cursor:raise ValueError('Within-segment step or sampling cursor discontinuity')
            if any(not math.isfinite(r[k]) for r in updates for k in ('loss','grad_norm','seconds','throughput_windows_per_sec','peak_mb')):raise ValueError('Nonfinite original-segment train log')
        committed_steps.update(r['global_step'] for r in updates if r['global_step']<=committed)
        ea=EventAccumulator(str(directory/'tensorboard'),size_guidance={'scalars':0});ea.Reload();tags=set(ea.Tags()['scalars']);all_tags|=tags
        update_tags={'train/loss','train/grad_norm','train/lr','train/optimizer_step','train/throughput_samples_per_sec','system/gpu_memory_peak_mb'}
        if updates and not update_tags<=tags:raise ValueError('Archived segment missing train TensorBoard tags')
        for tag in tags:
            if not all(math.isfinite(row.value) for row in ea.Scalars(tag)):raise ValueError('Archived TensorBoard contains nonfinite scalar')
        details.append(dict(segment=source.get('segment',index),directory=str(directory.relative_to(root)),status=raw['status'],runner_seconds=raw['seconds'],child_wall_seconds=wall,logged_updates=len(updates),first_step=updates[0]['global_step'] if updates else None,last_step=updates[-1]['global_step'] if updates else None,committed_last_step=committed,discarded_partial_updates=len(partial),discarded_partial_update_seconds=partial_seconds,tensorboard_tags=sorted(tags)))
        next_start=committed+1;previous_checkpoint=source.get('last_checkpoint_sha256') if source.get('archive_dir') else None
    if not required_tags<=all_tags:raise ValueError('All training segments combined lack required TensorBoard tags')
    if committed_steps!=set(range(1,summary['global_step']+1)):raise ValueError('Missing committed optimizer-step logs across all original segments')
    if logged_updates-discarded_updates!=summary['global_step']:raise ValueError('Logged updates minus discarded updates differs from final committed global step')
    if queue_entry and queue_entry.get('status')!='completed':raise ValueError('Training queue run not completed')
    return dict(segments=details,logged_updates=logged_updates,discarded_partial_updates=discarded_updates,discarded_partial_update_seconds=discarded_seconds,total_runner_seconds=runner_seconds,total_child_wall_seconds=child_wall_seconds,tensorboard_tags=sorted(all_tags))


def check_training(root,entries,protocol,a):
    entries=[e for e in entries if e['route'] in ('N0','N1') and e['split']=='holdout5000']
    checks=[]
    queue=json.loads((root/a.get('training_queue','evidence/stage4/training-queue.json')).read_text())
    if queue.get('status')!='completed':raise ValueError('Stage4 training queue is not completed')
    queue_runs={run['name']:run for run in queue['runs']}
    if set(queue_runs)!={f'{r}-seed{s}' for r in ('N0','N1') for s in (20261020,20261021)} or queue.get('authorized_gpu_uuid')!='GPU-f77a1700-de70-05d2-048f-18c8f9aa5974':raise ValueError('Unexpected training queue runs/GPU')
    stage3_queue=json.loads((root/'evidence/stage3/queue-develop-resolved.json').read_text())
    required_tags={'train/loss','train/lr','train/grad_norm','train/epoch_time_sec','train/optimizer_step','train/throughput_samples_per_sec','system/gpu_memory_peak_mb','val/precision','val/recall','val/f1','val/loss','val/nested_recall','val/character_f1'}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for entry in entries:
        run=entry.get('training_run')
        if not run:raise ValueError('Training provenance path required in local manifest')
        directory=root/run;s=json.loads((directory/'summary.json').read_text());m=json.loads((directory/'metadata.json').read_text());cfg=json.loads((directory/'config.json').read_text());records=s['records']
        if s['status'] not in ('completed','early_stopped') or s['epochs']!=len(records):raise ValueError('Incomplete training:'+run)
        if m['seed']!=entry['seed'] or m['method']!=entry['route'] or m['precision']!='fp32' or m['amp'] or m['tf32']:raise ValueError('Training method/seed/precision changed:'+run)
        expected=protocol['selected'][entry['route']]['config']
        for k,v in expected.items():
            if k not in ('seed','stage') and cfg.get(k)!=v:raise ValueError('Frozen training configuration changed:'+run+':'+k)
        if cfg['seed']!=entry['seed'] or cfg['micro_batch']*cfg['accumulation']!=32:raise ValueError('Seed/effective batch mismatch')
        if (m['train_samples'],m['dev_samples'],m['windows'])!=(13497,1503,13883):raise ValueError('Train/dev sample/window mismatch')
        if s.get('sealed_read') is True:raise ValueError('Training declared sealed label access')
        for split in ('train','dev'):
            if m['sources'][split]!=digest(root/f'data/processed/{split}.jsonl'):raise ValueError('Training split bytes changed')
        for rec in records:
            if rec['global_step']!=434*rec['epoch'] or rec['padded_training_windows']!=5 or rec['val']['samples']!=1503:raise ValueError('Training step/tail/dev denominator mismatch')
        best=max(records,key=lambda r:(r['val']['micro']['f1'],-r['epoch']))
        if best['epoch']!=s['best_epoch'] or best['val']['micro']['f1']!=s['best_f1']:raise ValueError('Checkpoint not selected exclusively by own internal dev')
        if entry.get('checkpoint_sha256')!=s['best_checkpoint_sha256']:raise ValueError('Inference checkpoint differs from selected dev-best:'+run)
        name=f"{entry['route']}-seed{entry['seed']}"
        queue_entry=queue_runs.get(name) if entry['seed']!=20261019 else None
        if entry['seed']!=20261019 and queue_entry is None:raise ValueError('Missing queue run:'+name)
        stage3_segment=next((r for r in stage3_queue['segments'] if r['name']==protocol['selected'][entry['route']]['run']),None) if entry['seed']==20261019 else None
        if entry['seed']==20261019 and stage3_segment is None:raise ValueError('Missing original stage3 source segment')
        segments=check_training_segments(root,directory,s,m,required_tags,queue_entry,stage3_segment,queue.get('source_sha256') if queue_entry else None)
        samples=[o['utilization_percent'] for o in queue_entry.get('runtime_observations',[]) if not o.get('foreign')] if queue_entry else []
        utilization=dict(samples=len(samples),sample_mean_percent=sum(samples)/len(samples) if samples else None,p50_percent=percentile(samples,.5) if samples else None,p95_percent=percentile(samples,.95) if samples else None,definition='10second hardware utilization snapshots while own training child active;foreign-compute snapshots excluded;not timeweighted kernel utilization')
        checks.append(dict(wait_seconds=queue_entry.get('wait_seconds',0) if queue_entry else None,gpu_utilization=utilization,route=entry['route'],seed=entry['seed'],training_run=run,status=s['status'],epochs=s['epochs'],best_epoch=s['best_epoch'],best_dev_f1=s['best_f1'],runner_seconds=segments['total_runner_seconds'],child_wall_seconds=segments['total_child_wall_seconds'],checkpoint_sha256=s['best_checkpoint_sha256'],segment_checks=segments))
    if {(r['route'],r['seed']) for r in checks}!={(r,s) for r in ('N0','N1') for s in SEEDS}:raise ValueError('Missing seed training provenance')
    return checks


def check_latencies(root,entries,gold_ids):
    results=[]
    for e in entries:
        if e['route'] not in ('D0','N0','N1'):continue
        if not e.get('timing_path') or not e.get('metadata_path'):raise ValueError('Local timing/metadata evidence required')
        timing=json.loads((root/e['timing_path']).read_text());meta=json.loads((root/e['metadata_path']).read_text())
        if e['route']!='D0' and (meta.get('gpu_uuid')!='GPU-f77a1700-de70-05d2-048f-18c8f9aa5974' or meta.get('precision')!='fp32' or meta.get('amp') is not False or meta.get('tf32') is not False):raise ValueError('Actual inference GPU or precision mismatch')
        detail=dict(route=e['route'],seed=e.get('seed'),split=e['split'],timing_path=e['timing_path'],load_seconds=timing['load_seconds'],device=meta.get('device'),gpu_uuid=meta.get('gpu_uuid'),timing_host='QHA server;D0 CPU or A40 GPU7')
        if e['split']=='common500':
            path=Path(e['timing_path']).parent/'per-sample-timing.jsonl';rows=read_jsonl(root/path)
            if [r['id'] for r in rows]!=gold_ids or any(not r.get('single_sample') for r in rows):raise ValueError('Single-sample latency IDs or semantics mismatch')
            times=[r['seconds'] for r in rows]
            if len(times)!=500 or any(not math.isfinite(t) or t<0 for t in times):raise ValueError('Invalid latency denominator/value')
            actual=dict(samples=500,total_seconds=sum(times),mean_seconds=sum(times)/500,p50_seconds=percentile(times,.5),p95_seconds=percentile(times,.95))
            recorded=timing['single_sample_latency']
            for k,v in actual.items():
                if not math.isclose(recorded[k],v,rel_tol=1e-9,abs_tol=1e-9):raise ValueError('Single-sample latency summary cannot be recomputed:'+k)
            detail['single_sample_latency']=actual;detail['same_as_main_batch_prediction']=sum(r.get('main_batch_prediction_equal') is True for r in rows)
            detail['different_batch_shape_predictions_not_used_for_main_score']=sum(r.get('main_batch_prediction_equal') is False for r in rows)
        elif e['route']=='D0':
            path=Path(e['timing_path']).parent/'per-sample-timing.jsonl';rows=read_jsonl(root/path)
            if len(rows)!=5000 or len({r['id'] for r in rows})!=5000 or any(not r.get('single_sample') for r in rows):raise ValueError('D0 full5000 must be actual CPU per-document timing')
            times=[r['seconds'] for r in rows];recorded=timing['single_sample_latency']
            actual=dict(samples=5000,total_seconds=sum(times),mean_seconds=sum(times)/5000,p50_seconds=percentile(times,.5),p95_seconds=percentile(times,.95))
            for k,v in actual.items():
                if not math.isclose(recorded[k],v,rel_tol=1e-9,abs_tol=1e-9):raise ValueError('D0 full latency cannot be recomputed:'+k)
            detail['cpu_single_sample_latency']=actual
        else:
            batch=timing['batch_evaluation']
            if batch['samples']!=5000 or not math.isfinite(batch['seconds']) or batch['seconds']<=0:raise ValueError('Full5000 batch timing mismatch')
            detail['batch_evaluation']=batch;detail['batch_average_is_not_response_latency']=True
        results.append(detail)
    return results


def independent_wire(row,wire):
    if not isinstance(wire,dict) or not isinstance(wire.get('entities'),list):raise ValueError('Wire entities missing')
    found=set()
    from stage4_metrics import TYPES
    for entity in wire['entities']:
        if not isinstance(entity,dict):raise ValueError('Wire entity not object')
        text=entity.get('text');n=entity.get('occurrence');t=entity.get('type')
        if not isinstance(text,str) or not text or type(n) is not int or n<0 or t not in TYPES:raise ValueError('Invalid wire type,surface or occurrence')
        offsets=[];cursor=0
        while True:
            cursor=row['text'].find(text,cursor)
            if cursor<0:break
            offsets.append(cursor);cursor+=1
        if n>=len(offsets):raise ValueError('Wire surface occurrence not present in original')
        found.add((offsets[n],offsets[n]+len(text),t))
    return [dict(start=s,end=e,type=t,text=row['text'][s:e]) for s,e,t in sorted(found)]


def request_bytes(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode('utf-8')


def expected_api_payload(row,route,examples,frozen):
    """Independent construction;does not import the production API prompt builder."""
    if route not in ('L0','L3'):raise ValueError('Unknown frozen API route')
    api=frozen['api']
    if (api['temperature'],api['thinking'],api['response_format'],api['max_output_tokens'])!=(0,'disabled','json_object',4096):raise ValueError('API decoding protocol changed')
    def user(r):return json.dumps(dict(text=r['text'],unicode_character_index=[f'{i}:{ch}' for i,ch in enumerate(r['text'])]),ensure_ascii=False)
    def example_wire(r):
        entities=[]
        for e in r['entities']:
            starts=[];cursor=0
            while True:
                cursor=r['text'].find(e['text'],cursor)
                if cursor<0:break
                starts.append(cursor);cursor+=1
            if e['start'] not in starts:raise ValueError('Frozen training example position not in original')
            entities.append(dict(text=e['text'],type=e['type'],occurrence=starts.index(e['start'])))
        return dict(entities=entities)
    messages=[dict(role='system',content=api['system'])]
    if route=='L3':
        for example in examples:messages.extend([dict(role='user',content=user(example)),dict(role='assistant',content=json.dumps(example_wire(example),ensure_ascii=False))])
    messages.append(dict(role='user',content=user(row)))
    return dict(model=api['model'],messages=messages,max_tokens=4096,temperature=0,response_format=dict(type='json_object'),thinking=dict(type='disabled'))


def validate_manifest_items(items,samples,examples,frozen,run_id):
    expected=[(route,row) for route in ('L0','L3') for row in samples]
    if len(samples)!=500 or len({r['id'] for r in samples})!=500 or len(examples)!=3 or len(items)!=1000:raise ValueError('Require exact frozen1000 API requests and500+3 records')
    if [(i.get('route'),i.get('sample_id')) for i in items]!=[(route,row['id']) for route,row in expected]:raise ValueError('API request manifest route/ID order differs from frozen1000')
    if len({i.get('key') for i in items})!=1000:raise ValueError('Duplicate API request manifest key')
    for item,(route,row) in zip(items,expected):
        if item['key']!=run_id+':'+route+':'+row['id']:raise ValueError('API manifest key not tied to route and original ID')
        payload=expected_api_payload(row,route,examples,frozen)
        if item.get('payload')!=payload or request_bytes(item['payload'])!=request_bytes(payload):raise ValueError('Request payload differs from independently constructed frozen prompt/input/examples')
        if hashlib.sha256(request_bytes(item['payload'])).hexdigest()!=item.get('payload_sha256'):raise ValueError('API manifest payload canonical digest mismatch')
    return {item['key']:item for item in items}


def check_api_request_freeze(root,a,gold_by_id):
    directory=root/a['api_run_dir'];protocol=json.loads((directory/'protocol-at-call.json').read_text())
    files={'frozen_protocol_sha256':'frozen-protocol-at-call.json','authorization_sha256':'authorization-at-call.json','price_sha256':'price-at-call.json','samples_sha256':'samples.jsonl','examples_sha256':'train-examples.jsonl','request_manifest_sha256':'requests-manifest.jsonl','privacy_inventory_sha256':'privacy-review-inputs.jsonl'}
    for field,name in files.items():
        if digest(directory/name)!=protocol[field]:raise ValueError('API pre-call frozen file changed:'+name)
    frozen=json.loads((directory/'frozen-protocol-at-call.json').read_text());price=json.loads((directory/'price-at-call.json').read_text());authorization=json.loads((directory/'authorization-at-call.json').read_text())
    if digest(root/'configs/stage4-frozen-protocol.json')!=protocol['frozen_protocol_sha256'] or digest(root/'configs/qha-stage4-contract.json')!=protocol['authorization_sha256']:raise ValueError('API frozen protocol/authorization differs from owning project')
    if authorization.get('stage')!=4 or authorization.get('authorization')!='explicit_user_stage4_2026-10-20' or authorization.get('explicit_final_api_authorized') is not True or authorization.get('final_holdout_authorized') is not True or authorization.get('api_budget_cny')!='100' or not authorization.get('authorization_request'):raise ValueError('Missing explicit final external API authorization')
    if (protocol.get('expected_calls'),protocol.get('samples'),protocol.get('routes'))!=(1000,500,['L0','L3']) or protocol.get('no_automatic_retries') is not True or protocol.get('no_fuzzy_repair') is not True:raise ValueError('API frozen run scope changed')
    if Decimal(price['input_peak_uncached_per_million'])!=2 or Decimal(price['output_peak_per_million'])!=8:raise ValueError('Frozen conservative peak2/8 CNY rates changed')
    if price.get('model')!=frozen['api']['model'] or price.get('version_reported_by_official_docs')!=frozen['api']['official_version_at_development'] or price.get('same_official_version_as_development') is not True or not price.get('source') or not price.get('updates_source'):raise ValueError('Missing matching official API model/price review')
    samples=read_jsonl(directory/'samples.jsonl');examples=read_jsonl(directory/'train-examples.jsonl')
    if [r['id'] for r in samples]!=list(gold_by_id) or any(row!=gold_by_id[row['id']] for row in samples):raise ValueError('API samples not exact same final500')
    if [r['id'] for r in examples]!=frozen['api']['example_ids'] or protocol.get('example_ids')!=frozen['api']['example_ids'] or protocol.get('sample_ids')!=list(gold_by_id):raise ValueError('Frozen API example/sample ID list changed')
    if digest(root/frozen['data']['common_subset'])!=protocol['samples_sha256'] or digest(root/frozen['api']['examples'])!=protocol['examples_sha256']:raise ValueError('API samples/examples source bytes changed')
    train={r['id']:r for r in read_jsonl(root/frozen['data']['train'])}
    if any(train.get(r['id'])!=r for r in examples) or len(set(gold_by_id)|{r['id'] for r in examples})!=503:raise ValueError('API examples must be exact training records disjoint from final500')
    from stage4_metrics import entity_keys
    for row in samples+examples:entity_keys(row)
    inventory=read_jsonl(directory/'privacy-review-inputs.jsonl');privacy=json.loads((directory/'privacy-review.json').read_text());started=json.loads((directory/'run-started.json').read_text())
    if len(inventory)!=503 or [r['id'] for r in inventory]!=[r['id'] for r in samples+examples]:raise ValueError('Privacy inventory does not cover exact503 external records')
    for index,(review_row,source) in enumerate(zip(inventory,samples+examples)):
        if review_row['text']!=source['text'] or review_row['text_sha256']!=hashlib.sha256(source['text'].encode('utf-8')).hexdigest() or review_row['source']!=('common500' if index<500 else 'frozen_training_example'):raise ValueError('Privacy inventory text/source differs from exact external send record')
    flagged=[r['id'] for r in inventory if r['regex_flags']]
    if flagged!=protocol['regex_flagged_ids'] or sorted(privacy.get('resolved_regex_flagged_ids',[]))!=sorted(flagged):raise ValueError('Privacy findings lack matching explicit resolution')
    if privacy.get('inventory_sha256')!=protocol['privacy_inventory_sha256'] or privacy.get('reviewed_ids')!=[r['id'] for r in inventory] or privacy.get('allow_external_send') is not True or privacy.get('semantic_review_completed') is not True or privacy.get('unresolved_findings')!=[] or not privacy.get('reviewer_kind') or not privacy.get('review_notes'):raise ValueError('Missing complete semantic privacy review and external-send permission')
    if started.get('protocol_sha256')!=digest(directory/'protocol-at-call.json') or started.get('privacy_review_sha256')!=digest(directory/'privacy-review.json'):raise ValueError('API run-started protocol/privacy evidence changed')
    if digest(root/protocol['opening_ledger_snapshot'])!=protocol['opening_ledger_sha256']:raise ValueError('API pre-call opening ledger snapshot changed')
    for name,sha in protocol.get('source_at_call',{}).items():
        checked_file(root,(directory/'source-at-call'/name).relative_to(root).as_posix(),sha)
    items=read_jsonl(directory/'requests-manifest.jsonl');manifest=validate_manifest_items(items,samples,examples,frozen,protocol['run_id'])
    if (directory/'requests-manifest.jsonl').read_bytes()!=b''.join(request_bytes(item)+b'\n' for item in items):raise ValueError('API request manifest canonical bytes changed')
    return dict(protocol=protocol,price=price,manifest=manifest,ordered_items=items,privacy_review_sha256=digest(directory/'privacy-review.json'),request_manifest_sha256=protocol['request_manifest_sha256'],independent_request_rebuild_verified=True)


def validate_api_call_frozen(call,raw,ledger_entry,item,price):
    if call.get('key')!=item['key'] or raw.get('key')!=item['key'] or call['route']!=item['route'] or call['sample_id']!=item['sample_id']:raise ValueError('API call identity not in frozen request manifest')
    if ledger_entry.get('request_key')!=item['key'] or ledger_entry.get('origin')!='cmeee_ner_stage4' or call.get('actual_model')!=price.get('model') or ledger_entry.get('actual_model')!=price.get('model'):raise ValueError('Settled API ledger request/model is not the frozen authorized call')
    payload_sha=hashlib.sha256(request_bytes(raw['payload'])).hexdigest()
    if raw['payload']!=item['payload'] or payload_sha!=item['payload_sha256'] or call.get('payload_sha256')!=item['payload_sha256'] or raw.get('payload_sha256')!=item['payload_sha256'] or ledger_entry.get('payload_sha256')!=item['payload_sha256']:raise ValueError('Raw/call/ledger payload differs from immutable request manifest')
    usage=call['usage']
    if type(usage.get('prompt_tokens')) is not int or type(usage.get('completion_tokens')) is not int or not 0<=usage['prompt_tokens']<=1048576 or not 0<=usage['completion_tokens']<=4096:raise ValueError('API token counts illegal or exceed frozen reservation')
    if 'total_tokens' in usage and (type(usage['total_tokens']) is not int or usage['total_tokens']!=usage['prompt_tokens']+usage['completion_tokens']):raise ValueError('API total tokens do not match input+output')
    if Decimal(price['input_peak_uncached_per_million'])!=2 or Decimal(price['output_peak_per_million'])!=8:raise ValueError('Expected frozen conservative peak2/8 CNY rates')
    charge=(Decimal(usage['prompt_tokens'])*Decimal(2)+Decimal(usage['completion_tokens'])*Decimal(8))/Decimal(1000000)
    if not charge.is_finite() or Decimal(call['charge_cny'])!=charge or Decimal(ledger_entry['conservative_charge_cny'])!=charge:raise ValueError('API charge differs from independently recomputed frozen token2/8 peak tariff')
    if ledger_entry.get('pricing')!=price:raise ValueError('Settled ledger pricing differs from frozen official price snapshot')
    return charge


APPROVED_UNKNOWN_ID='ner-final-85a1172afb6d4ff6b1be8cbcce79333b'
APPROVED_UNKNOWN_SAMPLE='cmeee-v2-dev-02327'


def approved_unknown_accounting(ledger,opening,proof,items):
    """Exactly one named operational exception;never create usage or mark it settled."""
    expected_bound=(Decimal(1048576)*2+Decimal(4096)*8)/Decimal(1000000)
    if proof.get('explicit_approved') is not True or not proof.get('human_reply') or proof.get('approved_unknown',{}).get('id')!=APPROVED_UNKNOWN_ID:raise ValueError('Exact existing-unknown human approval missing or altered')
    if proof.get('opening_rows')!=1074 or proof.get('max_remaining_calls')!=766 or proof.get('max_total_stage4_attempts')!=1000 or proof.get('no_resend_any_attempted_key') is not True or proof.get('new_unknown_rule')!='stop;only above exact existing unknown approved':raise ValueError('Unknown exception scope expanded')
    if proof.get('no_changes_to_frozen_payload_prompt_examples') is not True or proof.get('unknown_is_not_claimed_actual_settled') is not True or proof.get('cumulative_cap_cny')!='100' or proof.get('stage4_new_soft_cap_cny')!='80':raise ValueError('Exception must preserve science and cumulative budget')
    if len(items)!=1000 or items[233]['route']!='L0' or items[233]['sample_id']!=APPROVED_UNKNOWN_SAMPLE:raise ValueError('Approved exception is not original manifest item234')
    if len(opening['requests'])!=1074 or len(ledger['requests'])!=1840 or request_bytes(ledger['requests'][:1074])!=request_bytes(opening['requests']):raise ValueError('Complete1074-row prefix changed or final1840-row count wrong')
    rows=ledger['requests'];unresolved=[r for r in rows if r.get('status')!='settled']
    if len(unresolved)!=1 or unresolved[0].get('id')!=APPROVED_UNKNOWN_ID or unresolved[0].get('status')!='unknown_cost':raise ValueError('A second unknown/pending or a fabricated settlement is not authorized')
    unknown=unresolved[0];approved=proof['approved_unknown']
    if unknown.get('request_key')!=items[233]['key'] or unknown.get('payload_sha256')!=items[233]['payload_sha256'] or approved.get('request_key')!=unknown['request_key'] or approved.get('payload_sha256')!=unknown['payload_sha256'] or approved.get('status')!='unknown_cost':raise ValueError('Approved unknown ID/key/payload/status changed')
    if hashlib.sha256(request_bytes(unknown)).hexdigest()!=approved.get('entry_sha256'):raise ValueError('Original unknown ledger entry altered')
    if any(k in unknown for k in ('usage','conservative_charge_cny','settled_at','actual_model')):raise ValueError('Unknown charge cannot acquire invented usage/settlement metadata')
    if any(Decimal(str(value))!=expected_bound for value in (unknown.get('reserved_cny'),approved.get('reserved_cny'),proof.get('approved_unknown_bound_cny'))):raise ValueError('Unknown upper bound differs from independent fullcontext2/8 reservation')
    prior_stage=opening['requests'][840:]
    if [r.get('request_key') for r in prior_stage]!=[i['key'] for i in items[:234]] or len(prior_stage)!=234:raise ValueError('Original840+234 attempt prefix does not match frozen manifest')
    tail=rows[1074:]
    if [r.get('request_key') for r in tail]!=[i['key'] for i in items[234:]] or any(r.get('origin')!='cmeee_ner_stage4' or r.get('status')!='settled' or r.get('run_directory')!=proof['original_API_run'] for r in tail):raise ValueError('Only exact766 never-attempted tail requests may be appended')
    stage=rows[840:]
    if [r.get('request_key') for r in stage]!=[i['key'] for i in items] or len({r.get('id') for r in rows})!=1840:raise ValueError('Full1000 attempt order or unique request IDs changed')
    amounts=[Decimal(r['conservative_charge_cny']) for r in rows if r['status']=='settled']
    if any(not c.is_finite() or c<0 for c in amounts):raise ValueError('Invalid settled charge')
    known=sum(amounts,Decimal(0));upper=known+expected_bound
    opening_known=sum((Decimal(r['conservative_charge_cny']) for r in opening['requests'] if r['status']=='settled'),Decimal(0))
    if opening_known!=Decimal(proof['known_spent_before_cny']) or opening_known+expected_bound!=Decimal(proof['exposure_upper_before_cny']) or Decimal(proof['original_stage4_opening_cny'])!=Decimal('3.872064'):raise ValueError('Approval financial opening differs from actual1074 prefix')
    if Decimal(ledger['spent_cny'])!=known or upper>100 or upper-Decimal('3.872064')>80:raise ValueError('Known+approved unknown exposure exceeds cumulative/new-stage budget')
    expected_fields=dict(known_spent_cny=known,unknown_cost_upper_cny=expected_bound,exposure_upper_cny=upper,pending_reservation_cny=Decimal(0),exposure_including_pending_upper_cny=upper,upper_remaining_cny=100-upper)
    for key,value in expected_fields.items():
        if Decimal(ledger[key])!=value:raise ValueError('Bound-aware ledger total mismatch:'+key)
    if ledger.get('api_enabled') is not False or ledger.get('unknown_charge_is_not_settled') is not True or ledger.get('status')!='active_with_reserved_unknown_cost':raise ValueError('Final API must be disabled with unresolved charge honestly retained')
    return dict(known=known,unknown_bound=expected_bound,exposure_upper=upper,unknown=unknown,settled_calls=999,total_attempts=1000)


def check_approved_unknown_annex(root,a,ledger,request_freeze,calls,canonical):
    directory=root/a['api_run_dir'];proof_path=root/a['api_budget_amendment_path'];proof=json.loads(proof_path.read_text());annex_path=root/a['api_continuation_annex_path'];annex=json.loads(annex_path.read_text());base=annex_path.parent
    approval_copy=base/'approval-at-call.json'
    if digest(proof_path)!=annex['approval_sha256'] or approval_copy.read_bytes()!=proof_path.read_bytes():raise ValueError('Continuation annex approval copy differs from human proof')
    before_path=root/proof['continuation_opening_ledger'];opening=json.loads(before_path.read_text())
    if digest(before_path)!=proof['continuation_opening_ledger_sha256'] or annex['ledger_before_snapshot']!=proof['continuation_opening_ledger'] or annex['ledger_before_sha256']!=digest(before_path):raise ValueError('Approved1074-row opening snapshot changed')
    account=approved_unknown_accounting(ledger,opening,proof,request_freeze['ordered_items']);unknown=account['unknown'];items=request_freeze['ordered_items']
    if annex.get('name')!='continuation01' or annex.get('approved_unknown_request_ids')!=[APPROVED_UNKNOWN_ID] or annex.get('original_unknown_entry')!=unknown or annex.get('preserved_request_prefix_count')!=1074 or annex.get('preserved_request_prefix_sha256')!=hashlib.sha256(request_bytes(opening['requests'])).hexdigest():raise ValueError('Annex expands or changes approved unknown/prefix')
    if annex.get('remaining_keys')!=[i['key'] for i in items[234:]] or annex.get('remaining_count')!=766 or annex.get('max_total_attempts')!=1000 or annex.get('no_automatic_retry') is not True or annex.get('another_unknown_must_stop') is not True or annex.get('unknown_is_not_settled') is not True:raise ValueError('Continuation manifest tail/scope changed')
    for field,path in [('original_protocol_sha256',directory/'protocol-at-call.json'),('original_frozen_protocol_sha256',directory/'frozen-protocol-at-call.json'),('original_request_manifest_sha256',directory/'requests-manifest.jsonl')]:
        if annex[field]!=digest(path):raise ValueError('Original science/manifest changed across continuation')
    if Decimal(annex['unknown_bound_cny'])!=account['unknown_bound'] or annex.get('cumulative_limit_cny')!='100' or annex.get('new_stage_soft_cap_cny')!='80':raise ValueError('Annex budget bound changed')
    source='api_final_bounded_continue.py'
    checked_file(root,(base/'source-at-continuation'/source).relative_to(root).as_posix(),annex['source_sha256'])
    for name,sha in annex['prior_snapshots_sha256'].items():
        saved=base/'prior-state'/name
        if digest(saved)!=sha:raise ValueError('Preserved pre-continuation partial artifact changed:'+name)
        if name in ('summary.json','L0-predictions.jsonl','L3-predictions.jsonl','segments.jsonl') and digest(directory/name)!=sha:raise ValueError('Original partial output overwritten:'+name)
    original_calls=(directory/'calls.jsonl').read_bytes()[:annex['old_calls_bytes']]
    if annex.get('old_calls_count')!=233 or hashlib.sha256(original_calls).hexdigest()!=annex['old_calls_prefix_sha256'] or original_calls!=(base/'prior-state/calls.jsonl').read_bytes():raise ValueError('Original233 settled call byte prefix changed')
    failure_path=directory/annex['failure_file'];failure=json.loads(failure_path.read_text())
    if digest(failure_path)!=annex['failure_sha256'] or failure.get('id')!=APPROVED_UNKNOWN_ID or failure.get('key')!=unknown['request_key'] or failure.get('route')!='L0' or failure.get('sample_id')!=APPROVED_UNKNOWN_SAMPLE or not math.isfinite(failure['seconds']) or failure['seconds']<0:raise ValueError('Original unknown failure identity/timing changed')
    if (directory/'requests'/('raw-'+APPROVED_UNKNOWN_ID+'.json')).exists() or any(c['id']==APPROVED_UNKNOWN_ID for c in calls):raise ValueError('Unknown request gained fabricated response/call or was retried')
    pred=canonical['L0'][APPROVED_UNKNOWN_SAMPLE]
    if pred.get('status')!='transport_error' or pred.get('request_id')!=APPROVED_UNKNOWN_ID or pred.get('entities_unavailable') is not True or 'entities' in pred:raise ValueError('Unknown response must remain unavailable transport_error,never empty/success')
    for route in ('L0','L3'):
        if a['api_prediction_paths'][route]!=proof['canonical_outputs'][route]:raise ValueError('Canonical continuation path differs from approved destination')
    if a['api_summary_path']!=proof['summary_output']:raise ValueError('Continuation summary path differs from approved destination')
    adjustment=dict(id='api-continuation01-reserve-'+APPROVED_UNKNOWN_ID,kind='approved_unknown_cost_upper_bound_reservation',request_id=APPROVED_UNKNOWN_ID,amount_upper_cny=unknown['reserved_cny'],not_actual_settlement=True,original_unknown_status_preserved=True,approval_sha256=annex['approval_sha256'],annex_sha256=digest(annex_path),ledger_before_sha256=annex['ledger_before_sha256'],effective_from=proof['recorded_at'])
    if ledger.get('adjustments',[])!=opening.get('adjustments',[])+[adjustment] or json.loads((root/'private/budget/api-continuation01-adjustment.json').read_text())!=adjustment:raise ValueError('Unknown bound adjustment was changed,omitted or counted twice')
    # Full1000-attempt log (retrospective old records explicitly marked);never1000 settlements.
    attempts=read_jsonl(directory/'attempts-log-continuation01.jsonl')
    if len(attempts)!=1000 or [r.get('key') for r in attempts]!=[i['key'] for i in items] or len({r.get('id') for r in attempts})!=1000:raise ValueError('Attempt log must cover exact1000 frozen keys once in manifest order')
    call_by_key={c['key']:c for c in calls}
    for attempt in attempts:
        if attempt['id']==APPROVED_UNKNOWN_ID:
            if attempt.get('status')!='transport_error' or attempt.get('cost_status')!='unknown_cost' or attempt.get('usage_available') is not False or Decimal(attempt['charge_upper_cny'])!=account['unknown_bound'] or attempt['seconds']!=failure['seconds'] or 'charge_cny' in attempt or 'usage' in attempt:raise ValueError('Unknown attempt log cannot invent settlement/usage')
        else:
            call=call_by_key.get(attempt['key'])
            if call is None or attempt['id']!=call['id'] or attempt['status']!=call['status'] or attempt.get('cost_status')!='settled' or attempt.get('usage_available') is not True or attempt['seconds']!=call['seconds'] or Decimal(attempt['charge_cny'])!=Decimal(call['charge_cny']):raise ValueError('Settled attempt log differs from real durable response')
    started=json.loads((base/'run-started.json').read_text())
    if started.get('annex_sha256')!=digest(annex_path):raise ValueError('Continuation run-started annex differs')
    return dict(account=account,proof=proof,annex=annex,failure=failure,attempts=attempts,proof_sha256=digest(proof_path),annex_sha256=digest(annex_path))


def check_budget_api(root,a,gold_ids):
    ledger=json.loads((root/'private/budget/ledger.json').read_text());old=json.loads((root/'private/budget/original-ledger-20261018.json').read_text());rows=ledger['requests']
    if rows[:len(old['requests'])]!=old['requests']:raise ValueError('Historical paid requests changed')
    if ledger['api_enabled'] or Decimal(ledger['limit_cny'])!=100:raise ValueError('API must be closed and cumulative cap remain100')
    amendment=bool(a.get('api_budget_amendment_path'))
    if len({r['id'] for r in rows})!=len(rows) or (not amendment and any(r['status']!='settled' for r in rows)):raise ValueError('Duplicate or unauthorized unsettled cost records')
    amounts=[Decimal(r['conservative_charge_cny']) for r in rows if r['status']=='settled']
    if any(not n.is_finite() or n<0 for n in amounts):raise ValueError('Invalid charge')
    spent=sum(amounts,Decimal(0))
    if spent!=Decimal(ledger['spent_cny']) or spent>100:raise ValueError('Cumulative ledger mismatch or budget exceeded')
    if a.get('opening_ledger'):
        opening=json.loads((root/a['opening_ledger']).read_text())
        if rows[:len(opening['requests'])]!=opening['requests']:raise ValueError('Stage4 opening ledger history changed')
    gold_by_id={r['id']:r for r in read_jsonl(root/a.get('gold',{}).get('common500','data/sealed/common500.jsonl'))}
    request_freeze=check_api_request_freeze(root,a,gold_by_id)
    canonical={route:{p['id']:p for p in read_jsonl(root/a.get('api_prediction_paths',{}).get(route,str(Path(a['api_run_dir'])/(route+'-predictions.jsonl'))))} for route in ('L0','L3')}
    calls=read_jsonl(root/a['api_run_dir']/'calls.jsonl');by_id={r['id']:r for r in rows};routes={}
    continuation=None
    if amendment:
        if Path(a['api_continuation_annex_path']).parent.name=='continuation02':
            from stage4_transport_budget import check_transport_budget_annex
            continuation=check_transport_budget_annex(root,a,ledger,request_freeze,calls,canonical)
            continuation['unknown_entries']=continuation['account']['unknown']
        else:
            continuation=check_approved_unknown_annex(root,a,ledger,request_freeze,calls,canonical)
            continuation.update(version='continuation01',unknown_entries=[continuation['account']['unknown']],failures={APPROVED_UNKNOWN_ID:continuation['failure']})
    unknown_keys={r['request_key'] for r in continuation['unknown_entries']} if continuation else set()
    expected_items=[i for i in request_freeze['ordered_items'] if i['key'] not in unknown_keys]
    if len(calls)!=len(expected_items) or len({c['id'] for c in calls})!=len(expected_items) or [c.get('key') for c in calls]!=[i['key'] for i in expected_items]:raise ValueError('Final settled calls do not match frozen manifest excluding only explicitly approved transport unknowns')
    if a.get('opening_ledger'):
        opening=json.loads((root/a['opening_ledger']).read_text())
        if {r['id'] for r in rows[len(opening['requests']):] if r['status']=='settled'}!={c['id'] for c in calls}:raise ValueError('New cumulative settled records differ from the original1000 attempts excluding unavailable responses')
    for c in calls:
        if c['id'] not in by_id or Decimal(by_id[c['id']]['conservative_charge_cny'])!=Decimal(c['charge_cny']):raise ValueError('API raw call cannot be reconciled to settled ledger')
        if by_id[c['id']]['usage']!=c['usage']:raise ValueError('API recorded usage differs from ledger')
        raw_path=root/a['api_run_dir']/c['raw_file']
        if not raw_path.is_file():raise ValueError('Missing immutable raw API response')
        raw=json.loads(raw_path.read_text())
        item=request_freeze['manifest'].get(c.get('key'))
        if item is None:raise ValueError('API call not present in frozen request manifest')
        validate_api_call_frozen(c,raw,by_id[c['id']],item,request_freeze['price'])
        if type(c['seconds']) not in (int,float) or not math.isfinite(c['seconds']) or c['seconds']<0 or not math.isclose(c['seconds'],raw['seconds'],rel_tol=0,abs_tol=1e-12):raise ValueError('API call latency differs from immutable raw complete-response timing')
        if raw['sample_id']!=c['sample_id'] or raw['route']!=c['route'] or raw['id']!=c['id']:raise ValueError('Raw API response identity mismatch')
        if hashlib.sha256(raw['raw_body_utf8'].encode()).hexdigest()!=raw['raw_body_sha256'] or json.loads(raw['raw_body_utf8'])!=raw['response']:raise ValueError('Raw body and parsed API response differ')
        if raw['response']['usage']!=c['usage'] or raw['response']['model']!=c['actual_model']:raise ValueError('Raw API model/usage differs from settled call')
        pred=canonical[c['route']][c['sample_id']]
        if pred['status']!=c['status']:raise ValueError('Canonical API status differs from raw durable call')
        if c['status']!='valid' and 'entities' in pred:raise ValueError('Invalid API response cannot be silently filled with entities')
        if c['status']=='valid':
            choice=raw['response']['choices'][0]
            if choice['finish_reason']!='stop':raise ValueError('Truncated API response cannot be valid')
            grounded=independent_wire(gold_by_id[c['sample_id']],json.loads(choice['message']['content']))
            if pred['entities']!=grounded:raise ValueError('Canonical API spans cannot be reproduced from exact raw wire mapping')
    for route in ('L0','L3'):
        rr=[c for c in calls if c['route']==route]
        if [c['sample_id'] for c in rr]!=[i['sample_id'] for i in expected_items if i['route']==route] or len(rr)>500:raise ValueError('API route call IDs/order differ from frozen common500;no automatic retry')
        times=[r['seconds'] for r in rr]
        transport_seconds=[continuation['failures'][entry['id']]['seconds'] for entry in continuation['unknown_entries'] if request_freeze['manifest'][entry['request_key']]['route']==route] if continuation else []
        routes[route]=dict(attempts=500,settled_calls=len(rr),response_samples=len(times),transport_errors=len(transport_seconds),transport_error_seconds=transport_seconds,conservative_cny=str(sum((Decimal(r['charge_cny']) for r in rr),Decimal(0))),
            latency=dict(samples=len(times),mean_seconds=sum(times)/len(times) if times else None,p50_seconds=percentile(times,.5) if times else None,p95_seconds=percentile(times,.95) if times else None),
            latency_definition='complete known HTTP responses including invalid entity outputs;transport failure timing separate',raw_models=sorted({r['actual_model'] for r in rr}),errors=sum(r['status']!='valid' for r in rr))
    upper=continuation['account']['exposure_upper'] if continuation else spent
    unknown_bound=continuation['account']['unknown_bound'] if continuation else Decimal(0)
    summary_path=a.get('api_summary_path',str(Path(a['api_run_dir'])/'summary.json'));summary=json.loads((root/summary_path).read_text())
    if continuation:
        if summary.get('total_attempted')!=1000 or summary.get('settled_calls')!=len(calls) or summary.get('unknown_cost_calls')!=len(continuation['unknown_entries']) or Decimal(summary['known_cumulative_cny'])!=spent or Decimal(summary['unknown_cost_upper_cny'])!=unknown_bound or Decimal(summary['exposure_upper_cny'])!=upper:raise ValueError('Continuation summary misstates attempts,settlement or known/upper costs')
        for route in ('L0','L3'):
            sr=summary['routes'][route]
            if sr.get('total_attempts')!=500 or sr.get('settled_calls')!=routes[route]['settled_calls'] or sr.get('transport_error')!=routes[route]['transport_errors'] or sr.get('not_sent')!=0:raise ValueError('Continuation route summary differs from actual full500 attempts')
        if summary['routes']['L0'].get('main_strict_metrics') is not None:raise ValueError('L0 main score must remain N/A after missing original response')
    known_stage=sum((Decimal(c['charge_cny']) for c in calls),Decimal(0))
    return dict(status='PASS_WITH_APPROVED_UNKNOWN_BOUND' if continuation else 'PASS',independent_token_charge_recomputation=True,independent_frozen_payload_reconstruction=True,request_manifest_sha256=request_freeze['request_manifest_sha256'],privacy_review_sha256=request_freeze['privacy_review_sha256'],external_send_authorization_verified=True,requests=len(rows),historical_requests_preserved=len(old['requests']),cumulative_cny=str(spent),cumulative_cny_is_known_only=True,known_cumulative_cny=str(spent),unknown_cost_upper_cny=str(unknown_bound),exposure_upper_cny=str(upper),remaining_cny=str(100-upper),remaining_basis='known settled conservative charges plus approved unresolved fullbound',api_enabled=False,stage4_calls=len(calls),stage4_attempts=1000,stage4_settled_calls=len(calls),stage4_unknown_cost_calls=len(continuation['unknown_entries']) if continuation else 0,stage4_cny=str(known_stage),stage4_exposure_upper_cny=str(known_stage+unknown_bound),unknown_is_not_actual_charge_or_settlement=bool(continuation),approved_unknown_proof_sha256=continuation['proof_sha256'] if continuation else None,continuation_annex_sha256=continuation['annex_sha256'] if continuation else None,api_summary_path=summary_path,continuation_version=continuation.get('version') if continuation else None,unknown_cost_rule=continuation['proof'].get('future_unknown_rule','only_exact_existing_single_unknown_approved') if continuation else None,unknown_cost_approval_path=a.get('api_budget_amendment_path'),continuation_annex_path=a.get('api_continuation_annex_path'),routes=routes)



def run(root,execution,aggregation):
    start=time.perf_counter();recorded=json.loads((root/aggregation).read_text());fresh=calculate(root,execution);plan,a,protocol,entries,paths=load_manifest(root,execution)
    for field in ('status','execution_manifest_sha256','frozen_development_protocol_sha256','primary_seed','bootstrap','results','multi_seed','input_files_sha256'):
        if not equivalent(recorded[field],fresh[field]):raise ValueError('Aggregation differs from fresh independent raw-output replay:'+field)
    checks=check_training(root,entries,protocol,a)
    checkpoint_path=a.get('checkpoint_inspection','evidence/stage4/checkpoint-inspection.json');checkpoint=json.loads((root/checkpoint_path).read_text())
    if checkpoint.get('status')!='PASS' or checkpoint.get('device')!='cpu' or len(checkpoint.get('models',{}))!=6 or any(m.get('status')!='PASS' for m in checkpoint['models'].values()):raise ValueError('Six-model independent CPU checkpoint inspection required')
    gold_ids=[r['id'] for r in read_jsonl(root/paths['common500'])]
    latency=check_latencies(root,entries,gold_ids);budget=check_budget_api(root,a,gold_ids)
    if budget.get('stage4_unknown_cost_calls'):
        if fresh['results']['L0/common500']['main_metrics'] is not None:raise ValueError('Preserved L0 unknowns force full500 main metrics N/A')
        for route in ('L0','L3'):
            if fresh['results'][route+'/common500']['statuses'].get('transport_error',0)!=budget['routes'][route]['transport_errors']:raise ValueError('Canonical transport-failure counts differ from actual unknown attempts')
            if budget['routes'][route]['transport_errors'] and fresh['results'][route+'/common500']['main_metrics'] is not None:raise ValueError('Any unknown response makes complete500 main metrics unavailable')
        if any(pair['status']!='N/A' for name,pair in fresh['bootstrap']['differences'].items() if 'L0' in name.split()):raise ValueError('L0 paired intervals must remain N/A')
    evidence={}
    for path in a.get('required_evidence',[]):
        target=root/path
        expected=a.get('pre_evaluation_files',{}).get(path)
        if path.startswith('scripts/') and expected:target=historical_source(root,path,expected)
        if not target.is_file():raise ValueError('Missing required stage4 evidence:'+path)
        evidence[path]=digest(target)
    if not a.get('exit_resource_report'):raise ValueError('Final resource release evidence path required')
    exit_path=root/a['exit_resource_report'];exit_report=json.loads(exit_path.read_text())
    if exit_report.get('own_training_or_inference_pids')!=[] or exit_report.get('gpu_workload_stopped') is not True:raise ValueError('Own GPU workloads not verified exited')
    if fresh['status']=='INCOMPLETE':raise ValueError('Stage4 has unsent API items;cannot mark complete')
    return dict(status='PASS_WITH_WARNINGS' if fresh['status']=='COMPLETED_WITH_INVALID_API_OUTPUTS' else 'PASS',at=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,
        aggregation=str(aggregation),aggregation_sha256=digest(root/aggregation),recomputed_all16_prediction_entries=True,training=checks,latency=latency,budget=budget,
        same500_all5=True,full5000_local_all3seeds=True,api_primary_na_when_any_invalid=True,shared_bootstrap_indices=True,multi_seed_sample_sd_ddof=1,
        checkpoint_inspection=checkpoint_path,checkpoint_inspection_sha256=digest(root/checkpoint_path),primary_seed_fixed_before_final=True,gold_and_frozen_parameters_unchanged=True,own_gpu_workloads_exited=True,exit_resource_report=a['exit_resource_report'],required_evidence_sha256=evidence,
        limitations=fresh['limitations'],stage5_not_started=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--execution',type=Path,default=Path('configs/stage4-execution-continuation02.json'));p.add_argument('--aggregation',type=Path,default=Path('evidence/stage4/final-aggregation.json'));p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:result=run(args.root,args.execution,args.aggregation)
    except Exception as exc:
        result=dict(status='FAIL',at=datetime.now(timezone.utc).isoformat(),error=str(exc),traceback=traceback.format_exc());dump_new(args.root/args.output,result);print(json.dumps(result,ensure_ascii=False),flush=True);raise
    dump_new(args.root/args.output,result);print(json.dumps(dict(status=result['status'],seconds=result['seconds'],known_cumulative_cny=result['budget']['known_cumulative_cny'],exposure_upper_cny=result['budget']['exposure_upper_cny']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
