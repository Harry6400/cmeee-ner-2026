"""Independent operational accounting for the explicitly approved transport-budget rule.
No network,ledger writes,canonical repairs,training or changes to scientific metrics.
"""
from __future__ import annotations
import hashlib,json,math
from decimal import Decimal
EXISTING_UNKNOWN_IDS=('ner-final-85a1172afb6d4ff6b1be8cbcce79333b','ner-final-6494ce5c72594163a17a14497aa14603')
TRANSPORT_RULE='reserve_full_bound_skip_transport_failures_only;stop_on_non_transport_or_integrity_issue'
ALLOWED_FAMILIES={'urllib.error.URLError','ssl.SSLError','TimeoutError','ConnectionError'}


def canonical(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()
def number(value):
    n=Decimal(str(value))
    if not n.is_finite() or n<0:raise ValueError('Money must be finite and nonnegative')
    return n


def full_bound():return (Decimal(1048576)*2+Decimal(4096)*8)/Decimal(1000000)


def check_transport_failure(failure,entry,item):
    if failure.get('id')!=entry['id'] or failure.get('key')!=item['key'] or failure.get('route')!=item['route'] or failure.get('sample_id')!=item['sample_id']:raise ValueError('Future transport failure identity differs from original manifest')
    # The frozen operational source must set this only inside network request/read try-block.
    if failure.get('failure_phase')!='network_request_or_response_read' or failure.get('transport_policy_approved') is not True or failure.get('http_error') is not False or failure.get('exception_class') not in ALLOWED_FAMILIES or failure.get('payload_sha256')!=entry['payload_sha256'] or failure.get('approved_transport_exception_class')!=failure.get('exception_class'):raise ValueError('Future unknown is not an explicitly approved network-only interruption')
    if failure.get('no_retry') is not True or not failure.get('reason') or type(failure.get('seconds')) not in (int,float) or not math.isfinite(failure['seconds']) or failure['seconds']<0:raise ValueError('Future transport failure lacks honest reason,timing or no-retry evidence')
    if any(k in failure for k in ('usage','conservative_charge_cny','response','raw_body_utf8')):raise ValueError('Transport-only unknown must not discard a captured billable response or invent usage')


def audit_transport_rule(data,before,proof,items,failures,require_complete=True):
    if number(data.get('limit_cny'))!=100:raise ValueError('Cumulative ledger cap changed')
    if proof.get('explicit_approved') is not True or proof.get('human_reply')!='批准上述有上限规则，继续剩余511次（推荐）':raise PermissionError('Exact actual human transport-budget approval required')
    if proof.get('receiver')!='https://api.deepseek.com/chat/completions' or proof.get('original_API_run')!='runs/stage4-20261020/api-final' or proof.get('future_unknown_rule')!=TRANSPORT_RULE or set(proof.get('approved_transport_exception_classes',[]))!=ALLOWED_FAMILIES:raise ValueError('Approved transport rule/receiver changed')
    expected=dict(opening_rows=1329,already_attempted_stage4=489,max_remaining_calls=511,max_total_stage4_attempts=1000,no_resend_any_attempted_key=True,no_changes_to_frozen_payload_prompt_examples=True,unknown_is_not_claimed_actual_settled=True)
    if any(proof.get(k)!=v for k,v in expected.items()) or proof.get('cumulative_cap_cny')!='100' or proof.get('stage4_new_soft_cap_cny')!='80' or number(proof.get('original_stage4_opening_cny'))!=Decimal('3.872064'):raise ValueError('Approved511-tail scope/science/budget changed')
    bound=full_bound()
    if number(proof['per_call_hard_reservation_cny'])!=bound:raise ValueError('Per-call bound differs from independently calculated fullcontext2/8')
    if len(items)!=1000 or len({i['key'] for i in items})!=1000 or len(before['requests'])!=1329:raise ValueError('Wrong frozen request or1329 opening count')
    rows=data['requests']
    if len(rows)<1329 or len(rows)>1840 or canonical(rows[:1329])!=canonical(before['requests']):raise ValueError('Original1329 prefix changed or attempted more than1000')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate request ID')
    stage=rows[840:]
    if [r.get('request_key') for r in stage]!=[i['key'] for i in items[:len(stage)]] or any(r.get('origin')!='cmeee_ner_stage4' or r.get('run_directory')!=proof['original_API_run'] for r in stage):raise ValueError('Only frozen next unattempted keys may be appended;no retries or unrelated rows')
    approved=proof['approved_existing_unknowns'];old_unknown=[r for r in before['requests'] if r.get('status')!='settled']
    if [r.get('id') for r in approved]!=list(EXISTING_UNKNOWN_IDS) or [r['id'] for r in old_unknown]!=list(EXISTING_UNKNOWN_IDS):raise ValueError('Existing two unknown IDs or count changed')
    for entry,authorization in zip(old_unknown,approved):
        if entry.get('status')!='unknown_cost' or authorization.get('status')!='unknown_cost' or hashlib.sha256(canonical(entry)).hexdigest()!=authorization.get('entry_sha256') or any(entry.get(k)!=authorization.get(k) for k in ('id','request_key','payload_sha256')) or number(entry.get('reserved_cny'))!=bound or number(authorization.get('reserved_cny'))!=bound:raise ValueError('Original approved unknown entry/key/payload/bound altered')
    known=Decimal(0);unknown=[];pending=[]
    for index,row in enumerate(rows):
        status=row.get('status')
        if status=='settled':known+=number(row['conservative_charge_cny']);continue
        if status not in ('unknown_cost','pending') or index<840:raise ValueError('Unapproved non-stage4 unresolved ledger entry')
        if number(row.get('reserved_cny'))!=bound or row.get('input_token_bound')!=1048576 or row.get('max_output_tokens')!=4096:raise ValueError('Unknown/pending record lacks exact fullcontext reservation')
        if any(k in row for k in ('usage','conservative_charge_cny','settled_at','actual_model')):raise ValueError('Unresolved charge cannot be fabricated as known usage/settlement')
        item=items[index-840]
        if row.get('payload_sha256')!=item['payload_sha256']:raise ValueError('Unresolved payload differs from original manifest')
        if status=='pending':pending.append(row)
        else:
            unknown.append(row)
            if row['id'] not in EXISTING_UNKNOWN_IDS:
                failure=failures.get(row['id'])
                if failure is None:raise ValueError('Future unknown lacks original network failure evidence')
                check_transport_failure(failure,row,item)
    if len(pending)>1 or (pending and pending[0]['id']!=rows[-1]['id']):raise ValueError('At most one final in-flight reservation is permitted')
    unknown_upper=bound*len(unknown);pending_upper=bound*len(pending);upper=known+unknown_upper;including_pending=upper+pending_upper
    if known!=number(data['spent_cny']) or including_pending>100 or including_pending-Decimal('3.872064')>80:raise ValueError('Known+all unknown+pending exceeds approved total/new cap')
    opening_known=sum((number(r['conservative_charge_cny']) for r in before['requests'] if r['status']=='settled'),Decimal(0))
    if opening_known!=number(proof['known_spent_before_cny']) or bound*2!=number(proof['approved_existing_unknown_bound_cny']) or opening_known+bound*2!=number(proof['exposure_upper_before_cny']):raise ValueError('Approval opening known/upper totals do not match actual1329 snapshot')
    totals=dict(known_spent_cny=known,unknown_cost_upper_cny=unknown_upper,exposure_upper_cny=upper,pending_reservation_cny=pending_upper,exposure_including_pending_upper_cny=including_pending,upper_remaining_cny=100-including_pending)
    for key,value in totals.items():
        if number(data[key])!=value:raise ValueError('Dynamic unknown/pending exposure total differs:'+key)
    if require_complete and (len(stage)!=1000 or pending or data.get('api_enabled') is not False or data.get('unknown_charge_is_not_settled') is not True or data.get('status')!='active_with_reserved_unknown_cost'):raise ValueError('Final gate requires1000 attempted,no pending and API disabled')
    return dict(known=known,unknown_bound_each=bound,unknown_upper=unknown_upper,pending_upper=pending_upper,exposure_upper=upper,including_pending_upper=including_pending,total_attempts=len(stage),settled_calls=sum(r['status']=='settled' for r in stage),unknown_count=len(unknown),unknown=unknown,pending_count=len(pending),unknown_rows_never_settled=True,final_complete=len(stage)==1000 and not pending)


def check_transport_budget_annex(root,a,data,request_freeze,calls,canonical_predictions):
    from stage4_metrics import digest,read_jsonl
    def load(path):return json.loads(path.read_text())
    directory=root/a['api_run_dir'];proof_path=root/a['api_budget_amendment_path'];proof=load(proof_path);annex_path=root/a['api_continuation_annex_path'];annex=load(annex_path);base=annex_path.parent;items=request_freeze['ordered_items']
    before_path=root/proof['continuation_opening_ledger'];before=load(before_path)
    if digest(before_path)!=proof['continuation_opening_ledger_sha256'] or annex.get('ledger_before_snapshot')!=proof['continuation_opening_ledger'] or annex.get('ledger_before_sha256')!=digest(before_path):raise ValueError('Approved1329 opening snapshot differs from annex')
    if (base/'approval-at-call.json').read_bytes()!=proof_path.read_bytes() or digest(proof_path)!=annex.get('approval_sha256'):raise ValueError('Rule02 actual human approval copy changed')
    failures={}
    for entry in data['requests']:
        if entry.get('status')=='unknown_cost':
            path=directory/'requests'/('failure-'+entry['id']+'.json');failure=load(path);failures[entry['id']]=failure
            item=next(i for i in items if i['key']==entry['request_key'])
            if failure.get('id')!=entry['id'] or failure.get('route')!=item['route'] or failure.get('sample_id')!=item['sample_id'] or type(failure.get('seconds')) not in (int,float) or not math.isfinite(failure['seconds']) or failure['seconds']<0 or failure.get('key')!=entry['request_key'] or (directory/'requests'/('raw-'+entry['id']+'.json')).exists() or any(c['id']==entry['id'] for c in calls):raise ValueError('An unknown request was retried or gained fabricated response/usage')
    account=audit_transport_rule(data,before,proof,items,failures,require_complete=True)
    old_unknown=[r for r in before['requests'] if r['status']=='unknown_cost']
    if annex.get('name')!='continuation02' or annex.get('future_unknown_rule')!=TRANSPORT_RULE or annex.get('approved_existing_unknown_ids')!=list(EXISTING_UNKNOWN_IDS) or canonical(annex.get('original_unknown_entries'))!=canonical(old_unknown) or annex.get('preserved_request_prefix_count')!=1329 or annex.get('preserved_request_prefix_sha256')!=hashlib.sha256(canonical(before['requests'])).hexdigest():raise ValueError('Rule02 annex changes original unknowns or1329 prefix')
    if annex.get('remaining_keys')!=[i['key'] for i in items[489:]] or annex.get('remaining_count')!=511 or annex.get('max_total_attempts')!=1000 or annex.get('no_automatic_retry') is not True or annex.get('unknown_is_not_settled') is not True or annex.get('network_and_TLS_configuration_unchanged') is not True:raise ValueError('Rule02 remaining-tail/no-retry/TLS contract differs')
    for field,path in [('original_protocol_sha256',directory/'protocol-at-call.json'),('original_request_manifest_sha256',directory/'requests-manifest.jsonl')]:
        if annex.get(field)!=digest(path):raise ValueError('Original1000 scientific requests changed')
    if number(annex['per_call_hard_reservation_cny'])!=full_bound() or annex.get('cumulative_limit_cny')!='100' or annex.get('new_stage_soft_cap_cny')!='80':raise ValueError('Annex weakens hard reservation or budget caps')
    source='api_final_bounded_continue02.py'
    from historical_evidence import checked_file
    checked_file(root,(base/'source-at-continuation'/source).relative_to(root).as_posix(),annex['source_sha256'])
    for rel,sha in annex['prior_snapshots_sha256'].items():
        if digest(base/'prior-state'/rel)!=sha or (rel!='calls.jsonl' and digest(directory/rel)!=sha):raise ValueError('Earlier partial result/annex/failure was overwritten:'+rel)
    old_calls=(directory/'calls.jsonl').read_bytes()[:annex['old_calls_bytes']]
    if annex.get('old_calls_count')!=487 or hashlib.sha256(old_calls).hexdigest()!=annex['old_calls_prefix_sha256'] or old_calls!=(base/'prior-state/calls.jsonl').read_bytes():raise ValueError('Original487 settled call bytes changed')
    initial_hashes=annex.get('existing_unknown_failures_sha256',{})
    if set(initial_hashes)!=set(EXISTING_UNKNOWN_IDS):raise ValueError('Rule02 requires frozen raw failure evidence for both prior unknowns')
    for ident,sha in initial_hashes.items():
        if digest(directory/'requests'/('failure-'+ident+'.json'))!=sha:raise ValueError('Prior unknown raw failure/timing changed')
    for route in ('L0','L3'):
        if a['api_prediction_paths'][route]!=proof['canonical_outputs'][route]:raise ValueError('Rule02 canonical destination changed')
    if a['api_summary_path']!=proof['summary_output']:raise ValueError('Rule02 summary destination changed')
    by_key={i['key']:i for i in items}
    for entry in account['unknown']:
        item=by_key[entry['request_key']];pred=canonical_predictions[item['route']][item['sample_id']]
        if pred.get('status')!='transport_error' or pred.get('request_id')!=entry['id'] or pred.get('entities_unavailable') is not True or 'entities' in pred:raise ValueError('Unknown canonical result must stay unavailable,not an empty entity set')
    policy=dict(id='api-continuation02-policy',kind='approved_existing_and_future_transport_unknown_reservation_policy',request_ids=list(EXISTING_UNKNOWN_IDS),bounds_by_request={r['id']:r['reserved_cny'] for r in old_unknown},not_actual_settlement=True,future_unknown_rule=TRANSPORT_RULE,approval_sha256=annex['approval_sha256'],annex_sha256=digest(annex_path),ledger_before_sha256=annex['ledger_before_sha256'],effective_from=proof['recorded_at'])
    prefix=before.get('adjustments',[]);adjustments=data.get('adjustments',[])
    if canonical(adjustments[:len(prefix)])!=canonical(prefix) or not adjustments[len(prefix):] or adjustments[len(prefix)]!=policy or load(root/'private/budget/api-continuation02-policy.json')!=policy:raise ValueError('Old adjustment prefix or rule02 policy event changed')
    events=adjustments[len(prefix)+1:];new_unknown=[r for r in account['unknown'] if r['id'] not in EXISTING_UNKNOWN_IDS]
    if [e.get('request_id') for e in events]!=[r['id'] for r in new_unknown] or len({e.get('id') for e in events})!=len(events):raise ValueError('Future transport bound events missing,repeated or orphaned')
    for event,entry in zip(events,new_unknown):
        ident=entry['id'];failure=failures[ident];failure_path=directory/'requests'/('failure-'+ident+'.json');kind=failure['exception_class']
        fixed=dict(id='api-continuation02-bound-'+ident,kind='approved_future_transport_unknown_upper_bound',request_id=ident,amount_upper_cny=entry['reserved_cny'],not_actual_settlement=True,approval_sha256=annex['approval_sha256'],annex_sha256=digest(annex_path),failure_sha256=digest(failure_path),approved_transport_exception_class=kind)
        if any(event.get(k)!=v for k,v in fixed.items()) or not event.get('at') or load(root/'private/budget'/('api-continuation02-bound-'+ident+'.json'))!=event or entry.get('approved_transport_exception_class')!=kind or entry.get('continuation02_approval_sha256')!=annex['approval_sha256']:raise ValueError('Future transport bound event/classification/provenance differs')
    attempts=read_jsonl(directory/'attempts-log-continuation02.jsonl');stage=data['requests'][840:];call_by_key={c['key']:c for c in calls}
    if len(attempts)!=1000 or [r.get('key') for r in attempts]!=[i['key'] for i in items] or [r.get('id') for r in attempts]!=[r['id'] for r in stage]:raise ValueError('Final attempt log must cover original1000 exactly once,including every unknown')
    for attempt,entry in zip(attempts,stage):
        if entry['status']=='unknown_cost':
            if attempt.get('status')!='transport_error' or attempt.get('cost_status')!='unknown_cost' or attempt.get('usage_available') is not False or number(attempt['charge_upper_cny'])!=full_bound() or attempt.get('seconds')!=failures[entry['id']]['seconds'] or any(k in attempt for k in ('charge_cny','usage')):raise ValueError('Unknown attempt improperly became a known charge/empty response')
        else:
            call=call_by_key.get(entry['request_key'])
            if call is None or attempt['id']!=call['id'] or attempt.get('status')!=call['status'] or attempt.get('cost_status')!='settled' or attempt.get('usage_available') is not True or attempt.get('seconds')!=call['seconds'] or number(attempt['charge_cny'])!=number(call['charge_cny']):raise ValueError('Settled attempt differs from immutable response/cost')
    if load(base/'run-started.json').get('annex_sha256')!=digest(annex_path):raise ValueError('Rule02 actual run did not freeze this annex')
    # Compatibility with final audit's common known/upper interface;never pretend Nunknown=1.
    account['unknown_bound']=account['unknown_upper']
    return dict(account=account,proof=proof,annex=annex,failures=failures,attempts=attempts,proof_sha256=digest(proof_path),annex_sha256=digest(annex_path),version='continuation02')
