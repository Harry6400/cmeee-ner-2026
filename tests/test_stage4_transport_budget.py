"""Synthetic bounded-transport accounting and failure-policy counterexamples;no network or real data."""
import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from decimal import Decimal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage4_transport_budget import (audit_transport_rule,check_transport_budget_annex,canonical,full_bound,EXISTING_UNKNOWN_IDS,ALLOWED_FAMILIES,TRANSPORT_RULE)
from stage4_metrics import digest


def refresh(ledger):
    known=sum((Decimal(r['conservative_charge_cny']) for r in ledger['requests'] if r['status']=='settled'),Decimal(0))
    unknown=sum(r['status']=='unknown_cost' for r in ledger['requests'])*full_bound();pending=sum(r['status']=='pending' for r in ledger['requests'])*full_bound()
    ledger.update(spent_cny=str(known),known_spent_cny=str(known),unknown_cost_upper_cny=str(unknown),exposure_upper_cny=str(known+unknown),pending_reservation_cny=str(pending),exposure_including_pending_upper_cny=str(known+unknown+pending),upper_remaining_cny=str(100-known-unknown-pending))


def fixture(extra=()):
    run='runs/stage4-20261020/api-final';items=[]
    for i in range(1000):
        route='L0' if i<500 else 'L3';sample='synthetic-'+str(i%500)
        items.append(dict(key='synthetic-run:'+route+':'+sample,route=route,sample_id=sample,payload_sha256='synthetic-'+str(i)))
    history=[dict(id='synthetic-history-'+str(i),status='settled',conservative_charge_cny='3.872064' if i==0 else '0') for i in range(840)]
    stage=[];failures={}
    for i,item in enumerate(items):
        original_index={233:0,488:1};unknown=i in original_index or i in extra
        ident=EXISTING_UNKNOWN_IDS[original_index[i]] if i in original_index else 'synthetic-request-'+str(i)
        row=dict(id=ident,status='unknown_cost' if unknown else 'settled',request_key=item['key'],origin='cmeee_ner_stage4',run_directory=run,payload_sha256=item['payload_sha256'])
        if unknown:
            row.update(reserved_cny='2.12992',input_token_bound=1048576,max_output_tokens=4096,reason='synthetic transport interruption')
            failure=dict(id=ident,key=item['key'],route=item['route'],sample_id=item['sample_id'],seconds=.25,reason='synthetic network EOF',no_retry=True)
            if i in extra:failure.update(failure_phase='network_request_or_response_read',transport_policy_approved=True,http_error=False,exception_class='urllib.error.URLError',exception_family='urllib.error.URLError',approved_transport_exception_class='urllib.error.URLError',payload_sha256=item['payload_sha256'])
            failures[ident]=failure
        else:row['conservative_charge_cny']='.891080' if i==0 else '0'
        stage.append(row)
    before=dict(requests=history+copy.deepcopy(stage[:489]),adjustments=[dict(id='synthetic-preserved-old-policy')])
    proof=dict(explicit_approved=True,human_reply='批准上述有上限规则，继续剩余511次（推荐）',receiver='https://api.deepseek.com/chat/completions',original_API_run=run,future_unknown_rule=TRANSPORT_RULE,approved_transport_exception_classes=sorted(ALLOWED_FAMILIES),opening_rows=1329,already_attempted_stage4=489,max_remaining_calls=511,max_total_stage4_attempts=1000,no_resend_any_attempted_key=True,no_changes_to_frozen_payload_prompt_examples=True,unknown_is_not_claimed_actual_settled=True,cumulative_cap_cny='100',stage4_new_soft_cap_cny='80',original_stage4_opening_cny='3.872064',per_call_hard_reservation_cny='2.129920',known_spent_before_cny='4.763144',approved_existing_unknown_bound_cny='4.259840',exposure_upper_before_cny='9.022984')
    proof['approved_existing_unknowns']=[dict(id=r['id'],status='unknown_cost',request_key=r['request_key'],payload_sha256=r['payload_sha256'],reserved_cny=r['reserved_cny'],entry_sha256=hashlib.sha256(canonical(r)).hexdigest()) for r in before['requests'] if r['status']=='unknown_cost']
    ledger=dict(requests=history+stage,limit_cny='100',api_enabled=False,unknown_charge_is_not_settled=True,status='active_with_reserved_unknown_cost');refresh(ledger)
    return ledger,before,proof,items,failures


class TransportBudgetTests(unittest.TestCase):
    def test_dynamic_two_and_future_three_or_four_unknowns_account_once(self):
        for extra in ((),(700,),(700,800)):
            data,before,proof,items,failures=fixture(extra);saved=copy.deepcopy(data)
            result=audit_transport_rule(data,before,proof,items,failures)
            self.assertEqual(result['unknown_count'],2+len(extra));self.assertEqual(result['settled_calls'],998-len(extra));self.assertEqual(result['unknown_upper'],Decimal('2.129920')*(2+len(extra)));self.assertEqual(result['total_attempts'],1000);self.assertEqual(data,saved)
    def test_approval_scope_original_ids_prefix_and_bound_tampering_rejected(self):
        changes=[lambda d,b,p,i,f:p.update(explicit_approved=False),lambda d,b,p,i,f:p['approved_existing_unknowns'][0].update(id='changed'),lambda d,b,p,i,f:p.update(per_call_hard_reservation_cny='1'),lambda d,b,p,i,f:d['requests'][0].update(conservative_charge_cny='99'),lambda d,b,p,i,f:p.update(max_remaining_calls=512),lambda d,b,p,i,f:d['requests'][-1].update(request_key=i[-2]['key'])]
        for change in changes:
            args=fixture();change(*args)
            with self.assertRaises((ValueError,PermissionError)):audit_transport_rule(*args)
    def test_future_without_failure_or_nontransport_or_http_or_nan_rejected(self):
        changes=[lambda f:f.clear(),lambda f:f.update(failure_phase='response_parse_settlement_or_persistence'),lambda f:f.update(exception_class='urllib.error.HTTPError'),lambda f:f.update(http_error=True),lambda f:f.update(exception_class='ValueError'),lambda f:f.update(seconds=float('nan')),lambda f:f.update(usage={'prompt_tokens':1}),lambda f:f.update(payload_sha256='wrong'),lambda f:f.update(no_retry=False)]
        for change in changes:
            data,before,proof,items,failures=fixture((700,));change(failures['synthetic-request-700'])
            with self.assertRaises(ValueError):audit_transport_rule(data,before,proof,items,failures)
    def test_pending_is_counted_under_caps_but_blocks_final_gate(self):
        data,before,proof,items,failures=fixture();data['requests'][-1].pop('conservative_charge_cny');data['requests'][-1].update(status='pending',reserved_cny='2.12992',input_token_bound=1048576,max_output_tokens=4096);data['api_enabled']=True;refresh(data)
        result=audit_transport_rule(data,before,proof,items,failures,require_complete=False);self.assertEqual(result['pending_count'],1);self.assertEqual(result['including_pending_upper'],Decimal('11.152904'))
        with self.assertRaises(ValueError):audit_transport_rule(data,before,proof,items,failures)
        data['requests'][-2]['conservative_charge_cny']='79';refresh(data)
        with self.assertRaises(ValueError):audit_transport_rule(data,before,proof,items,failures,require_complete=False)
    def test_unavailable_adapter_does_not_turn_unknown_into_empty_gold_or_change_math(self):
        from aggregate_stage4 import evaluate_predictions_with_transport
        from stage4_metrics import evaluate_route,paired_bootstrap
        gold=[dict(id='synthetic',text='x',entities=[])];pred=[dict(id='synthetic',text='x',status='transport_error',entities_unavailable=True)]
        unavailable=evaluate_predictions_with_transport(gold,pred,set(),True);valid=evaluate_predictions_with_transport(gold,gold,set(),True)
        self.assertEqual(valid,evaluate_route(gold,gold,set(),True));self.assertIsNone(unavailable['main_metrics'])
        bootstrap=paired_bootstrap({'D0':valid,'L0':unavailable,'L3':unavailable},10,20261019)
        self.assertTrue(all(v['status']=='N/A' for k,v in bootstrap['differences'].items() if 'L' in k))


class TransportAnnexTests(unittest.TestCase):
    def test_full_1000_chain_dynamic_future_and_event_tamper(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-stage4-transport-annex-') as temp:
            root=Path(temp);data,before,proof,items,failures=fixture((700,));directory=root/proof['original_API_run'];base=directory/'continuation02'
            def put(path,value):
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canonical(value));return p
            def rows(path,values):
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b''.join(canonical(v)+b'\n' for v in values));return p
            before_path=put('private/budget/synthetic-opening.json',before)
            proof.update(recorded_at='2026-10-20T03:28:09+00:00',continuation_opening_ledger=str(before_path.relative_to(root)),continuation_opening_ledger_sha256=digest(before_path),canonical_outputs={r:str((directory/(r+'-predictions-continuation02.jsonl')).relative_to(root)) for r in ('L0','L3')},summary_output=str((directory/'summary-continuation02.json').relative_to(root)))
            proof_path=put('evidence/synthetic-approval.json',proof);put(str((base/'approval-at-call.json').relative_to(root)),proof)
            protocol=put(str((directory/'protocol-at-call.json').relative_to(root)),dict(synthetic=True));manifest=rows(str((directory/'requests-manifest.jsonl').relative_to(root)),items)
            calls=[];attempts=[];canonical_predictions={'L0':{},'L3':{}}
            for i,item in enumerate(items):
                entry=data['requests'][840+i]
                if entry['status']=='unknown_cost':
                    ident=entry['id'];failure=failures[ident];put(str((directory/'requests'/('failure-'+ident+'.json')).relative_to(root)),failure)
                    attempts.append(dict(id=ident,key=item['key'],status='transport_error',cost_status='unknown_cost',usage_available=False,charge_upper_cny='2.12992',seconds=.25))
                    canonical_predictions[item['route']][item['sample_id']]=dict(status='transport_error',request_id=ident,entities_unavailable=True)
                else:
                    call=dict(id=entry['id'],key=item['key'],status='valid',seconds=.01,charge_cny=entry['conservative_charge_cny']);calls.append(call);attempts.append(dict(call,cost_status='settled',usage_available=True))
            rows(str((directory/'calls.jsonl').relative_to(root)),calls);old_calls=rows(str((base/'prior-state/calls.jsonl').relative_to(root)),calls[:487]);rows(str((directory/'attempts-log-continuation02.jsonl').relative_to(root)),attempts)
            prior_hashes={}
            for name in ('summary.json','summary-continuation01.json','requests/failure-'+EXISTING_UNKNOWN_IDS[0]+'.json','requests/failure-'+EXISTING_UNKNOWN_IDS[1]+'.json'):
                p=directory/name
                if not p.exists():put(str(p.relative_to(root)),dict(synthetic=True,old=True))
                saved=base/'prior-state'/name;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(p.read_bytes());prior_hashes[name]=digest(p)
            source=put('scripts/api_final_bounded_continue02.py',dict(synthetic_source=True));put(str((base/'source-at-continuation/api_final_bounded_continue02.py').relative_to(root)),dict(synthetic_source=True))
            old_unknown=[r for r in before['requests'] if r['status']=='unknown_cost']
            annex=dict(name='continuation02',future_unknown_rule=TRANSPORT_RULE,approval_sha256=digest(proof_path),ledger_before_snapshot=proof['continuation_opening_ledger'],ledger_before_sha256=digest(before_path),approved_existing_unknown_ids=list(EXISTING_UNKNOWN_IDS),original_unknown_entries=old_unknown,preserved_request_prefix_count=1329,preserved_request_prefix_sha256=hashlib.sha256(canonical(before['requests'])).hexdigest(),remaining_keys=[i['key'] for i in items[489:]],remaining_count=511,max_total_attempts=1000,no_automatic_retry=True,unknown_is_not_settled=True,network_and_TLS_configuration_unchanged=True,original_protocol_sha256=digest(protocol),original_request_manifest_sha256=digest(manifest),per_call_hard_reservation_cny='2.12992',cumulative_limit_cny='100',new_stage_soft_cap_cny='80',source_sha256=digest(source),prior_snapshots_sha256=prior_hashes,old_calls_count=487,old_calls_bytes=old_calls.stat().st_size,old_calls_prefix_sha256=digest(old_calls),existing_unknown_failures_sha256={ident:digest(directory/'requests'/('failure-'+ident+'.json')) for ident in EXISTING_UNKNOWN_IDS})
            annex_path=put(str((base/'annex-at-call.json').relative_to(root)),annex)
            policy=dict(id='api-continuation02-policy',kind='approved_existing_and_future_transport_unknown_reservation_policy',request_ids=list(EXISTING_UNKNOWN_IDS),bounds_by_request={r['id']:r['reserved_cny'] for r in old_unknown},not_actual_settlement=True,future_unknown_rule=TRANSPORT_RULE,approval_sha256=digest(proof_path),annex_sha256=digest(annex_path),ledger_before_sha256=digest(before_path),effective_from=proof['recorded_at'])
            put('private/budget/api-continuation02-policy.json',policy)
            ident='synthetic-request-700';entry=data['requests'][1540];entry.update(approved_transport_exception_class='urllib.error.URLError',continuation02_approval_sha256=digest(proof_path))
            event=dict(id='api-continuation02-bound-'+ident,kind='approved_future_transport_unknown_upper_bound',request_id=ident,amount_upper_cny='2.12992',not_actual_settlement=True,approval_sha256=digest(proof_path),annex_sha256=digest(annex_path),failure_sha256=digest(directory/'requests'/('failure-'+ident+'.json')),approved_transport_exception_class='urllib.error.URLError',at='2026-10-20T03:36:00+00:00')
            put('private/budget/api-continuation02-bound-'+ident+'.json',event);data['adjustments']=before['adjustments']+[policy,event];put(str((base/'run-started.json').relative_to(root)),dict(annex_sha256=digest(annex_path)))
            audit=dict(api_run_dir=proof['original_API_run'],api_budget_amendment_path=str(proof_path.relative_to(root)),api_continuation_annex_path=str(annex_path.relative_to(root)),api_prediction_paths=proof['canonical_outputs'],api_summary_path=proof['summary_output'])
            result=check_transport_budget_annex(root,audit,data,dict(ordered_items=items),calls,canonical_predictions);self.assertEqual(result['account']['unknown_count'],3);self.assertEqual(len(result['attempts']),1000)
            data['adjustments'][-1]['amount_upper_cny']='0'
            with self.assertRaises(ValueError):check_transport_budget_annex(root,audit,data,dict(ordered_items=items),calls,canonical_predictions)


if __name__=='__main__':unittest.main()
