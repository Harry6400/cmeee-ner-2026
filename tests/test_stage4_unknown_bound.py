"""Synthetic one-exact-unknown scope counterexamples;no actual ledger/network/data."""
import copy,hashlib,sys,unittest
from pathlib import Path
from decimal import Decimal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_stage4 import approved_unknown_accounting,request_bytes,APPROVED_UNKNOWN_ID,APPROVED_UNKNOWN_SAMPLE
from aggregate_stage4 import evaluate_predictions_with_transport
from stage4_metrics import evaluate_route,paired_bootstrap


def fixture():
    items=[]
    for i in range(1000):
        sample=APPROVED_UNKNOWN_SAMPLE if i==233 else 'synthetic-'+str(i%500);route='L0' if i<500 else 'L3'
        items.append(dict(route=route,sample_id=sample,key='synthetic-run:'+route+':'+sample,payload_sha256='synthetic-payload-'+str(i)))
    history=[dict(id='synthetic-historic-'+str(i),status='settled',conservative_charge_cny='3.872064' if i==0 else '0') for i in range(840)]
    stage=[]
    for i,item in enumerate(items):
        row=dict(id=APPROVED_UNKNOWN_ID if i==233 else 'synthetic-paid-'+str(i),status='unknown_cost' if i==233 else 'settled',request_key=item['key'],origin='cmeee_ner_stage4',run_directory='synthetic-api',payload_sha256=item['payload_sha256'])
        if i==233:row.update(reserved_cny='2.12992',reason='synthetic EOF;no response or usage')
        else:row['conservative_charge_cny']='.441236' if i==0 else '0'
        stage.append(row)
    unknown=stage[233]
    opening=dict(requests=history+copy.deepcopy(stage[:234]),spent_cny='4.313300')
    proof=dict(explicit_approved=True,human_reply='synthetic scoped approval',approved_unknown=dict(id=APPROVED_UNKNOWN_ID,status='unknown_cost',request_key=unknown['request_key'],payload_sha256=unknown['payload_sha256'],reserved_cny='2.12992',entry_sha256=hashlib.sha256(request_bytes(unknown)).hexdigest()),opening_rows=1074,max_remaining_calls=766,max_total_stage4_attempts=1000,no_resend_any_attempted_key=True,new_unknown_rule='stop;only above exact existing unknown approved',no_changes_to_frozen_payload_prompt_examples=True,unknown_is_not_claimed_actual_settled=True,cumulative_cap_cny='100',stage4_new_soft_cap_cny='80',original_API_run='synthetic-api',approved_unknown_bound_cny='2.129920',known_spent_before_cny='4.313300',exposure_upper_before_cny='6.443220',original_stage4_opening_cny='3.872064')
    ledger=dict(requests=history+stage,spent_cny='4.313300',known_spent_cny='4.313300',unknown_cost_upper_cny='2.12992',exposure_upper_cny='6.443220',pending_reservation_cny='0',exposure_including_pending_upper_cny='6.443220',upper_remaining_cny='93.556780',api_enabled=False,unknown_charge_is_not_settled=True,status='active_with_reserved_unknown_cost')
    return ledger,opening,proof,items

class UnknownBoundTests(unittest.TestCase):
    def test_exact_one_unknown_preserved_and_counted_once(self):
        ledger,opening,proof,items=fixture();before=copy.deepcopy(ledger);v=approved_unknown_accounting(ledger,opening,proof,items)
        self.assertEqual(v['known'],Decimal('4.313300'));self.assertEqual(v['unknown_bound'],Decimal('2.12992'));self.assertEqual(v['exposure_upper'],Decimal('6.443220'));self.assertEqual(v['settled_calls'],999);self.assertEqual(v['total_attempts'],1000);self.assertEqual(ledger,before)
    def test_tampered_approval_id_bound_or_prefix_rejected(self):
        for mutate in (lambda l,o,p,i:p['approved_unknown'].update(id='different'),lambda l,o,p,i:p.update(approved_unknown_bound_cny='1'),lambda l,o,p,i:l['requests'][0].update(conservative_charge_cny='9'),lambda l,o,p,i:p.update(explicit_approved=False)):
            ledger,opening,proof,items=fixture();mutate(ledger,opening,proof,items)
            with self.assertRaises(ValueError):approved_unknown_accounting(ledger,opening,proof,items)
    def test_second_unknown_unrelated_tail_or_fake_settlement_rejected(self):
        for mutate in (lambda l:l['requests'][-1].update(status='unknown_cost'),lambda l:l['requests'][-1].update(origin='unrelated-project'),lambda l:l['requests'][1073].update(status='settled',conservative_charge_cny='0')):
            ledger,opening,proof,items=fixture();mutate(ledger)
            with self.assertRaises(ValueError):approved_unknown_accounting(ledger,opening,proof,items)
    def test_transport_adapter_preserves_metric_math_and_na_pairs(self):
        gold=[dict(id='synthetic',text='甲',entities=[dict(start=0,end=1,type='bod',text='甲')])];pred=[dict(id='synthetic',text='甲',status='transport_error',entities_unavailable=True)]
        v=evaluate_predictions_with_transport(gold,pred,set(),True);self.assertIsNone(v['main_metrics']);self.assertEqual(v['statuses'],{'transport_error':1});self.assertNotIn('entities',pred[0]);self.assertEqual(v['valid'],0)
        valid=evaluate_predictions_with_transport(gold,gold,set(),True);self.assertEqual(valid,evaluate_route(gold,gold,set(),True))
        intervals=paired_bootstrap({'D0':valid,'L0':v,'L3':valid},10,20261019);self.assertEqual(intervals['differences']['L0 minus D0']['status'],'N/A');self.assertEqual(intervals['differences']['L3 minus D0']['status'],'available')
        with self.assertRaises(ValueError):evaluate_predictions_with_transport(gold,[dict(pred[0],entities=[])],set(),True)
    def test_scientific_metrics_file_unchanged(self):
        root=Path(__file__).resolve().parents[1]
        original=root/'source-before-continuation01/stage4_metrics.py'
        if original.exists():self.assertEqual(original.read_bytes(),(root/'scripts/stage4_metrics.py').read_bytes())

class UnknownAnnexIOTests(unittest.TestCase):
    def test_full_annex_chain_and_duplicate_bound_adjustment_rejected(self):
        import tempfile,json
        from verify_stage4 import check_approved_unknown_annex
        from stage4_metrics import digest
        with tempfile.TemporaryDirectory(prefix='synthetic-approved-unknown-annex-') as temp:
            root=Path(temp);ledger,opening,proof,items=fixture();directory=root/'synthetic-api';base=directory/'continuation01'
            def put(path,value):
                path=root/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False));return path
            def rows(path,values):
                path=root/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b''.join(request_bytes(v)+b'\n' for v in values));return path
            before=put('private/budget/synthetic-opening.json',opening)
            proof.update(continuation_opening_ledger='private/budget/synthetic-opening.json',continuation_opening_ledger_sha256=digest(before),recorded_at='2026-10-20T03:05:00+00:00',canonical_outputs={r:'synthetic-api/'+r+'-predictions-continuation01.jsonl' for r in ('L0','L3')},summary_output='synthetic-api/summary-continuation01.json')
            proof_path=put('evidence/synthetic-proof.json',proof);put('synthetic-api/continuation01/approval-at-call.json',proof)
            for name in ('protocol-at-call.json','frozen-protocol-at-call.json'):put('synthetic-api/'+name,dict(synthetic=True))
            manifest=rows('synthetic-api/requests-manifest.jsonl',items)
            call_rows=[]
            for index,item in enumerate(items):
                if index==233:continue
                entry=ledger['requests'][840+index];call_rows.append(dict(id=entry['id'],key=item['key'],status='valid',seconds=.01,charge_cny=entry['conservative_charge_cny']))
            old_calls=rows('synthetic-api/continuation01/prior-state/calls.jsonl',call_rows[:233]);rows('synthetic-api/calls.jsonl',call_rows)
            for name in ('summary.json','L0-predictions.jsonl','L3-predictions.jsonl','segments.jsonl'):
                path=put('synthetic-api/'+name,dict(synthetic=True,original_partial=True));saved=base/'prior-state'/name;saved.write_bytes(path.read_bytes())
            failure=dict(id=APPROVED_UNKNOWN_ID,key=items[233]['key'],route='L0',sample_id=APPROVED_UNKNOWN_SAMPLE,seconds=.2)
            failure_path=put('synthetic-api/requests/failure-'+APPROVED_UNKNOWN_ID+'.json',failure)
            wrapper=put('scripts/api_final_bounded_continue.py',dict(synthetic_source_fixture=True));saved=base/'source-at-continuation/api_final_bounded_continue.py';saved.parent.mkdir(parents=True);saved.write_bytes(wrapper.read_bytes())
            annex=dict(name='continuation01',approval_sha256=digest(proof_path),ledger_before_snapshot=proof['continuation_opening_ledger'],ledger_before_sha256=digest(before),approved_unknown_request_ids=[APPROVED_UNKNOWN_ID],original_unknown_entry=ledger['requests'][1073],preserved_request_prefix_count=1074,preserved_request_prefix_sha256=hashlib.sha256(request_bytes(opening['requests'])).hexdigest(),remaining_keys=[i['key'] for i in items[234:]],remaining_count=766,max_total_attempts=1000,no_automatic_retry=True,another_unknown_must_stop=True,unknown_is_not_settled=True,original_protocol_sha256=digest(directory/'protocol-at-call.json'),original_frozen_protocol_sha256=digest(directory/'frozen-protocol-at-call.json'),original_request_manifest_sha256=digest(manifest),unknown_bound_cny='2.12992',cumulative_limit_cny='100',new_stage_soft_cap_cny='80',source_sha256=digest(wrapper),prior_snapshots_sha256={p.name:digest(p) for p in (base/'prior-state').iterdir()},old_calls_count=233,old_calls_bytes=old_calls.stat().st_size,old_calls_prefix_sha256=digest(old_calls),failure_file=str(failure_path.relative_to(directory)),failure_sha256=digest(failure_path))
            annex_path=put('synthetic-api/continuation01/annex-at-call.json',annex)
            adjustment=dict(id='api-continuation01-reserve-'+APPROVED_UNKNOWN_ID,kind='approved_unknown_cost_upper_bound_reservation',request_id=APPROVED_UNKNOWN_ID,amount_upper_cny='2.12992',not_actual_settlement=True,original_unknown_status_preserved=True,approval_sha256=annex['approval_sha256'],annex_sha256=digest(annex_path),ledger_before_sha256=annex['ledger_before_sha256'],effective_from=proof['recorded_at'])
            ledger['adjustments']=[adjustment];put('private/budget/api-continuation01-adjustment.json',adjustment);put('synthetic-api/continuation01/run-started.json',dict(annex_sha256=digest(annex_path)))
            attempts=[];by_key={r['key']:r for r in call_rows}
            for index,item in enumerate(items):
                if index==233:attempts.append(dict(id=APPROVED_UNKNOWN_ID,key=item['key'],status='transport_error',cost_status='unknown_cost',usage_available=False,charge_upper_cny='2.12992',seconds=.2))
                else:
                    call=by_key[item['key']];attempts.append(dict(id=call['id'],key=item['key'],status='valid',cost_status='settled',usage_available=True,charge_cny=call['charge_cny'],seconds=.01))
            rows('synthetic-api/attempts-log-continuation01.jsonl',attempts)
            audit=dict(api_run_dir='synthetic-api',api_budget_amendment_path='evidence/synthetic-proof.json',api_continuation_annex_path=str(annex_path.relative_to(root)),api_prediction_paths=proof['canonical_outputs'],api_summary_path=proof['summary_output'])
            canonical={'L0':{APPROVED_UNKNOWN_SAMPLE:dict(status='transport_error',request_id=APPROVED_UNKNOWN_ID,entities_unavailable=True)},'L3':{}}
            result=check_approved_unknown_annex(root,audit,ledger,dict(ordered_items=items),call_rows,canonical);self.assertEqual(len(result['attempts']),1000)
            ledger['adjustments'].append(copy.deepcopy(adjustment))
            with self.assertRaises(ValueError):check_approved_unknown_annex(root,audit,ledger,dict(ordered_items=items),call_rows,canonical)


if __name__=='__main__':unittest.main()
