"""SYNTHETIC continuation fixtures;no real API or production ledger writes."""
import copy
import json
import sys
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import test_stage4_api as fixtures
write,rows=fixtures.write,fixtures.rows
import api_final as api
import api_final_bounded_continue as continuation


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.FinalAPITests('test_preflight_exact1000_payloads_and_context_reservations')
        self.fixture.setUp();self.root=self.fixture.root;self.directory=self.fixture.run
        self.fixture.samples[233]['id']=continuation.EXPECTED_FAILED_SAMPLE
        rows(self.root/self.fixture.frozen['data']['common_subset'],self.fixture.samples)
        self.protocol=self.fixture.prepared();self.fixture.privacy_pass(self.protocol)
        self.items=api.read_rows(self.directory/'requests-manifest.jsonl')
        data=copy.deepcopy(self.fixture.opening)
        for index,item in enumerate(self.items[:233]):
            ident='synthetic-final-'+str(index)
            raw=self.fixture.fake_raw(item,ident,valid=index not in (5,10,20))
            raw['response']['usage']={'prompt_tokens':220618 if index==0 else 0,'completion_tokens':0}
            api.create_json(self.directory/'requests'/('raw-'+ident+'.json'),raw)
            entry=dict(id=ident,request_key=item['key'],origin=api.ORIGIN,status='settled',
                run_directory=str(self.directory.relative_to(self.root)),conservative_charge_cny='0.441236' if index==0 else '0',
                usage=raw['response']['usage'],actual_model='deepseek-flash',reserved_cny=item['reserved_cny'],
                payload_sha256=item['payload_sha256'],input_token_bound=item['input_token_bound'])
            data['requests'].append(entry)
            api.append_json(self.directory/'calls.jsonl',api.call_record(self.root,self.directory,item,raw,entry))
        item=self.items[233]
        unknown=dict(id=continuation.ALLOWED_UNKNOWN,request_key=item['key'],purpose=item['key'],origin=api.ORIGIN,
            run_directory=str(self.directory.relative_to(self.root)),status='unknown_cost',reserved_cny='2.12992',
            input_token_bound=1048576,max_output_tokens=4096,payload_sha256=item['payload_sha256'],reason='SYNTHETIC SSL EOF')
        data['requests'].append(unknown);data.update(status='stopped_unknown_cost',spent_cny='4.313300',api_enabled=False)
        write(self.root/'private/budget/ledger.json',data)
        self.before=copy.deepcopy(data)
        api.create_json(self.directory/'requests'/('failure-'+continuation.ALLOWED_UNKNOWN+'.json'),
            dict(id=continuation.ALLOWED_UNKNOWN,key=item['key'],seconds=5.00908525,no_retry=True))
        api.append_json(self.directory/'segments.jsonl',dict(status='SYNTHETIC stopped',seconds=477.0))
        api.summarize(self.root,self.directory,self.protocol,'SYNTHETIC interrupted')
        write(self.root/'evidence/stage4/training-queue.json',dict(status='completed',synthetic=True))
        write(self.directory/'training-completion-at-call.json',dict(training_queue_sha256=api.sha(self.root/'evidence/stage4/training-queue.json')))
        self.opening_path=self.root/'private/budget/stage4-continuation01-opening-ledger.json'
        api.create_bytes(self.opening_path,(self.root/'private/budget/ledger.json').read_bytes(),private=True)
        self.approval=dict(explicit_approved=True,human_reply='批准按最坏费用预留，继续未发送的766次（推荐）',
            approved_question='SYNTHETIC authorization fixture,not a real human action',recorded_at=api.now(),
            receiver='https://api.deepseek.com/chat/completions',original_API_run='runs/stage4-20261020/api-final',
            approved_unknown=dict(id=unknown['id'],status='unknown_cost',reserved_cny='2.12992',request_key=unknown['request_key'],
                payload_sha256=unknown['payload_sha256'],entry_sha256=api.sha_bytes(api.canonical(unknown))),
            continuation_opening_ledger=str(self.opening_path.relative_to(self.root)),continuation_opening_ledger_sha256=api.sha(self.opening_path),
            max_remaining_calls=766,max_total_stage4_attempts=1000,no_resend_any_attempted_key=True,
            new_unknown_rule='stop;only above exact existing unknown approved',cumulative_cap_cny='100',stage4_new_soft_cap_cny='80',
            original_stage4_opening_cny='3.872064',no_changes_to_frozen_payload_prompt_examples=True,unknown_is_not_claimed_actual_settled=True,
            canonical_outputs={r:str((self.directory/(r+'-predictions-continuation01.jsonl')).relative_to(self.root)) for r in ('L0','L3')},
            summary_output=str((self.directory/'summary-continuation01.json').relative_to(self.root)))
        write(self.root/'approval.json',self.approval)
        self.annex=continuation.prepare(self.root,self.directory,self.root/'approval.json')
        self.ledger=continuation.BoundLedger(self.root,self.directory,self.before,self.fixture.price,self.annex)

    def tearDown(self):self.fixture.tearDown()

    def settle_fixture(self,item):
        ident=self.ledger.reserve(item)
        raw=self.fixture.fake_raw(item,ident)
        api.create_json(self.directory/'requests'/('raw-'+ident+'.json'),raw)
        self.ledger.settle(ident,raw['response']['usage'],'deepseek-flash')
        return ident

    def test_prepare_is_read_only_for_original_ledger_science_and_partial_outputs(self):
        self.assertEqual(api.read_json(self.ledger.path),self.before)
        self.assertEqual(self.annex['remaining_count'],766)
        self.assertEqual(self.annex['exposure_upper_cny'],'6.443220')
        self.assertEqual(api.sha(self.directory/'requests-manifest.jsonl'),self.protocol['request_manifest_sha256'])
        continuation.validate_annex(self.root,self.directory)

    def test_adjustment_preserves_unknown_and1074prefix_and_never_retries(self):
        self.ledger.activate();self.ledger.activate()
        data=self.ledger.read()
        self.assertEqual(data['requests'][:1074],self.before['requests'])
        self.assertEqual(data['requests'][-1]['status'],'unknown_cost')
        self.assertNotIn('usage',data['requests'][-1])
        self.assertNotIn('conservative_charge_cny',data['requests'][-1])
        self.assertEqual(len(data['adjustments']),1)
        self.assertEqual(data['exposure_upper_cny'],'6.443220')
        with self.assertRaises(ValueError):self.ledger.reserve(self.items[233])
        with self.assertRaises(ValueError):self.ledger.reserve(self.items[0])

    def test_upper_bound_soft_cap_includes_unknown_reserve(self):
        self.ledger.activate();data=self.ledger.read();item=self.items[234]
        data['requests'].append(dict(id='synthetic-high-known-charge',request_key=item['key'],origin=api.ORIGIN,
            run_directory=str(self.directory.relative_to(self.root)),status='settled',conservative_charge_cny='76'))
        data['spent_cny']='80.313300';write(self.ledger.path,data)
        self.assertEqual(continuation.exposure(data)['upper'],Decimal('82.443220'))
        with self.assertRaisesRegex(ValueError,'budget'):self.ledger.reserve(self.items[235])
        self.assertEqual(len(api.read_json(self.ledger.path)['requests']),1075)

    def test_second_unknown_stops_and_cannot_inherit_prior_approval(self):
        self.ledger.activate();ident=self.ledger.reserve(self.items[234]);self.ledger.unknown(ident,'SYNTHETIC second EOF')
        self.assertEqual(api.read_json(self.ledger.path)['status'],'stopped_unknown_cost')
        with self.assertRaises(ValueError):self.ledger.read()
        with self.assertRaises(ValueError):self.ledger.activate()

    def test_settled_before_log_crash_reconciles_without_resend_and_keeps_old_outputs(self):
        self.ledger.activate();before={p:api.sha(self.directory/p) for p in ('summary.json','L0-predictions.jsonl','L3-predictions.jsonl','segments.jsonl')}
        self.settle_fixture(self.items[234])
        with patch.object(continuation.urllib.request,'build_opener',side_effect=AssertionError('network forbidden')):
            records,entries=continuation.reconcile(self.directory,self.ledger,self.items)
            self.assertEqual(len(records),234)
            self.assertEqual(len(entries),235)
            self.assertEqual(len(continuation.reconcile(self.directory,self.ledger,self.items)[0]),234)
        continuation.attempt_log(self.directory,entries,records)
        summary=continuation.summarize(self.root,self.directory,self.annex,'SYNTHETIC partial continuation')
        self.assertEqual(summary['settled_calls'],234)
        self.assertEqual(summary['unknown_cost_calls'],1)
        pred=api.read_rows(self.directory/'L0-predictions-continuation01.jsonl')[233]
        self.assertEqual(pred['status'],'transport_error');self.assertNotIn('entities',pred)
        self.assertIsNone(summary['routes']['L0']['main_strict_metrics'])
        self.assertIsNone(summary['routes']['L3']['main_strict_metrics'])
        self.assertEqual(before,{p:api.sha(self.directory/p) for p in before})
        self.assertEqual(len(api.read_rows(self.directory/'attempts-log-continuation01.jsonl')),235)

    def test_mock_runtime_new_transport_failure_is_one_attempt_and_blocks_resume(self):
        old_hash=api.sha(self.directory/'summary.json')
        fake_key='SYNTHETIC_CONTINUATION_SECRET'
        with patch.object(api,'secret',return_value=fake_key):
            with patch.object(continuation.urllib.request,'build_opener') as opener:
                opener.return_value.open.side_effect=api.urllib.error.URLError('SYNTHETIC EOF '+fake_key)
                with self.assertRaises(RuntimeError):continuation.run(self.root,self.directory)
                self.assertEqual(opener.return_value.open.call_count,1)
        data=api.read_json(self.ledger.path)
        self.assertEqual(data['requests'][:1074],self.before['requests'])
        self.assertEqual(len(data['requests']),1075)
        self.assertEqual(sum(r['status']=='unknown_cost' for r in data['requests']),2)
        self.assertEqual(api.sha(self.directory/'summary.json'),old_hash)
        self.assertFalse(data['api_enabled'])
        for path in self.root.rglob('*.json'):self.assertNotIn(fake_key,path.read_text())
        with patch.object(api,'secret',side_effect=AssertionError('must stop before credentials')):
            with self.assertRaises(ValueError):continuation.run(self.root,self.directory,resume=True)

    def test_full1000_attempt_fixture_is999_settled_plus_one_unknown(self):
        self.ledger.activate();data=self.ledger.read()
        for index,item in enumerate(self.items[234:]):
            ident='synthetic-completion-'+str(index)
            raw=self.fixture.fake_raw(item,ident)
            raw['response']['usage']={'prompt_tokens':0,'completion_tokens':0}
            api.create_json(self.directory/'requests'/('raw-'+ident+'.json'),raw)
            entry=dict(id=ident,request_key=item['key'],origin=api.ORIGIN,status='settled',
                run_directory=str(self.directory.relative_to(self.root)),conservative_charge_cny='0',
                usage=raw['response']['usage'],actual_model='deepseek-flash',reserved_cny=item['reserved_cny'],
                payload_sha256=item['payload_sha256'],input_token_bound=item['input_token_bound'],
                continuation_annex_sha256=self.ledger.adjustment['annex_sha256'])
            data['requests'].append(entry)
            api.append_json(self.directory/'calls.jsonl',api.call_record(self.root,self.directory,item,raw,entry))
        self.ledger.save(data);self.ledger.read()
        records,entries=continuation.reconcile(self.directory,self.ledger,self.items)
        continuation.attempt_log(self.directory,entries,records)
        log=api.read_rows(self.directory/'attempts-log-continuation01.jsonl')
        self.assertEqual(len(log),1000)
        self.assertEqual([r['key'] for r in log],[r['key'] for r in self.items])
        summary=continuation.summarize(self.root,self.directory,self.annex,'SYNTHETIC complete attempts')
        self.assertEqual(summary['total_attempted'],1000)
        self.assertEqual(summary['settled_calls'],999)
        self.assertEqual(summary['unknown_cost_calls'],1)
        self.assertEqual(summary['exposure_upper_cny'],'6.443220')
        self.assertIsNone(summary['routes']['L0']['main_strict_metrics'])
        self.assertEqual(summary['routes']['L3']['main_strict_metrics']['samples'],500)
        self.assertFalse(summary['full_five_route_comparison_available'])
        with self.assertRaises(ValueError):self.ledger.reserve(self.items[-1])

    def test_scope_amendment_does_not_allow_a_future_unknown_or_retry(self):
        bad=copy.deepcopy(self.approval);bad['new_unknown_rule']='automatically allow all future failures'
        with self.assertRaises(ValueError):continuation.approval_valid(bad)
        bad=copy.deepcopy(self.approval);bad['max_remaining_calls']=767
        with self.assertRaises(ValueError):continuation.approval_valid(bad)


if __name__=='__main__':unittest.main()
