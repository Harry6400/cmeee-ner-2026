"""SYNTHETIC bounded-transport continuation02 tests;no actual API/ledger writes."""
import copy
import json
import ssl
import unittest
from decimal import Decimal
from unittest.mock import patch

import test_stage4_api_continuation as prior
import api_final as api
import api_final_bounded_continue02 as two


class Continuation02Tests(unittest.TestCase):
    def setUp(self):
        self.old=prior.ContinuationTests('test_prepare_is_read_only_for_original_ledger_science_and_partial_outputs');self.old.setUp()
        self.root,self.directory,self.items=self.old.root,self.old.directory,self.old.items
        self.old.ledger.activate();data=self.old.ledger.read()
        for index,item in enumerate(self.items[234:488]):
            ident='synthetic-continuation01-'+str(index);raw=self.old.fixture.fake_raw(item,ident)
            raw['response']['usage']={'prompt_tokens':224922 if index==0 else 0,'completion_tokens':0}
            api.create_json(self.directory/'requests'/('raw-'+ident+'.json'),raw)
            entry=dict(id=ident,request_key=item['key'],origin=api.ORIGIN,status='settled',run_directory=str(self.directory.relative_to(self.root)),
                conservative_charge_cny='0.449844' if index==0 else '0',usage=raw['response']['usage'],actual_model='deepseek-flash',reserved_cny='2.12992',
                input_token_bound=1048576,payload_sha256=item['payload_sha256'],continuation_annex_sha256=self.old.ledger.adjustment['annex_sha256'])
            data['requests'].append(entry);api.append_json(self.directory/'calls.jsonl',api.call_record(self.root,self.directory,item,raw,entry))
        item=self.items[488];ident=two.INITIAL_UNKNOWNS[1]
        data['requests'].append(dict(id=ident,request_key=item['key'],origin=api.ORIGIN,status='unknown_cost',run_directory=str(self.directory.relative_to(self.root)),
            reserved_cny='2.12992',input_token_bound=1048576,max_output_tokens=4096,payload_sha256=item['payload_sha256'],reason='SYNTHETIC TLS protocol error'))
        data.update(status='stopped_unknown_cost',api_enabled=False,spent_cny='4.763144');self.old.ledger.save(data)
        api.create_json(self.directory/'requests'/('failure-'+ident+'.json'),dict(id=ident,key=item['key'],seconds=5.9,no_retry=True))
        api.append_json(self.directory/'segments-continuation01.jsonl',dict(status='SYNTHETIC TLS failure',seconds=375))
        prior.continuation.summarize(self.root,self.directory,self.old.annex,'SYNTHETIC stopped')
        entries={r['request_key']:r for r in data['requests'] if r.get('origin')==api.ORIGIN};records={r['key']:r for r in api.read_rows(self.directory/'calls.jsonl')}
        prior.continuation.attempt_log(self.directory,entries,records)
        self.before=api.read_json(self.root/'private/budget/ledger.json')
        snapshot=self.root/'private/budget/stage4-continuation02-opening-ledger.json';api.create_bytes(snapshot,(self.root/'private/budget/ledger.json').read_bytes(),private=True)
        unknown=[r for r in self.before['requests'] if r['status']=='unknown_cost']
        proof=dict(explicit_approved=True,human_reply='批准上述有上限规则，继续剩余511次（推荐）',recorded_at=api.now(),approved_question='SYNTHETIC fixture,no human action',
            approved_existing_unknowns=[dict(id=r['id'],request_key=r['request_key'],reserved_cny=r['reserved_cny'],payload_sha256=r['payload_sha256'],entry_sha256=api.sha_bytes(api.canonical(r))) for r in unknown],
            continuation_opening_ledger=str(snapshot.relative_to(self.root)),continuation_opening_ledger_sha256=api.sha(snapshot),max_remaining_calls=511,max_total_stage4_attempts=1000,
            no_resend_any_attempted_key=True,future_unknown_rule=two.FUTURE_RULE,approved_transport_exception_classes=list(two.TRANSPORT_CLASSES),per_call_hard_reservation_cny='2.129920',
            cumulative_cap_cny='100',stage4_new_soft_cap_cny='80',original_stage4_opening_cny='3.872064',receiver='https://api.deepseek.com/chat/completions',original_API_run='runs/stage4-20261020/api-final',
            no_changes_to_frozen_payload_prompt_examples=True,unknown_is_not_claimed_actual_settled=True,
            canonical_outputs={r:str((self.directory/(r+'-predictions-continuation02.jsonl')).relative_to(self.root)) for r in ('L0','L3')},summary_output=str((self.directory/'summary-continuation02.json').relative_to(self.root)))
        prior.write(self.root/'approval02.json',proof);self.proof=proof
        self.annex=two.prepare(self.root,self.directory,self.root/'approval02.json')
        self.ledger=two.BoundLedger(self.root,self.directory,self.before,self.old.fixture.price,self.annex)

    def tearDown(self):self.old.tearDown()

    def test_prepare_preserves1329_and_prior_partial_outputs(self):
        self.assertEqual(api.read_json(self.ledger.path),self.before);two.validate_annex(self.root,self.directory)
        self.assertEqual(self.annex['remaining_keys'],[x['key'] for x in self.items[489:]])
        self.assertEqual(len(self.annex['remaining_keys']),511)

    def test_classifier_excludes_http_integrity_and_persistence(self):
        self.assertEqual(two.transport_class(ssl.SSLError('SYNTHETIC TLS')),'ssl.SSLError')
        self.assertEqual(two.transport_class(api.urllib.error.URLError('SYNTHETIC EOF')),'urllib.error.URLError')
        self.assertEqual(two.transport_class(TimeoutError()),'TimeoutError')
        self.assertEqual(two.transport_class(ConnectionError()),'ConnectionError')
        self.assertIsNone(two.transport_class(api.urllib.error.HTTPError('https://x',500,'error',{},None)))
        self.assertIsNone(two.transport_class(ValueError('bad usage')))
        self.assertIsNone(two.transport_class(OSError('disk failure')))

    def test_policy_append_only_and_old_unknowns_not_settled(self):
        self.ledger.activate();data=self.ledger.read()
        self.assertEqual(data['requests'][:1329],self.before['requests'])
        self.assertEqual(data['adjustments'][:1],self.before['adjustments'])
        self.assertEqual(len(data['adjustments']),2)
        self.assertEqual(data['exposure_upper_cny'],'9.022984')
        for item in (self.items[233],self.items[488],self.items[0]):
            with self.assertRaises(ValueError):self.ledger.reserve(item)

    def test_future_transport_requires_network_failure_evidence(self):
        self.ledger.activate();item=self.items[489];ident=self.ledger.reserve(item)
        api.create_json(self.directory/'requests'/('failure-'+ident+'.json'),dict(id=ident,key=item['key'],seconds=.1,
            approved_transport_exception_class='ssl.SSLError',failure_phase='network_request_or_response_read',http_error=False,transport_policy_approved=True,payload_sha256=item['payload_sha256']))
        self.ledger.transport_unknown(ident,'SYNTHETIC TLS','ssl.SSLError');data=self.ledger.read()
        self.assertEqual(sum(r['status']=='unknown_cost' for r in data['requests']),3)
        self.assertEqual(data['exposure_upper_cny'],'11.152904')
        self.assertEqual(data['requests'][:1329],self.before['requests'])
        with self.assertRaises(ValueError):self.ledger.reserve(item)
        next_id=self.ledger.reserve(self.items[490]);self.ledger.unknown(next_id,'SYNTHETIC bad model')
        with self.assertRaises(ValueError):self.ledger.read()

    def test_mock_repeated_transport_fails_stop_at_soft_upper_cap_without_retry(self):
        old_summary=api.sha(self.directory/'summary-continuation01.json')
        with patch.object(api,'secret',return_value='SYNTHETIC-SECRET'):
            with patch.object(two.urllib.request,'build_opener') as opener:
                opener.return_value.open.side_effect=ssl.SSLError('SYNTHETIC TLS')
                with patch('builtins.print'):result=two.run(self.root,self.directory)
                self.assertEqual(opener.return_value.open.call_count,35)
        data=api.read_json(self.ledger.path)
        self.assertEqual(data['requests'][:1329],self.before['requests'])
        self.assertEqual(result['total_attempted'],524)
        self.assertEqual(result['unknown_cost_calls'],37)
        self.assertTrue(result['status'].startswith('stopped_at_budget'))
        self.assertLessEqual(Decimal(data['exposure_upper_cny'])-Decimal('3.872064'),80)
        self.assertFalse(data['api_enabled'])
        self.assertEqual(api.sha(self.directory/'summary-continuation01.json'),old_summary)
        attempts=api.read_rows(self.directory/'attempts-log-continuation02.jsonl')
        self.assertEqual([r['key'] for r in attempts],[r['key'] for r in self.items[:524]])

    def test_mock_one_transport_then510_responses_completes_dynamic1000_attempts(self):
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self):return api.canonical(dict(model='deepseek-flash',usage=dict(prompt_tokens=0,completion_tokens=0),choices=[dict(finish_reason='stop',message=dict(content='{"entities":[]}'))]))
        count=[0]
        def opened(*args,**kwargs):
            count[0]+=1
            if count[0]==1:raise ssl.SSLError('SYNTHETIC first TLS')
            return Response()
        with patch.object(api,'secret',return_value='SYNTHETIC-SECRET'):
            with patch.object(two.urllib.request,'build_opener') as opener:
                opener.return_value.open.side_effect=opened
                with patch('builtins.print'):result=two.run(self.root,self.directory)
        self.assertEqual(count[0],511)
        self.assertEqual(result['total_attempted'],1000)
        self.assertEqual(result['settled_calls'],997)
        self.assertEqual(result['unknown_cost_calls'],3)
        self.assertTrue(result['status'].startswith('completed1000'))
        self.assertIsNone(result['routes']['L0']['main_strict_metrics'])
        self.assertEqual(result['routes']['L3']['main_strict_metrics']['samples'],500)
        data=api.read_json(self.ledger.path);self.assertEqual(data['requests'][:1329],self.before['requests'])
        attempts=api.read_rows(self.directory/'attempts-log-continuation02.jsonl')
        self.assertEqual([x['key'] for x in attempts],[x['key'] for x in self.items])
        self.assertFalse(data['api_enabled'])

    def test_mock_http_error_stops_instead_of_using_transport_waiver(self):
        with patch.object(api,'secret',return_value='SYNTHETIC-SECRET'):
            with patch.object(two.urllib.request,'build_opener') as opener:
                opener.return_value.open.side_effect=api.urllib.error.HTTPError('https://api.deepseek.com',500,'SYNTHETIC',{},None)
                with self.assertRaises(RuntimeError):two.run(self.root,self.directory)
                self.assertEqual(opener.return_value.open.call_count,1)
        data=api.read_json(self.ledger.path)
        self.assertEqual(data['status'],'stopped_unknown_cost')
        self.assertEqual(len(data['requests']),1330)
        self.assertNotIn('approved_transport_exception_class',data['requests'][-1])
        self.assertFalse(data['api_enabled'])


if __name__=='__main__':unittest.main()
