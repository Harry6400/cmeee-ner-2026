"""Synthetic final-request integrity and tariff counterexamples;no network or actual final samples."""
import copy,hashlib,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_stage4 import request_bytes,expected_api_payload,validate_manifest_items,validate_api_call_frozen


def row(ident,text='甲😀甲'):
    return dict(id=ident,text=text,entities=[])


def frozen():
    return dict(api=dict(system='synthetic frozen prompt',temperature=0,thinking='disabled',response_format='json_object',max_output_tokens=4096,model='synthetic-frozen-alias'))

class ApiIntegrityTests(unittest.TestCase):
    def call_fixture(self):
        price=dict(input_peak_uncached_per_million='2',output_peak_per_million='8',source='synthetic tariff fixture only',model='synthetic-frozen-alias')
        payload=expected_api_payload(row('synthetic-sample'),'L0',[],frozen());sha=hashlib.sha256(request_bytes(payload)).hexdigest()
        item=dict(key='synthetic-run:L0:synthetic-sample',route='L0',sample_id='synthetic-sample',payload=payload,payload_sha256=sha)
        raw=dict(key=item['key'],payload=copy.deepcopy(payload),payload_sha256=sha)
        call=dict(key=item['key'],route='L0',sample_id='synthetic-sample',payload_sha256=sha,usage=dict(prompt_tokens=100,completion_tokens=10,total_tokens=110),charge_cny='.00028',actual_model=price['model'])
        ledger=dict(request_key=item['key'],origin='cmeee_ner_stage4',actual_model=price['model'],payload_sha256=sha,pricing=price,conservative_charge_cny='.00028')
        return call,raw,ledger,item,price
    def test_frozen_charge_exact_decimal_and_coherent_tamper_rejected(self):
        call,raw,ledger,item,price=self.call_fixture();self.assertEqual(str(validate_api_call_frozen(call,raw,ledger,item,price)),'0.00028')
        call['charge_cny']='.00001';ledger['conservative_charge_cny']='.00001'
        with self.assertRaises(ValueError):validate_api_call_frozen(call,raw,ledger,item,price)
    def test_raw_payload_tamper_with_updated_local_digests_rejected(self):
        call,raw,ledger,item,price=self.call_fixture();raw['payload']['messages'][0]['content']='changed synthetic prompt'
        sha=hashlib.sha256(request_bytes(raw['payload'])).hexdigest();raw['payload_sha256']=sha;call['payload_sha256']=sha;ledger['payload_sha256']=sha
        with self.assertRaises(ValueError):validate_api_call_frozen(call,raw,ledger,item,price)
    def test_independent_payload_only_input_and_three_training_examples(self):
        examples=[row('synthetic-example-'+str(i),'甲甲甲') for i in range(3)];examples[0]['entities']=[dict(start=1,end=3,type='bod',text='甲甲')]
        l0=expected_api_payload(row('synthetic-target'),'L0',examples,frozen());l3=expected_api_payload(row('synthetic-target'),'L3',examples,frozen())
        self.assertEqual(len(l0['messages']),2);self.assertEqual(len(l3['messages']),8);self.assertIn('"occurrence": 1',l3['messages'][2]['content']);self.assertEqual(l0['messages'][-1],l3['messages'][-1]);self.assertNotIn('entities',l0['messages'][-1]['content'])
    def manifest_fixture(self):
        samples=[row('synthetic-sample-'+str(i)) for i in range(500)];examples=[row('synthetic-example-'+str(i)) for i in range(3)];items=[]
        for route in ('L0','L3'):
            for r in samples:
                payload=expected_api_payload(r,route,examples,frozen());items.append(dict(key='synthetic-run:'+route+':'+r['id'],route=route,sample_id=r['id'],payload=payload,payload_sha256=hashlib.sha256(request_bytes(payload)).hexdigest()))
        return items,samples,examples
    def test_manifest_coherent_payload_tamper_and_order_rejected(self):
        items,samples,examples=self.manifest_fixture();self.assertEqual(len(validate_manifest_items(items,samples,examples,frozen(),'synthetic-run')),1000)
        items[0]['payload']['messages'][0]['content']='different prompt';items[0]['payload_sha256']=hashlib.sha256(request_bytes(items[0]['payload'])).hexdigest()
        with self.assertRaises(ValueError):validate_manifest_items(items,samples,examples,frozen(),'synthetic-run')
        items,samples,examples=self.manifest_fixture();items[0]['payload']['temperature']=False;items[0]['payload_sha256']=hashlib.sha256(request_bytes(items[0]['payload'])).hexdigest()
        with self.assertRaises(ValueError):validate_manifest_items(items,samples,examples,frozen(),'synthetic-run')
        items,samples,examples=self.manifest_fixture();items[0],items[1]=items[1],items[0]
        with self.assertRaises(ValueError):validate_manifest_items(items,samples,examples,frozen(),'synthetic-run')

if __name__=='__main__':unittest.main()
