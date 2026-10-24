"""只测试本地预算，不发送网络请求。"""
import json,sys,tempfile,unittest
from pathlib import Path
from decimal import Decimal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from api_development import Ledger

class BudgetTests(unittest.TestCase):
 def setup_ledger(self,d):
  p=Path(d);r=dict(id='historical',status='settled',conservative_charge_cny='3.792488')
  original=dict(requests=[r]);(p/'original-ledger-20261018.json').write_text(json.dumps(original))
  (p/'ledger.json').write_text(json.dumps(dict(requests=[r],spent_cny='3.792488',limit_cny='100',status='active',api_enabled=True)))
  ledger=Ledger();ledger.path=p/'ledger.json';ledger.opening=Decimal('3.792488');return ledger,r
 def test_preserve_history_reserve_settle(self):
  with tempfile.TemporaryDirectory() as d:
   ledger,r=self.setup_ledger(d);ident=ledger.reserve('fixture');ledger.settle(ident,dict(prompt_tokens=500,completion_tokens=100),'deepseek-flash')
   result=ledger.read();self.assertEqual(result['spent_cny'],'3.794288');self.assertEqual(result['requests'][0],r)
 def test_subcap_is_shared_across_batches(self):
  with tempfile.TemporaryDirectory() as d:
   ledger,r=self.setup_ledger(d)
   p=ledger.path;value=json.loads(p.read_text());value['requests'].append(dict(id='prior_batch',status='settled',conservative_charge_cny='9'));value['spent_cny']='12.792488';p.write_text(json.dumps(value))
   ledger.opening=Decimal('12.792488');ledger.stage_opening=Decimal('3.792488')
   with self.assertRaises(ValueError):ledger.reserve('new batch must share cap')
 def test_unknown_cost_stops_future_requests(self):
  with tempfile.TemporaryDirectory() as d:
   ledger,r=self.setup_ledger(d);ident=ledger.reserve('fixture');ledger.unknown(ident,'synthetic network failure')
   with self.assertRaises(ValueError):ledger.reserve('must not send')
 def test_pending_and_duplicate_settlement_stop(self):
  with tempfile.TemporaryDirectory() as d:
   ledger,r=self.setup_ledger(d);ident=ledger.reserve('fixture')
   with self.assertRaises(ValueError):ledger.reserve('pending')
   ledger.settle(ident,dict(prompt_tokens=10,completion_tokens=5),'deepseek-flash')
   with self.assertRaises(ValueError):ledger.settle(ident,dict(prompt_tokens=10,completion_tokens=5),'deepseek-flash')

if __name__=='__main__':unittest.main()
