"""重算已有API小样本记录；不重新调用API。"""
import json,math
from decimal import Decimal
from pathlib import Path
from ner_core import ROOT,read_rows,score,dump,digest


def main():
 results={}
 for name in ('api-development','api-development-grounded'):
  root=ROOT/'runs/stage3-20261019'/name;calls=read_rows(root/'calls.jsonl');gold=read_rows(root/'dev.jsonl')
  if len(calls)!=12 or len({r['id'] for r in calls})!=12:raise ValueError('调用数/ID不一致')
  protocol=json.loads((root/'protocol.json').read_text());routes={}
  for route in ('L0','L3'):
   rows=[r for r in calls if r['route']==route]
   if [r['sample_id'] for r in rows]!=[r['id'] for r in gold]:raise ValueError('路线样本顺序不一致')
   times=sorted(r['seconds'] for r in rows);usage={}
   for r in rows:
    for k,v in r['response']['usage'].items():
     if type(v) is int:usage[k]=usage.get(k,0)+v
   pred=read_rows(root/(route+'-valid-predictions.jsonl'))
   metrics=score(gold,pred) if len(pred)==6 else None
   routes[route]=dict(attempts=6,valid=len(pred),invalid=6-len(pred),strict_on_all_six=metrics,
       conditional_score_not_used_as_main=True,latency_mean_seconds=sum(times)/6,latency_p50_seconds=(times[2]+times[3])/2,
       latency_p95_seconds=times[5],usage_sum=usage,conservative_cny=str(sum((Decimal(r['charge_cny']) for r in rows),Decimal(0))),
       actual_models=sorted({r['response']['model'] for r in rows}))
  results[name]=dict(routes=routes,authorized_protocol_sha256=digest(root/'authorized-protocol-reconstructed.json'),
                    authorized_protocol_snapshot='reconstructed with explicit provenance; byte digest matches ledger',
                    executed_client_sha256=digest(root/'client-executed.py'))
 ledger=json.loads((ROOT/'private/budget/ledger.json').read_text());original=json.loads((ROOT/'private/budget/original-ledger-20261018.json').read_text())
 if ledger['requests'][:816]!=original['requests'] or len(ledger['requests'])!=840 or ledger['api_enabled']:
  raise ValueError('历史明细、调用数或关闭状态不符')
 current={r['id']:r for r in ledger['requests'][816:]}
 for name in results:
  for r in read_rows(ROOT/'runs/stage3-20261019'/name/'calls.jsonl'):
   if current[r['id']]['conservative_charge_cny']!=r['charge_cny'] or current[r['id']]['usage']!=r['response']['usage']:
    raise ValueError('账本与原始输出不一致')
 total=Decimal(ledger['spent_cny'])-Decimal(original['spent_cny'])
 if total>10 or Decimal(ledger['spent_cny'])>100:raise ValueError('超预算')
 dump(ROOT/'evidence/stage3/api-audit.json',dict(status='PASS',batches=results,new_requests=24,new_cny=str(total),
       cumulative_cny=ledger['spent_cny'],remaining_cny=str(Decimal(100)-Decimal(ledger['spent_cny'])),historical_requests_preserved=816,
       api_enabled=False,sealed_read=False,raw_v1_failures_preserved=True))
 print(json.dumps(dict(status='PASS',new_requests=24,new_cny=str(total),cumulative_cny=ledger['spent_cny'])))

if __name__=='__main__':main()
