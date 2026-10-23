# -*- coding: utf-8 -*-
"""Synthetic report bookkeeping fixtures;not real experiment conclusions."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage4_metrics import evaluate_route,SEEDS
from write_stage4_report import generate

class ReportTests(unittest.TestCase):
    def test_dynamic_continuation02_counts_sources_and_scientific_limits(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-stage4-report02-') as temp:
            root=Path(temp);gold=[dict(id='synthetic',text='x',entities=[])]
            result=evaluate_route(gold,gold,set(),False);result['valid']=500
            results={}
            for route in ('D0','N0','N1'):
                for seed in ((None,) if route=='D0' else SEEDS):
                    for split in ('common500','holdout5000'):results[route+(f'-seed{seed}' if seed else '')+'/'+split]=copy.deepcopy(result)
            for route,valid,status in [('L0',497,dict(valid=497,invalid=1,transport_error=2)),('L3',499,dict(valid=499,transport_error=1))]:
                results[route+'/common500']=dict(main_metrics=None,valid=valid,statuses=status,conditional_valid_only_metrics=copy.deepcopy(result['main_metrics']))
            aggregation=dict(results=results,multi_seed={r:{s:dict(f1=dict(mean=.5,sample_sd=.01)) for s in ('common500','holdout5000')} for r in ('N0','N1')},bootstrap=dict(differences={'L3 minus D0':dict(status='N/A')}),execution_manifest='configs/stage4-execution-continuation02.json')
            budget=dict(stage4_cny='1',cumulative_cny='5',known_cumulative_cny='5',unknown_cost_upper_cny='6.38976',exposure_upper_cny='11.38976',remaining_cny='88.61024',stage4_unknown_cost_calls=3,continuation_version='continuation02',unknown_cost_approval_path='evidence/stage4/transport-budget-authorization02.json',continuation_annex_path='runs/stage4-20261020/api-final/continuation02/annex-at-call.json',api_summary_path='runs/stage4-20261020/api-final/summary-continuation02.json',routes={})
            for route,settled,unknown in [('L0',498,2),('L3',499,1)]:budget['routes'][route]=dict(attempts=500,settled_calls=settled,errors=1 if route=='L0' else 0,transport_errors=unknown,transport_error_seconds=[.25]*unknown,conservative_cny='.5',latency=dict(mean_seconds=.01,p50_seconds=.01,p95_seconds=.01))
            def save(path,value):
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False))
            save('aggregation.json',aggregation);save('verification.json',dict(status='PASS_WITH_WARNINGS',training=[],latency=[],budget=budget))
            generate(root,Path('aggregation.json'),Path('verification.json'),Path('docs/report.md'),Path('evidence/role.json'))
            text=(root/'docs/report.md').read_text();self.assertIn('1000次尝试为997次已结算完整响应加3笔费用未知',text);self.assertIn('transport-budget-authorization02.json',text);self.assertIn('attempts-log-continuation02.jsonl',text);self.assertIn('五路线完整可比主F1是否可用',text);self.assertIn('窗口未完整覆盖参考实体数为0',text);self.assertIn('同跨度异类参考实体数为0',text);self.assertNotIn('999已结算加1未知',text)
            self.assertIn('阶段四报告生成时的历史快照',text);self.assertIn('不反映当前项目进度',text)
            roles=json.loads((root/'evidence/role.json').read_text())
            self.assertEqual(roles['planned_totals_percent'],{'吴皓睿':52,'周栩乐':48})
            self.assertEqual([r['planned_weight_percent'] for r in roles['roles'] if r['role']=='吴皓睿'],[21,16,10,5])
            self.assertEqual([r['planned_weight_percent'] for r in roles['roles'] if r['role']=='周栩乐'],[10,14,14,10])
            self.assertIn('snapshot',roles['scope'])
            self.assertTrue(roles['stage5_error_analysis_not_yet_claimed'])
            self.assertTrue(all('后续未知必须停止' not in r['cross_check'] for r in roles['roles']))

if __name__=='__main__':unittest.main()
