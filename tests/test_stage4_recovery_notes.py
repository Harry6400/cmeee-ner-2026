"""Synthetic recovery-document fixtures;not actual stage4 evidence."""
import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from write_stage4_recovery_notes import collect,render,digest


def save(root,path,value):
    p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False));return p

class RecoveryNoteTests(unittest.TestCase):
    def test_actual_success_separate_from_preserved_historical_pending_and_wall(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-stage4-recovery-') as temp:
            root=Path(temp);base=root/'evidence/stage4';base.mkdir(parents=True)
            issue=dict(id='ST4-I04',kind='CPU_checker_representation_bug',phenomenon='synthetic tuple/list mismatch',cause_basis='synthetic matching values, different representation',change='synthetic normalization',retest='pending')
            path=base/'issues.jsonl';path.write_text(json.dumps(issue)+'\n');before=path.read_bytes()
            save(root,'evidence/stage4/checkpoint-inspection.json',dict(status='PASS',device='cpu',seconds=25.558,models={str(i):{'status':'PASS'} for i in range(6)}))
            save(root,'evidence/stage4/optimizer-settings-regression.json',dict(operation='optimizer-settings-regression',returncode=0,seconds=.03))
            save(root,'evidence/stage4/post-training-controller.json',dict(status='failed',operations=[dict(name='checkpoint-inspection',status='failed',returncode=1,seconds=4.,console='evidence/stage4/old.log')]))
            save(root,'evidence/stage4/post-training-controller-recovery01.json',dict(status='completed',operations=[dict(name='checkpoint-inspection-recovery01',status='completed',returncode=0,seconds=28.,console='evidence/stage4/new.log')]))
            data=collect(root);self.assertEqual(path.read_bytes(),before);self.assertEqual(data['issues'][0]['historical_retest_note'],'pending');self.assertEqual(data['issues'][0]['retest_status'],'PASS')
            text=render(root,data);self.assertIn('25.558秒',text);self.assertIn('4.000秒',text);self.assertIn('28.000秒',text);self.assertIn('不能直接相加',text);self.assertEqual(data['status'],'PARTIAL_EVIDENCE_SNAPSHOT')
    def test_operation_prefix_distinct_from_result_and_missing_time_not_zero(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-stage4-recovery-') as temp:
            root=Path(temp)
            save(root,'evidence/stage4/operation-final-aggregation.json',dict(operation='operation-final-aggregation',returncode=0,seconds=1.1))
            result=save(root,'evidence/stage4/final-aggregation.json',dict(seconds=.9,results={}));before=result.read_bytes()
            save(root,'evidence/stage4/local-evaluation-queue.json',dict(status='evaluating',runs=[dict(name='synthetic-model',status='evaluating',segments=[dict(status='running',wall_seconds=None)],wait_seconds=.2)]))
            data=collect(root);self.assertEqual(result.read_bytes(),before);self.assertEqual(len(data['operation_records']),1)
            self.assertEqual(next(t['seconds'] for t in data['timings'] if t['name']=='独立指标汇总'),.9)
            self.assertIsNone(next(t['seconds'] for t in data['timings'] if t['name']=='六模型GPU评估子进程墙钟'))
            self.assertFalse(any('同名覆盖' in w for w in data['warnings']))
    def test_existing_dependency_retest_and_preventive_review_not_claimed_failure(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-stage4-recovery-') as temp:
            root=Path(temp);base=root/'evidence/stage4';base.mkdir(parents=True)
            module=root/'scripts/original-synthetic.py';module.parent.mkdir();module.write_text('# synthetic dependency fixture\n')
            issues=[dict(id='ST4-I05',kind='deployment_dependency_absent',phenomenon='synthetic import failure',change='copy exact synthetic original',retest='pending'),dict(id='ST4-I06',kind='preventive_code_review',phenomenon='synthetic potential operation/result naming collision',change='operation prefix before any execution')]
            (base/'issues.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in issues))
            save(root,'evidence/stage4/baseline-dependency-recovery.json',dict(existing_original_files={'scripts/original-synthetic.py':digest(module)}))
            save(root,'evidence/stage4/baseline-dependency-transfer.json',dict(operation='baseline-dependency-transfer',returncode=0,seconds=1.))
            save(root,'runs/stage4-20261020/local-evaluation/D0/summary.json',dict(status='completed',seconds=2.))
            data=collect(root);self.assertEqual(data['issues'][0]['retest_status'],'PASS');self.assertEqual(data['issues'][1]['retest_status'],'PREVENTIVE_REVIEW_NOTE');self.assertIn('不表述为已经发生',data['issues'][1]['retest_description'])

    def test_dynamic_continuation02_attempts_and_independent_segment_walls(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-stage4-recovery02-') as temp:
            root=Path(temp);api='runs/stage4-20261020/api-final'
            save(root,'configs/stage4-execution-continuation02.json',dict(audit=dict(api_summary_path=api+'/summary-continuation02.json')))
            save(root,api+'/summary-continuation01.json',dict(total_attempted=489,settled_calls=487,unknown_cost_calls=2))
            save(root,api+'/summary-continuation02.json',dict(status='completed',total_attempted=1000,settled_calls=997,unknown_cost_calls=3,known_cumulative_cny='5',unknown_cost_upper_cny='6.38976',exposure_upper_cny='11.38976'))
            def rows(path,values):
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r)+'\n' for r in values))
            rows(api+'/calls.jsonl',[dict(id='synthetic-'+str(i),seconds=.01) for i in range(997)])
            rows(api+'/attempts-log-continuation02.jsonl',[dict(key='synthetic-'+str(i),cost_status='unknown_cost' if i in (233,488,700) else 'settled',seconds=.25 if i in (233,488,700) else .01) for i in range(1000)])
            rows(api+'/segments-continuation01.jsonl',[dict(seconds=20,status='stopped_unknown')]);rows(api+'/segments-continuation02.jsonl',[dict(seconds=30,status='completed')])
            ledger=save(root,'private/budget/ledger.json',dict(api_enabled=False,requests=[]));data=collect(root)
            self.assertTrue(data['critical_evidence']['API1000_attempts_finished']);self.assertEqual(data['api_accounting']['unknown_cost_calls'],3)
            timings={t['name']:t['seconds'] for t in data['timings']};self.assertEqual(timings['API运维续跑continuation01流程墙钟'],20);self.assertEqual(timings['API运维续跑continuation02流程墙钟'],30);self.assertEqual(timings['全部费用未知传输失败耗时'],.75);self.assertAlmostEqual(timings['API完整网络响应逐次seconds合计'],9.97)
            text=render(root,data);self.assertIn('997项已有完整响应',text);self.assertIn('3笔仍为费用未知',text);self.assertNotIn('999次已结算',text)
            ledger.write_text(json.dumps(dict(api_enabled=True,requests=[dict(status='pending')])));pending=collect(root);self.assertFalse(pending['critical_evidence']['API1000_attempts_finished'])

if __name__=='__main__':unittest.main()
