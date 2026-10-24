"""Synthetic counterexamples for the actual pre-stage6 audit gaps."""
import copy,json,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_stage5 import validate_api_records,validate_cases,metric
from stage5_repairs import Engine,load_policy,features

def put(root,path,value,jsonl=False):
    p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in value) if jsonl else json.dumps(value,ensure_ascii=False));return p

class ReviewChecks(unittest.TestCase):
    def api_fixture(self,root):
        gold=[dict(id='a',text='肺炎',entities=[dict(start=0,end=2,type='dis',text='肺炎')]),dict(id='b',text='GH',entities=[dict(start=0,end=2,type='ite',text='GH')])]
        calls=[];records=[];base='runs/stage4-20261020/api-final/'
        for route,ident,wire in [('L0','a',dict(entities=[dict(text='肺炎',type='dis',occurrence=0)])),('L3','b',dict(entities=[dict(text='GH',type='ite',occurrence=0),dict(text='GH',type='ite',occurrence=1)]))]:
            cid='call-'+ident;raw='requests/'+cid+'.json';status='valid' if ident=='a' else 'invalid';row=gold[0 if ident=='a' else 1]
            calls.append(dict(id=cid,route=route,sample_id=ident,status=status,raw_file=raw))
            put(root,base+raw,dict(id=cid,route=route,sample_id=ident,response=dict(choices=[dict(message=dict(content=json.dumps(wire))) ])))
            action=[] if ident=='a' else [dict(action='drop_overflow_duplicate',index=1,entity=wire['entities'][1],actual_occurrences=1)]
            records.append(dict(call_id=cid,route=route,sample_id=ident,original_status=status,raw_file=base+raw,status='valid_unchanged' if ident=='a' else 'repaired',prediction=row,actions=action,issues=[]))
            put(root,base+route+'-predictions.jsonl',[row],True)
        put(root,base+'calls.jsonl',calls,True);put(root,'configs/stage4-execution-continuation02.json',dict(audit=dict(api_prediction_paths={x:base+x+'-predictions.jsonl' for x in ['L0','L3']})))
        summary={route:dict(responses=1,repaired=int(route=='L3'),remaining_invalid=0,conditional_valid_samples=1,network_calls=0,conditional_diagnostic_micro=metric([gold[i]],[gold[i]]),primary_F1='N/A retrospective') for i,route in enumerate(['L0','L3'])}
        return gold,records,summary
    def test_original_fixture_and_route_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);gold,records,summary=self.api_fixture(root);validate_api_records(root,gold,records,summary)
            changed=copy.deepcopy(records);changed[0]['route']='L3'
            with self.assertRaisesRegex(ValueError,'identity'):validate_api_records(root,gold,changed,summary)
    def test_summary_and_raw_path_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);gold,records,summary=self.api_fixture(root);summary['L0']['conditional_valid_samples']=2
            with self.assertRaisesRegex(ValueError,'summary'):validate_api_records(root,gold,records,summary)
            gold,records,summary=self.api_fixture(root);records[0]['raw_file']=records[1]['raw_file']
            with self.assertRaisesRegex(ValueError,'path'):validate_api_records(root,gold,records,summary)
    def test_invalid_reason_is_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);gold,records,summary=self.api_fixture(root);records[1].update(status='invalid',prediction=None,issues=[dict(reason='arbitrary invalid')])
            with self.assertRaisesRegex(ValueError,'status'):validate_api_records(root,gold,records,summary)
    def test_repair_action_content_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);gold,records,summary=self.api_fixture(root);records[1]['actions'][0]['actual_occurrences']=3
            with self.assertRaisesRegex(ValueError,'action content'):validate_api_records(root,gold,records,summary)
    def test_displayed_cases_and_recovery_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=dict(start=0,end=2,type='dis',text='肺炎');g=dict(id='a',text='肺炎',entities=[target]);put(root,'before.jsonl',[dict(g,entities=[])],True);put(root,'runs/stage5-20261020/common500-diagnostic/combined-predictions.jsonl',[g],True)
            cases=[dict(case_id=str(i),id='a',route='D0',input='肺炎',reference=[target],source_prediction='before.jsonl',original_prediction=[],repaired_prediction=[target],target=target,target_recovered=True,category=str(i),observed_facts='synthetic fixture',attribution='synthetic fixture',limitations='synthetic fixture') for i in range(3)]
            validate_cases(root,[g],[],cases,cases)
            changed=copy.deepcopy(cases);changed[0]['input']='其他'
            with self.assertRaisesRegex(ValueError,'scientific content'):validate_cases(root,[g],[],cases,changed)
            changed=copy.deepcopy(cases);changed[0]['target_recovered']=False
            with self.assertRaisesRegex(ValueError,'recovery'):validate_cases(root,[g],[],changed,changed)
    def test_external_policy_changes_actual_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);policy=load_policy();train=[dict(id=str(i),text='喂养困难',entities=[dict(start=0,end=4,type='sym',text='喂养困难')]) for i in range(3)]
            class FixedTokens:
                def tokenize(self,text,HMM=True):return [(text,0,len(text))]
            cfg=put(root,'policy.json',dict(repair_policy=policy));engine=Engine(train,root/'cache',cfg);engine.tokenizer=FixedTokens();row=dict(id='q',text='婴儿喂养困难')
            self.assertEqual(len(engine.predict(row,'lexicon')[0]['entities']),1)
            policy['min_term_count']=4;put(root,'policy.json',dict(repair_policy=policy));engine2=Engine(train,root/'cache',cfg);engine2.tokenizer=FixedTokens()
            self.assertEqual(len(engine2.predict(row,'lexicon')[0]['entities']),0)
            self.assertNotEqual(features('abc肺def',3,4,2),features('abc肺def',3,4,3))

if __name__=='__main__':unittest.main(verbosity=2)
