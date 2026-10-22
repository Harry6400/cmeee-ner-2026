"""Offline teacher HTTP and isolated process lifecycle contracts; no paid API/model calls."""
import http.client,json,os,signal,sys,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from teacher_project import TeacherProject,ReproductionJobs,BusyError
from serve_stage5 import make_server

class FakeAPI:
    def __init__(self,root): pass
    def budget(self):return dict(baseline=16.819840,trial_limit=15,spent=0)
    def check(self,config):return dict(valid=True,network_call=False)
    def models(self,config):return dict(status='models_checked',models=[dict(id='synthetic-model')],selected_model_found=True,generation_requests=0)
    def extract(self,data):raise RuntimeError('secret-api-key-must-not-escape')

class TeacherServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'ui').mkdir();(self.root/'ui/stage5.html').write_text('<h1>教师验收</h1>')
        (self.root/'ui/teacher.js').write_text('// local')
        (self.root/'private').mkdir();(self.root/'private/secret.js').write_text('do not publish')
        (self.root/'ui/demo-cases.json').write_text(json.dumps({'cases':[{'case':{'case_id':'F04','id':'example','category':'boundary','route':'N1'},'routes':{'N1':{'source':'frozen'}}}]}))
        self.project=TeacherProject(self.root,api_factory=FakeAPI)
        self.server=make_server(port=0,project=self.project);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    def tearDown(self):
        self.project.close();self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()
    def request(self,path,body=None,headers=None,method=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        method=method or ('POST' if body is not None else 'GET');header={'Content-Type':'application/json'} if body is not None else {}
        if headers:header.update(headers)
        conn.request(method,path,json.dumps(body) if body is not None else None,header);response=conn.getresponse();data=response.read();status=response.status;conn.close()
        try:return status,json.loads(data)
        except ValueError:return status,data.decode()
    def test_empty_assets_page_project_and_replay_work(self):
        self.assertEqual(self.request('/')[0],200);status,data=self.request('/api/project')
        self.assertEqual(status,200);self.assertFalse(data['assets']['N1']['ready']);self.assertFalse(data['assets']['data']['ready'])
        self.assertEqual(self.request('/api/cases')[1][0]['case_id'],'F04')
        self.assertEqual(self.request('/api/case?id=F04')[1]['routes']['N1']['source'],'frozen')
        self.assertEqual(self.request('/api/case?id=absent')[0],404)
    def test_model_directory_is_public_local_keyless_and_budget_free(self):
        with patch('teacher_api._resolve_hostname',side_effect=AssertionError('preset lookup must be local')):
            status,data=self.request('/api/api-presets')
        self.assertEqual(status,200);self.assertEqual(data['default_model'],'deepseek-flash')
        self.assertEqual([row['id'] for row in data['models']],['deepseek-flash','deepseek-v4-pro'])
        self.assertEqual(data['external_requests'],0)
        self.assertEqual(data['provider'],'deepseek')
        self.assertEqual(len(data['providers']),7)
        self.assertTrue(all(row['input_price_cny'] and row['output_price_cny'] and row['context_max_tokens']
                            and row['price_source'] and row['price_checked_at'] for row in data['models']))
        self.assertNotIn('budget',json.dumps(data).lower())
        self.assertNotIn('api_key',json.dumps(data).lower())
        self.assertFalse((self.root/'output/teacher-api').exists())
    def test_static_routes_do_not_publish_private_or_traversal(self):
        self.assertEqual(self.request('/ui/teacher.js')[0],200)
        for path in ['/private/secret.js','/ui/../private/secret.js','/ui/%2e%2e/private/secret.js','/ui/%252e%252e/private/secret.js','/ui/%5c..%5cprivate/secret.js','/ui/demo-cases.json']:
            self.assertIn(self.request(path)[0],(400,404),path)
    def test_post_origin_host_json_and_bad_input_are_rejected(self):
        self.assertEqual(self.request('/api/extract',{'text':'症状'},headers={'Origin':'https://evil.example'})[0],403)
        self.assertEqual(self.request('/api/extract',{'text':'症状'},headers={'Host':'evil.example:'+str(self.server.server_port)})[0],403)
        self.assertEqual(self.request('/api/extract',{'text':'症状'},headers={'Host':'127.0.0.1:1'})[0],403)
        self.assertEqual(self.request('/api/extract',{'text':' '})[0],400)
        self.assertEqual(self.request('/api/%2e%2e/extract',{'text':'发热'})[0],400)
        self.assertEqual(self.request('/api/extract',{'text':'症状','method':'other'})[0],400)
        self.assertEqual(self.request('/api/extract',{'text':'症状'},headers={'Content-Type':'text/plain'})[0],400)
        self.assertEqual(self.request('/api/reproduction/jobs',{'command':'no'})[0],400)
    def test_neural_inference_does_not_load_dictionary_or_training(self):
        directory=self.root/'models/trained';directory.mkdir(parents=True);(directory/'N1.pt').write_bytes(b'synthetic')
        (directory/'manifest.json').write_text(json.dumps({'items':[{'method':'N1','file':'N1.pt'}]}))
        class Model:
            def predict_many(self,rows,method):return [dict(rows[0],entities=[dict(start=0,end=2,text='发热',type='sym')])],{'device':'cpu'}
        self.project.models_factory=lambda root:Model()
        self.project.engine_factory=lambda root:(_ for _ in ()).throw(AssertionError('D0 must not load'))
        status,data=self.request('/api/extract',{'text':'发热','method':'N1'})
        self.assertEqual(status,200);self.assertEqual(data['status'],'valid');self.assertEqual(data['prediction'][0]['type'],'sym');self.assertIsNone(data['baseline']);self.assertIsNone(self.project.engine)
        self.assertEqual(self.request('/api/extract',{'text':'发热','method':'D0'})[0],503)
    def test_cpu_lock_excludes_overlap_and_secrets_never_echo(self):
        self.project.cpu_lock.acquire()
        try:self.assertEqual(self.request('/api/extract',{'text':'发热'})[0],409)
        finally:self.project.cpu_lock.release()
        status,data=self.request('/api/extract-api',{'config':{'api_key':'secret-api-key-must-not-escape'}})
        self.assertEqual(status,503);self.assertNotIn('secret-api-key',json.dumps(data))
        self.assertNotIn('secret-api-key',(self.project.log_dir/'ui-server.jsonl').read_text())
        self.assertEqual(self.request('/api/api-check',{'config':{}})[1]['network_call'],False)
    def test_history_is_flat_preserved_and_corrupt_budget_keeps_page_available(self):
        (self.root/'ui/project-data.json').write_text(json.dumps({'previous_reproductions':[{'source':'existing','at':'2026-10-19','seconds':12,'status':'PASS','label':'old'}]}))
        run=self.root/'runs/stage6-candidate-reproduction';run.mkdir(parents=True)
        (run/'summary.json').write_text(json.dumps({'status':'PASS','finished_at':'2026-10-20','seconds':113.67}))
        data=self.request('/api/project')[1]
        self.assertEqual(data['previous_reproductions'][0]['source'],'existing')
        self.assertEqual(data['previous_reproductions'][1]['at'],'2026-10-20')
        self.assertEqual(data['previous_reproductions'][1]['kind'],'historical')
        self.project.api().budget=lambda:(_ for _ in ()).throw(ValueError('bad ledger'))
        status,data=self.request('/api/project');self.assertEqual(status,200)
        self.assertEqual(data['budget']['remaining_cny'],'0.000000')
        self.assertEqual(data['budget']['baseline_cny'],'16.819840')
        self.assertEqual(data['budget']['demo_cap_cny'],'15.000000')
    def test_malformed_json_and_oversized_body_are_bad_requests(self):
        for body in ['{broken','[]','x'*65537]:
            connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
            connection.request('POST','/api/extract',body,{'Content-Type':'application/json'})
            response=connection.getresponse();response.read();self.assertEqual(response.status,400);connection.close()
    def test_foreign_full_reproduction_guard_blocks_local_inference(self):
        self.project.jobs.directory.mkdir(parents=True);self.project.jobs.guard.write_text(json.dumps({'owner_token':'foreign-controller'}))
        status,data=self.request('/api/extract',{'text':'发热','method':'N1'})
        self.assertEqual(status,409);self.assertIn('保护锁',data['error'])
        self.assertEqual(json.loads(self.project.jobs.guard.read_text())['owner_token'],'foreign-controller')
    def test_api_configuration_errors_are_useful_but_not_arbitrary_echo(self):
        self.project.api().check=lambda config:(_ for _ in ()).throw(ValueError('预算预留不足，未发送API请求'))
        status,data=self.request('/api/api-check',{'config':{}})
        self.assertEqual(status,400);self.assertEqual(data['error_code'],'request_protected');self.assertIs(data['request_sent'],False)
        self.project.api().check=lambda config:(_ for _ in ()).throw(ValueError('provider echoed secret-key-do-not-echo'))
        status,data=self.request('/api/api-check',{'config':{}})
        self.assertEqual(status,400);self.assertNotIn('secret-key-do-not-echo',data['error'])
    def test_model_directory_route_and_structured_storage_error(self):
        status,data=self.request('/api/api-models',{'config':{'provider':'custom','base_url':'https://example.com/v1','api_key':'synthetic-key'}})
        self.assertEqual(status,200);self.assertEqual(data['generation_requests'],0)
        self.assertEqual(data['models'],[{'id':'synthetic-model'}])
        from teacher_api import APIError
        self.project.api().check=lambda config:(_ for _ in ()).throw(APIError('storage_not_writable','local_storage',False))
        status,data=self.request('/api/api-check',{'config':{'api_key':'secret-key-do-not-echo'}})
        self.assertEqual(status,503);self.assertEqual(data['error_code'],'storage_not_writable')
        self.assertIs(data['request_sent'],False)
        self.assertEqual(data['stage'],'local_storage')
        log=(self.project.log_dir/'ui-server.jsonl').read_text()
        self.assertIn('local_storage',log);self.assertNotIn('secret-key-do-not-echo',log)
    def test_artifact_lookup_is_fixed(self):
        (self.root/'报告.pdf').write_bytes(b'%PDF-synthetic')
        self.assertEqual(self.request('/artifacts/report')[0],200)
        self.assertEqual(self.request('/artifacts/secret')[0],404)
        self.assertTrue(next(a for a in self.request('/api/project')[1]['artifacts'] if a['id']=='report')['available'])

class ReproductionJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'scripts').mkdir();self.cpu=threading.Lock();self.jobs=ReproductionJobs(self.root,self.cpu)
    def tearDown(self):self.jobs.close();self.tmp.cleanup()
    def script(self,source):(self.root/'scripts/teacher_reproduce.py').write_text(source)
    def wait(self,ident,timeout=5):
        limit=time.monotonic()+timeout
        while time.monotonic()<limit:
            state=self.jobs.get(ident)
            if state['status']!='running':return state
            time.sleep(.02)
        self.fail('synthetic child did not finish')
    def test_unique_job_progress_success_and_fixed_command(self):
        self.script("import argparse,json,pathlib,time\np=argparse.ArgumentParser();p.add_argument('--archive');p.add_argument('--output');a=p.parse_args();o=pathlib.Path(a.output);o.mkdir(parents=True);print(json.dumps(dict(event='progress',method='N1',completed_windows=1,total_windows=5)),flush=True);time.sleep(.15);(o/'summary.json').write_text(json.dumps(dict(status='PASS',samples=5000)))\n")
        job=self.jobs.start()
        with self.assertRaises(BusyError):self.jobs.start()
        result=self.wait(job['id']);self.assertEqual(result['status'],'passed');self.assertEqual(result['summary']['samples'],5000);self.assertEqual(result['exit_code'],0)
        self.assertEqual(result['progress']['completed_windows'],1);self.assertTrue((self.root/result['stdout']).is_file())
        self.assertIn('--archive',result['command']);self.assertNotIn('--split',result['command']);self.assertFalse(self.cpu.locked());self.assertFalse(self.jobs.guard.exists())
        self.assertTrue((self.jobs.metadata/(job['id']+'.json')).is_file())
        self.assertEqual(self.jobs.cancel(job['id'])['status'],'passed')
    def test_nonzero_child_or_missing_pass_never_passes(self):
        self.script("import sys;print('actual synthetic failure',flush=True);sys.exit(7)")
        result=self.wait(self.jobs.start()['id']);self.assertEqual(result['status'],'failed');self.assertEqual(result['exit_code'],7);self.assertIn('actual synthetic failure',result['log'])
    def test_cancel_only_owned_process_group_and_keep_partial_outputs(self):
        self.script("import argparse,pathlib,time\np=argparse.ArgumentParser();p.add_argument('--archive');p.add_argument('--output');a=p.parse_args();o=pathlib.Path(a.output);o.mkdir(parents=True);(o/'partial.txt').write_text('retained');print('child ready',flush=True);time.sleep(30)\n")
        ident=self.jobs.start()['id'];limit=time.monotonic()+3
        while 'child ready' not in self.jobs.get(ident)['log'] and time.monotonic()<limit:time.sleep(.02)
        self.jobs.cancel(ident);result=self.wait(ident)
        self.assertEqual(result['status'],'cancelled');self.assertTrue((self.root/result['output']/'partial.txt').is_file());self.assertFalse(self.cpu.locked());self.assertFalse(self.jobs.guard.exists())
    def test_foreign_persistent_guard_blocks_restart_without_deleting_or_killing(self):
        self.jobs.directory.mkdir(parents=True);self.jobs.guard.write_text(json.dumps({'owner_token':'old-controller','controller_pid':99999999}))
        with patch('teacher_project.os.killpg',create=True) as kill:
            restored=ReproductionJobs(self.root,threading.Lock())
            with self.assertRaisesRegex(BusyError,'保护锁'):restored.start()
            self.assertEqual(json.loads(self.jobs.guard.read_text())['owner_token'],'old-controller');kill.assert_not_called();restored.close()
        self.assertFalse(self.cpu.locked())
    def test_restart_marks_unfinished_failed_without_touching_pid(self):
        self.jobs.metadata.mkdir(parents=True);job=dict(id='interrupted-001',status='running',started_at='2026-10-20',started_unix=0,seconds=0,log='',output='output/not-owned')
        (self.jobs.metadata/'interrupted-001.json').write_text(json.dumps(job))
        with patch('teacher_project.os.killpg',create=True) as kill:
            restored=ReproductionJobs(self.root,threading.Lock());self.assertEqual(restored.get('interrupted-001')['status'],'failed');kill.assert_not_called();restored.close()
if __name__=='__main__':unittest.main()
