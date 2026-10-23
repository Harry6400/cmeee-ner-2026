"""Synthetic collector fixtures, never evidence of Windows or actual model acceptance."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import windows_acceptance as acceptance
import owned_process

class EvidenceTests(unittest.TestCase):
    def test_missing_checks_cannot_finalize_pass_and_inventory_hashes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'中文 路径源码';source.mkdir();(source/'ui').mkdir()
            (source/'ui/example.txt').write_text('测试',encoding='utf-8')
            archive=root/'source.zip'
            with zipfile.ZipFile(archive,'w') as z:z.write(source/'ui/example.txt','ui/example.txt')
            directory=root/'evidence';directory.mkdir()
            acceptance.write(directory/'session.json',dict(root=str(source),archive=str(archive),archive_sha256=acceptance.sha(archive),commit='synthetic',operator='synthetic operator'))
            result=acceptance.finalize(directory)
            self.assertEqual(result['status'],'BLOCKED')
            report=acceptance.load(directory/'report.json')
            self.assertEqual(report['checks']['json-readback']['status'],'NOT_RUN')
            self.assertEqual(result['sha256'],acceptance.sha(Path(result['archive'])))
            with zipfile.ZipFile(result['archive']) as z:
                manifest=json.loads(z.read('manifest.json'))
                for name,item in manifest.items():self.assertEqual(item['sha256'],hashlib.sha256(z.read(name)).hexdigest())
            (source/'ui/example.txt').write_text('changed')
            self.assertEqual(acceptance.binding(acceptance.load(directory/'session.json'))['status'],'FAIL')
    def test_manual_pass_needs_evidence_and_history_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            args=type('Args',(),dict(status='PASS',check='json-readback',note='actually saved synthetic JSON',evidence=[]))()
            with self.assertRaises(ValueError):acceptance.manual(directory,args)
            file=directory/'synthetic.json';file.write_text('{}')
            acceptance.record(directory,args.check,'FAIL','first failure retained')
            args.evidence=[file];acceptance.manual(directory,args)
            events=[acceptance.load(p) for p in sorted(directory.glob('event-*.json'))]
            self.assertEqual([x['status'] for x in events],['FAIL','PASS'])
            self.assertTrue(events[-1]['detail']['declared_by_operator'])
    def test_missing_command_records_blocked_and_real_nonzero_records_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            with patch.object(acceptance.subprocess,'Popen',side_effect=FileNotFoundError):
                event=acceptance.run_check(directory,'frontend-tests')
            self.assertEqual(event['status'],'BLOCKED');self.assertIsNone(event['detail']['exit_code'])
            fake=MagicMock();fake.stdout=io.StringIO('synthetic failure\n');fake.wait.return_value=7;fake.poll.return_value=7
            with patch.object(acceptance.subprocess,'Popen',return_value=fake):
                event=acceptance.run_check(directory,'frontend-tests')
            self.assertEqual(event['status'],'FAIL');self.assertEqual(event['detail']['exit_code'],7)
    def test_independent_recount_rejects_rounded_scores_or_changed_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir();(root/'data/sealed').mkdir(parents=True)
            output=root/'new';(output/'evaluation').mkdir(parents=True)
            rows=[dict(id=str(i),text='😀发热',entities=[dict(start=1,end=3,type='sym',text='发热')]) for i in range(5000)]
            text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
            gold=root/'data/sealed/holdout.jsonl';gold.write_text(text,encoding='utf-8')
            metric=dict(tp=5000,predicted=5000,gold=5000,precision=1.0,recall=1.0,f1=1.0)
            acceptance.write(root/'configs/reproduction-expected.json',dict(data_sha256={'sealed/holdout.jsonl':acceptance.sha(gold)},metrics={'holdout5000':{m:metric for m in ('D0','N0','N1')}}))
            base=dict(status='PASS',samples=5000,split='holdout5000',API_calls=0,GPU_started=False,training=False)
            acceptance.write(output/'summary.json',base)
            acceptance.write(output/'evaluation/summary.json',dict(base,results={m:dict(metrics=dict(micro=metric)) for m in ('D0','N0','N1')}))
            for m in ('D0','N0','N1'):(output/'evaluation'/f'{m}-predictions.jsonl').write_text(text,encoding='utf-8')
            self.assertEqual(acceptance.independent_cpu(root,output)['status'],'PASS')
            rows[-1]['entities'][0]['start']=0
            (output/'evaluation/N1-predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'text/type'):acceptance.independent_cpu(root,output)

class OwnedProcessTests(unittest.TestCase):
    def test_windows_bootstrap_assigns_job_before_allowing_project_execution(self):
        fake=MagicMock();fake.stdin=MagicMock();job=MagicMock();order=[]
        job.assign.side_effect=lambda p:order.append('assigned')
        fake.stdin.write.side_effect=lambda token:order.append(token)
        with patch.object(owned_process.os,'name','nt'),patch.object(owned_process,'WindowsJob',return_value=job):
            factory=MagicMock(return_value=fake)
            result=owned_process.spawn([sys.executable,'-B','-u','C:/中文 路径/script.py','--output','a'],popen=factory,text=True)
        self.assertIs(result,fake);self.assertEqual(order,['assigned','1'])
        args,kwargs=factory.call_args
        self.assertEqual(args[0][-3:],['C:/中文 路径/script.py','--output','a'])
        self.assertNotIn('start_new_session',kwargs);self.assertEqual(kwargs['creationflags'],0x200)
    def test_assignment_failure_never_releases_project_bootstrap(self):
        fake=MagicMock();fake.poll.return_value=None;job=MagicMock();job.assign.side_effect=OSError('job assignment denied')
        with patch.object(owned_process.os,'name','nt'),patch.object(owned_process,'WindowsJob',return_value=job):
            with self.assertRaises(OSError):owned_process.spawn([sys.executable,'-B','-u','script.py'],popen=MagicMock(return_value=fake),text=True)
        fake.stdin.write.assert_not_called();fake.kill.assert_called_once();job.close.assert_called_once()
    @unittest.skipUnless(os.name=='nt','requires native Windows Job Object; macOS mock checks are not Windows evidence')
    def test_native_job_cancels_grandchild_but_keeps_unrelated_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=root/'子进程.py';pidfile=root/'pid.txt'
            child.write_text("import subprocess,sys,time,pathlib\np=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\npathlib.Path(sys.argv[1]).write_text(str(p.pid))\nprint('ready',flush=True)\ntime.sleep(60)\n",encoding='utf-8')
            unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
            process=None
            try:
                process=owned_process.spawn([sys.executable,'-B','-u',str(child),str(pidfile)],stdout=subprocess.PIPE,text=True,encoding='utf-8')
                limit=time.monotonic()+10
                while not pidfile.exists() and time.monotonic()<limit:time.sleep(.02)
                self.assertTrue(pidfile.exists())
                owned_process.terminate(process);process.wait(timeout=10)
                self.assertTrue(owned_process.gone(process));self.assertIsNone(unrelated.poll())
            finally:
                if process is not None:
                    owned_process.terminate(process);process.wait(timeout=10);process.stdout.close();owned_process.release(process)
                unrelated.terminate();unrelated.wait(timeout=10)
if __name__=='__main__':unittest.main()
