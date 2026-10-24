"""Foreground dependency closure: collect and audit completed authorized outputs;no API or training launch."""
import hashlib,json,os,subprocess,time,signal
from pathlib import Path
from datetime import datetime,timezone
r=Path(__file__).resolve().parents[1];assert r.name=='医学实体抽取';p=r/'evidence/stage4/finalize-controller.json';assert not p.exists()
def now():return datetime.now(timezone.utc).isoformat()
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def save():
 t=p.with_suffix('.writing.json');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
stop=[False];child=None;signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True));start=time.perf_counter()
names=['collect_stage4.py','aggregate_stage4.py','stage4_metrics.py','verify_stage4.py','write_stage4_report.py']
d=dict(status='waiting_remote_final_predictions_and_local_API',started_at=now(),controller_pid=os.getpid(),observations=[],operations=[],source_sha256={n:sha(r/'scripts'/n) for n in names},no_API_or_training_launch_here=True);save()
remote="""import json
from pathlib import Path
p=Path('/nfsdata_a40/cyf/codes/CT/cmeee_ner/evidence/stage4/post-training-controller.json');q=json.loads(p.read_text());print(json.dumps({'status':q['status'],'operations':[{k:x.get(k) for k in ['name','status','returncode']} for x in q['operations']]}))
"""
try:
 while True:
  if stop[0]:raise InterruptedError('user stopped dependent final audit')
  if time.perf_counter()-start>84600:raise TimeoutError('bounded dependent final audit wait')
  try:
   a=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','qha40gpu-2','python3 -'],input=remote,text=True,capture_output=True,timeout=30);a.check_returncode();q=json.loads(a.stdout);api=json.loads((r/'evidence/stage4/api-controller.json').read_text())
  except Exception as e:
   d['observations'].append(dict(at=now(),status='transient_read_retry',error=repr(e)));save();time.sleep(60);continue
  state=dict(remote=q['status'],api=api['status'])
  if not d['observations'] or d['observations'][-1].get('state')!=state:d['observations'].append(dict(at=now(),state=state));save()
  if q['status'] in ['failed','user_stopped'] or api['status'] in ['failed','user_stopped','API_stopped_with_issue;no_retry']:raise RuntimeError('Required stage4 producer stopped:'+str(state))
  if q['status']=='completed_local_stage4' and api['status']=='finished_API_calls' and api.get('returncode')==0:break
  time.sleep(60)
 d.update(wait_seconds=time.perf_counter()-start,status='independent_final_audit');save()
 for n,w in d['source_sha256'].items():assert sha(r/'scripts'/n)==w,'Audit source changed while queued:'+n
 for name,script,args in [('final-evidence-transfer','collect_stage4.py',['--execute']),('final-aggregation','aggregate_stage4.py',[]),('final-verification','verify_stage4.py',[]),('final-report-generation','write_stage4_report.py',[])]:
  if stop[0]:raise InterruptedError('user stopped dependent final audit')
  op=dict(name=name,script='scripts/'+script,started_at=now(),status='running');d['operations'].append(op);save();t=time.perf_counter()
  command=[str(r/'.venv/bin/python'),'scripts/record_operation.py','--name',name,'--directory','evidence/stage4',str(r/'.venv/bin/python'),'scripts/'+script]+args
  child=subprocess.Popen(command,cwd=r,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
  while child.poll() is None:
   if stop[0]:child.send_signal(signal.SIGTERM)
   time.sleep(1)
  code=child.wait();child=None;op.update(returncode=code,finished_at=now(),seconds=time.perf_counter()-t,status='completed' if code==0 else 'failed');save()
  if code:raise RuntimeError('Independent final audit failed:'+name)
 v=json.loads((r/'evidence/stage4/final-verification.json').read_text());assert v['status'] in ['PASS','PASS_WITH_WARNINGS']
 d.update(status='completed_stage4_audit_report',finished_at=now(),verification_status=v['status'],cumulative_cny=v['budget']['cumulative_cny'],no_stage5_started=True);save();print(json.dumps(d,ensure_ascii=False),flush=True)
except BaseException as e:
 d.update(status='user_stopped' if isinstance(e,InterruptedError) else 'failed',error=repr(e),finished_at=now());save();raise
finally:
 if child is not None and child.poll() is None:
  child.send_signal(signal.SIGTERM)
  try:child.wait(timeout=120)
  except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
