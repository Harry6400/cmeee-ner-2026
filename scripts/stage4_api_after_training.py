"""Human-approved DeepSeek final evaluation, after all training and CPU checks;foreground task only."""
import os,json,time,subprocess,signal,hashlib,base64
from pathlib import Path
from datetime import datetime,timezone
r=Path(__file__).resolve().parents[1];proof=r/'evidence/stage4/external-send-authorization.json';approval=json.loads(proof.read_text())
assert approval['explicit_receiver_approved'] and approval['receiver']=='https://api.deepseek.com/chat/completions' and approval['max_requests']==1000
p=r/'evidence/stage4/api-controller.json';assert not p.exists()
def now():return datetime.now(timezone.utc).isoformat()
def save(d):
 t=p.with_suffix('.writing.json');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True));child=None;start=time.perf_counter()
d=dict(status='waiting_all_training_and_CPU_checkpoint_PASS',started_at=now(),controller_pid=os.getpid(),observations=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),external_receiver_authorization_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),no_automatic_API_retry=True);save(d)
remote="""import json,base64
from pathlib import Path
r=Path('/nfsdata_a40/cyf/codes/CT/cmeee_ner');p=r/'evidence/stage4/training-queue.json';q=json.loads(p.read_text());out={'status':q['status'],'runs':[{k:x.get(k) for k in ['name','status','epochs','best_epoch','best_dev_f1']} for x in q['runs']]}
if q['status']=='completed':
 out['queue_raw_base64']=base64.b64encode(p.read_bytes()).decode();cp=r/'evidence/stage4/checkpoint-inspection.json'
 if cp.exists():
  c=json.loads(cp.read_text());out['checkpoint_status']=c.get('status');out['checkpoint_raw_base64']=base64.b64encode(cp.read_bytes()).decode()
print(json.dumps(out))
"""
try:
 while True:
  if stop[0]:raise InterruptedError('user stopped dependent API task')
  if time.perf_counter()-start>84600:raise TimeoutError('bounded wait for all supplementary training')
  try:
   a=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','qha40gpu-2','python3 -'],input=remote,text=True,capture_output=True,timeout=30);a.check_returncode();q=json.loads(a.stdout)
  except Exception as e:
   d['observations'].append(dict(at=now(),status='read_only_transport_or_atomic_write_retry',error=repr(e)));save(d);time.sleep(60);continue
  d['observations'].append(dict(at=now(),status=q['status'],runs=q['runs'],checkpoint_status=q.get('checkpoint_status')));save(d)
  if q['status'] in ['failed','user_stopped']:raise RuntimeError('training queue not complete:'+q['status'])
  if q['status']=='completed' and q.get('checkpoint_status')=='PASS':break
  time.sleep(60)
 assert hashlib.sha256(proof.read_bytes()).hexdigest()==d['external_receiver_authorization_sha256']
 for name,key in [('training-queue.json','queue_raw_base64'),('checkpoint-inspection.json','checkpoint_raw_base64')]:
  target=r/'evidence/stage4'/name;raw=base64.b64decode(q[key]);assert not target.exists();target.write_bytes(raw)
 d.update(status='running_frozen_final_API',training_and_CPU_wait_seconds=time.perf_counter()-start,API_started_at=now());save(d);t=time.perf_counter();console=r/'evidence/stage4/api-final-controller.log'
 with console.open('x') as f:
  child=subprocess.Popen([str(r/'.venv/bin/python'),str(r/'scripts/api_final.py'),'run'],cwd=r,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True);d['API_child_pid']=child.pid;save(d)
  while child.poll() is None:
   if stop[0]:child.send_signal(signal.SIGTERM)
   calls=r/'runs/stage4-20261020/api-final/calls.jsonl'
   if calls.exists():d['settled_calls_logged']=len(calls.read_text().splitlines());save(d)
   time.sleep(5)
  code=child.wait();child=None
 d.update(status='finished_API_calls' if code==0 else 'API_stopped_with_issue;no_retry',returncode=code,API_seconds=time.perf_counter()-t,finished_at=now(),console=str(console.relative_to(r)));save(d);print(json.dumps(d,ensure_ascii=False));raise SystemExit(code)
except BaseException as e:
 if isinstance(e,SystemExit):raise
 d.update(status='user_stopped' if isinstance(e,InterruptedError) else 'failed',error=repr(e),finished_at=now());save(d);raise
finally:
 if child is not None and child.poll() is None:
  child.send_signal(signal.SIGTERM)
  try:child.wait(timeout=120)
  except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
