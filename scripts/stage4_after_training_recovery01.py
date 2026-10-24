"""Foreground dependent stage4 closure; waits for all supplement seeds before any final evaluation."""
import os,json,subprocess,time,signal,hashlib
from pathlib import Path
from datetime import datetime,timezone
r=Path('/nfsdata_a40/cyf/codes/CT/cmeee_ner');assert r.resolve()==r
u='GPU-f77a1700-de70-05d2-048f-18c8f9aa5974';p=r/'evidence/stage4/post-training-controller-recovery01.json';assert not p.exists()
def now():return datetime.now(timezone.utc).isoformat()
def save(d):
 t=p.with_suffix('.writing.json');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
d=dict(status='waiting_all_four_supplementary_training_runs',started_at=now(),controller_pid=os.getpid(),operations=[],source_sha256=sha(Path(__file__)),no_API_here=True,recovers='evidence/stage4/post-training-controller.json',diagnostic_fix='normalize optimizer tuple/list JSON representation only');save(d);started=time.perf_counter();stop=[False];active=None
signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
env=dict(os.environ,TMPDIR=str(r/'.tmp'),XDG_CACHE_HOME=str(r/'.cache'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
try:
 while True:
  if stop[0]:raise InterruptedError('user stopped dependent stage4 closure')
  q=json.loads((r/'evidence/stage4/training-queue.json').read_text())
  if q['status']=='completed':break
  if q['status'] in ['failed','user_stopped']:raise RuntimeError('training queue not complete:'+q['status'])
  if time.perf_counter()-started>84600:raise TimeoutError('bounded dependent wait')
  time.sleep(30)
 d.update(training_wait_seconds=time.perf_counter()-started,training_queue_sha256=sha(r/'evidence/stage4/training-queue.json'));save(d)
 spec=json.loads((r/'configs/stage4-execution.json').read_text());contract=json.loads((r/'configs/qha-stage4-contract.json').read_text());assert contract['final_holdout_authorized']
 for rel,want in spec['audit']['pre_evaluation_files'].items():
  # API/privacy artifacts and some Mac-only evidence stay local; every remote scientific/data file is checked here.
  f=r/rel
  if f.exists():assert sha(f)==want,rel
 for name,script in [('checkpoint-inspection','inspect_stage4_checkpoints.py'),('dictionary-evaluation','dictionary_evaluate_stage4.py'),('local-evaluation-controller','local_evaluation_queue_stage4.py')]:
  if stop[0]:raise InterruptedError('user stopped dependent closure')
  console=r/'evidence/stage4'/(name+'-recovery01.log');assert not console.exists();op=dict(name=name,script='scripts/'+script,source_sha256=sha(r/'scripts'/script),started_at=now(),status='running',console=str(console.relative_to(r)));d['operations'].append(op);d['status']=name;save(d);t=time.perf_counter()
  with console.open('x') as f:
   active=subprocess.Popen([str(r/'.venv/bin/python'),str(r/'scripts'/script)],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);op['pid']=active.pid;save(d)
   while active.poll() is None:
    if stop[0]:active.send_signal(signal.SIGTERM)
    if time.perf_counter()-t>84600:active.send_signal(signal.SIGTERM);stop[0]=True
    time.sleep(2)
   code=active.wait();active=None
  op.update(status='completed' if code==0 else 'failed',returncode=code,finished_at=now(),seconds=time.perf_counter()-t);save(d)
  if code:raise RuntimeError('stage4 dependent check failed:'+name)
 q=json.loads((r/'evidence/stage4/local-evaluation-queue.json').read_text());train=json.loads((r/'evidence/stage4/training-queue.json').read_text());pids=[s['pid'] for x in train['runs'] for s in x['segments'] if 'pid' in s]+[s['pid'] for x in q['runs'] for s in x['segments'] if 'pid' in s]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True);on_card=[x for x in apps.splitlines() if u in x];own=[int(x.split(',')[0]) for x in on_card if int(x.split(',')[0]) in pids]
 resource=dict(recorded_at=now(),gpu=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),all_GPU7_compute_processes=on_card,own_training_or_inference_pids=own,gpu_workload_stopped=not own,foreign_processes_not_modified=True,source_training_queue='evidence/stage4/training-queue.json',source_inference_queue='evidence/stage4/local-evaluation-queue.json')
 (r/'evidence/stage4/post-run-gpu.json').write_text(json.dumps(resource,ensure_ascii=False,indent=2)+'\n');assert not own
 d.update(status='completed_local_stage4',finished_at=now(),API_final='not run remotely;local Mac task only');save(d);print(json.dumps(d,ensure_ascii=False))
except BaseException as e:
 d.update(status='user_stopped' if isinstance(e,InterruptedError) else 'failed',error=repr(e),finished_at=now());save(d);raise
finally:
 if active is not None and active.poll() is None:
  active.send_signal(signal.SIGTERM)
  try:active.wait(timeout=120)
  except subprocess.TimeoutExpired:active.kill();active.wait(timeout=10)
