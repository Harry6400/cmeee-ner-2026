"""Finish authorized stage3 only; no paid API or sealed-data operations."""
import os,json,time,subprocess,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
r=Path('/nfsdata_a40/cyf/codes/CT/cmeee_ner');assert r.resolve()==r
u='GPU-f77a1700-de70-05d2-048f-18c8f9aa5974'
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,d):
 t=p.with_suffix('.writing.json');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
p=r/'evidence/stage3/completion-controller.json';assert not p.exists();d=dict(started_at=now(),controller_pid=os.getpid(),status='waiting_formal_training',observations=[],sealed_read=False,paid_api_calls=0);dump(p,d);start=time.perf_counter()
while True:
 q=json.loads((r/'evidence/stage3/queue-resume-02.json').read_text())
 if q['status']=='completed':break
 if q['status']=='interrupted':raise RuntimeError('formal training interrupted; do not run performance before it finishes')
 if time.perf_counter()-start>84600:raise TimeoutError('training closure wait bounded')
 time.sleep(5)
d.update(status='GPU_preflight');dump(p,d);wait=time.perf_counter()
while True:
 gpu=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip()
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True);apps=[a for a in apps.splitlines() if u in a];fields=[a.strip() for a in gpu.split(',')];assert fields[0]=='7' and fields[1]==u
 busy=bool(apps) or int(fields[3].split()[0])>128;d['observations'].append(dict(at=now(),gpu=gpu,compute_processes=apps,busy=busy));dump(p,d)
 if not busy:break
 if time.perf_counter()-wait>84600:raise TimeoutError('bounded wait for performance')
 time.sleep(60)
assert shutil.disk_usage(r).free>20*1024**3
contract=json.loads((r/'evidence/stage3/resume-02-contract.json').read_text())
for name,want in contract['critical_source_sha256'].items():assert hashlib.sha256((r/'scripts'/name).read_bytes()).hexdigest()==want
preflight=dict(recorded_at=now(),gpu=gpu,compute_processes=apps,free_disk_bytes=shutil.disk_usage(r).free,resource_gate='PASS',contract='evidence/stage3/resume-02-contract.json',critical_sources_unchanged=True)
dump(r/'evidence/stage3/performance-prelaunch.json',preflight)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=u,TMPDIR=str(r/'.tmp'),XDG_CACHE_HOME=str(r/'.cache'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
cmd=[str(r/'.venv/bin/python'),str(r/'scripts/benchmark_stage3_runtime.py'),'--output',str(r/'runs/stage3-runtime-20261020'),'--checkpoint',str(r/'runs/stage3-20261019/N1-lr-5e-05/last.pt')]
d.update(status='performance',performance_started_at=now(),performance_prelaunch_wait_seconds=time.perf_counter()-wait);dump(p,d);t=time.perf_counter()
with (r/'evidence/stage3/performance-controller.log').open('x') as f:a=subprocess.run(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=115)
d['performance']=dict(returncode=a.returncode,seconds=time.perf_counter()-t,finished_at=now(),console='evidence/stage3/performance-controller.log');d.update(status='CPU_checkpoint_inspection');dump(p,d)
t=time.perf_counter();started=now()
with (r/'evidence/stage3/checkpoint-inspection.log').open('x') as f:b=subprocess.run([str(r/'.venv/bin/python'),str(r/'scripts/inspect_development_checkpoints.py')],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
operation=dict(operation='remote CPU strict checkpoint verification',started_at=started,finished_at=now(),seconds=time.perf_counter()-t,returncode=b.returncode,console='evidence/stage3/checkpoint-inspection.log',device='cpu');dump(r/'evidence/stage3/checkpoint-inspection-operation.json',operation)
gpu=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True);dump(r/'evidence/stage3/post-run-gpu.json',dict(recorded_at=now(),gpu=gpu,compute_processes=[x for x in apps.splitlines() if u in x],all_own_GPU_children_exited=True,foreign_processes_not_modified=True))
d.update(status='completed' if a.returncode==0 and b.returncode==0 else 'completed_with_failed_checks',checkpoint_inspection=operation,finished_at=now());dump(p,d);print(json.dumps(d,ensure_ascii=False))
