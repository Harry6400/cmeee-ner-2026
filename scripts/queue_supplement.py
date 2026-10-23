"""Sequential authorized GPU7 development supplement; no API or sealed data access."""
from __future__ import annotations
import json,os,signal,subprocess,sys,time,shutil,hashlib
from datetime import datetime,timezone
from pathlib import Path
from ner_core import ROOT,dump,digest

def now():return datetime.now(timezone.utc).isoformat()
def gpu_observation(own_pgid=None):
 uuid='GPU-f77a1700-de70-05d2-048f-18c8f9aa5974'
 line=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip()
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
 foreign=[];own=[]
 for item in apps.splitlines():
  if uuid not in item:continue
  pid=int(item.split(',')[0]);is_own=False
  if own_pgid:
   try:is_own=os.getpgid(pid)==own_pgid
   except ProcessLookupError:pass
  (own if is_own else foreign).append(dict(pid=pid,record=item))
 fields=[v.strip() for v in line.split(',')]
 if fields[0]!=uuid or fields[1]!='NVIDIA A40':raise RuntimeError('wrong registered GPU identity')
 return dict(at=now(),raw=line,used_mib=int(fields[2].split()[0]),free_mib=int(fields[3].split()[0]),foreign=foreign,own=own)

def main():
 if ROOT.resolve()!=Path('/nfsdata_a40/cyf/codes/CT/cmeee_ner'):raise ValueError('registered remote root required')
 study=json.loads((ROOT/'configs/stage6-supplement.json').read_text());contract=json.loads((ROOT/'configs/qha-stage6-supplement-contract.json').read_text())
 if digest(ROOT/'configs/stage6-supplement.json')!=contract['study_sha256']:raise ValueError('study differs from explicit contract')
 E=ROOT/study['evidence_root'];E.mkdir(parents=True,exist_ok=True);ledger_path=E/'gpu-queue.json'
 with ledger_path.open('x') as f:f.write('{}')
 ledger=dict(status='RUNNING',pid=os.getpid(),started_at=now(),gpu_uuid=study['gpu_uuid'],jobs=[],observations=[],API_calls=0,sealed_read=False,continuous_run_budget=84600)
 def save():dump(ledger_path,ledger)
 save()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=study['gpu_uuid'],TMPDIR=str(ROOT/'.tmp'),XDG_CACHE_HOME=str(ROOT/'.cache'),HF_HOME=str(ROOT/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
 try:
  for job in study['training_jobs']:
   output=ROOT/study['run_root']/job['name'];resume=None;segments=[]
   job_record=dict(**job,status='WAITING',segments=segments);ledger['jobs'].append(job_record);save()
   while True:
    while True:
     obs=gpu_observation();ledger['observations'].append(obs);ledger['heartbeat_at']=now();save()
     if not obs['foreign'] and obs['used_mib']<=128 and obs['free_mib']>=12000:break
     ledger['status']='WAITING_RESOURCE';job_record['status']='WAITING_RESOURCE';save();time.sleep(60)
    if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('less than20GiB filesystem headroom')
    project_bytes=int(subprocess.check_output(['du','-sb',str(ROOT)],text=True).split()[0])
    if project_bytes>50*1024**3:raise RuntimeError('project larger than50GiB')
    # Fresh full resource observation immediately before launch; text retained verbatim.
    probe_cmds=[['nvidia-smi','-q','-i',study['gpu_uuid']],['nvidia-smi','topo','-m'],['df','-B1',str(ROOT)]]
    preflight=dict(at=now(),project_bytes=project_bytes,observations=[dict(command=c,output=subprocess.check_output(c,text=True)) for c in probe_cmds],last_gate=gpu_observation())
    if preflight['last_gate']['foreign'] or preflight['last_gate']['used_mib']>128:continue
    segment_id=len(segments);dump(E/(job['name']+f'-prelaunch-{segment_id:02d}.json'),preflight)
    cmd=[sys.executable,str(ROOT/'scripts/train_supplement.py'),'--job',job['name']]
    if resume:cmd+=['--resume',str(resume)]
    log_path=E/(job['name']+f'-segment-{segment_id:02d}.log');clock=time.perf_counter()
    with log_path.open('x') as log:
     child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     segment=dict(id=segment_id,pid=child.pid,command=cmd,started_at=now(),status='RUNNING',console=str(log_path.relative_to(ROOT)),resume=str(resume.relative_to(ROOT)) if resume else None)
     segments.append(segment);job_record['status']='RUNNING';ledger['status']='RUNNING';save();stop_reason=None
     while child.poll() is None:
      time.sleep(20)
      if child.poll() is not None:break
      obs=gpu_observation(child.pid);ledger['observations'].append(obs);ledger['heartbeat_at']=now()
      if obs['foreign']:stop_reason='foreign GPU process;own training gracefully stopped'
      if time.perf_counter()-clock>=study['max_segment_seconds']-180:stop_reason='continuous segment budget'
      if stop_reason:
       segment['stop_reason']=stop_reason;child.send_signal(signal.SIGTERM);save()
       try:child.wait(timeout=120)
       except subprocess.TimeoutExpired:
        # Only this exact child session is owned; never signal observed foreign PIDs.
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
       break
      save()
     code=child.wait()
    segment.update(returncode=code,finished_at=now(),seconds=time.perf_counter()-clock,status='EXITED');save()
    summary_path=output/'summary.json'
    if code or not summary_path.exists():raise RuntimeError('training subprocess failed;log retained: '+job['name'])
    summary=json.loads(summary_path.read_text());segment['runner_status']=summary['status'];segment['completed_epochs']=summary['epochs'];save()
    if summary['status'] in ['completed','early_stopped']:
     job_record.update(status='COMPLETED',best_epoch=summary['best_epoch'],best_f1=summary['best_f1'],epochs=summary['epochs'],seconds=sum(s['seconds'] for s in segments));save();break
    if summary['status']!='segment_limit':raise RuntimeError('unexpected training status')
    archive=output/'segments'/f'segment-{segment_id:02d}';archive.mkdir(parents=True,exist_ok=False)
    for name in ['summary.json','metadata.json','progress.json','config.json','metrics.jsonl','restore-check.json']:
     if (output/name).is_file():shutil.copy2(output/name,archive/name)
    last=output/'last.pt'
    if not last.exists():raise RuntimeError('safe stop before first checkpoint;manual restart required')
    last_step=summary['records'][-1]['global_step'] if summary['records'] else 0
    lines=[json.loads(s) for s in (output/'metrics.jsonl').read_text().splitlines() if s]
    discarded=[r for r in lines if r.get('event')=='train' and r.get('global_step',0)>last_step]
    segment.update(discarded_partial_updates=len(discarded),discarded_partial_update_seconds=sum(r.get('seconds',0) for r in discarded),archive=str(archive.relative_to(ROOT)),resume_checkpoint_sha256=digest(last));save()
    if stop_reason=='continuous segment budget':raise RuntimeError('segment budget reached;explicit continuation needed')
    resume=last
  ledger.update(status='COMPLETED',finished_at=now(),post_run_gpu=gpu_observation());save()
 except BaseException as ex:
  ledger.update(status='FAILED',error_type=type(ex).__name__,error=str(ex),finished_at=now());save();raise
 print(json.dumps(dict(status=ledger['status'],jobs=[{k:v for k,v in j.items() if k!='segments'} for j in ledger['jobs']]),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
