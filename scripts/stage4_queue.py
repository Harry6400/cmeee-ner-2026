"""GPU7 sequential stage4 queue. Independent logs/checkpoints per interrupted segment; no automatic API calls."""
import argparse,json,os,signal,subprocess,sys,time,shutil
from datetime import datetime,timezone
from pathlib import Path
from ner_core import ROOT,dump,digest

def now():return datetime.now(timezone.utc).isoformat()
def observe(uuid):
    raw=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
    fields=[x.strip() for x in raw.split(',')]
    if fields[0]!='7' or fields[1]!=uuid or 'A40' not in fields[2]:raise RuntimeError('GPU index/UUID/model mismatch')
    return dict(at=now(),gpu=raw,memory_used_mb=int(fields[3].split()[0]),memory_free_mb=int(fields[4].split()[0]),utilization_percent=int(fields[5].split()[0]),compute_processes=[x.strip() for x in apps.splitlines() if uuid in x])
def checked_path(path):
    p=Path(path).resolve();root=ROOT.resolve()
    if root!=Path('/nfsdata_a40/cyf/codes/CT/cmeee_ner') or not p.is_relative_to(root):raise RuntimeError('remote path isolation failed')
    return p

def archive(run,number):
    """Archive ended segment. Hard links preserve checkpoints before next atomic replacement."""
    dest=run/'segments'/f'segment-{number:02d}';dest.mkdir(parents=True,exist_ok=False)
    for name in ['config.json','metadata.json','summary.json','failure.json','progress.json','restore-check.json']:
        p=run/name
        if p.exists():shutil.copy2(p,dest/name)
    for name in ['metrics.jsonl','tensorboard']:
        p=run/name
        if p.exists():shutil.move(str(p),dest/name)
    for name in ['best.pt','last.pt']:
        p=run/name
        if p.exists():os.link(p,dest/name)
    step=0
    if (dest/'last.pt').exists():
        # CPU-only metadata read after child process is gone. Never initializes CUDA.
        import torch
        state=torch.load(dest/'last.pt',map_location='cpu',weights_only=False);step=state['global_step']
    updates=[]
    if (dest/'metrics.jsonl').exists():
        updates=[json.loads(x) for x in (dest/'metrics.jsonl').read_text().splitlines() if x]
        updates=[x for x in updates if x.get('event')=='train' and x['global_step']>step]
    result=dict(archive_dir=str(dest.relative_to(ROOT)),last_complete_global_step=step,discarded_partial_updates=len(updates),discarded_partial_update_seconds=sum(x['seconds'] for x in updates),last_checkpoint_sha256=digest(dest/'last.pt') if (dest/'last.pt').exists() else None)
    dump(dest/'archive.json',result);return dest/'last.pt' if (dest/'last.pt').exists() else None,result

def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,default=ROOT/'configs/qha-stage4-contract.json');p.add_argument('--protocol',type=Path,default=ROOT/'configs/stage4-frozen-protocol.json');p.add_argument('--output',type=Path,default=ROOT/'runs/stage4-20261020');p.add_argument('--journal',type=Path,default=ROOT/'evidence/stage4/training-queue.json');a=p.parse_args()
    checked_path(a.output);checked_path(a.journal);checked_path(a.protocol);checked_path(a.contract)
    contract=json.loads(a.contract.read_text())
    if contract['stage']!=4 or contract['authorization']!='explicit_user_stage4_2026-10-20' or contract['frozen_protocol_sha256']!=digest(a.protocol):raise ValueError('stage4 contract/protocol mismatch')
    if a.output.resolve()!=(ROOT/contract['run_root']).resolve():raise ValueError('runroot differs from contract')
    protocol=json.loads(a.protocol.read_text());cfg=protocol['selected']['N1']['config'];uuid=cfg['gpu_uuid'];limit=cfg['max_segment_seconds'];assert limit<=84600
    if uuid!=contract['gpu_uuid']:raise ValueError('GPU differs from contract')
    if a.journal.exists():raise RuntimeError('Existing queue journal; use a new journal, never overwrite lifecycle')
    a.output.mkdir(parents=True,exist_ok=True);a.journal.parent.mkdir(parents=True,exist_ok=True)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    q=dict(status='running',controller_pid=os.getpid(),started_at=now(),protocol_sha256=digest(a.protocol),contract_sha256=digest(a.contract),authorized_gpu_uuid=uuid,runs=[],source_sha256={n:digest(ROOT/'scripts'/n) for n in ['train_stage4.py','stage4_queue.py','ner_core.py','ner_model.py','ner.py','stage3_common.py']},precision='fp32; AMP/TF32 disabled',runtime='frozen baseline; micro4 accumulate8',api_calls=0,holdout_used_for_selection=False)
    dump(a.journal,q)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,TMPDIR=str(ROOT/'.tmp'),XDG_CACHE_HOME=str(ROOT/'.cache'),HF_HOME=str(ROOT/'.cache/huggingface'),TORCH_HOME=str(ROOT/'.cache/torch'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key in ['TMPDIR','XDG_CACHE_HOME','HF_HOME','TORCH_HOME']:checked_path(env[key]);Path(env[key]).mkdir(parents=True,exist_ok=True)
    try:
        for method in ['N1','N0']:
            for seed in [20261020,20261021]:
                if stop[0]:q['status']='user_stopped';return
                name=f'{method}-seed{seed}';run=a.output/name;entry=dict(name=name,method=method,seed=seed,status='waiting',segments=[],observations=[],runtime_observations=[],wait_seconds=0);q['runs'].append(entry);dump(a.journal,q)
                parent=None;number=0
                if run.exists():raise RuntimeError('run exists; explicit recovery journal required:'+name)
                while True:
                    wait=time.perf_counter();entry['status']='waiting';dump(a.journal,q)
                    while True:
                        if stop[0]:q['status']='user_stopped';return
                        obs=observe(uuid);obs['busy']=bool(obs['compute_processes']) or obs['memory_used_mb']>128 or obs['memory_free_mb']<12000
                        entry['observations'].append(obs);dump(a.journal,q)
                        if not obs['busy']:break
                        print(now()+' '+name+' GPU7 busy; waiting60s',flush=True);time.sleep(60)
                    entry['wait_seconds']+=time.perf_counter()-wait
                    if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('disk free below20GiB')
                    if sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file())>50*1024**3:raise RuntimeError('project exceeds50GiB')
                    if digest(a.contract)!=q['contract_sha256']:raise RuntimeError('contract changed during queue')
                    if digest(a.protocol)!=q['protocol_sha256']:raise RuntimeError('protocol changed during queue')
                    for n,h in q['source_sha256'].items():
                        if digest(ROOT/'scripts'/n)!=h:raise RuntimeError('source changed during queue:'+n)
                    # Recheck after expensive disk and source checks, immediately before spawning.
                    obs=observe(uuid);obs['busy']=bool(obs['compute_processes']) or obs['memory_used_mb']>128 or obs['memory_free_mb']<12000;obs['phase']='immediate_prelaunch';entry['observations'].append(obs);dump(a.journal,q)
                    if obs['busy']:
                        waited=time.perf_counter();time.sleep(60);entry['wait_seconds']+=time.perf_counter()-waited;continue
                    cmd=[sys.executable,str(ROOT/'scripts/train_stage4.py'),'--method',method,'--seed',str(seed),'--protocol',str(a.protocol),'--contract',str(a.contract),'--output',str(run)]
                    if parent:cmd+=['--resume',str(parent)]
                    console=a.journal.parent/f'{name}-segment-{number:02d}.log';start=time.perf_counter();segment=dict(segment=number,started_at=now(),planned_stop_at_seconds=limit,command=cmd,console=str(console.relative_to(ROOT)),parent_checkpoint=str(parent.relative_to(ROOT)) if parent else None,status='running');entry['segments'].append(segment)
                    with console.open('x') as f:
                        child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);segment['pid']=child.pid;entry['status']='training';dump(a.journal,q)
                        while child.poll() is None:
                            time.sleep(10)
                            if child.poll() is not None:break
                            obs=observe(uuid);foreign=[x for x in obs['compute_processes'] if int(x.split(',')[0])!=child.pid];obs['foreign']=foreign;entry['runtime_observations'].append(obs);dump(a.journal,q)
                            reason='user_stop' if stop[0] else 'foreign_compute_process' if foreign else 'segment_time_cap' if time.perf_counter()-start>limit-120 else None
                            if reason:
                                segment['interruption_reason']=reason;child.send_signal(signal.SIGTERM)
                                try:child.wait(timeout=120)
                                except subprocess.TimeoutExpired:
                                    segment['forced_own_child_termination']=True;child.kill();child.wait(timeout=10)
                                break
                        code=child.wait()
                    segment.update(returncode=code,finished_at=now(),wall_seconds=time.perf_counter()-start)
                    summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else None
                    finished=code==0 and summary and summary['status'] in ('completed','early_stopped')
                    segment['status']='completed' if finished else 'interrupted' if 'interruption_reason' in segment else 'failed'
                    if finished:
                        entry.update(status='completed',summary=str((run/'summary.json').relative_to(ROOT)),best_checkpoint=str((run/'best.pt').relative_to(ROOT)),best_checkpoint_sha256=digest(run/'best.pt'),best_epoch=summary['best_epoch'],best_dev_f1=summary['best_f1'],epochs=summary['epochs'],finished_at=now());dump(a.journal,q);print(json.dumps(entry,ensure_ascii=False),flush=True);break
                    if segment['status']=='failed':
                        raw=console.read_text()
                        if any(msg in raw for msg in ['GPU7存在计算进程；等待，不抢卡','GPU7显存状态不满足空闲启动门槛']):
                            segment['status']='prelaunch_busy_race';segment['discarded_partial_updates']=0;number+=1;dump(a.journal,q);continue
                        raise RuntimeError('training child failed:'+name+';see '+str(console))
                    parent,arch=archive(run,number);segment.update(arch);number+=1;dump(a.journal,q)
                    if parent is None:
                        for file in run.iterdir():
                            if file.name!='segments':
                                if file.is_dir():raise RuntimeError('unexpected nonarchived run subdirectory')
                                file.unlink()
                    if stop[0]:entry['status']='user_stopped';q['status']='user_stopped';return
                    # Occupancy only. Immediate next poll; wait interval remains60s if the foreign process is still present.
        q.update(status='completed',finished_at=now());print(json.dumps(dict(status='completed',runs=[{k:x[k] for k in ['name','best_dev_f1','best_epoch','epochs']} for x in q['runs']])),flush=True)
    except Exception as exc:q.update(status='failed',error=repr(exc),finished_at=now());raise
    finally:dump(a.journal,q)

if __name__=='__main__':main()
