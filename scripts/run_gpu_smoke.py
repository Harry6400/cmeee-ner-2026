"""只调度本项目GPU7，所有GPU子进程累计墙钟预算600秒。"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from ner_core import ROOT,dump,digest


def timestamp():return datetime.now(timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);args=p.parse_args()
    run=args.run.resolve()
    if not run.is_relative_to(ROOT.resolve()):raise ValueError('输出必须在本项目')
    cfg=json.loads((ROOT/'configs/stage2.json').read_text())
    run.mkdir(parents=True,exist_ok=False)
    ledger=dict(started_at=timestamp(),budget_seconds=cfg['gpu_total_seconds'],
                consumed_seconds=0.0,segments=[],queue_observations=[],
                accounting='conservative subprocess wall time includes import/load/save and release; no overlapping GPU processes')
    dump(run/'gpu-budget.json',ledger)
    if __import__('shutil').disk_usage(ROOT).free < 20*1024**3:raise RuntimeError('磁盘空间不足')
    project_bytes=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    if project_bytes > 50*1024**3:raise RuntimeError('项目超过50GiB')
    # 权重必须与官方固定下载清单一致，失败则在CUDA初始化之前停止。
    artifact_manifest=json.loads((ROOT/cfg['bert_directory']/'download-manifest.json').read_text())
    if artifact_manifest['revision']!=cfg['bert_revision']:raise ValueError('BERT版本不一致')
    for artifact in artifact_manifest['files']:
        if digest(ROOT/cfg['bert_directory']/artifact['file'])!=artifact['sha256']:
            raise ValueError('BERT文件校验失败:'+artifact['file'])
    dump(run/'model-integrity.json',dict(status='PASS',revision=cfg['bert_revision'],files=len(artifact_manifest['files'])))
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],OMP_NUM_THREADS=str(cfg['cpu_threads']),
               MKL_NUM_THREADS=str(cfg['cpu_threads']),TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
               TMPDIR=str(ROOT/'.tmp'),XDG_CACHE_HOME=str(ROOT/'.cache'))
    for method in ('N0','N1'):
        for resume in (False,True):
            remaining=cfg['gpu_total_seconds']-ledger['consumed_seconds']
            if remaining < 45:
                ledger['status']='budget_exhausted';dump(run/'gpu-budget.json',ledger);return 2
            # 每个新进程都再次等待与核查，其他卡上的进程不影响本卡资格。
            while True:
                gpu=subprocess.check_output(['nvidia-smi','-i',cfg['gpu_uuid'],
                    '--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip()
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
                fields=[x.strip() for x in gpu.split(',')]
                busy=any(cfg['gpu_uuid'] in line for line in apps.splitlines()) or int(fields[2].split()[0])>128
                observation=dict(at=timestamp(),gpu=gpu,busy=busy)
                ledger['queue_observations'].append(observation);dump(run/'gpu-budget.json',ledger)
                print(json.dumps(observation),flush=True)
                if not busy:break
                time.sleep(60)
            name=method+('-resume' if resume else '-initial')
            target=cfg['initial_steps']+(cfg['resume_steps'] if resume else 0)
            command=[sys.executable,str(ROOT/'scripts/ner.py'),'train','--method',method,
                     '--subset',str(ROOT/'runs/stage2-subset'),'--output',str(run/name),
                     '--target-step',str(target),'--max-seconds',str(remaining-8)]
            if resume:command+=['--resume',str(run/(method+'-initial')/'last.pt')]
            start=time.perf_counter();started=timestamp()
            with (run/(name+'.log')).open('x') as console:
                child=subprocess.Popen(command,stdout=console,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                try:
                    result=child.wait(timeout=max(1,remaining-15))
                except subprocess.TimeoutExpired:
                    # 只对自己创建的PID发信号，给保存和退出保留15秒。
                    child.send_signal(signal.SIGTERM)
                    try:result=child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill();result=child.wait(timeout=5)
            elapsed=time.perf_counter()-start
            ledger['consumed_seconds']+=elapsed
            segment=dict(name=name,started_at=started,finished_at=timestamp(),seconds=elapsed,returncode=result)
            ledger['segments'].append(segment);dump(run/'gpu-budget.json',ledger)
            print(json.dumps(segment),flush=True)
            summary=run/name/'summary.json'
            if result or not summary.exists() or json.loads(summary.read_text())['global_step']!=target:
                ledger['status']='failed_or_incomplete';dump(run/'gpu-budget.json',ledger);return 1
    ledger['status']='completed';ledger['finished_at']=timestamp();dump(run/'gpu-budget.json',ledger)
    return 0

if __name__=='__main__':raise SystemExit(main())
