"""本项目GPU7顺序队列：占用即等待，不干预外部进程。"""
import argparse,json,os,signal,subprocess,sys,time,shutil
from datetime import datetime,timezone
from pathlib import Path
from ner_core import ROOT,dump,digest


def now():return datetime.now(timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['diagnose','develop']);a=p.parse_args()
    cfg=json.loads((ROOT/'configs/stage3.json').read_text());base=ROOT/'runs/stage3-20261019'
    ledger_path=ROOT/'evidence/stage3'/('queue-'+a.mode+'.json')
    if ledger_path.exists():raise ValueError('队列记录已存在，禁止重复启动覆盖')
    ledger=dict(started_at=now(),mode=a.mode,status='running',segments=[],gpu_uuid=cfg['gpu_uuid'],observations=[])
    dump(ledger_path,ledger)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],TMPDIR=str(ROOT/'.tmp'),XDG_CACHE_HOME=str(ROOT/'.cache'),
         OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def run(name,cmd,limit):
        while True:
            gpu=subprocess.check_output(['nvidia-smi','-i',cfg['gpu_uuid'],'--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip()
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
            foreign=any(cfg['gpu_uuid'] in line for line in apps.splitlines())
            used=int(gpu.split(',')[1].split()[0]);busy=foreign or used>128
            ledger['observations'].append(dict(at=now(),gpu=gpu,busy=busy));dump(ledger_path,ledger)
            if not busy:break
            print('GPU7 busy; waiting60s',flush=True);time.sleep(60)
        if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('磁盘余量不足')
        project_size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        if project_size>50*1024**3:raise RuntimeError('项目超过50GiB')
        start=time.perf_counter();segment=dict(name=name,started_at=now(),command=cmd,pid=None,status='running')
        ledger['segments'].append(segment);dump(ledger_path,ledger)
        console=ROOT/'evidence/stage3'/(name+'.log')
        with console.open('x') as f:
            child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            segment['pid']=child.pid;dump(ledger_path,ledger)
            try:code=child.wait(timeout=limit-20)
            except subprocess.TimeoutExpired:
                child.send_signal(signal.SIGTERM)
                try:code=child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();code=child.wait(timeout=5)
        segment.update(finished_at=now(),seconds=time.perf_counter()-start,returncode=code,status='completed' if code==0 else 'failed')
        dump(ledger_path,ledger);print(json.dumps(segment),flush=True)
        if code:raise RuntimeError('子任务失败:'+name)
    try:
        if a.mode=='diagnose':
            run('n1-train-only-overfit',[sys.executable,str(ROOT/'scripts/diagnose_n1.py'),'overfit','--output',str(base/'n1-overfit')],300)
            if json.loads((base/'n1-overfit/diagnosis.json').read_text())['status']!='PASS':raise RuntimeError('诊断未通过')
        else:
            if json.loads((base/'n1-overfit/diagnosis.json').read_text())['status']!='PASS':raise RuntimeError('未通过先行诊断')
            def train(name,method,lr,rope=True):
                cmd=[sys.executable,str(ROOT/'scripts/train_development.py'),'--method',method,'--encoder-lr',str(lr),'--output',str(base/name)]
                if not rope:cmd+=['--no-rope']
                run(name,cmd,cfg['max_segment_seconds'])
                s=json.loads((base/name/'summary.json').read_text())
                if s['status'] not in ('completed','early_stopped'):raise RuntimeError('训练段未完整结束:'+name)
                return s
            train('N0','N0',cfg['n1_encoder_lr'])
            candidates=[]
            for lr in cfg['learning_rates']:
                name='N1-lr-'+format(lr,'.0e');summary=train(name,'N1',lr)
                candidates.append(dict(name=name,lr=lr,f1=summary['best_f1'],best_epoch=summary['best_epoch'],epochs=summary['epochs']))
            selected=max(candidates,key=lambda c:(c['f1'],-c['lr']))
            dump(base/'lr-selection.json',dict(candidates=candidates,selected=selected,selection_split='internal dev only',tie='lower LR',final_holdout_accessed=False))
            train('N1-rope-off','N1',selected['lr'],False)
        ledger.update(status='completed',finished_at=now())
    except Exception as e:
        ledger.update(status='failed',error=repr(e),finished_at=now());raise
    finally:dump(ledger_path,ledger)

if __name__=='__main__':main()
