"""Wait for four authorized new-seed runs, then frozen six-checkpoint final evaluation onGPU7."""
import argparse,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
from ner_core import ROOT,dump,digest
from stage4_queue import observe,checked_path,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,default=ROOT/'configs/qha-stage4-contract.json');p.add_argument('--protocol',type=Path,default=ROOT/'configs/stage4-frozen-protocol.json');p.add_argument('--training-journal',type=Path,default=ROOT/'evidence/stage4/training-queue.json');p.add_argument('--output',type=Path,default=ROOT/'runs/stage4-20261020/local-evaluation');p.add_argument('--journal',type=Path,default=ROOT/'evidence/stage4/local-evaluation-queue.json');a=p.parse_args()
    for path in [a.contract,a.protocol,a.training_journal,a.output,a.journal]:checked_path(path)
    contract=json.loads(a.contract.read_text());protocol=json.loads(a.protocol.read_text());uuid=contract['gpu_uuid'];limit=84600
    if contract['stage']!=4 or contract['authorization']!='explicit_user_stage4_2026-10-20' or not contract['final_holdout_authorized'] or contract['frozen_protocol_sha256']!=digest(a.protocol):raise ValueError('contract/frozen protocol mismatch')
    if a.journal.exists():raise ValueError('journal exists; no overwrite')
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    a.output.mkdir(parents=True,exist_ok=True);a.journal.parent.mkdir(parents=True,exist_ok=True)
    q=dict(status='waiting_for_training',controller_pid=os.getpid(),started_at=now(),contract_sha256=digest(a.contract),protocol_sha256=digest(a.protocol),gpu_uuid=uuid,runs=[],training_wait_seconds=0,source_sha256={n:digest(ROOT/'scripts'/n) for n in ['local_evaluation_queue_stage4.py','local_evaluate_stage4.py','stage4_queue.py','ner_core.py','ner_model.py','stage3_common.py']});dump(a.journal,q);active=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,TMPDIR=str(ROOT/'.tmp'),XDG_CACHE_HOME=str(ROOT/'.cache'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        t=time.perf_counter()
        while True:
            if stop[0]:q['status']='user_stopped';return
            if a.training_journal.exists():
                training=json.loads(a.training_journal.read_text())
                if training['status']=='completed':break
                if training['status'] in ['failed','user_stopped']:raise RuntimeError('training queue not complete:'+training['status'])
            time.sleep(60)
        q['training_wait_seconds']=time.perf_counter()-t;q['training_journal_sha256']=digest(a.training_journal);q['status']='evaluating';dump(a.journal,q)
        selections=[]
        for method in ['N0','N1']:
            for seed in [20261019,20261020,20261021]:
                name=f'{method}-seed{seed}';run=ROOT/('runs/stage3-20261019/'+protocol['selected'][method]['run']) if seed==20261019 else ROOT/'runs/stage4-20261020'/name
                summary=json.loads((run/'summary.json').read_text());checkpoint=run/'best.pt'
                if summary['status'] not in ['completed','early_stopped'] or summary['method']!=method:raise ValueError('checkpoint run not complete')
                expected=protocol['selected'][method]['best_epoch'] if seed==20261019 else summary['best_epoch']
                if expected!=summary['best_epoch'] or abs(max(rec['val']['micro']['f1'] for rec in summary['records'])-summary['best_f1'])>1e-12:raise ValueError('checkpoint not own devbest')
                selections.append(dict(name=name,method=method,seed=seed,training_run=str(run.relative_to(ROOT)),best_epoch=expected,best_dev_f1=summary['best_f1'],checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest(checkpoint),training_summary_sha256=digest(run/'summary.json')))
        dump(a.output/'selected-checkpoints.json',dict(frozen_before_final_predictions_at=now(),selection='own internal dev best per each seed; never bestseed onholdout',selections=selections,contract_sha256=q['contract_sha256'],protocol_sha256=q['protocol_sha256']))
        for selected in selections:
            if stop[0]:q['status']='user_stopped';return
            name=selected['name'];out=a.output/name;entry=dict(name=name,status='waiting',observations=[],runtime_observations=[],segments=[],wait_seconds=0);q['runs'].append(entry);number=0
            if out.exists():raise ValueError('output exists:'+str(out))
            while True:
                t=time.perf_counter()
                while True:
                    if stop[0]:q['status']='user_stopped';return
                    obs=observe(uuid);obs['busy']=bool(obs['compute_processes']) or obs['memory_used_mb']>128 or obs['memory_free_mb']<12000;entry['observations'].append(obs);dump(a.journal,q)
                    if not obs['busy']:break
                    print(now()+' '+name+' GPU7 busy; waiting60s',flush=True);time.sleep(60)
                entry['wait_seconds']+=time.perf_counter()-t
                if shutil.disk_usage(ROOT).free<20*1024**3:raise ValueError('disk free below20GiB')
                if sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file())>50*1024**3:raise ValueError('project exceeds50GiB')
                if digest(a.contract)!=q['contract_sha256'] or digest(a.protocol)!=q['protocol_sha256']:raise ValueError('contract/protocol changed')
                if digest(ROOT/selected['checkpoint'])!=selected['checkpoint_sha256']:raise ValueError('selected checkpoint changed')
                for n,want in q['source_sha256'].items():
                    if digest(ROOT/'scripts'/n)!=want:raise ValueError('evaluation source changed:'+n)
                obs=observe(uuid);obs['busy']=bool(obs['compute_processes']) or obs['memory_used_mb']>128 or obs['memory_free_mb']<12000;obs['phase']='immediate_prelaunch';entry['observations'].append(obs);dump(a.journal,q)
                if obs['busy']:t=time.perf_counter();time.sleep(60);entry['wait_seconds']+=time.perf_counter()-t;continue
                cmd=[sys.executable,str(ROOT/'scripts/local_evaluate_stage4.py'),'--contract',str(a.contract),'--protocol',str(a.protocol),'--method',selected['method'],'--seed',str(selected['seed']),'--checkpoint',str(ROOT/selected['checkpoint']),'--output',str(out)]
                console=a.journal.parent/f'evaluate-{name}-segment-{number:02d}.log';seg=dict(segment=number,started_at=now(),console=str(console.relative_to(ROOT)),command=cmd,status='running');entry['segments'].append(seg);entry['status']='evaluating';t=time.perf_counter()
                with console.open('x') as f:
                    active=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);seg['pid']=active.pid;dump(a.journal,q)
                    while active.poll() is None:
                        time.sleep(10)
                        if active.poll() is not None:break
                        obs=observe(uuid);foreign=[x for x in obs['compute_processes'] if int(x.split(',')[0])!=active.pid];obs['foreign']=foreign;entry['runtime_observations'].append(obs);dump(a.journal,q)
                        reason='user_stop' if stop[0] else 'foreign_compute_process' if foreign else 'segment_time_cap' if time.perf_counter()-t>limit-120 else None
                        if reason:
                            seg['interruption_reason']=reason;active.send_signal(signal.SIGTERM)
                            try:active.wait(timeout=120)
                            except subprocess.TimeoutExpired:seg['forced_own_child_termination']=True;active.kill();active.wait(timeout=10)
                            break
                    code=active.wait();active=None
                seg.update(returncode=code,finished_at=now(),wall_seconds=time.perf_counter()-t)
                finished=code==0 and (out/'summary.json').exists() and json.loads((out/'summary.json').read_text())['status']=='completed';seg['status']='completed' if finished else 'interrupted' if 'interruption_reason' in seg else 'failed'
                if finished:entry.update(status='completed',summary=str((out/'summary.json').relative_to(ROOT)));dump(a.journal,q);break
                if seg['status']=='failed' and any(msg in console.read_text() for msg in ['GPU7存在计算进程；等待，不抢卡','GPU7显存状态不满足空闲启动门槛']):seg['status']='prelaunch_busy_race';number+=1;dump(a.journal,q);continue
                if seg['status']=='failed':raise RuntimeError('evaluation child failed; see '+str(console))
                if out.exists():
                    archived=a.output/'interrupted-segments'/f'{name}-segment-{number:02d}';archived.parent.mkdir(parents=True,exist_ok=True);out.rename(archived);seg['archived_incomplete_output']=str(archived.relative_to(ROOT))
                number+=1;dump(a.journal,q)
                if stop[0]:q['status']='user_stopped';return
        entries=[json.loads(p.read_text()) for selected in selections for p in (a.output/selected['name']).glob('*/manifest-entry.json')]
        if (a.output/'D0/summary.json').exists():entries+=[json.loads(p.read_text()) for p in (a.output/'D0').glob('*/manifest-entry.json')]
        dump(a.output/'manifest.json',dict(status='completed_local_models',entries=entries,common500_not_independent_test=True,created_at=now(),protocol_sha256=q['protocol_sha256'],contract_sha256=q['contract_sha256']))
        q.update(status='completed',finished_at=now());print(json.dumps(dict(status='completed',entries=len(entries))),flush=True)
    except Exception as exc:q.update(status='failed',error=repr(exc),finished_at=now());raise
    finally:
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGTERM)
            try:active.wait(timeout=120)
            except subprocess.TimeoutExpired:active.kill();active.wait(timeout=10)
            q['own_child_cleanup_after_controller_error']=dict(pid=active.pid,returncode=active.returncode,at=now())
        dump(a.journal,q)
if __name__=='__main__':main()
