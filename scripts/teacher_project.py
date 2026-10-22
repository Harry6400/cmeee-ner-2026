"""Portable teacher UI services; lazy CPU assets and isolated reproduction jobs."""
from __future__ import annotations
import json, os, signal, subprocess, sys, threading, time, uuid
import owned_process
from datetime import datetime, timezone
from pathlib import Path


def now(): return datetime.now(timezone.utc).isoformat()
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


class BusyError(RuntimeError): pass
class AssetError(RuntimeError): pass


class ReproductionJobs:
    """Only child process groups created here may be cancelled."""
    def __init__(self, root, cpu_lock, popen=None):
        self.root=Path(root); self.cpu_lock=cpu_lock; self.popen=popen or subprocess.Popen
        self.directory=self.root/'output/teacher-reproduction'; self.metadata=self.directory/'.jobs'
        self.lock=threading.RLock(); self.jobs={}; self.processes={}; self.threads={}; self.active=None; self.closed=False
        self.guard=self.directory/'.active.lock'
        if self.metadata.is_dir():
            for p in sorted(self.metadata.glob('*.json')):
                try:
                    job=read_json(p)
                    if job.get('id')!=p.stem: continue
                    if job.get('status')=='running':
                        job.update(status='failed',stage='interrupted',error='服务曾中断；保留部分输出，本次未通过。',finished_at=now())
                        self._save(job)
                    self.jobs[job['id']]=job
                except (OSError,ValueError,KeyError): continue
    def _save(self, job):
        self.metadata.mkdir(parents=True,exist_ok=True)
        dest=self.metadata/(job['id']+'.json'); tmp=dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(job,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); tmp.replace(dest)
    def _acquire_guard(self, job):
        self.directory.mkdir(parents=True,exist_ok=True)
        try: fd=os.open(self.guard,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError: raise BusyError('完整复现保护锁已存在；可能仍有旧任务运行。请核查该任务及output/teacher-reproduction/.active.lock，未核查前禁止再次启动。') from None
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as handle: json.dump(dict(owner_token=job['owner_token'],job_id=job['id'],controller_pid=os.getpid(),started_at=job['started_at']),handle)
        except Exception:
            # No child has been spawned and this exact create succeeded, so this instance owns this incomplete file.
            self.guard.unlink(missing_ok=True); raise
    def _release_guard(self, job):
        try:
            guard=read_json(self.guard)
            if guard.get('owner_token')==job['owner_token']: self.guard.unlink()
        except (OSError,ValueError,AttributeError): pass
    @staticmethod
    def _group_gone(process):
        return owned_process.gone(process)
    def _public(self, job):
        value=dict(job)
        if value['status']=='running': value['seconds']=max(0,time.time()-value['started_unix'])
        value.pop('started_unix',None); return value
    def list(self):
        with self.lock:
            jobs=[self._public(j) for j in sorted(self.jobs.values(),key=lambda x:x['started_at'],reverse=True)]
            return {'jobs':jobs,'active':self._public(self.jobs[self.active]) if self.active else None,'protection':dict(locked=self.guard.exists(),path='output/teacher-reproduction/.active.lock',message='保护锁存在时禁止重启第二个全量任务；遗留锁需人工核查。')}
    def get(self, ident):
        with self.lock:
            if ident not in self.jobs: raise KeyError('复现任务不存在')
            return self._public(self.jobs[ident])
    def start(self):
        with self.lock:
            if self.closed: raise BusyError('服务正在关闭')
            if self.active or not self.cpu_lock.acquire(blocking=False): raise BusyError('已有本地CPU任务运行，请稍后重试。')
            ident=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:10]
            relative=Path('output/teacher-reproduction')/ident
            job=dict(id=ident,owner_token=uuid.uuid4().hex,controller_pid=os.getpid(),lock='output/teacher-reproduction/.active.lock',status='running',output=relative.as_posix(),started_at=now(),started_unix=time.time(),finished_at=None,seconds=0,stage='starting',progress=None,log='',summary=None,error=None,exit_code=None,cancel_requested=False,stdout=(Path('output/teacher-reproduction/.jobs')/(ident+'.stdout.log')).as_posix())
            try: self._acquire_guard(job)
            except Exception: self.cpu_lock.release(); raise
            self.jobs[ident]=job; self.active=ident
            try:
                self._save(job)
                thread=threading.Thread(target=self._run,args=(ident,),daemon=True,name='teacher-reproduction-'+ident)
                self.threads[ident]=thread; thread.start()
            except Exception:
                self.active=None; self.jobs.pop(ident,None); self.threads.pop(ident,None); self._release_guard(job); self.cpu_lock.release(); raise
            return self._public(job)
    def _consume(self, job, line):
        job['log']=(job['log']+line)[-30000:]
        try: event=json.loads(line)
        except ValueError: return
        if not isinstance(event,dict): return
        if event.get('event')=='progress':
            job['stage']=str(event.get('method',''))+' CPU推理'
            job['progress']={k:event[k] for k in ['method','completed_windows','total_windows','seconds'] if k in event}
        elif event.get('event')=='method_complete':
            job['stage']=str(event.get('method',''))+' 已完成'
            job['progress']={k:event[k] for k in ['method','f1','frozen_metric_equal','seconds'] if k in event}
        elif event.get('event'):
            job['stage']=str(event['event'])
    def _run(self, ident):
        job=self.jobs[ident]; output=self.root/job['output']; process=None
        try:
            # teacher_reproduce refuses existing output; store stdout outside its output until it creates that directory.
            stdout_path=self.metadata/(ident+'.stdout.log')
            command=[sys.executable,'-B','-u',str(self.root/'scripts/teacher_reproduce.py'),'--archive',str(self.root/'assets/CMeEE-V2.zip'),'--output',str(output)]
            with self.lock:
                if job['cancel_requested']: return
                process=owned_process.spawn(command,popen=self.popen,cwd=self.root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                self.processes[ident]=process; job['child_pid']=process.pid; job['child_pgid']=process.pid; job['command']=[str(x) for x in command]; self._save(job)
            with stdout_path.open('x',encoding='utf-8') as log:
                for line in process.stdout:
                    log.write(line); log.flush()
                    with self.lock: self._consume(job,line); self._save(job)
            code=process.wait()
            with self.lock: job['exit_code']=code
            if output.is_dir():
                import shutil
                shutil.copyfile(stdout_path,output/'job.stdout.log')
                with self.lock: job['stdout']=(Path(job['output'])/'job.stdout.log').as_posix()
            summary_path=output/'summary.json'
            summary=read_json(summary_path) if summary_path.is_file() else None
            with self.lock:
                job['summary']=summary
                if job['cancel_requested']: job.update(status='cancelled',stage='已取消',error='仅终止本任务；部分输出已保留。')
                elif code==0 and summary and summary.get('status')=='PASS': job.update(status='passed',stage='完整5000条CPU复现通过')
                else: job.update(status='failed',stage='复现未通过',error='执行失败或指标不一致，请检查保留的日志与输出。')
        except Exception as exc:
            with self.lock: job.update(status='cancelled' if job['cancel_requested'] else 'failed',stage='已取消' if job['cancel_requested'] else '执行失败',error='复现进程未完成（'+type(exc).__name__+'）；请检查日志。')
        finally:
            # A logging/metadata failure must not unlock while our child still runs.
            if process is not None and process.poll() is None:
                try: owned_process.terminate(process)
                except ProcessLookupError: pass
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try: owned_process.terminate(process,force=True)
                    except ProcessLookupError: pass
                    process.wait(timeout=5)
            if process is not None and process.stdout is not None: process.stdout.close()
            group_gone=self._group_gone(process)
            owned_process.release(process)
            with self.lock:
                if job['status']=='running': job.update(status='cancelled' if job['cancel_requested'] else 'failed',stage='已取消' if job['cancel_requested'] else '执行中断')
                job['finished_at']=now(); job['seconds']=time.time()-job['started_unix']; self.processes.pop(ident,None)
                if self.active==ident:self.active=None
                if not group_gone:
                    job.update(protection_retained=True,error='未能确认本任务全部子进程退出；已保留保护锁，禁止启动新CPU任务，请人工核查。')
                    if job['status']=='passed':job.update(status='failed',stage='子进程状态未核清')
                try: self._save(job)
                finally:
                    if group_gone:self._release_guard(job)
                    self.cpu_lock.release(); self.threads.pop(ident,None)
    def cancel(self, ident):
        with self.lock:
            if ident not in self.jobs: raise KeyError('复现任务不存在')
            job=self.jobs[ident]
            if job['status']!='running': return self._public(job)
            process=self.processes.get(ident)
            if process is not None and process.poll() is not None: return self._public(job)
            job['cancel_requested']=True; job['stage']='正在取消'; self._save(job)
            if process is not None:
                try: owned_process.terminate(process)
                except ProcessLookupError: pass
            return self._public(job)
    def close(self):
        with self.lock: self.closed=True; ident=self.active
        if ident:
            self.cancel(ident)
            with self.lock: process=self.processes.get(ident)
            if process:
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try: owned_process.terminate(process,force=True)
                    except ProcessLookupError: pass
                    process.wait(timeout=5)
            with self.lock: thread=self.threads.get(ident)
            if thread is not None and thread is not threading.current_thread(): thread.join(timeout=20)


class TeacherProject:
    def __init__(self,root,log_dir=None,models_factory=None,engine_factory=None,api_factory=None,popen=None):
        self.root=Path(root).resolve(); self.log_dir=Path(log_dir) if log_dir else self.root/'output/local-ui'
        self.log_dir.mkdir(parents=True,exist_ok=True); self.log_lock=threading.Lock(); self.cpu_lock=threading.Lock()
        self.models_factory=models_factory; self.engine_factory=engine_factory; self.api_factory=api_factory
        self.models=None; self.engine=None; self._api=None; self.api_lock=threading.Lock()
        self.jobs=ReproductionJobs(self.root,self.cpu_lock,popen)
    def record(self,**event):
        with self.log_lock:
            with (self.log_dir/'ui-server.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(dict(at=now(),**event),ensure_ascii=False)+'\n')
    def api(self):
        with self.api_lock:
            if self._api is None:
                if self.api_factory: self._api=self.api_factory(self.root)
                else:
                    from teacher_api import TeacherAPI
                    self._api=TeacherAPI(self.root)
            return self._api
    def assets(self):
        data=self.root/'data'; trained=self.root/'models/trained'
        train=(data/'processed/train.jsonl').is_file(); holdout=(data/'sealed/holdout.jsonl').is_file()
        ready=(data/'manifests/readiness.json').is_file() and train and holdout
        archive=next((p for p in [self.root/'assets/CMeEE-V2.zip',data/'raw/CMeEE-V2.zip'] if p.is_file()),None)
        result=dict(data=dict(ready=ready,message='已准备训练与完整留出数据；复现时校验内容。' if ready else '缺少准备清单、训练集或完整留出集。'),archive=dict(ready=archive is not None,message='官方压缩包可用。' if archive else '尚未放入官方CMeEE-V2.zip。'),L3=dict(ready=train,message='训练数据可用；调用前校验三个冻结示例。' if train else '三样本需要先准备官方训练数据。'))
        try: items=read_json(trained/'manifest.json')['items']
        except (OSError,ValueError,KeyError): items=[]
        for method in ['N0','N1']:
            item=next((x for x in items if isinstance(x,dict) and x.get('method')==method),None)
            name=item.get('file') if item else None
            valid=isinstance(name,str) and name==Path(name).name and name not in ('','.') and (trained/name).is_file()
            result[method]=dict(ready=valid,message='正式权重可用；首次推理验证文件内容。' if valid else '正式权重或清单缺失。')
        return result
    def case_payloads(self):
        try:
            cases=read_json(self.root/'ui/demo-cases.json')['cases']
            return [r for r in cases if isinstance(r,dict) and isinstance(r.get('case'),dict)]
        except (OSError,ValueError,KeyError): return []
    def cases(self): return [{k:r['case'].get(k) for k in ['case_id','id','category','route']} for r in self.case_payloads()]
    def case(self,ident):
        payload=next((r for r in self.case_payloads() if r['case'].get('case_id')==ident),None)
        if payload is None: raise KeyError('案例不存在')
        self.record(event='view_case',case_id=ident,source='portable_frozen_examples'); return payload
    def _model(self):
        if self.models is None:
            if self.models_factory: self.models=self.models_factory(self.root)
            else:
                from cpu_inference import CPUModels
                self.models=CPUModels(self.root)
        return self.models
    def _engine(self):
        if self.engine is None:
            if not (self.root/'data/processed/train.jsonl').is_file(): raise AssetError('D0需要已准备的官方训练数据。')
            if self.engine_factory: self.engine=self.engine_factory(self.root)
            else:
                from stage5_repairs import Engine,rows
                self.engine=Engine(rows(self.root/'data/processed/train.jsonl'),self.log_dir/'cache',self.root/'configs/review-runtime.json')
        return self.engine
    def extract(self,data):
        text=data.get('text'); method=data.get('method','N1')
        if method not in ('D0','N0','N1'): raise ValueError('未知抽取路线')
        if not isinstance(text,str) or not text.strip() or len(text)>4000: raise ValueError('请输入1—4000个Unicode字符。')
        if self.jobs.guard.exists(): raise BusyError('完整复现保护锁存在；请等待正在运行的任务，若服务曾中断则先人工核查遗留任务与保护锁。')
        if not self.cpu_lock.acquire(blocking=False): raise BusyError('已有本地CPU任务运行，请稍后重试。')
        try:
            started=time.perf_counter(); row=dict(id='live-local-input',text=text)
            baseline=repaired=tokens=neural_timing=None; actions=[]
            if method=='D0':
                engine=self._engine(); original,_,tokens=engine.predict(row,'baseline'); fixed,actions,_=engine.predict(row,'combined')
                prediction=baseline=original['entities']; repaired=fixed['entities']
            else:
                if not self.assets()[method]['ready']: raise AssetError(method+'正式权重或清单缺失。')
                output,neural_timing=self._model().predict_many([row],method); prediction=output[0]['entities']
            result=dict(text=text,method=method,status='valid',prediction=prediction,baseline=baseline,repaired=repaired,actions=actions,tokens=tokens,seconds=time.perf_counter()-started,neural_timing=neural_timing,runtime='local CPU;fresh inference',new_api_calls=0,source='本次现场CPU推理')
            self.record(event='live_extract',**result); return result
        finally: self.cpu_lock.release()
    def artifact_paths(self):
        candidate=self.root/'deliverables/stage6/candidate'
        specifications=[('report','报告PDF','报告.pdf'),('abstract','项目摘要','项目摘要.md'),('statement-wu','吴皓睿贡献声明','贡献声明_*_吴皓睿.md'),('statement-zhou','周栩乐贡献声明','贡献声明_*_周栩乐.md'),('readme','复现说明','README.md')]
        result={}
        for ident,label,name in specifications:
            roots=[candidate,self.root] if ident!='readme' else [self.root,candidate]
            found=next((match for base in roots for match in (base.glob(name) if '*' in name else [base/name]) if match.is_file()),None)
            result[ident]=(label,found)
        zip_roots=[self.root/'deliverables/stage6',self.root]
        source_zip=next((match for base in zip_roots for match in base.glob('*_中文医学实体抽取_本地候选.zip') if match.is_file()),None)
        result['source-zip']=('源码本地候选ZIP',source_zip)
        result['weights']=('正式推理权重包',next((p for p in [self.root/'deliverables/stage6/upload/中文医学实体抽取_正式推理权重_seed20261019.zip'] if p.is_file()),None))
        return result
    def project(self):
        try: result=read_json(self.root/'ui/project-data.json')
        except (OSError,ValueError): result={}
        if not isinstance(result,dict): result={}
        try: budget=self.api().budget()
        except (OSError,ValueError):
            from teacher_api import BASELINE,DEMO_CAP,TOTAL_CAP,CONFIRMED_PLATFORM_SPENT,PRIOR_UNRESOLVED_RESERVE,money
            budget=dict(baseline_cny=money(BASELINE),demo_cap_cny=money(DEMO_CAP),total_cap_cny=money(TOTAL_CAP),remaining_cny='0.000000',calls=None,confirmed_platform_spent_cny=money(CONFIRMED_PLATFORM_SPENT),prior_unresolved_reserved_cny=money(PRIOR_UNRESOLVED_RESERVE),reservations_are_not_actual_spending=True,is_invoice=False,error='历史费用侧账不可读；新调用使用独立用量记录。')
        result.update(assets=self.assets(),budget=budget)
        result['artifacts']=[dict(id=i,label=label,available=p is not None,url='/artifacts/'+i if p else None,status='本地候选' if p else '未生成或未放入本目录',file_name=p.name if p else None,file_size=p.stat().st_size if p else None,modified_at=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat() if p else None) for i,(label,p) in self.artifact_paths().items()]
        previous=[dict(r,kind='historical') for r in result.get('previous_reproductions',[]) if isinstance(r,dict)]
        paths=[self.root/'runs/stage6-candidate-reproduction/summary.json',self.root/'evidence/stage6-teacher-upgrade/candidate-reproduction-summary.json']
        for p in paths:
            if p.is_file():
                try:
                    summary=read_json(p); source=p.relative_to(self.root).as_posix()
                    if not any(r.get('source')==source for r in previous): previous.append(dict(source=source,at=summary.get('finished_at',summary.get('at',summary.get('started_at'))),seconds=summary.get('seconds',summary.get('total_seconds')),status=summary.get('status','UNKNOWN'),kind='historical',label='已保存的历史复现；不是本次运行'))
                except (OSError,ValueError): pass
        result['previous_reproductions']=previous; result['reproduction_jobs']=self.jobs.list(); return result
    def close(self): self.jobs.close()
