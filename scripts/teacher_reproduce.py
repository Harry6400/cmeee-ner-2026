"""One-command teacher CPU reproduction. No network, SSH, ledger or historical-answer dependency."""
from __future__ import annotations
import argparse,hashlib,json,platform,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def verify_model_assets(root):
    directory=root/'models/trained';manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if {x['method'] for x in manifest['items']}!={'N0','N1'}:raise ValueError('Required formal N0/N1 assets missing')
    for item in manifest['items']:
        name=Path(item['file'])
        if name.name!=str(name) or name.is_absolute():raise ValueError('Unsafe model asset path')
        if item['seed']!=20261019:raise ValueError('Wrong frozen primary seed')
        p=directory/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Model asset mismatch: '+item['method'])
    return {'methods':['N0','N1'],'seed':20261019,'device':'cpu'}
def execute(root,archive,output,split='holdout5000'):
    root=Path(root).resolve();archive=Path(archive).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Output exists;choose a new directory: '+str(output))
    if split not in ('common500','holdout5000'):raise ValueError('Unknown fixed split')
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();at=datetime.now(timezone.utc).isoformat();state=dict(status='RUNNING',started_at=at,split=split,API_calls=0,GPU_started=False,training=False,private_ledger_required=False,SSH_required=False,platform=platform.platform(),python=sys.version)
    with (output/'run.log').open('x',encoding='utf-8') as log:
        def note(event,**kw):
            line=json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),event=event,**kw),ensure_ascii=False);log.write(line+'\n');log.flush();print(line,flush=True)
        try:
            note('model_assets_check',**verify_model_assets(root))
            from prepare_data import prepare,verify
            data=root/'data'
            if (data/'manifests/readiness.json').exists():note('data_verify',**verify(data))
            else:
                if any((data/x).exists() for x in ['processed','sealed']):raise ValueError('Partial data directory;keep it and use a clean extraction')
                if not archive.is_file():raise FileNotFoundError('Official CMeEE-V2.zip missing: '+str(archive))
                prepared=prepare(archive,data);note('data_prepared',status=prepared['status'] if 'status' in prepared else 'prepared')
                note('data_verify',**verify(data))
            cmd=[sys.executable,'-B','-u',str(root/'scripts/reproduce.py'),'--root',str(root),'evaluate','--split',split,'--output',str(output/'evaluation')]
            note('fresh_CPU_evaluation_start',command=cmd)
            child=subprocess.Popen(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait();state['evaluation_exit_code']=code
            if code:raise RuntimeError('CPU evaluator failed with exit '+str(code))
            result=json.loads((output/'evaluation/summary.json').read_text(encoding='utf-8'));state['result']=result
            if result['status']!='PASS':raise ValueError('Fresh metrics differ from frozen expectations;outputs retained,do not replace scores')
            state['status']='PASS';note('teacher_reproduction_complete',samples=result['samples'])
        except Exception as exc:
            state.update(status='FAIL',error_type=type(exc).__name__,error=str(exc));traceback.print_exc(file=log);note('failed',error_type=type(exc).__name__,error=str(exc))
    state.update(finished_at=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start)
    with (output/'summary.json').open('x',encoding='utf-8') as f:json.dump(state,f,ensure_ascii=False,indent=2);f.write('\n')
    return state

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,default=Path('assets/CMeEE-V2.zip'));p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['common500','holdout5000'],default='holdout5000');a=p.parse_args()
    try:result=execute(ROOT,a.archive,a.output,a.split)
    except (OSError,ValueError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
    print(json.dumps({k:result[k] for k in ['status','split','seconds','API_calls','GPU_started']},ensure_ascii=False));raise SystemExit(0 if result['status']=='PASS' else 1)
if __name__=='__main__':main()
