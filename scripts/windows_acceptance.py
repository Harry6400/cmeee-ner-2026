"""Evidence collector for the Windows handoff. Standard library; never starts API/GPU/training."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
AUTOMATIC = ('environment', 'source', 'install', 'prepare', 'assets', 'python-tests', 'frontend-tests', 'cpu')
MANUAL = ('edge-extraction', 'edge-comparison', 'edge-errors', 'edge-experiments',
          'json-readback', 'stale-result', 'jobs-concurrency', 'jobs-cancel',
          'jobs-refresh', 'jobs-interruption', 'jobs-complete')
CHECKS = AUTOMATIC + MANUAL
STATUSES = ('PASS', 'FAIL', 'BLOCKED', 'NOT_RUN')


def now(): return datetime.now(timezone.utc).isoformat()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def write(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2); f.write('\n')
def capture(command, cwd=None):
    try:
        r = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
        return dict(command=command, exit_code=r.returncode, stdout=r.stdout.strip(), stderr=r.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(command=command, exit_code=None, error=type(exc).__name__)
def record(directory, check, status, detail, evidence=()):
    directory = Path(directory)
    event = dict(check=check, status=status, at=now(), detail=detail, evidence=list(evidence))
    index = len(list(directory.glob('event-*.json'))) + 1
    write(directory / f'event-{index:04d}.json', event)
    return event

def binding(state):
    root = Path(state['root']); archive = Path(state['archive'])
    errors = []
    if sha(archive) != state['archive_sha256']: errors.append('source archive changed')
    names = set()
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if entry.is_dir(): continue
            name = entry.filename; rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in name: raise ValueError('unsafe source archive')
            names.add(name)
            target = root / rel
            if target.is_symlink() or not target.is_file() or sha(target) != hashlib.sha256(z.read(entry)).hexdigest():
                errors.append('source differs: ' + name)
    for area in ('scripts', 'ui', 'configs', 'tests'):
        for path in (root / area).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.relative_to(root).as_posix() not in names:
                errors.append('extra runtime source: ' + path.relative_to(root).as_posix())
    return dict(status='FAIL' if errors else 'PASS', errors=errors, commit=state['commit'], archive_sha256=state['archive_sha256'])

def initialize(args):
    directory = args.output.resolve()
    if directory.is_relative_to(ROOT): raise ValueError('evidence session must be outside clean source')
    directory.mkdir(parents=True, exist_ok=False)
    repo = args.source_repo.resolve(); archive = args.archive.resolve()
    commit_result = capture(['git', 'rev-parse', args.commit + '^{commit}'], repo)
    if commit_result['exit_code'] != 0: raise ValueError('cannot resolve source commit')
    commit = commit_result['stdout']
    original = subprocess.run(['git', 'archive', '--format=zip', commit], cwd=repo, capture_output=True, check=True).stdout
    if hashlib.sha256(original).hexdigest() != sha(archive): raise ValueError('archive is not git archive of supplied commit')
    state = dict(root=str(ROOT), archive=str(archive), archive_sha256=sha(archive), commit=commit,
                 source_repo=str(repo), started_at=now(), operator=args.operator,
                 repository=capture(['git', 'remote', 'get-url', 'origin'], repo),
                 branch=capture(['git', 'branch', '--show-current'], repo),
                 initial_worktree=capture(['git', 'status', '--short'], repo),
                 checklist={key:'NOT_RUN' for key in CHECKS})
    write(directory/'session.json', state)
    result = binding(state); record(directory, 'source', result['status'], result)
    info = dict(system=platform.system(), platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), python=sys.version, executable=sys.executable,
                disk_free_bytes=shutil.disk_usage(ROOT).free,
                uv=capture(['uv', '--version']), node=capture(['node', '--version']))
    if os.name == 'nt':
        info['hardware'] = capture(['powershell.exe','-NoProfile','-Command',
            '[Console]::OutputEncoding=[Text.UTF8Encoding]::new(); $c=Get-CimInstance Win32_Processor; $o=Get-CimInstance Win32_OperatingSystem; @{cpu=@($c.Name);memory_bytes=[long]$o.TotalVisibleMemorySize*1024;os=$o.Caption;build=$o.BuildNumber}|ConvertTo-Json -Compress'])
        info['edge'] = capture(['powershell.exe','-NoProfile','-Command',
            "[Console]::OutputEncoding=[Text.UTF8Encoding]::new(); $p=@(\"${env:ProgramFiles(x86)}\\Microsoft\\Edge\\Application\\msedge.exe\",\"$env:ProgramFiles\\Microsoft\\Edge\\Application\\msedge.exe\",\"$env:LOCALAPPDATA\\Microsoft\\Edge\\Application\\msedge.exe\"); $v=@($p|Where-Object {Test-Path $_}|ForEach-Object {(Get-Item $_).VersionInfo.ProductVersion}); if(!$v){exit 1}; $v|ConvertTo-Json -Compress"])
    write(directory/'environment.json', info)
    ready = (os.name == 'nt' and sys.version_info[:2] == (3,11) and
             all(info[k]['exit_code'] == 0 for k in ('uv','node','hardware','edge')))
    record(directory, 'environment', 'PASS' if ready else 'BLOCKED',
           'Requires real Windows, Python 3.11, uv, Node, hardware and Edge version capture.', ['environment.json'])


def assets(directory):
    expected = load(ROOT/'configs/windows-assets.json')
    missing, errors = [], []
    for rel, expected_hash in expected['files'].items():
        path = ROOT / rel
        if not path.is_file(): missing.append(rel)
        elif sha(path) != expected_hash: errors.append(rel)
    for rel in expected['required_display_files']:
        if not (ROOT/rel).is_file(): missing.append(rel)
    status = 'FAIL' if errors else 'BLOCKED' if missing else 'PASS'
    return record(directory, 'assets', status, dict(missing=missing, hash_mismatches=errors,
                  note='Data source/license confirmation is recorded separately in the acceptance report.'))


def run_check(directory, check):
    commands = {
        'install': ['uv','sync','--locked'],
        'prepare': ['uv','run','--locked','python','-X','utf8','scripts/prepare_data.py','--archive','assets/CMeEE-V2.zip','--output','data'],
        'python-tests': ['uv','run','--locked','python','-X','utf8','-B','-m','unittest','discover','-s','tests','-v'],
        'frontend-tests': ['node','tests/test_teacher_ui.js'],
    }
    command = commands[check]
    index = len(list(directory.glob('event-*.json'))) + 1
    log = directory / f'{index:04d}-{check}.log'
    started = now(); code = None; error = None
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    try:
        with log.open('x', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(dict(command=command, cwd=str(ROOT), started_at=started),ensure_ascii=False)+'\n'); f.flush()
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,encoding='utf-8',errors='replace')
            try:
                for line in child.stdout: f.write(line); f.flush(); print(line,end='')
                code = child.wait()
            finally:
                if child.poll() is None: child.terminate(); child.wait()
                child.stdout.close()
    except OSError as exc: error = type(exc).__name__
    return record(directory,check,'BLOCKED' if error else 'PASS' if code == 0 else 'FAIL',
                  dict(command=command,exit_code=code,error=error,started_at=started,finished_at=now()),[log.name])


def entity_set(row):
    text = row['text']; result = set()
    for e in row['entities']:
        a,b,t = e['start'],e['end'],e['type']
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text): raise ValueError('invalid entity boundary')
        if t not in ('dis','sym','dru','equ','pro','bod','ite','mic','dep') or e['text'] != text[a:b]: raise ValueError('invalid entity text/type')
        result.add((a,b,t))
    return result

def independent_cpu(root, output):
    """Recompute from local new predictions; do not import project metric implementations."""
    expected = load(root/'configs/reproduction-expected.json')
    gold_path = root/'data/sealed/holdout.jsonl'
    if sha(gold_path) != expected['data_sha256']['sealed/holdout.jsonl']: raise ValueError('gold hash differs')
    gold = [json.loads(line) for line in gold_path.read_text(encoding='utf-8').splitlines()]
    if len(gold) != 5000 or len({g['id'] for g in gold}) != 5000: raise ValueError('gold count/IDs differ')
    top = load(output/'summary.json'); summary = load(output/'evaluation/summary.json')
    if top['status'] != 'PASS' or summary['status'] != 'PASS' or summary['samples'] != 5000 or summary['split'] != 'holdout5000': raise ValueError('not a complete passing holdout5000 run')
    for value in (top,summary):
        if value.get('API_calls') != 0 or value.get('GPU_started') is not False or value.get('training') is not False: raise ValueError('unexpected workload scope')
    result = {}
    for method in ('D0','N0','N1'):
        path = output/'evaluation'/f'{method}-predictions.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if len(rows) != 5000 or len({r['id'] for r in rows}) != 5000: raise ValueError(method+': wrong prediction count/IDs')
        counts = Counter()
        for g,p in zip(gold,rows):
            if (g['id'],g['text']) != (p['id'],p['text']): raise ValueError(method+': prediction order/text differs')
            a,b = entity_set(g),entity_set(p)
            counts.update(tp=len(a & b),predicted=len(b),gold=len(a))
        tp,n,g = (counts[k] for k in ('tp','predicted','gold'))
        measured = dict(counts,precision=tp/n if n else 0,recall=tp/g if g else 0,f1=2*tp/(n+g) if n+g else 0)
        target = expected['metrics']['holdout5000'][method]
        equal = all(abs(measured[k]-v)<1e-12 for k,v in target.items())
        reported = summary['results'][method]['metrics']['micro']
        if any(abs(measured[k]-reported[k])>=1e-12 for k in target): raise ValueError(method+': reported metrics differ from predictions')
        result[method] = dict(samples=len(rows),metrics=measured,frozen_equal=equal,predictions_sha256=sha(path))
    return dict(status='PASS' if all(x['frozen_equal'] for x in result.values()) else 'FAIL',
                results=result,run_summary=top,evaluation_summary=summary,
                input_summary_sha256=sha(output/'summary.json'),evaluation_summary_sha256=sha(output/'evaluation/summary.json'))


def collect_cpu(directory, output):
    try:
        output=output.resolve()
        if not output.is_relative_to(ROOT/'output/teacher-reproduction'): raise ValueError('requires this clean source browser-job output')
        session=load(directory/'session.json')
        job=load(ROOT/'output/teacher-reproduction/.jobs'/f'{output.name}.json')
        if job['status']!='passed' or (ROOT/job['output']).resolve()!=output: raise ValueError('browser job not passed or path differs')
        if datetime.fromisoformat(job['started_at']) < datetime.fromisoformat(session['started_at']): raise ValueError('historical job predates acceptance session')
        data = independent_cpu(ROOT,output); status = data['status']
        data['browser_job']=dict(id=job['id'],started_at=job['started_at'],finished_at=job['finished_at'],exit_code=job['exit_code'])
    except (OSError,ValueError,KeyError,TypeError) as exc:
        data = dict(status='FAIL',error_type=type(exc).__name__,error=str(exc)); status='FAIL'
    filename = f'cpu-{len(list(directory.glob("event-*.json")))+1:04d}.json'
    write(directory/filename,data)
    return record(directory,'cpu',status,dict(output=str(output),note='Independent standard-library recount; raw predictions stay local.'),[filename])


def manual(directory,args):
    if args.status == 'PASS' and (not args.evidence or not args.note.strip()): raise ValueError('PASS requires actual evidence and an observation')
    copied = []
    target = directory / f'manual-{len(list(directory.glob("event-*.json")))+1:04d}'
    for index, path in enumerate(args.evidence or []):
        if path.is_symlink() or not path.is_file(): raise ValueError('evidence must be an explicit regular file')
        if path.suffix.lower() not in ('.png','.jpg','.jpeg','.json','.txt','.md','.log'): raise ValueError('unsupported evidence type')
        if any(term in path.name.lower() for term in ('prediction','ledger','.env','secret','token','key')): raise ValueError('private/bulk evidence filename rejected')
        target.mkdir(exist_ok=True)
        dest = target / f'{index:02d}{path.suffix.lower()}'; shutil.copyfile(path,dest); copied.append(dest.relative_to(directory).as_posix())
    return record(directory,args.check,args.status,dict(observation=args.note,declared_by_operator=True,
                  warning='Evidence must be independently reviewed; collector does not visually verify browser behavior.'),copied)


def finalize(directory):
    state=load(directory/'session.json'); source=binding(state)
    record(directory,'source',source['status'],source)
    events=[load(p) for p in sorted(directory.glob('event-*.json'))]
    current={k:dict(status='NOT_RUN') for k in CHECKS}
    for e in events: current[e['check']]=e
    overall='FAIL' if any(x['status']=='FAIL' for x in current.values()) else 'PASS' if all(x['status']=='PASS' for x in current.values()) else 'BLOCKED'
    report=dict(status=overall,finished_at=now(),commit=state['commit'],source_archive_sha256=state['archive_sha256'],
                operator=state['operator'],checks=current,api='NOT_RUN: excluded by authorization',windows_status='Pending independent evidence review',
                materials='Report/video/share links are separate delivery items; list missing assets in PR report.')
    # Single-use finalization; further fixes/attempts use a new evidence session.
    write(directory/'report.json',report)
    files=[p for p in sorted(directory.rglob('*')) if p.is_file()]
    manifest={p.relative_to(directory).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    write(directory/'manifest.json',manifest)
    archive=directory.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(directory.rglob('*')):
            if path.is_file(): z.write(path,path.relative_to(directory).as_posix())
    write(Path(str(archive)+'.sha256.json'),dict(file=archive.name,sha256=sha(archive),bytes=archive.stat().st_size))
    return dict(status=overall,archive=str(archive),sha256=sha(archive))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    init=sub.add_parser('init');init.add_argument('--output',type=Path,required=True);init.add_argument('--source-repo',type=Path,required=True);init.add_argument('--archive',type=Path,required=True);init.add_argument('--commit',required=True);init.add_argument('--operator',required=True)
    for name in ('run','assets','cpu','record','finalize'):
        q=sub.add_parser(name);q.add_argument('--session',type=Path,required=True)
        if name=='run': q.add_argument('--check',choices=('install','prepare','python-tests','frontend-tests'),required=True)
        if name=='cpu': q.add_argument('--output',type=Path,required=True)
        if name=='record':
            q.add_argument('--check',choices=MANUAL,required=True);q.add_argument('--status',choices=STATUSES,required=True);q.add_argument('--note',required=True);q.add_argument('--evidence',type=Path,nargs='+')
    args=p.parse_args()
    try:
        if args.action=='init': initialize(args); result=dict(session=str(args.output))
        else:
            directory=args.session.resolve();state=load(directory/'session.json')
            if Path(state['root']).resolve()!=ROOT: raise ValueError('run collector from the bound clean source directory')
            if (directory/'report.json').exists(): raise ValueError('session finalized; create a new session')
            if args.action=='run':result=run_check(directory,args.check)
            elif args.action=='assets':result=assets(directory)
            elif args.action=='cpu':result=collect_cpu(directory,args.output)
            elif args.action=='record':result=manual(directory,args)
            else:result=finalize(directory)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result.get('status','PASS')=='PASS' else 1
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2
if __name__=='__main__': raise SystemExit(main())
