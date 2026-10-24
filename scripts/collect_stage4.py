"""Collect completed remote stage4 evidence without weights,secrets,overwrite or remote writes."""
import argparse,hashlib,json,subprocess,time,tarfile,io
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--producer-journal',default='evidence/stage4/post-training-controller.json');a=p.parse_args()
r=Path.cwd();assert r.name=='医学实体抽取';out=r/'evidence/stage4/transfer-manifest.json';assert not out.exists();start=time.perf_counter();at=datetime.now(timezone.utc).isoformat()
remote='''import json,hashlib
from pathlib import Path
r=Path('/nfsdata_a40/cyf/codes/CT/cmeee_ner')
p=r/PRODUCER_JOURNAL;s=json.loads(p.read_text());assert s['status']=='completed_local_stage4',s['status']
files=[]
for base in ['evidence/stage4','runs/stage4-20261020']:
 for f in sorted((r/base).rglob('*')):
  if f.is_symlink():raise ValueError('No symlinks:'+str(f))
  if not f.is_file() or f.suffix in ['.pt','.pth','.bin','.safetensors']:continue
  raw=f.read_bytes();files.append(dict(path=str(f.relative_to(r)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
print(json.dumps(dict(files=files,remote_completed=True)))
'''
remote=remote.replace('PRODUCER_JOURNAL',repr(a.producer_journal))
res=subprocess.run(['ssh','-o','BatchMode=yes','qha40gpu-2','python3 -'],input=remote,text=True,capture_output=True,check=True);inventory=json.loads(res.stdout);missing=[];same=[]
for x in inventory['files']:
 rel=Path(x['path']);assert not rel.is_absolute() and '..' not in rel.parts
 f=r/rel
 if f.exists():
  assert hashlib.sha256(f.read_bytes()).hexdigest()==x['sha256'],'Existing evidence conflict;never overwrite:'+str(rel)
  same.append(x)
 else:missing.append(x)
print(json.dumps(dict(remote_files=len(inventory['files']),already_identical=len(same),missing_files=len(missing),missing_bytes=sum(x['bytes'] for x in missing),execute=a.execute)))
if not a.execute:raise SystemExit(0)
if missing:
 expected={x['path']:x for x in missing}
 child=subprocess.Popen(['ssh','-o','BatchMode=yes','qha40gpu-2','cd /nfsdata_a40/cyf/codes/CT/cmeee_ner && tar -cf - --files-from=-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 child.stdin.write(('\n'.join(expected)+'\n').encode());child.stdin.close()
 seen=set()
 with tarfile.open(fileobj=child.stdout,mode='r|') as archive:
  for member in archive:
   rel=member.name.removeprefix('./');assert rel in expected and member.isfile() and not member.issym() and not member.islnk(),rel
   f=r/rel;f.parent.mkdir(parents=True,exist_ok=True)
   with f.open('xb') as target:target.write(archive.extractfile(member).read())
   seen.add(rel)
 err=child.stderr.read();assert child.wait()==0,err.decode();assert seen==set(expected)
for x in inventory['files']:
 f=r/x['path'];assert f.stat().st_size==x['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==x['sha256'],x['path']
record=dict(started_at=at,finished_at=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,source='qha40gpu-2:/nfsdata_a40/cyf/codes/CT/cmeee_ner',producer_journal=a.producer_journal,remote_writes=False,weights_and_secrets_excluded=True,files=inventory['files'],transferred_files=len(missing),transferred_bytes=sum(x['bytes'] for x in missing),all_sha256_verified=True)
with out.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status='PASS',seconds=record['seconds'],transferred_files=record['transferred_files'],transferred_bytes=record['transferred_bytes'])))
