"""One-shot completion stage of the approved four-job study;no GPU/API operations."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time,os
ROOT=Path(__file__).resolve().parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def main():
 if ROOT!=Path('/nfsdata_a40/cyf/codes/CT/cmeee_ner'):raise ValueError('registered mirror required')
 E=ROOT/'evidence/stage6-template-supplement';status=E/'gpu-study-completion.json';state=dict(status='WAITING_TRAINING_QUEUE',pid=os.getpid(),started_at=now(),API_calls=0,GPU_started=False)
 def save():status.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
 with status.open('x') as f:f.write('{}')
 save()
 while True:
  try:q=json.loads((E/'gpu-queue.json').read_text())
  except json.JSONDecodeError:time.sleep(5);continue
  if q.get('status')=='FAILED':state.update(status='TRAINING_QUEUE_FAILED',at=now(),queue_error=q.get('error'));save();return 1
  if q.get('status')=='COMPLETED':break
  time.sleep(60)
 state.update(status='VERIFYING',at=now());save()
 with (E/'gpu-verification-console.log').open('x') as f:r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/verify_supplement_gpu.py'),'--output',str(E/'gpu-independent-verification.json')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 state.update(status='COMPLETED' if r.returncode==0 else 'VERIFICATION_FAILED',finished_at=now(),returncode=r.returncode);save();return r.returncode
if __name__=='__main__':raise SystemExit(main())
