"""Independent standard-library recount of CPU supplementary experiments; no model/API/GPU."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
TYPES={'dis','sym','dru','equ','pro','bod','ite','mic','dep'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 if 'sealed' in Path(p).resolve().parts:raise ValueError('supplement verifier cannot read sealed')
 return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def entities(r):
 assert isinstance(r['id'],str) and isinstance(r['text'],str) and isinstance(r['entities'],list)
 out=set()
 for e in r['entities']:
  a,b,t=e['start'],e['end'],e['type'];assert type(a)is int and type(b)is int and 0<=a<b<=len(r['text']) and t in TYPES and e['text']==r['text'][a:b];out.add((a,b,t))
 return out
def calc(gold,pred):
 assert len(gold)==len(pred)==1503;total=[0,0,0];char=[0,0,0]
 for g,p in zip(gold,pred):
  assert set(p)=={'id','text','entities'};assert g['id']==p['id'] and g['text']==p['text'];a,b=entities(g),entities(p)
  for i,n in enumerate([len(a&b),len(b),len(a)]):total[i]+=n
  a={(i,t) for s,e,t in a for i in range(s,e)};b={(i,t) for s,e,t in b for i in range(s,e)}
  for i,n in enumerate([len(a&b),len(b),len(a)]):char[i]+=n
 def rates(v):
  t,p,g=v;return dict(tp=t,predicted=p,gold=g,precision=t/p if p else 0,recall=t/g if g else 0,f1=2*t/(p+g) if p+g else 0)
 return dict(micro=rates(total),character_type_micro=rates(char))
def close(a,b):return abs(a-b)<=1e-12

def verify_cpu(study):
 base=ROOT/study['run_root'];gold=rows(ROOT/'data/processed/dev.jsonl');records=[];threshold=json.loads((base/'cpu-threshold/summary.json').read_text())
 assert threshold['status']=='COMPLETED' and len(threshold['results'])==10 and threshold['samples']==1503
 for method,run in [('N0','N0'),('N1','N1-lr-5e-05')]:
  previous=None
  for level in study['thresholds']:
   r=next(x for x in threshold['results'] if x['method']==method and x['threshold']==level);pred=rows(ROOT/r['predictions']);m=calc(gold,pred)
   for family in m:
    assert all(close(v,r['metrics'][family][k]) for k,v in m[family].items()),'metric mismatch'
   current=[entities(p) for p in pred]
   if previous is not None:assert all(n<=p for n,p in zip(current,previous)), 'threshold candidates not nested'
   previous=current
   if level==0:
    oldsummary=json.loads((ROOT/f'runs/stage3-20261019/{run}/summary.json').read_text());original=rows(ROOT/f'runs/stage3-20261019/{run}/dev-epoch-{oldsummary["best_epoch"]:02d}.jsonl');assert all(entities(a)==entities(b) for a,b in zip(original,pred))
   records.append(dict(method=method,threshold=level,independent_metrics=m))
 latency=json.loads((base/'cpu-latency/summary.json').read_text());assert latency['status']=='COMPLETED' and set(latency['results'])=={'D0','N0','N1'}
 timing_records=[]
 for method,r in latency['results'].items():
  assert r['warmup_samples']==10 and len(r['repetitions'])==3;values=[];previous=None
  for repeat in r['repetitions']:
   p=rows(ROOT/repeat['predictions']);t=rows(ROOT/repeat['timing']);assert len(t)==1503 and [x['id'] for x in t]==[x['id'] for x in gold]
   assert all(math.isfinite(x['seconds']) and x['seconds']>0 for x in t)
   if previous is not None:assert all(entities(a)==entities(b) for a,b in zip(previous,p))
   previous=p;values.extend(x['seconds'] for x in t)
  m=calc(gold,previous)
  for family in m:assert all(close(v,r['metrics'][family][k]) for k,v in m[family].items())
  ordered=sorted(values);index=(len(values)-1)*.95;lo=math.floor(index);hi=math.ceil(index);p95=ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)
  assert len(values)==4509 and close(statistics.mean(values),r['aggregate']['mean_seconds']) and close(p95,r['aggregate']['p95_seconds'])
  timing_records.append(dict(method=method,samples=1503,timing_observations=len(values),mean_seconds=statistics.mean(values),p95_seconds=p95,independent_metrics=m))
 return dict(threshold_conditions=records,latency=timing_records,zero_threshold_identical_to_original_development=True,monotonic_candidate_sets=True,latency_predictions_repeat_equal=True)

def verify_parameters(study):
 base=ROOT/study['run_root'];gold=rows(ROOT/'data/processed/dev.jsonl');summary=json.loads((base/'cpu-parameter-latency/summary.json').read_text());assert summary['status']=='COMPLETED';records=[]
 for r in summary['results']:
  method,level=r['method'],r['threshold'];assert (method,level) in [('N0',-.5),('N1',.5)];expected=rows(base/f'cpu-threshold/{method}/threshold-{level:g}/predictions.jsonl');values=[]
  for repeat in r['repetitions']:
   pred=rows(ROOT/repeat['predictions']);timing=rows(ROOT/repeat['timing']);assert len(timing)==1503;assert [x['id'] for x in timing]==[x['id'] for x in gold];assert all(entities(a)==entities(b) for a,b in zip(expected,pred));values.extend(x['seconds'] for x in timing)
  m=calc(gold,pred)
  for family in m:assert all(close(v,r['metrics'][family][k]) for k,v in m[family].items())
  assert len(values)==4509 and all(math.isfinite(v) and v>0 for v in values);ordered=sorted(values);i=(len(values)-1)*.95;l=math.floor(i);h=math.ceil(i);q=ordered[l]+(ordered[h]-ordered[l])*(i-l);assert close(statistics.mean(values),r['aggregate']['mean_seconds']) and close(q,r['aggregate']['p95_seconds']);records.append(dict(method=method,threshold=level,observations=len(values),mean_seconds=statistics.mean(values),p95_seconds=q,independent_metrics=m))
 assert len(records)==2;return dict(conditions=records,total_observations=9018,all_match_original_threshold_predictions=True)

def verify_protected(study):
 E=ROOT/study['evidence_root'];frozen=json.loads((E/'frozen-before.json').read_text());protected=json.loads((E/'protected-before.json').read_text())
 assert len(frozen)==1447
 approval=E/'review-approval.json';allowed=set()
 if approval.exists():
  consent=json.loads(approval.read_text());assert consent['approved'] is True
  allowed=set(consent['allowed_artifact_changes'])
 immutable={p:v for p,v in (frozen|protected).items() if p not in allowed}
 assert not set(frozen)&allowed
 bad=[p for p,v in immutable.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=v]
 assert not bad, 'protected files changed:'+repr(bad)
 return dict(frozen_files=1447,extra_protected_files=len(protected),immutable_unchanged=True,approved_report_artifact_changes=sorted(allowed),note='report edits allowed only after explicit recorded draft approval;models,data,budget,frozen results stay protected')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();start=time.perf_counter();study=json.loads((ROOT/'configs/stage6-supplement.json').read_text());result=dict(status='RUNNING',at=datetime.now(timezone.utc).isoformat(),implementation='independent Python standard library;not independent human review',API_calls=0,GPU=False,sealed_read=False)
 try:result.update(cpu=verify_cpu(study),parameter_latency=verify_parameters(study),protected=verify_protected(study),status='PASS')
 except BaseException as ex:result.update(status='FAIL',error_type=type(ex).__name__,error=str(ex));raise
 finally:result['seconds']=time.perf_counter()-start;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(status=result['status'],seconds=result['seconds'],threshold_conditions=10,timing_observations=22545,frozen_files=1447),ensure_ascii=False))
if __name__=='__main__':main()
