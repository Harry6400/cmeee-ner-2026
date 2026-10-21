"""Standard-library independent processed-dev recount for all six grid and paired-seed runs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,statistics,time
from verify_supplement import rows,calc,sha,close
ROOT=Path(__file__).resolve().parents[1]
def evaluate_study(root):
 study=json.loads((root/'configs/stage6-supplement.json').read_text());cfgbase=json.loads((root/study['base_config']).read_text());assert sha(root/study['base_config'])==study['base_config_sha256'];assert sha(root/'data/processed/dev.jsonl')==study['dev_sha256'];gold=rows(root/'data/processed/dev.jsonl');paths=list(study['reuse_runs'])+[study['run_root']+'/'+j['name'] for j in study['training_jobs']];results=[]
 fixed=['window_chars','overlap_chars','micro_batch','accumulation','precision','head_size','bert_model','bert_revision','n1_head_lr','decode_threshold','n1_max_epochs','patience','gpu_uuid']
 for path in paths:
  run=root/path;c=json.loads((run/'config.json').read_text());s=json.loads((run/'summary.json').read_text());assert all(c[k]==cfgbase[k] for k in fixed),'control changed:'+path;assert s['status'] in ['completed','early_stopped'] and s['method']=='N1' and not s['sealed_read'] and s['paid_api_calls']==0
  best=s['best_epoch'];bestm=calc(gold,rows(run/f'dev-epoch-{best:02d}.jsonl'));third=next(r for r in s['records'] if r['epoch']==3);thirdm=calc(gold,rows(run/'dev-epoch-03.jsonl'));assert third['global_step']==1302
  for family in ['micro','character_type_micro']:
   assert all(close(v,s['metrics'][family][k]) for k,v in bestm[family].items()), 'best mismatch:'+path
   assert all(close(v,third['val'][family][k]) for k,v in thirdm[family].items()), 'third mismatch:'+path
  assert close(bestm['micro']['f1'],s['best_f1']);assert best==max(s['records'],key=lambda r:(r['val']['micro']['f1'],-r['epoch']))['epoch']
  results.append(dict(run=path,origin='new_supplement' if path.startswith(study['run_root']) else 'reused_completed_development',seed=c['seed'],encoder_lr=c['n1_encoder_lr'],rope=c['rope'],fixed_epoch3=thirdm,best_dev=bestm,best_epoch=best,actual_epochs=s['epochs'],epoch_optimizer_steps=[r['global_step'] for r in s['records']],training_order=[r['permutation_sha256'] for r in s['records']],train_seconds=sum(r['train_seconds'] for r in s['records']),validation_seconds=sum(r['validation']['seconds'] for r in s['records']),checkpoint_save_seconds=sum(r.get('save_seconds',0) for r in s['records']),peak_memory_mb=s['peak_memory_mb'],prediction_file=str((run/f'dev-epoch-{best:02d}.jsonl').relative_to(root)),prediction_sha256=sha(run/f'dev-epoch-{best:02d}.jsonl')))
 assert len(results)==10
 grid=[]
 for lr in study['learning_rates']:
  on=next(r for r in results if r['seed']==20261019 and r['encoder_lr']==lr and r['rope']);off=next(r for r in results if r['seed']==20261019 and r['encoder_lr']==lr and not r['rope']);n=min(len(on['training_order']),len(off['training_order']));assert on['training_order'][:n]==off['training_order'][:n]
  grid.append(dict(encoder_lr=lr,seed=20261019,on=on['best_dev']['micro']['f1'],off=off['best_dev']['micro']['f1'],delta_on_minus_off=on['best_dev']['micro']['f1']-off['best_dev']['micro']['f1'],fixed3_on=on['fixed_epoch3']['micro']['f1'],fixed3_off=off['fixed_epoch3']['micro']['f1']))
 paired=[]
 for seed in study['seeds']:
  on=next(r for r in results if r['seed']==seed and r['encoder_lr']==5e-5 and r['rope']);off=next(r for r in results if r['seed']==seed and r['encoder_lr']==5e-5 and not r['rope']);n=min(len(on['training_order']),len(off['training_order']));assert on['training_order'][:n]==off['training_order'][:n]
  paired.append(dict(seed=seed,on_best=on['best_dev']['micro']['f1'],off_best=off['best_dev']['micro']['f1'],delta_best=on['best_dev']['micro']['f1']-off['best_dev']['micro']['f1'],on_fixed3=on['fixed_epoch3']['micro']['f1'],off_fixed3=off['fixed_epoch3']['micro']['f1'],delta_fixed3=on['fixed_epoch3']['micro']['f1']-off['fixed_epoch3']['micro']['f1']))
 def agg(key):
  v=[r[key] for r in paired];return dict(values=v,mean=statistics.mean(v),sample_sd=statistics.stdev(v),ddof=1)
 return dict(status='PASS',runs=results,learning_rate_rope_grid=grid,paired_seeds=paired,paired_statistics={k:agg(k) for k in ['on_best','off_best','delta_best','on_fixed3','off_fixed3','delta_fixed3']},source_predictions_recounted=20,dev_samples=1503,training_data_only=True,API_calls=0,sealed_read=False,precision='fp32',fixed_epoch3_updates=1302,controls_checked=fixed,interpretation='retrospective internal-dev supplement;grid is one seed;selected-LR RoPE comparison is3paired seeds;no final score replacement or significance claim')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();at=datetime.now(timezone.utc).isoformat();start=time.perf_counter()
 try:r=evaluate_study(ROOT)
 except BaseException as ex:
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='FAIL',at=at,error_type=type(ex).__name__,error=str(ex),seconds=time.perf_counter()-start),ensure_ascii=False,indent=2));raise
 r.update(at=at,seconds=time.perf_counter()-start,verifier='AI-authored independent standard-library recount;not human review');a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=r['status'],runs=10,source_predictions_recounted=20,seconds=r['seconds']),ensure_ascii=False))
if __name__=='__main__':main()
