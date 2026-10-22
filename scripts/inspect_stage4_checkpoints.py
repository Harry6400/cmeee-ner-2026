"""Independent CPU-only inspection of all six selected/last checkpoints and real training evidence."""
import argparse,ast,gc,hashlib,json,math,random,time
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ner_core import ROOT,digest,dump
from ner_model import Extractor

def finite(obj):
    if isinstance(obj,torch.Tensor):return bool(torch.isfinite(obj).all())
    if isinstance(obj,dict):return all(finite(v) for v in obj.values())
    if isinstance(obj,(list,tuple)):return all(finite(v) for v in obj)
    return not isinstance(obj,float) or math.isfinite(obj)
def json_equivalent(actual,expected):
    # JSON freezes sequences as arrays; AdamW returns betas as a tuple.
    # Normalize only JSON representation while retaining every value and group key.
    def canonical(value):return json.loads(json.dumps(value,allow_nan=False))
    return canonical(actual)==canonical(expected)

def epoch_loop(path):
    t=ast.parse(path.read_text());nodes=[n for n in ast.walk(t) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch']
    if len(nodes)!=1:raise ValueError('epoch loop ambiguous')
    return ast.dump(nodes[0],include_attributes=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,default=ROOT/'configs/stage4-frozen-protocol.json');p.add_argument('--contract',type=Path,default=ROOT/'configs/qha-stage4-contract.json');p.add_argument('--output',type=Path,default=ROOT/'evidence/stage4/checkpoint-inspection.json');a=p.parse_args()
    if a.output.exists():raise ValueError('inspection evidence already exists; do not overwrite')
    torch.set_num_threads(4);protocol=json.loads(a.protocol.read_text());contract=json.loads(a.contract.read_text());start=time.perf_counter();results={}
    if contract['frozen_protocol_sha256']!=digest(a.protocol):raise ValueError('protocol changed')
    if epoch_loop(ROOT/'scripts/train_development.py')!=epoch_loop(ROOT/'scripts/train_stage4.py'):raise ValueError('scientific training/validation/save loop differs')
    for name in ['ner_core.py','ner_model.py','stage3_common.py']:
        if digest(ROOT/'scripts'/name)!=digest(ROOT/'evidence/stage3/source-snapshot'/name):raise ValueError('original scientific module changed:'+name)
    for method in ['N0','N1']:
        for seed in protocol['training']['seeds']:
            t=time.perf_counter();name=f'{method}-seed{seed}';run=ROOT/('runs/stage3-20261019/'+protocol['selected'][method]['run']) if seed==20261019 else ROOT/'runs/stage4-20261020'/name
            s=json.loads((run/'summary.json').read_text());cfg=dict(protocol['selected'][method]['config']);cfg['seed']=seed
            if seed!=20261019:cfg['stage']=4
            if s['status'] not in ['completed','early_stopped'] or json.loads((run/'config.json').read_text())!=cfg:raise ValueError('summary/config mismatch:'+name)
            if s['global_step']!=s['epochs']*434:raise ValueError('completed epoch/globalstep mismatch:'+name)
            if s['best_epoch']!=min(r['epoch'] for r in s['records'] if abs(r['val']['micro']['f1']-s['best_f1'])<1e-12):raise ValueError('own devbest selection mismatch')
            for rec in s['records']:
                order=list(range(13883));random.Random(seed+rec['epoch']).shuffle(order);order+=order[:5]
                if hashlib.sha256(json.dumps(order).encode()).hexdigest()!=rec['permutation_sha256'] or rec['padded_training_windows']!=5:raise ValueError('training window permutation mismatch')
            metadata=json.loads((run/'metadata.json').read_text())
            if metadata['seed']!=seed or metadata['precision']!='fp32' or metadata['amp'] or metadata['tf32'] or (metadata['batch'],metadata['micro_batch'],metadata['accumulation'])!=(32,4,8):raise ValueError('precision/batch/seed metadata mismatch')
            actual=digest(run/'best.pt')
            if actual!=s['best_checkpoint_sha256']:raise ValueError('best content changed')
            best=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
            if any(x.is_floating_point() and x.dtype!=torch.float32 for x in best['model'].values()):raise ValueError('best actual dtype differs fromFP32')
            if best['config']!=cfg or best['method']!=method or best['epoch']!=s['best_epoch'] or not finite(best['model']):raise ValueError('best state/config mismatch')
            model=Extractor(method,cfg,best['vocab'],best['bert_config']);model.load_state_dict(best['model'],strict=True)
            if method=='N0':
                torch.manual_seed(seed);initial=Extractor(method,cfg,best['vocab'],best['bert_config']);delta=(best['model']['embedding.weight']-initial.state_dict()['embedding.weight']).abs().max().item();del initial
                basis='N0 embedding vs own fixed-seed module initialization'
            else:
                from safetensors import safe_open
                with safe_open(str(ROOT/cfg['bert_directory']/'model.safetensors'),framework='pt',device='cpu') as source:
                    key=next(k for k in source.keys() if k.endswith('embeddings.word_embeddings.weight'));original=source.get_tensor(key);delta=(best['model']['encoder.embeddings.word_embeddings.weight']-original).abs().max().item();del original
                basis='N1 encoder embedding vs pinned original BERT'
            if not delta>0:raise ValueError('parameters did not update')
            del best,model;gc.collect()
            last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
            if any(x.is_floating_point() and x.dtype!=torch.float32 for x in last['model'].values()):raise ValueError('last actual dtype differs fromFP32')
            if last['config']!=cfg or not finite(last['model']) or not finite(last['optimizer']):raise ValueError('last nonfinite/config')
            model=Extractor(method,cfg,last['vocab'],last['bert_config']);model.load_state_dict(last['model'],strict=True);del model;gc.collect()
            steps=sorted({int(v['step'].item()) for v in last['optimizer']['state'].values() if 'step' in v})
            if steps!=[s['global_step']] or (last['epoch'],last['global_step'],last['best_epoch'])!=(s['epochs'],s['global_step'],s['best_epoch']):raise ValueError('last optimizer/step mismatch')
            groups=[{k:v for k,v in group.items() if k!='params'} for group in last['optimizer']['param_groups']]
            if not json_equivalent(groups,protocol['training']['optimizer']['actual_groups'][method]):raise ValueError('optimizer group settings differ')
            metricpaths=sorted(run.glob('segments/*/metrics.jsonl'))+([run/'metrics.jsonl'] if (run/'metrics.jsonl').exists() else []);updates=0;partial_replayed=0
            for metric in metricpaths:
                records=[json.loads(x) for x in metric.read_text().splitlines() if x];tr=[x for x in records if x.get('event')=='train'];updates+=len(tr)
                if any(not math.isfinite(x[k]) for x in tr for k in ['loss','grad_norm','seconds','throughput_windows_per_sec','peak_mb']):raise ValueError('nonfinite train event')
                if any(y['global_step']!=x['global_step']+1 for x,y in zip(tr,tr[1:])):raise ValueError('within-segment globalstep discontinuity')
                if tr and any(x['window_cursor']%32 for x in tr):raise ValueError('effective windowbatch incorrect')
            if updates<s['global_step']:raise ValueError('missing training logs')
            tags=set();eventfiles=0
            for tbdir in sorted(run.glob('segments/*/tensorboard'))+([run/'tensorboard'] if (run/'tensorboard').exists() else []):
                acc=EventAccumulator(str(tbdir),size_guidance={'scalars':0});acc.Reload();eventfiles+=len(list(tbdir.glob('events.out.tfevents.*')));available=acc.Tags()['scalars'];tags.update(available)
                for tag in available:
                    if any(not math.isfinite(x.value) for x in acc.Scalars(tag)):raise ValueError('nonfinite TensorBoard scalar')
            required={'train/loss','train/grad_norm','train/lr','train/optimizer_step','train/throughput_samples_per_sec','train/epoch_time_sec','system/gpu_memory_peak_mb','val/precision','val/recall','val/f1','val/nested_recall','val/character_f1'}
            if not required<=tags or eventfiles==0:raise ValueError('required TensorBoard tag/event missing')
            results[name]=dict(status='PASS',training_run=str(run.relative_to(ROOT)),best_epoch=s['best_epoch'],last_epoch=s['epochs'],global_step=s['global_step'],best_sha256=actual,last_sha256=digest(run/'last.pt'),cpu_strict_load=True,finite_weights_and_optimizer=True,optimizer_steps=steps,optimizer_groups=groups,parameter_update_check=dict(basis=basis,max_absolute_delta=delta,scope='one named tensor; not every parameter'),train_events=updates,replayed_partial_train_events=updates-s['global_step'],effective_batch=32,window_permutation_all_epochs_verified=True,precision='fp32',amp=False,tf32=False,tensorboard=dict(eventfiles=eventfiles,required_tags=sorted(required),all_scalars_finite=True),seconds=time.perf_counter()-t)
            del last;gc.collect();print(json.dumps(dict(name=name,status='PASS',seconds=time.perf_counter()-t)),flush=True)
    dump(a.output,dict(status='PASS',device='cpu',models=results,source_epoch_loop_identical=True,scientific_core_modules_unchanged=True,holdout_accessed=False,paid_api_calls=0,seconds=time.perf_counter()-start,protocol_sha256=digest(a.protocol),contract_sha256=digest(a.contract)));print(json.dumps(dict(status='PASS',models=list(results),seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
