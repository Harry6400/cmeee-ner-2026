"""仅核查阶段二已有证据，不训练、不读holdout、不调用API。"""
import argparse,json,math,sys
from pathlib import Path
from ner_core import ROOT,digest,read_rows,score
from historical_evidence import historical_source,historical_budget,historical_isolation
from stage4_metrics import dump_new


def verify(root,run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    cfg=json.loads((root/'configs/stage2.json').read_text())
    budget=json.loads((run/'gpu-budget.json').read_text())
    if budget['status']!='completed' or budget['consumed_seconds']>cfg['gpu_total_seconds']:
        raise ValueError('GPU阶段未完成或超时')
    subset=root/'runs/stage2-subset'
    gold=read_rows(subset/'dev.jsonl')
    if len(gold)!=64 or len(read_rows(subset/'train.jsonl'))!=128:
        raise ValueError('样本规模不符')
    results={}
    train_tags={'train/loss','train/lr','train/grad_norm','train/optimizer_step',
                'train/throughput_samples_per_sec','system/gpu_memory_peak_mb'}
    for method in ('N0','N1'):
        first=run/(method+'-initial');second=run/(method+'-resume')
        a=json.loads((first/'summary.json').read_text());b=json.loads((second/'summary.json').read_text())
        restore=json.loads((second/'restore-check.json').read_text())
        if (a['global_step'],b['global_step'],a['cursor'],b['cursor'])!=(4,6,128,192):
            raise ValueError('步数或采样位置不连续')
        if not (a['parameter_updated'] and b['parameter_updated']):raise ValueError('参数未更新')
        if not all(restore[k] for k in ('model_equal','python_rng_equal','numpy_rng_equal','torch_rng_equal','cuda_rng_equal')):
            raise ValueError('恢复状态不一致')
        if restore['optimizer_steps']!=[4]:raise ValueError('优化器步数未恢复')
        histories=[];tags={}
        for segment in (first,second):
            rows=read_rows(segment/'metrics.jsonl')
            start=rows[0]
            for name,expected in start['code_sha256'].items():
                historical_source(root,'scripts/'+name,expected)
            histories += [r for r in rows if r['event']=='optimizer_step']
            ea=EventAccumulator(str(segment/'tensorboard'));ea.Reload()
            present=set(ea.Tags()['scalars'])
            if not train_tags<=present:raise ValueError('TensorBoard缺失训练标签')
            for tag in train_tags:
                if not all(math.isfinite(e.value) for e in ea.Scalars(tag)):raise ValueError('非有限TensorBoard值')
            tags[segment.name]=sorted(present)
        if [r['global_step'] for r in histories]!=[1,2,3,4,5,6]:raise ValueError('训练历史不连续')
        if [x['cursor'] for r in histories for x in r['samples']]!=list(range(192)):
            raise ValueError('采样cursor存在缺失或重复')
        manifest=json.loads((subset/'manifest.json').read_text())
        from ner_core import windows
        import random
        cfg=json.loads((root/'configs/stage2.json').read_text())
        train=read_rows(subset/'train.jsonl')
        ws=[w for r in train for w in windows(r,cfg['window_chars'],cfg['overlap_chars'])]
        order=list(range(len(ws)));random.Random(cfg['seed']).shuffle(order)
        for r in histories:
            for x in r['samples']:
                w=ws[order[x['cursor']%len(order)]]
                if (x['id'],x['start'])!=(w['id'],w['start']):raise ValueError('恢复后的实际样本顺序不符')
        cpu=root/'runs'/('stage2-'+method+'-cpu')
        metrics=score(gold,read_rows(cpu/'predictions.jsonl'))
        stored=json.loads((cpu/'metrics.json').read_text())
        if metrics!=stored:raise ValueError('指标无法重算')
        cpus=json.loads((cpu/'summary.json').read_text())
        if cpus['device']!='cpu' or cpus['samples']!=64:raise ValueError('CPU检查未通过')
        if digest(second/'inference.pt')!=b['inference_sha256'] or cpus['checkpoint_sha256']!=b['inference_sha256']:
            raise ValueError('本机权重与远端检查点不一致')
        ea=EventAccumulator(str(cpu/'tensorboard'));ea.Reload()
        if not {'val/precision','val/recall','val/f1','val/nested_recall'}<=set(ea.Tags()['scalars']):
            raise ValueError('CPU开发指标未记录到TensorBoard')
        results[method]=dict(steps=6,cursor=192,restore=restore,cpu=cpus,tensorboard_tags=tags,
                             peak_memory_mb=max(a['peak_memory_mb'],b['peak_memory_mb']))
    ledger,original,budget_check=historical_budget(root,2)
    if ledger['requests'][:len(original['requests'])]!=original['requests'] or original['spent_cny']!='3.792488' or ledger['api_enabled']:
        raise ValueError('费用或API状态改变')
    isolation=historical_isolation(root,2)
    return dict(status='PASS_WITH_WARNINGS',gpu_budget=budget,models=results,
                external_isolation=isolation,historical_budget=budget_check,paid_api_calls=0,
                cost_cny=original['spent_cny'],historical_cumulative_cny=ledger['spent_cny'],limit_cny=ledger['limit_cny'],
                sealed_contents_read_for_development=False,full_training=False,
                note='smoke validation only; not formal model quality; synthetic tests separate')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Historical verification output already exists')
    result=verify(ROOT,a.run);dump_new(a.output,result)
    print(json.dumps(dict(status=result['status'],gpu_seconds=result['gpu_budget']['consumed_seconds'],
                          models=list(result['models']),cost_cny=result['cost_cny']),ensure_ascii=False))
