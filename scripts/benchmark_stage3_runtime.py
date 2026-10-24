"""Bounded, independent FP32 runtime audit; never replaces formal trial weights.

Install this adapter beside the existing project scripts. Only processed train/dev
are read. A complete last.pt is the shared starting state; see amendment evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

UUID = 'GPU-f77a1700-de70-05d2-048f-18c8f9aa5974'
REMOTE_ROOT = Path('/nfsdata_a40/cyf/codes/CT/cmeee_ner')


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def check_path(path, root):
    path = Path(path).resolve()
    path.relative_to(root)
    return path


def tensor_tree_cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: tensor_tree_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tensor_tree_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(tensor_tree_cpu(v) for v in value)
    return value


def compare_trees(a, b, atol, rtol):
    """Compare every tensor, including AdamW steps/moments and full gradients."""
    import numpy as np
    import torch
    result = dict(passed=True, tensors_checked=0, exact_tensors=0,
                  max_absolute_delta=0., failures=[], finite=True)

    def fail(path, reason):
        result['passed'] = False
        if len(result['failures']) < 16:
            result['failures'].append(dict(path=path, reason=reason))

    def visit(x, y, path):
        if torch.is_tensor(x):
            result['tensors_checked'] += 1
            if not torch.is_tensor(y) or x.shape != y.shape or x.dtype != y.dtype:
                fail(path, 'tensor shape/dtype changed'); return
            x, y = x.detach().cpu(), y.detach().cpu()
            if x.is_floating_point():
                if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                    result['finite'] = False; fail(path, 'nonfinite'); return
            if torch.equal(x, y):
                result['exact_tensors'] += 1; return
            if not x.is_floating_point():
                fail(path, 'non-floating tensor changed'); return
            delta = float((x - y).abs().max()) if x.numel() else 0.
            result['max_absolute_delta'] = max(result['max_absolute_delta'], delta)
            if not torch.allclose(x, y, atol=atol, rtol=rtol):
                fail(path, 'numeric tolerance exceeded')
        elif isinstance(x, np.ndarray):
            if not isinstance(y, np.ndarray) or not np.array_equal(x, y):
                fail(path, 'numpy RNG state changed')
        elif isinstance(x, dict):
            if not isinstance(y, dict) or x.keys() != y.keys():
                fail(path, 'dictionary keys changed'); return
            for k in x: visit(x[k], y[k], f'{path}/{k}')
        elif isinstance(x, (list, tuple)):
            if type(x) is not type(y) or len(x) != len(y):
                fail(path, 'sequence shape changed'); return
            for i, (xx, yy) in enumerate(zip(x, y)): visit(xx, yy, f'{path}/{i}')
        elif x != y:
            fail(path, 'non-tensor value changed')
    visit(a, b, '')
    return result


def worker(args):
    # Imports and checkpoint setup count in the parent's 110 second active budget.
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    sys.path.insert(0, str(args.root / 'scripts'))
    from ner import cuda_gate
    from ner_core import read_rows, windows, write_rows
    from ner_model import Extractor, configure, collate, loss_fn
    from stage3_common import evaluate
    start = time.perf_counter()
    out = args.output / args.worker
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    contract = plan['frozen']
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = parent['config']
    assert parent['method'] == 'N1' and cfg['micro_batch'] == 4 and cfg['accumulation'] == 8
    assert cfg['seed'] == contract['seed'] and not cfg['api_enabled']
    assert all(k in parent for k in ('optimizer', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'))
    gate = cuda_gate(cfg)
    configure(cfg['seed'], cfg['cpu_threads'])
    assert torch.cuda.device_count() == 1
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    model = Extractor('N1', cfg, parent['vocab'], parent['bert_config'])
    model.load_state_dict(parent['model'], strict=True)
    model.to('cuda')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('all model parameters must remain FP32')
    if torch.is_autocast_enabled('cuda'):
        raise ValueError('AMP/autocast is outside this benchmark contract')
    opt = torch.optim.AdamW([
        dict(params=model.encoder.parameters(), lr=cfg['n1_encoder_lr']),
        dict(params=model.head.parameters(), lr=cfg['n1_head_lr'])])
    opt.load_state_dict(parent['optimizer'])
    vocab = parent['vocab']
    initial_step, epoch = parent['global_step'], parent['epoch'] + 1
    initial_head = model.head.proj.weight.detach().clone()
    train = read_rows(args.root / 'data/processed/train.jsonl')
    ws = [w for r in train for w in windows(r, cfg['window_chars'], cfg['overlap_chars'])]
    order = list(range(len(ws)))
    random.Random(cfg['seed'] + epoch).shuffle(order)
    order += order[:(-len(order)) % 32]
    identities = [dict(id=ws[i]['id'], start=ws[i]['start'], end=ws[i]['end'], window_index=i)
                  for i in order[:42*32]]
    write_json(out / 'sample-order.json', dict(epoch=epoch, windows=identities,
        full_permutation_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest()))

    def rng():
        return dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

    def restore_rng(state):
        random.setstate(state['python']); np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])

    start_rng = dict(python=parent['python_rng'], numpy=parent['numpy_rng'],
                     torch=parent['torch_rng'], cuda=parent['cuda_rng'])
    # State is restored after module construction; data preparation uses no RNG.
    restore_rng(start_rng)
    del parent
    cached = []
    prep_start = time.perf_counter()
    if args.worker == 'cached_cpu_collation':
        # Exactly the same dynamic lengths, per-microbatch contents and dtype.
        # Two additional updates are reserved for the real-cycle replay below.
        # Their preparation cost is conservatively charged to the candidate.
        for j in range(0, 42*32, 4):
            batch = [ws[i] for i in order[j:j+4]]
            cached.append(collate(batch, vocab, 'N1', 'cpu'))
        bytes_cached = sum(x.numel()*x.element_size() for b in cached for x in b)
        if bytes_cached > 4*1024**3:
            raise ValueError('bounded CPU cache exceeds 4GiB')
    else:
        bytes_cached = 0
    preparation_seconds = time.perf_counter() - prep_start
    writer = SummaryWriter(str(out / 'tensorboard'))
    metrics_path = out / 'metrics.jsonl'
    global_step = initial_step
    trace = []
    profiles = []
    torch.cuda.reset_peak_memory_stats()

    def update(index, runtime, record=True):
        nonlocal global_step
        model.train(); opt.zero_grad(set_to_none=True)
        t = time.perf_counter(); collation_seconds = 0.; compute_seconds = 0.; value = 0.
        for micro in range(8):
            j = index*32 + micro*4
            tt = time.perf_counter()
            if runtime == 'cached_cpu_collation' and index < 42:
                ids, attn, chars, gold = (x.to('cuda') for x in cached[index*8+micro])
            else:
                batch = [ws[i] for i in order[j:j+4]]
                ids, attn, chars, gold = collate(batch, vocab, 'N1', 'cuda')
            collation_seconds += time.perf_counter() - tt
            tt = time.perf_counter()
            loss = loss_fn(model(ids, attn, chars), gold)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            (loss/8).backward(); value += float(loss.detach())/8
            compute_seconds += time.perf_counter() - tt
        tt = time.perf_counter()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step(); global_step += 1; torch.cuda.synchronize()
        optimizer_seconds = time.perf_counter() - tt
        seconds = time.perf_counter() - t
        row = dict(event='train', global_step=global_step, epoch=epoch,
                   window_cursor=(index+1)*32, loss=value, grad_norm=float(grad),
                   seconds=seconds, throughput_windows_per_sec=32/seconds,
                   peak_mb=torch.cuda.max_memory_allocated()/2**20,
                   runtime=runtime, scope='independent_runtime_benchmark_only')
        if record:
            trace.append(row)
            profiles.append(dict(collation_seconds=collation_seconds,
                                 compute_seconds=compute_seconds, optimizer_seconds=optimizer_seconds))
            with metrics_path.open('a') as stream: stream.write(json.dumps(row)+'\n')
            for key, number in {'train/loss':value, 'train/grad_norm':float(grad),
                'train/lr':opt.param_groups[0]['lr'], 'train/optimizer_step':global_step,
                'train/throughput_samples_per_sec':32/seconds,
                'system/gpu_memory_peak_mb':row['peak_mb']}.items():
                writer.add_scalar(key, number, global_step)
            if len(trace) % 10 == 0: print(json.dumps(row), flush=True)
        return row

    def snapshot(include_gradients=False):
        state = dict(model=tensor_tree_cpu(model.state_dict()),
                     optimizer=tensor_tree_cpu(opt.state_dict()), rng=tensor_tree_cpu(rng()),
                     global_step=global_step, scheduler=None)
        if include_gradients:
            state['gradients'] = {n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}
        return state

    def restore(state):
        nonlocal global_step
        model.load_state_dict(state['model'], strict=True)
        opt.load_state_dict(state['optimizer']); restore_rng(state['rng'])
        global_step = state['global_step']

    try:
        setup_seconds = time.perf_counter() - start
        torch.cuda.synchronize(); warm_start = time.perf_counter()
        for i in range(10): update(i, args.worker)
        torch.cuda.synchronize(); warm_seconds = time.perf_counter() - warm_start
        measure_start = time.perf_counter()
        for i in range(10,40): update(i, args.worker)
        torch.cuda.synchronize(); measured_seconds = time.perf_counter() - measure_start
        state = snapshot(include_gradients=True)
        parameter_changed = bool((model.head.proj.weight - initial_head).abs().max() > 0)
        assert parameter_changed
        report = dict(schema='autodl_gpu_tuning_report_v1', name=args.worker,
            workload_type='training', status='PENDING_EQUIVALENCE', complexity=0 if args.worker=='baseline' else 1,
            wall_time_sec=measured_seconds, throughput=30*32/measured_seconds,
            training_end_to_end_seconds=preparation_seconds+warm_seconds+measured_seconds,
            training_end_to_end_scope='all40 updates plus candidate cache preparation; unchanged checkpoint load is separately timed setup',
            throughput_unit='training windows/s', warmup_steps=10, measured_steps=30,
            setup_seconds=setup_seconds, warmup_seconds=warm_seconds,
            cpu_cache_preparation_seconds=preparation_seconds, cpu_cache_bytes=bytes_cached,
            cache_scope='42 deterministic updates;40 timed and2 reserved for real-cycle; all preparation charged',
            preparation_in_setup_and_total_wall=True, measured_step_seconds=[x['seconds'] for x in trace[10:]],
            phase_wall_diagnostics=profiles[10:], gate=gate, device=torch.cuda.get_device_name(),
            precision='fp32', amp=False, tf32=False, effective_batch_preserved=True,
            sample_identity_passed=True, aggregation_passed=True, leased_gpu_count=1, active_gpu_count=1,
            ddp_scaling_efficiency=None, ddp_efficiency_reason='not applicable; single authorized GPU',
            parameter_changed=parameter_changed, initial_global_step=initial_step,
            final_global_step=global_step, scheduler=None, error_type=None,
            parameters=dict(micro_batch=4, accumulation=8, effective_batch=32, workers=0,
                cpu_threads=cfg['cpu_threads'], runtime_change='none' if args.worker=='baseline' else
                'cache exact deterministic CPU collate tensors in bounded chunks before training; same blocking H2D'),
            evidence=dict(trace=str(metrics_path.relative_to(args.root)), samples=str((out/'sample-order.json').relative_to(args.root))))
        write_json(out/'candidate.json', report)
        if args.worker == 'baseline':
            torch.save(state, out/'end-state.pt')
            write_json(out/'trace.json', trace)
            report['worker_wall_seconds'] = time.perf_counter()-start
            write_json(out/'candidate.json', report)
            return
        baseline_dir = args.output/'baseline'
        baseline = torch.load(baseline_dir/'end-state.pt', map_location='cpu', weights_only=False)
        baseline_trace = json.loads((baseline_dir/'trace.json').read_text())
        tol = plan['equivalence']; equivalence_start = time.perf_counter()
        comparisons = {
            'parameters':compare_trees(baseline['model'], state['model'], tol['parameters_abs'], tol['relative']),
            'optimizer':compare_trees(baseline['optimizer'], state['optimizer'], tol['optimizer_state_abs'], tol['relative']),
            'gradients':compare_trees(baseline['gradients'], state['gradients'], tol['gradient_norm_abs'], tol['relative']),
            'rng':compare_trees(baseline['rng'], state['rng'], 0., 0.)}
        loss_delta=max(abs(a['loss']-b['loss']) for a,b in zip(baseline_trace,trace))
        grad_delta=max(abs(a['grad_norm']-b['grad_norm']) for a,b in zip(baseline_trace,trace))
        sequence_equal=len(trace)==len(baseline_trace)==40 and all(
            (a['global_step'],a['epoch'],a['window_cursor'])==(b['global_step'],b['epoch'],b['window_cursor'])
            for a,b in zip(baseline_trace,trace))
        samples_equal=(out/'sample-order.json').read_bytes()==(baseline_dir/'sample-order.json').read_bytes()
        equivalence=all(c['passed'] for c in comparisons.values()) and sequence_equal and samples_equal and \
            loss_delta<=tol['loss_abs'] and grad_delta<=tol['gradient_norm_abs'] and state['global_step']==baseline['global_step']
        eq=dict(passed=equivalence, full_state=comparisons, loss_max_abs=loss_delta,
            gradient_norm_max_abs=grad_delta, sample_sequence_equal=samples_equal,
            step_sequence_equal=sequence_equal, compared_updates=40,
            seconds=time.perf_counter()-equivalence_start, tolerances=tol)
        write_json(out/'equivalence.json',eq)
        report.update(status='PASS' if equivalence else 'NUMERIC_FAIL', equivalence_passed=equivalence,
                      error_type=None if equivalence else 'strict equivalence failed')
        base_report=json.loads((baseline_dir/'candidate.json').read_text())
        gain=report['throughput']/base_report['throughput']-1
        end_to_end_gain=base_report['training_end_to_end_seconds']/report['training_end_to_end_seconds']-1
        selected='cached_cpu_collation' if equivalence and end_to_end_gain>=plan['measurement']['material_gain_fraction'] else 'baseline'
        # Always restore the selected exact branch, before the real-cycle check.
        restore(state if selected=='cached_cpu_collation' else baseline)
        del state, baseline
        confirmation_start=time.perf_counter()
        before=update(40,selected,record=False)
        dev=read_rows(args.root/'data/processed/dev.jsonl')[:8]
        predictions, metrics, validation=evaluate(model,dev,vocab,cfg,'N1')
        write_rows(out/'confirmation-dev-predictions.jsonl',predictions)
        writer.add_scalar('val/loss',validation['loss'],1)
        writer.add_scalar('val/f1',metrics['micro']['f1'],1)
        saved=snapshot()
        save_start=time.perf_counter()
        checkpoint_path=out/'confirmation.pt';torch.save(saved,checkpoint_path)
        save_seconds=time.perf_counter()-save_start
        first=update(41,selected,record=False)
        uninterrupted=snapshot()
        restore_start=time.perf_counter()
        reloaded=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        # Full byte-content-level equality of serialized vs in-memory state.
        checkpoint_equal=compare_trees(saved,reloaded,0.,0.)
        restore(reloaded);del saved,reloaded
        resumed=update(41,selected,record=False)
        actual=snapshot()
        continuation_equal=compare_trees(uninterrupted,actual,tol['parameters_abs'],tol['relative'])
        trace_equal=abs(first['loss']-resumed['loss'])<=tol['loss_abs'] and \
            abs(first['grad_norm']-resumed['grad_norm'])<=tol['gradient_norm_abs'] and first['global_step']==resumed['global_step']
        restore_seconds=time.perf_counter()-restore_start
        confirmation=dict(status='PASS' if checkpoint_equal['passed'] and continuation_equal['passed'] and trace_equal else 'FAIL',
            sequence='real train update -> 8 real internal-dev samples -> checkpoint -> uninterrupted update vs restored same update',
            validation_scope='first 8 internal dev records; pipeline audit only, never model selection or final evaluation',
            validation=validation, metrics=metrics, sample_ids=[r['id'] for r in dev], train_before_validation=before,
            uninterrupted_next=first,resumed_next=resumed,checkpoint_exact_check=checkpoint_equal,
            continuation_full_state=continuation_equal,continuation_trace_equal=trace_equal,
            save_seconds=save_seconds,restore_and_replay_check_seconds=restore_seconds,
            seconds=time.perf_counter()-confirmation_start,checkpoint=str(checkpoint_path.relative_to(args.root)),
            checkpoint_sha256=sha(checkpoint_path),sealed_read=False,paid_api_calls=0)
        write_json(out/'confirmation.json',confirmation)
        report.update(selected_runtime=selected,throughput_gain_fraction=gain,
            end_to_end_gain_fraction=end_to_end_gain,
            selection_metric='40-update end-to-end windows/s including CPU cache preparation; 5% simplicity rule',
            selected_equivalence_passed=(equivalence if selected=='cached_cpu_collation' else True) and confirmation['status']=='PASS',
            confirmation=confirmation,worker_wall_seconds=time.perf_counter()-start)
        write_json(out/'candidate.json',report)
    finally:
        writer.flush();writer.close()


def telemetry():
    line=subprocess.check_output(['nvidia-smi','-i',UUID,
        '--query-gpu=uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=.75).strip()
    fields=[x.strip() for x in line.split(',')]
    if fields[0]!=UUID: raise RuntimeError('GPU UUID mismatch')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader,nounits'],text=True,timeout=.75)
    pids=[int(x.split(',')[0]) for x in apps.splitlines() if UUID in x]
    return dict(at=utc(),gpu_uuid=fields[0],utilization_pct=float(fields[1]),
        memory_used_mib=float(fields[2]),memory_total_mib=float(fields[3]),
        power_w=float(fields[4]),temperature_c=float(fields[5]),compute_pids=pids)


def controller(args):
    started=time.perf_counter();deadline=started+109.0-args.prior_attempt_seconds
    plan=json.loads(args.plan.read_text())
    if plan['budget_seconds']!=110: raise ValueError('fixed 110 second budget required')
    if args.root!=REMOTE_ROOT.resolve(): raise ValueError('remote root must be exact authorized project')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=UUID: raise ValueError('set authorized CUDA UUID explicitly')
    args.output.mkdir(parents=True,exist_ok=False)
    if args.summary.exists(): raise FileExistsError('performance summary exists; preserve it before a new attempt')
    attempts=[];failure=None;foreign=False
    write_json(args.output/'plan-amendment.json',dict(recorded_at=utc(),
        original_plan=str(args.plan.relative_to(args.root)),original_plan_sha256=sha(args.plan),
        original_checkpoint=plan['checkpoint'],actual_checkpoint=str(args.checkpoint.relative_to(args.root)),
        actual_checkpoint_sha256=sha(args.checkpoint),
        critical_sources={name:sha(args.root/'scripts'/name) for name in
            ('ner_core.py','ner_model.py','ner.py','train_development.py','stage3_common.py')},
        reason='best.pt contains inference state only, not optimizer/RNG; same-trial last.pt contains complete reproducible state',
        formal_best_and_selection_unchanged=True,scope='independent runtime benchmark only',
        authorization='user asks resume stage3 and high utilization; full-state checkpoint correction recorded before benchmark'))
    for name in ('baseline','cached_cpu_collation'):
        if time.perf_counter()>=deadline: failure='active tuning deadline before next candidate';break
        samples=[];next_sample=0.;t=time.perf_counter()
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',name,'--root',str(args.root),
             '--output',str(args.output),'--plan',str(args.plan),'--checkpoint',str(args.checkpoint),
             '--summary',str(args.summary)]
        console=args.output/(name+'.log')
        with console.open('x') as stream:
            process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=args.root)
            while process.poll() is None:
                now=time.perf_counter()
                if now>=deadline:
                    failure='active tuning deadline';process.kill();process.wait(timeout=.5);break
                if now>=next_sample and now<deadline-2:
                    try:
                        item=telemetry();samples.append(item)
                        if any(pid!=process.pid for pid in item['compute_pids']):
                            foreign=True;failure='foreign GPU process appeared; own benchmark stopped'
                            process.kill();process.wait(timeout=.5);break
                    except Exception as error:
                        failure='telemetry/gate failure: '+repr(error)
                        process.kill();process.wait(timeout=.5);break
                    next_sample=now+1
                time.sleep(.05)
        report_path=args.output/name/'candidate.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else dict(
            schema='autodl_gpu_tuning_report_v1',name=name,status='ERROR',equivalence_passed=False)
        report.update(subprocess_returncode=process.returncode,parent_observed_wall_seconds=time.perf_counter()-t,
                      console=str(console.relative_to(args.root)))
        # Telemetry contains setup, warm-up, measured loop and state comparisons;
        # retain timestamps and never label full-child mean as training-only.
        util=[s['utilization_pct'] for s in samples]
        report.update(gpu_util_mean_pct=[sum(util)/len(util)] if util else [],
            gpu_util_p10_pct=[sorted(util)[max(0,math.ceil(len(util)*.1)-1)]] if util else [],
            gpu_memory_peak_mib=[max(s['memory_used_mib'] for s in samples)] if samples else [],
            memory_total_mib=[samples[-1]['memory_total_mib']] if samples else [],
            telemetry_scope='entire fresh subprocess including setup/state checking; raw timestamps retained',
            telemetry=samples)
        write_json(args.output/(name+'-report.json'),report);attempts.append(report)
        if process.returncode!=0 or failure:
            failure=failure or 'candidate subprocess failure';break
    finished=time.perf_counter();elapsed=finished-started
    completed=len(attempts)==2 and not failure and attempts[-1].get('selected_equivalence_passed',False)
    selected=attempts[-1].get('selected_runtime') if attempts else None
    selected_report=next((r for r in attempts if r['name']==selected),{})
    memory_pass=all(not r.get('gpu_memory_peak_mib') or r['gpu_memory_peak_mib'][0] <
        r['memory_total_mib'][0]*plan['measurement']['memory_ceiling_fraction'] for r in attempts)
    completed=completed and memory_pass and elapsed+args.prior_attempt_seconds<=110
    if completed:
        attempts[0].update(status='PASS',equivalence_passed=True,
            baseline_gate_basis='original reference operations; full sample/step contract; selected-runtime validation/checkpoint/replay passed')
    for report in attempts:
        write_json(args.output/(report['name']+'-report.json'),report)
    baseline_report=attempts[0] if attempts else {}
    phases=baseline_report.get('phase_wall_diagnostics',[])
    collation=sum(x['collation_seconds'] for x in phases)
    all_phases=sum(sum(x.values()) for x in phases)
    summary=dict(status='completed' if completed else 'incomplete',recorded_at=utc(),
        tuning_budget_seconds=110,tuning_total_seconds=elapsed+args.prior_attempt_seconds,
        this_attempt_seconds=elapsed,prior_failed_setup_seconds=args.prior_attempt_seconds,
        active_wall_definition='parent preflight/amendment, fresh-process imports/setup, all warmup/measurement, cache preparation, equivalence, real cycle, telemetry; queue not included',
        selected_runtime=selected if completed else 'original runtime retained; bounded audit incomplete',
        selected_equivalence_passed=bool(completed),candidates=attempts,
        selected_equivalence_basis='full loss/grad-norm sequence, final full gradients/model/optimizer/RNG, sample sequence; baseline retained if candidate fails or gain<5%',
        baseline_preparation_fraction=collation/all_phases if all_phases else None,
        baseline_throughput=baseline_report.get('throughput'),selected_throughput=selected_report.get('throughput'),
        baseline_end_to_end_seconds=baseline_report.get('parent_observed_wall_seconds'),
        selected_end_to_end_seconds=selected_report.get('parent_observed_wall_seconds'),
        confirmation=attempts[-1].get('confirmation') if attempts else None,
        memory_gate_passed=memory_pass,foreign_process_detected=foreign,
        good_enough_gate_reached=completed and bool(selected_report.get('gpu_util_mean_pct')) and
            selected_report['gpu_util_mean_pct'][0]>=85,
        rejected_reasons=[dict(name=r['name'],reason=r.get('error_type') or 'less than 5% gain; prefer simpler original loop')
            for r in attempts if r['name']!=selected],
        remaining_bottleneck='single A40; FP32, micro4/accum8 and dynamic original shapes fixed; no more tuning budget; telemetry includes intentional setup/verification stalls',
        error=failure,checkpoint=str(args.checkpoint.relative_to(args.root)),
        plan_amendment=str((args.output/'plan-amendment.json').relative_to(args.root)),
        formal_stage3_weights_unchanged=True, sealed_read=False, paid_api_calls=0,
        estimated_N1_train_seconds_if_same_window_mix=(434*32/selected_report['throughput'])
            if selected_report.get('throughput') else None,
        eta_note='one epoch training extrapolation only; add observed full-dev validation and checkpoint overhead; no full-job runtime promise')
    write_json(args.summary,summary)
    print(json.dumps({k:summary[k] for k in ('status','tuning_total_seconds','selected_runtime','selected_equivalence_passed','error')}),flush=True)
    return 0 if completed else 2


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=REMOTE_ROOT)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--summary',type=Path)
    parser.add_argument('--worker',choices=['baseline','cached_cpu_collation'])
    parser.add_argument('--prior-attempt-seconds',type=float,default=0.)
    args=parser.parse_args();args.root=args.root.resolve()
    args.plan=check_path(args.plan or args.root/'configs/performance-tuning-plan.json',args.root)
    args.checkpoint=check_path(args.checkpoint or args.root/'runs/stage3-20261019/N1-lr-5e-05/last.pt',args.root)
    args.output=check_path(args.output,args.root)
    args.summary=check_path(args.summary or args.root/'evidence/stage3/performance-summary.json',args.root)
    if args.worker:
        try: worker(args)
        except Exception as error:
            write_json(args.output/(args.worker+'-failure.json'),dict(error=repr(error),at=utc(),traceback=traceback.format_exc()))
            raise
        return 0
    return controller(args)


if __name__=='__main__':
    raise SystemExit(main())
