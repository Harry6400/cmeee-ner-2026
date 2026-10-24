"""Summarize preserved stage4 engineering failures, actual recoveries and timing scopes.
Read-only local evidence;no network,training,API,budget mutation or metric recomputation.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from datetime import datetime,timezone,timedelta
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def read_lines(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
def finite(value):return type(value) in (int,float) and math.isfinite(value) and value>=0

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()


def wall(data):
    if not data.get('started_at') or not data.get('finished_at'):return None
    start=datetime.fromisoformat(data['started_at']);end=datetime.fromisoformat(data['finished_at'])
    if start.tzinfo is None or end.tzinfo is None:return None
    seconds=(end-start).total_seconds()
    return seconds if seconds>=0 else None


def local_time(value):
    if not value:return '未记录'
    try:
        dt=datetime.fromisoformat(value)
        if dt.tzinfo is None:return value+'（未注明时区）'
        return dt.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')+'（北京时间）'
    except ValueError:return value


def sec(value):return f'{value:.3f}秒' if finite(value) else '未记录/待取回'


def normalized_path(root,value):
    if not value:return None
    path=Path(value)
    for prefix in (str(root),'/nfsdata_a40/cyf/codes/CT/cmeee_ner'):
        if str(path).startswith(prefix+'/'):return str(path)[len(prefix)+1:]
    return str(path) if not path.is_absolute() else None


def link(root,value):
    rel=normalized_path(root,value)
    if rel and (root/rel).is_file():return f'[{Path(rel).name}](../{rel})'
    return f'待取回：`{rel or value}`'


def collect(root,execution=None):
    base=root/'evidence/stage4';source_paths=set();warnings=[]
    def optional(rel):
        path=root/rel
        if not path.is_file():return None
        source_paths.add(rel);return read(path)
    issues_path=base/'issues.jsonl';issues=read_lines(issues_path) if issues_path.is_file() else []
    if issues_path.is_file():source_paths.add('evidence/stage4/issues.jsonl')
    else:warnings.append('缺少issues.jsonl；不能把未记录的失败补写成已留存证据。')
    operations=[]
    for path in sorted(base.glob('*.json')):
        data=read(path)
        if isinstance(data,dict) and isinstance(data.get('operation'),str) and finite(data.get('seconds')):
            rel=str(path.relative_to(root));source_paths.add(rel)
            operations.append(dict(name=data['operation'],seconds=data['seconds'],returncode=data.get('returncode'),started_at=data.get('started_at'),finished_at=data.get('finished_at'),record=rel,console=normalized_path(root,data.get('console')),scope='record_operation outer subprocess wall;not measured CPU utilization'))
    controllers=[]
    paths=set(base.glob('post-training-controller*.json'))|set(base.glob('finalize-controller*.json'))
    paths|=set(base.glob('api*controller*.json'))
    for path in sorted(paths):
        data=read(path);rel=str(path.relative_to(root));source_paths.add(rel)
        controller=dict(path=rel,status=data.get('status'),started_at=data.get('started_at'),finished_at=data.get('finished_at'),timestamp_wall_seconds=wall(data),error=data.get('error'),operations=[])
        for op in data.get('operations',[]):
            controller['operations'].append({k:op.get(k) for k in ('name','status','returncode','seconds','started_at','finished_at','console','script')})
        retries=[o for o in data.get('observations',[]) if 'retry' in str(o.get('status','')).lower()]
        controller['read_only_transport_retry_count']=len(retries);controller['read_only_transport_retry_observations']=retries
        for key in ('training_wait_seconds','wait_seconds','training_and_CPU_wait_seconds','API_seconds','API_started_at','settled_calls_logged'):
            if key in data:controller[key]=data[key]
        controllers.append(controller)
    post_ops=[dict(op,controller=c['path']) for c in controllers if Path(c['path']).name.startswith('post-training-controller') for op in c['operations']]
    checkpoints=optional('evidence/stage4/checkpoint-inspection.json')
    checkpoint_pass=bool(checkpoints and checkpoints.get('status')=='PASS' and len(checkpoints.get('models',{}))==6 and all(m.get('status')=='PASS' for m in checkpoints['models'].values()))
    d0=optional('runs/stage4-20261020/local-evaluation/D0/summary.json')
    d0_pass=bool(d0 and d0.get('status')=='completed')
    regression=optional('evidence/stage4/optimizer-settings-regression.json')
    source_transfer=optional('evidence/stage4/source-snapshot-transfer-check.json')
    dependency=optional('evidence/stage4/baseline-dependency-recovery.json')
    dependency_checks={}
    if dependency:
        for rel,want in dependency.get('existing_original_files',{}).items():
            if (root/rel).is_file():source_paths.add(rel);dependency_checks[rel]=digest(root/rel)==want
            else:dependency_checks[rel]=None
    interpreted=[]
    for raw in issues:
        item=dict(id=raw.get('id','未编号'),kind=raw.get('kind'),phenomenon=raw.get('phenomenon'),cause_basis=raw.get('cause_evidence',raw.get('cause_basis')),change=raw.get('change'),historical_retest_note=raw.get('retest'),recorded_at=raw.get('recorded_at',raw.get('at')),source='evidence/stage4/issues.jsonl',not_an_extraction_error=True,retest_status='RECORDED_NOTE_ONLY',retest_description='仅保留当时结构化记录；未将其当作新的实测通过。',evidence_paths=list(raw.get('evidence',raw.get('evidence_paths',[]))))
        ident=item['id']
        if any(word in str(raw.get('kind','')).lower() for word in ('preventive','prevention','code_review')):
            item.update(retest_status='PREVENTIVE_REVIEW_NOTE',retest_description='执行前代码审查发现及预防性处理，不表述为已经发生的结果覆盖或运行失败；实际是否尚未执行步骤由对应原journal核对，后续实跑终态另列。',evidence_paths=list(raw.get('old_evidence',[])))
        if ident=='ST4-I01':
            retest=optional('evidence/stage4/metrics-synthetic-tests-retest.json');passed=bool(retest and retest.get('returncode')==0)
            item.update(retest_status='PASS' if passed else 'PENDING',retest_description='实际反例测试进程返回0；属于合成指标/集成检查，不是模型效果实验。' if passed else '缺少成功反例测试运行记录，待核对。',evidence_paths=['evidence/stage4/metrics-synthetic-tests.log','evidence/stage4/metrics-synthetic-tests.json','evidence/stage4/metrics-synthetic-tests-retest.log','evidence/stage4/metrics-synthetic-tests-retest.json'])
        elif ident=='ST4-I02':
            passed=bool(source_transfer and source_transfer.get('status')=='PASS' and source_transfer.get('original_stage3_snapshot_transferred') and source_transfer.get('running_training_source_unchanged'))
            item.update(retest_status='PASS' if passed else 'PENDING',retest_description='既有源快照传输后，本地/远端内容一致检查通过，记录显示训练源未修改。' if passed else '源快照传输校验待取回。',evidence_paths=['evidence/stage4/source-snapshot-transfer.json','evidence/stage4/source-snapshot-transfer.log','evidence/stage4/source-snapshot-transfer-check.json'])
        elif ident=='ST4-I03':
            item.update(retest_status='RETROSPECTIVE_TOOL_NOTE',retest_description='原工具PermissionError与内存compile成功由结构化工具错误记录追记；没有独立原控制台文件，不补造历史终端或耗时。',evidence_paths=['evidence/stage4/issues.jsonl'])
        elif ident=='ST4-I04':
            passed=checkpoint_pass and bool(regression and regression.get('returncode')==0)
            item.update(retest_status='PASS' if passed else 'PENDING',retest_description=f"JSON序列表示归一的回归检查返回0；六模型实际CPU严格加载、权重/优化器有限性及状态检查{'通过' if checkpoint_pass else '待检'}；检查器自报耗时{sec(checkpoints.get('seconds') if checkpoints else None)}。训练参数或检查点未据此重新优化。" if passed else '优化器表示回归检查或六模型实际CPU复测仍缺终态证据。',evidence_paths=['evidence/stage4/checkpoint-inspection.log','evidence/stage4/post-training-controller.json','evidence/stage4/finalize-controller.json','evidence/stage4/source-before-recovery/inspect_stage4_checkpoints.py','scripts/inspect_stage4_checkpoints.py','evidence/stage4/optimizer-settings-regression.json','evidence/stage4/optimizer-settings-regression.log','evidence/stage4/checkpoint-inspection.json'])
        elif ident=='ST4-I05':
            transfer=optional('evidence/stage4/baseline-dependency-transfer.json');passed=d0_pass and bool(transfer and transfer.get('returncode')==0) and dependency_checks and all(v is True for v in dependency_checks.values())
            item.update(retest_status='PASS' if passed else 'PENDING',retest_description='原有基线模块、词典和开发汇总补传后，传输命令返回0；本机文件仍与补传记录中的原文件一致，真实D0全流程已完成。后续GPU评估终态另查队列，不把D0通过等同于全部阶段四完成。' if passed else '依赖传输、原文件一致性或真实D0完成记录尚不齐备，待取回。',evidence_paths=['evidence/stage4/dictionary-evaluation-recovery01.log','evidence/stage4/post-training-controller-recovery01.json','evidence/stage4/finalize-controller-recovery01.json','evidence/stage4/baseline-dependency-recovery.json','evidence/stage4/baseline-dependency-transfer.json','evidence/stage4/baseline-dependency-transfer.log','runs/stage4-20261020/local-evaluation/D0/summary.json','evidence/stage4/post-training-controller-recovery02.json'])
        for path in item['evidence_paths']:
            if (root/path).is_file():source_paths.add(path)
        item['unavailable_evidence_paths']=[p for p in item['evidence_paths'] if not (root/p).is_file()]
        interpreted.append(item)
    timings=[]
    def time_row(name,value,source,scope,overlap,kind='measured',status=None):
        timings.append(dict(name=name,seconds=value if finite(value) else None,source=source,scope=scope,overlap_group=overlap,kind=kind,status=status or ('available' if finite(value) else 'pending')))
    train=optional('evidence/stage4/training-queue.json');train_runs=[]
    if train:
        child=wait=runner=discarded_seconds=0.;discarded_updates=0;runner_complete=child_complete=wait_complete=discard_complete=True;epoch_train=epoch_dev=epoch_save=0.;epoch_complete=True
        for run in train.get('runs',[]):
            if run.get('seed') not in (20261020,20261021):warnings.append('训练队列含非补充种子，未计入新增训练汇总。');continue
            detail=dict(name=run['name'],status=run.get('status'),wait_seconds=run.get('wait_seconds'),segments=[],reused_historical=False)
            wait+=run.get('wait_seconds',0) if finite(run.get('wait_seconds')) else 0
            if not finite(run.get('wait_seconds')):wait_complete=False
            for segment in run.get('segments',[]):
                value=segment.get('wall_seconds');child+=value if finite(value) else 0
                if not finite(value):child_complete=False;warnings.append('训练分段缺少终态wall_seconds：'+run['name'])
                path=(segment.get('archive_dir')+'/summary.json') if segment.get('archive_dir') else f"runs/stage4-20261020/{run['name']}/summary.json"
                summary=optional(path)
                if summary and finite(summary.get('seconds')) and segment.get('status')!='prelaunch_busy_race':runner+=summary['seconds']
                elif segment.get('status')!='prelaunch_busy_race':runner_complete=False
                if segment.get('status')=='interrupted' and not finite(segment.get('discarded_partial_update_seconds')):discard_complete=False
                discarded_updates+=segment.get('discarded_partial_updates',0);discarded_seconds+=segment.get('discarded_partial_update_seconds',0)
                detail['segments'].append(dict(status=segment.get('status'),wall_seconds=value,runner_summary=path,runner_seconds=summary.get('seconds') if summary else None,discarded_partial_updates=segment.get('discarded_partial_updates',0),console=segment.get('console')))
            final=optional(f"runs/stage4-20261020/{run['name']}/summary.json")
            if final:
                for record in final.get('records',[]):
                    values=(record.get('train_seconds'),record.get('validation',{}).get('seconds'),record.get('save_seconds'))
                    if all(finite(x) for x in values):epoch_train+=values[0];epoch_dev+=values[1];epoch_save+=values[2]
                    else:epoch_complete=False
            else:epoch_complete=False
            train_runs.append(detail)
        time_row('四次补训：训练子进程墙钟合计',child if child_complete else None,'evidence/stage4/training-queue.json','仅seed20/21；含初始化、开发验证、保存及控制轮询尾部；所有真实分段计入','training_outer',status=train.get('status'))
        time_row('四次补训：训练器自报全部分段seconds',runner if runner_complete else None,'runs/stage4-20261020/*/summary.json','运行器记录的分段时钟；含在训练子进程墙钟内，不能与其相加','training_outer')
        time_row('补训资源门槛轮询累计等待',wait if wait_complete else None,'evidence/stage4/training-queue.json','包含只读资源观察；不代表新增GPU计算；不包含所有控制器/磁盘检查开销','wait')
        for name,value in [('已提交完整轮的训练循环',epoch_train),('已提交完整轮的内部开发验证',epoch_dev),('已提交完整轮的保存',epoch_save)]:time_row(name,value if epoch_complete else None,'runs/stage4-20261020/*/summary.json','只从每run最终完整records求和，避免恢复后重复累计；均含在训练器seconds内','training_inner')
        time_row('被回退的未完成轮更新耗时',discarded_seconds if discard_complete else None,'evidence/stage4/training-queue.json',f'原队列实记回退更新{discarded_updates}次；已含于各段训练运行时钟','training_inner')
    else:warnings.append('训练队列尚未取回，训练时间待检。')
    time_row('六模型CPU检查器实际seconds',checkpoints.get('seconds') if checkpoints else None,'evidence/stage4/checkpoint-inspection.json','主动CPU检查流程的实测墙钟，不是纯CPU核时；包含六组检查，不重复加六个内层计时','cpu_check_inner',status='PASS' if checkpoint_pass else 'pending')
    for op in post_ops:
        if any(word in str(op.get('name','')) for word in ('checkpoint','dictionary')):time_row('控制器子进程：'+str(op.get('name')),op.get('seconds'),op['controller'],'外层子进程墙钟，含启动与2秒轮询尾部；成功CPU检查或D0内层seconds已经包含其中','cpu_or_d0_outer',status=op.get('status'))
    time_row('D0 CPU完整评估流程',d0.get('seconds') if d0 else None,'runs/stage4-20261020/local-evaluation/D0/summary.json','词典构建、分词初始化、完整5000匹配及共同500派生/核算；非GPU时长','d0_inner',status=d0.get('status') if d0 else 'pending')
    evaluation=optional('evidence/stage4/local-evaluation-queue.json');eval_details=[]
    if evaluation:
        child=wait=runner=loading=batch_time=single_time=0.;complete=child_complete=wait_complete=True
        for run in evaluation.get('runs',[]):
            wait+=run.get('wait_seconds',0) if finite(run.get('wait_seconds')) else 0
            if not finite(run.get('wait_seconds')):wait_complete=False
            for segment in run.get('segments',[]):
                value=segment.get('wall_seconds');child+=value if finite(value) else 0
                if not finite(value):complete=False;child_complete=False
            summary_path=run.get('summary',f"runs/stage4-20261020/local-evaluation/{run['name']}/summary.json");summary=optional(summary_path)
            full_timing=optional(f"runs/stage4-20261020/local-evaluation/{run['name']}/holdout5000/timing.json")
            common_timing=optional(f"runs/stage4-20261020/local-evaluation/{run['name']}/common500/timing.json")
            if summary and finite(summary.get('seconds')) and finite(summary.get('loading_seconds')):runner+=summary['seconds'];loading+=summary['loading_seconds']
            else:complete=False
            if full_timing and finite(full_timing.get('batch_evaluation',{}).get('seconds')):batch_time+=full_timing['batch_evaluation']['seconds']
            else:complete=False
            if common_timing and finite(common_timing.get('single_sample_latency',{}).get('total_seconds')):single_time+=common_timing['single_sample_latency']['total_seconds']
            else:complete=False
            eval_details.append(dict(name=run['name'],status=run.get('status'),summary=summary_path,runner_seconds=summary.get('seconds') if summary else None))
        time_row('六模型GPU评估子进程墙钟',child if child_complete else None,'evidence/stage4/local-evaluation-queue.json','全部种子均为本阶段新增推理；包括加载、完整5000批处理及共同500单样本检查','gpu_eval_outer',status=evaluation.get('status'))
        time_row('六模型评估器自报seconds',runner if complete else None,'runs/stage4-20261020/local-evaluation/*/summary.json','内层真实执行时钟；已含在GPU评估子进程墙钟内，不等于纯CUDA kernel时间','gpu_eval_inner')
        for name,value in [('六模型加载',loading),('六模型完整5000批量评价',batch_time),('六模型共同500逐条延迟检查',single_time)]:time_row(name,value if complete else None,'runs/stage4-20261020/local-evaluation/*/*/timing.json','评估流程内层计时，不能与评估器seconds重复相加；共同500主预测从同一次5000输出派生','gpu_eval_subphase')
        time_row('评估资源门槛轮询累计等待',wait if wait_complete else None,'evidence/stage4/local-evaluation-queue.json','资源等待，不是GPU计算','wait')
    api=optional('evidence/stage4/api-controller.json')
    if api:
        time_row('API等待四次补训与CPU检查',api.get('training_and_CPU_wait_seconds'),'evidence/stage4/api-controller.json','依赖等待，与训练/检查同期重叠，不能再算作主动CPU/GPU/API调用时间','overlapping_dependency_wait')
        time_row('API子进程整体墙钟',api.get('API_seconds'),'evidence/stage4/api-controller.json','包含请求预留、网络响应、落盘、结算及轮询尾部；未完成时不填推算值','api_outer',status=api.get('status'))
    api_root='runs/stage4-20261020/api-final'
    calls_path=root/api_root/'calls.jsonl';calls=[];attempts=[];continuation_summary=None;continuation_attempts_path=None
    execution_candidates=[execution] if execution else ['configs/stage4-execution-continuation02.json','configs/stage4-execution-continuation01.json','configs/stage4-execution.json']
    execution_data=None
    for candidate in execution_candidates:
        if candidate and (root/candidate).is_file():execution_data=optional(str(candidate));break
    selected_summary=execution_data.get('audit',{}).get('api_summary_path') if execution_data else None
    if not selected_summary:
        selected_summary=next((api_root+'/summary-'+version+'.json' for version in ('continuation02','continuation01') if (root/(api_root+'/summary-'+version+'.json')).is_file()),None)
    if selected_summary:continuation_summary=optional(selected_summary)
    continuation_version=Path(selected_summary).stem.removeprefix('summary-') if selected_summary and 'continuation' in selected_summary else None
    if continuation_version:continuation_attempts_path=root/api_root/('attempts-log-'+continuation_version+'.jsonl')
    for version in ('continuation01','continuation02'):
        segments_path=root/api_root/('segments-'+version+'.jsonl')
        if segments_path.is_file():
            continuation_segments=read_lines(segments_path);source_paths.add(str(segments_path.relative_to(root)))
            values=[r.get('seconds') for r in continuation_segments]
            time_row('API运维续跑'+version+'流程墙钟',sum(values) if values and all(finite(t) for t in values) else None,str(segments_path.relative_to(root)),'各续跑段只计自身墙钟；原未知项不重发；与逐条网络响应/失败计时重叠，预留不当结算','api_continuation_outer',status=continuation_segments[-1].get('status') if continuation_segments else 'pending')
    if continuation_attempts_path and continuation_attempts_path.is_file():
        attempts=read_lines(continuation_attempts_path);source_paths.add(str(continuation_attempts_path.relative_to(root)))
        unknown_times=[r.get('seconds') for r in attempts if r.get('cost_status')=='unknown_cost']
        time_row('全部费用未知传输失败耗时',sum(unknown_times) if unknown_times and all(finite(t) for t in unknown_times) else None,str(continuation_attempts_path.relative_to(root)),f'当前{len(unknown_times)}笔仅有失败计时、没有原响应/usage；按授权逐笔预算预留，不视作实际结算；从最终完整尝试日志只计一次','api_transport_failure')
    if calls_path.is_file():
        calls=read_lines(calls_path);source_paths.add(str(calls_path.relative_to(root)));times=[c.get('seconds') for c in calls]
        full_attempts=bool(continuation_summary and continuation_summary.get('total_attempted')==1000 and len(calls)+continuation_summary.get('unknown_cost_calls',0)==1000)
        time_row('API完整网络响应逐次seconds合计',sum(times) if all(finite(t) for t in times) else None,str(calls_path.relative_to(root)),f'当前实记{len(calls)}次有完整原始响应的已结算调用；包括内容非法响应；所有费用未知传输失败另列；只从append-only calls统计一次，不把各续跑快照相加；包含于原/续跑流程墙钟','api_inner',status='completed1000_attempts_with_dynamic_unsettled_count' if full_attempts else 'completed1000_responses' if len(calls)==1000 else 'partial')
    ledger=optional('private/budget/ledger.json')
    pending=[r for r in ledger.get('requests',[]) if r.get('status')=='pending'] if ledger else None
    api_account=dict(summary_path=selected_summary,continuation_version=continuation_version,attempts=len(attempts),settled_calls=len(calls),unknown_cost_calls=sum(r.get('cost_status')=='unknown_cost' for r in attempts),pending_count=len(pending) if pending is not None else None,known_cumulative_cny=continuation_summary.get('known_cumulative_cny') if continuation_summary else None,unknown_cost_upper_cny=continuation_summary.get('unknown_cost_upper_cny') if continuation_summary else None,exposure_upper_cny=continuation_summary.get('exposure_upper_cny') if continuation_summary else None,unknown_never_claimed_settled=True)
    api_finished=bool(continuation_summary and len(attempts)==1000 and continuation_summary.get('total_attempted')==1000 and continuation_summary.get('settled_calls')==len(calls) and continuation_summary.get('unknown_cost_calls')==api_account['unknown_cost_calls'] and len(calls)+api_account['unknown_cost_calls']==1000 and len({r.get('key') for r in attempts})==1000 and ledger and ledger.get('api_enabled') is False and pending==[])
    if not continuation_version:api_finished=bool(api and api.get('status')=='finished_API_calls' and api.get('returncode')==0 and len(calls)==1000 and ledger and ledger.get('api_enabled') is False and pending==[])
    for item in interpreted:
        if item['id'] in ('ST4-I08','ST4-I10'):
            item['retest_status']='COMPLETED_ATTEMPTS_COST_UNSETTLED' if api_finished else 'PARTIAL_ATTEMPTS_COST_UNSETTLED'
            item['retest_description']=f"当前实记{api_account['attempts']}项尝试，{api_account['settled_calls']}项已有响应、{api_account['unknown_cost_calls']}笔费用仍未知；原unknown请求不重发、不补usage或结算。最终1000项/无pending/API关闭门槛{'已满足' if api_finished else '待完成'}；独立最终核查另列，不把工程运维恢复当抽取错误或真实费用结清。"
            item['evidence_paths'] += [api_root+'/requests/failure-'+ident+'.json' for ident in ('ner-final-85a1172afb6d4ff6b1be8cbcce79333b','ner-final-6494ce5c72594163a17a14497aa14603')]
            item['evidence_paths'] += [p for p in (selected_summary,str(continuation_attempts_path.relative_to(root)) if continuation_attempts_path else None,'evidence/stage4/transport-budget-authorization02.json',api_root+'/continuation02/annex-at-call.json') if p]
            item['evidence_paths']=list(dict.fromkeys(item['evidence_paths']))
            item['unavailable_evidence_paths']=[p for p in item['evidence_paths'] if not (root/p).is_file()]
            for path in item['evidence_paths']:
                if (root/path).is_file():source_paths.add(path)
    transfer=optional('evidence/stage4/transfer-manifest.json')
    time_row('最终证据传输与校验流程',transfer.get('seconds') if transfer else None,'evidence/stage4/transfer-manifest.json','包含远端只读清点、传输及本地校验；没有复制无关权重/密钥，不加算外层record_operation','transfer_inner')
    aggregation=optional('evidence/stage4/final-aggregation.json');verification=optional('evidence/stage4/final-verification.json')
    time_row('独立指标汇总',aggregation.get('seconds') if aggregation and 'results' in aggregation else None,'evidence/stage4/final-aggregation.json','仅历史输出CPU核算；不重新调用API或训练','audit_inner')
    time_row('独立最终核查',verification.get('seconds') if verification and verification.get('status') in ('PASS','PASS_WITH_WARNINGS') else None,'evidence/stage4/final-verification.json','包括重新读取输出、指标/配对统计核算、训练和API/账本/延迟证据核对；不是最终模型再次调参','audit_inner')
    for operation in operations:
        if 'transfer' in operation['name']:time_row('传输命令外层：'+operation['name'],operation['seconds'],operation['record'],'record_operation实记，可能包住同一传输helper内层时间；此行和内层不可重复相加','transfer_outer',status='returncode='+str(operation['returncode']))
    critical=dict(four_supplementary_training_completed=bool(train and train.get('status')=='completed' and len(train.get('runs',[]))==4),six_CPU_checkpoint_checks_passed=checkpoint_pass,D0_completed=d0_pass,six_GPU_evaluations_completed=bool(evaluation and evaluation.get('status')=='completed' and len(evaluation.get('runs',[]))==6),API1000_attempts_finished=api_finished,final_transfer_available=bool(transfer),independent_verification_passed=bool(verification and verification.get('status') in ('PASS','PASS_WITH_WARNINGS')))
    if aggregation and 'operation' in aggregation:warnings.append('final-aggregation.json包含operation计时结构，不能当作主汇总结果；应核查是否发生同名覆盖。')
    if verification and 'operation' in verification:warnings.append('final-verification.json包含operation计时结构，不能当作独立核查结果。')
    incomplete=[k for k,value in critical.items() if not value]
    if incomplete:warnings.append('当前仅是证据快照，尚缺终态：'+','.join(incomplete))
    return dict(schema_version=1,created_at=datetime.now(timezone.utc).isoformat(),status='COMPLETE_EVIDENCE_SNAPSHOT' if not incomplete else 'PARTIAL_EVIDENCE_SNAPSHOT',scope='stage4 engineering recovery and measured timing only;not extraction-error analysis or independent effect re-evaluation',issues=interpreted,controllers=controllers,operation_records=operations,timings=timings,training_runs=train_runs,evaluation_runs=eval_details,dependency_original_file_checks=dependency_checks,critical_evidence=critical,warnings=warnings,reused_seed20261019_training_counted_as_new=False,all_time_rows_are_nonadditive=True,raw_issue_notes_untouched=True,approved_unknown_cost_remains_unsettled=bool(api_account['unknown_cost_calls']),api_accounting=api_account,actual_AI_production='AI implemented,debugged,launched,recovered and checked these artifacts. Namedstudent roles are a teacher-created simulation;no real student hours or signatures are inferred.',source_files_sha256={rel:digest(root/rel) for rel in sorted(source_paths)})


def render(root,data):
    lines=['# 阶段四工程与恢复记录','',f"证据快照：{local_time(data['created_at'])}；状态：**{data['status']}**。本页为阶段四工程记录的历史快照，不反映当前项目进度；仅从已取回的真实JSON与原始入口整理工程问题和计时，不重算模型效果、不修改原预测、金标、冻结参数、预算或历史日志。",'',
        '工程故障与实体抽取错误分别记录。下面的导入失败、检查器表示误判及只读通信复查，不能充当项目要求的三类实体抽取失败。旧日志和失败journal全部保留，成功复测使用独立文件；issues.jsonl中的“pending”是当时状态，本页依据后续终态另列当前结果。','']
    for issue in data['issues']:
        lines += ['## '+issue['id'],'',f"**现象：** {issue.get('phenomenon') or '未记录'}",'',f"**原因依据：** {issue.get('cause_basis') or '未记录'}",'',f"**修改：** {issue.get('change') or '未记录'}",'',f"**实际复测 / 待检：** **{issue['retest_status']}**。{issue['retest_description']}", '', '真实入口：'+'；'.join(link(root,p) for p in issue['evidence_paths'])+'。','']
        related=[op for c in data['controllers'] if Path(c['path']).name.startswith('post-training-controller') for op in c['operations'] if (issue['id']=='ST4-I04' and 'checkpoint' in str(op.get('name'))) or (issue['id']=='ST4-I05' and 'dictionary' in str(op.get('name')))]
        if related:
            lines += ['| 原控制器步骤 | 终态 | 外层子进程墙钟 | 原控制台 |','|---|---|---:|---|']
            for op in related:lines.append(f"| {op['name']} | {op.get('status')} / code {op.get('returncode')} | {sec(op.get('seconds'))} | {link(root,op.get('console'))} |")
            lines+=['','这些是控制器包住子进程的实测墙钟，含启动和轮询尾部。CPU检查器自己的seconds及D0流程seconds见下表，不能相加冒充额外计算。','']
    retry=[c for c in data['controllers'] if c['read_only_transport_retry_count']]
    lines+=['## 只读通信复查与依赖守卫','',f"已取回控制器中，共记录只读传输/原子文件读取复查{sum(c['read_only_transport_retry_count'] for c in retry)}次。它们是SSH读取或依赖状态复查，不能当作重发付费API请求；付费请求是否完整且无重复，由独立最终核查的1000项冻结请求链验证。没有为未单列的复查主动耗时填入估算值。",'',
        '| 控制器 | 当前/最终状态 | 时间戳差推得的整体墙钟 | 只读复查次数 |','|---|---|---:|---:|']
    for c in data['controllers']:lines.append(f"| {link(root,c['path'])} | {c.get('status')} | {sec(c.get('timestamp_wall_seconds'))} | {c['read_only_transport_retry_count']} |")
    lines+=['','控制器整体墙钟含长时间等待上游训练、CPU检查或API完成。下游因为守卫发现上游失败而退出，不表示它实际执行了完整评估；这些并行或重叠墙钟不累计为GPU、CPU检查或API主动时间。','',
        '## 分阶段真实计时与口径','',
        '种子20261019的训练复用阶段三检查点，不计作本阶段新增训练；它在本阶段的最终推理仍是新增执行。阶段四只累计20261020、20261021共四次补训。恢复时只从最终完整轮records统计一次训练/开发/保存分项，所有原始分段的额外回退另列。','',
        '| 项目 | 实记耗时 | 口径与重叠说明 | 来源 |','|---|---:|---|---|']
    for t in data['timings']:lines.append(f"| {t['name']} | {sec(t['seconds'])} | {t['scope']}；状态：{t['status']} | {link(root,t['source']) if '*' not in t['source'] else '`'+t['source']+'`'} |")
    lines+=['','以上各行**不能直接相加**：内层训练/开发/保存已经包含在训练器及训练子进程墙钟中；CPU检查器seconds包含在控制器对应子进程墙钟中；GPU推理分项包含在评估器内；API逐次响应包含在API子进程整体时间中；控制器等待与其它任务并行。主动流程seconds也是墙钟测量，不是进程CPU核时或纯GPU kernel时间。原始记录缺项时明确写待取回，不将未知值填0。','',
        '## 原始命令计时记录','',
        '| 操作 | 返回码 | 外层子进程墙钟 | 原记录 / 控制台 |','|---|---:|---:|---|']
    for operation in data['operation_records']:lines.append(f"| {operation['name']} | {operation.get('returncode')} | {sec(operation['seconds'])} | {link(root,operation['record'])} / {link(root,operation.get('console'))} |")
    lines+=['','这些命令记录保留失败与复测，合成反例明确属于工程/指标验证。它们不替代真实留出集输出、正式效果或真实CPU/GPU验收。','',
        '## AI实际制作与模拟双人分工','',
        '实际的实现、排错、部署、运行、恢复与证据整理由AI完成。吴皓睿、周栩乐是教师示范角色：工程恢复、训练/推理集成和复现示范主责映射至吴皓睿（E05、E13、E16、E18、E20）；原文、指标分母、成本与证据交叉核对映射至周栩乐（E06、E14、E15、E17、E19）。这是职责安排，不是已发生的人工经历。不得从GPU/控制器时长倒填两人的实际工时，也不代签或伪造多人提交身份。','',
        '正文、⑧分工表和两份贡献材料继续引用同一证据映射；最终PDF附录二集中披露整个示范由AI制作、角色工时和确认属于模拟。当前教学责任映射统一为吴皓睿52%、周栩乐48%；这是示范职责安排，不代表实际人工贡献、工时或签署。本页工程状态按阶段四历史快照理解，原始日志和当时记录保持原值。', '',
        '定位入口：'+link(root,'docs/示范角色与AI制作边界.md')+'；'+link(root,'docs/REPORT_EVIDENCE.md')+'；'+link(root,'evidence/stage4/role-evidence.json')+'；'+link(root,'evidence/stage4/final-verification.json')+'。','',
        '## 证据缺口与后续边界','']
    if data.get('approved_unknown_cost_remains_unsettled'):
        a=data['api_accounting']
        lines += [f"完整尝试日志当前实记{a['attempts']}项，其中{a['settled_calls']}项已有完整响应并按usage结算估费，{a['unknown_cost_calls']}笔仍为费用未知，待处理预留{a['pending_count']}笔。累计已知usage保守估费¥{a['known_cumulative_cny']}；全部未知逐笔最坏预留合计¥{a['unknown_cost_upper_cny']}；总暴露上界¥{a['exposure_upper_cny']}。未补造未知项的响应、usage或结算；原记录、旧部分结果和每段失败journal保留。",'', '用户批准的有上限规则允许同类网络传输中断逐笔预留全上下文最坏¥2.129920后跳过，继续原冻结1000清单中从未尝试的项；每个key最多尝试一次。known加全部unknown加当前pending均受累计¥100和本阶段新增¥80约束。非传输、响应完整性、模型/usage、持久化或账本问题必须停止。此运维规则不改变提示、参数、样本、指标；含非法/不可用项的路线主成绩和相关配对区间仍为N/A。', '', '授权与证据：'+link(root,'evidence/stage4/unknown-cost-authorization.json')+'；'+link(root,'evidence/stage4/transport-budget-authorization02.json')+'；'+link(root,'runs/stage4-20261020/api-final/continuation02/annex-at-call.json')+'；'+link(root,a['summary_path'])+'。','']
    if data['warnings']:lines += ['- '+w for w in data['warnings']]
    else:lines+=['本次快照要求的阶段四终态入口齐备；历史追记与实际原始日志的区别仍按上文保留。']
    lines+=['','至少三个真实失败案例、覆盖至少三类有依据的抽取原因，以及独立修复复测、可视化系统属于下一阶段。当前工程恢复记录不宣称已完成这些内容，也不回写冻结主成绩。','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--summary',type=Path,default=Path('evidence/stage4/recovery-summary.json'));parser.add_argument('--report',type=Path,default=Path('docs/阶段四工程与恢复记录.md'));parser.add_argument('--require-complete',action='store_true');parser.add_argument('--execution',type=Path,help='Explicit execution manifest selecting latest operational summary');args=parser.parse_args()
    if (args.root/args.summary).exists() or (args.root/args.report).exists():raise FileExistsError('Recovery note output exists;use new paths and preserve prior snapshot')
    result=collect(args.root,args.execution)
    if args.require_complete and result['status']!='COMPLETE_EVIDENCE_SNAPSHOT':raise ValueError('Required final evidence missing:'+str(result['critical_evidence']))
    text=render(args.root,result)
    for path in (args.root/args.summary,args.root/args.report):path.parent.mkdir(parents=True,exist_ok=True)
    with (args.root/args.summary).open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with (args.root/args.report).open('x',encoding='utf-8') as f:f.write(text)
    print(json.dumps(dict(status=result['status'],issues=len(result['issues']),controller_count=len(result['controllers']),timing_rows=len(result['timings']),summary=str(args.summary),report=str(args.report)),ensure_ascii=False))

if __name__=='__main__':main()
