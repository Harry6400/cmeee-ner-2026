"""Generate stage4 audit evidence and teaching-role mapping from verified real results;no final PDF/ZIP/signatures."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from stage4_metrics import SEEDS,dump_new


def fmt(value,digits=2):return 'N/A' if value is None else f'{value*100:.{digits}f}%'
def seconds(value):return f'{value:.3f}秒'
def write_new(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(text)


def generate(root,aggregation,verification,report_path,role_path):
    a=json.loads((root/aggregation).read_text());v=json.loads((root/verification).read_text())
    if v['status'] not in ('PASS','PASS_WITH_WARNINGS'):raise ValueError('Stage4 report requires successful independent verification')
    results=a['results'];budget=v['budget'];unknown_count=budget.get('stage4_unknown_cost_calls',0);settled_count=sum(r.get('settled_calls',r['attempts']) for r in budget['routes'].values());attempt_count=sum(r['attempts'] for r in budget['routes'].values());continuation_version=budget.get('continuation_version','continuation01');api_root='runs/stage4-20261020/api-final';primary={r:results[r+('-seed20261019' if r in ('N0','N1') else '')+'/common500'] for r in ('D0','N0','N1','L0','L3')}
    lines=['# 阶段四核查：多种子与冻结留出集公平评估','', '**状态说明：本报告及下文阶段五、阶段六状态均为阶段四报告生成时的历史快照，不反映当前项目进度。**','',
        f"核查状态：**{v['status']}**。已完成冻结本地评估与全部1000次API尝试；五路线完整可比主F1是否可用按下表标示。下述结果来自真实原始预测的独立核算。阶段五界面、正式错误归因与独立修复复测尚未执行（仅表示阶段四报告生成时的状态）。",'',
        '## 协议与分母','',
        '五条路线使用同一固定500条。N0/N1主对照固定种子20261019，另报20261020、20261021及三种子均值与样本标准差（ddof=1）；不按最终结果挑种子。本地D0及N0/N1全部三种子另报完整5000条。共同500是5000的子集，不能当两份独立测试。', '',
        '严格指标按原文Unicode字符右开区间的起点、终点、类别完全一致计数；精确重复只计一次，不做文本归一化。所有参考实体保留在分母。窗口未完整覆盖的实体在机器证据中单列。非法API输出不视为正常空实体；一条非法即令该路线完整500主P/R/F1及相关配对区间为N/A，条件有效子集只作诊断。', '',
        '## 五路线共同500条主结果','',
        '| 路线 | 主种子 | 有效/全部 | 严格P | 严格R | 严格F1 |','|---|---:|---:|---:|---:|---:|']
    for route,r in primary.items():
        m=r['main_metrics']['micro'] if r['main_metrics'] else None
        lines.append(f"| {route} | {20261019 if route in ('N0','N1') else '—'} | {r['valid']}/500 | {fmt(m['precision'] if m else None)} | {fmt(m['recall'] if m else None)} | {fmt(m['f1'] if m else None)} |")
    lines+=['','API完整样本有效率和失败记录保留；若下面存在条件成绩，其分母与主表不同，不能据此给路线排序。','',
        '| API路线 | 内容非法 / 传输失败 / 未发送 | 条件有效样本数 | 条件严格F1（仅诊断） |','|---|---:|---:|---:|']
    for route in ('L0','L3'):
        r=primary[route];m=r['conditional_valid_only_metrics'];lines.append(f"| {route} | {r['statuses'].get('invalid',0)} / {r['statuses'].get('transport_error',0)} / {r['statuses'].get('not_sent',0)} | {m['samples'] if m else '—'} | {fmt(m['micro']['f1'] if m else None)} |")
    lines+=['','## 本地完整5000条与三种子','',
        '| 路线 | 种子 | 严格P | 严格R | 严格F1 |','|---|---:|---:|---:|---:|']
    for route in ('D0','N0','N1'):
        for seed in ((None,) if route=='D0' else SEEDS):
            key=route+(f'-seed{seed}' if seed else '')+'/holdout5000';m=results[key]['main_metrics']['micro']
            lines.append(f"| {route} | {seed if seed else '—'} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} |")
    lines+=['','| 路线 | 范围 | 三种子F1均值 ± 样本标准差（百分点） |','|---|---|---:|']
    for route in ('N0','N1'):
        for split in ('common500','holdout5000'):
            s=a['multi_seed'][route][split]['f1'];lines.append(f"| {route} | {split} | {s['mean']*100:.2f} ± {s['sample_sd']*100:.2f} |")
    lines+=['','本项目的BERT监督微调构成课程大模型方案；D0、N0与N1在完整5000条上的冻结对照承担主要效果证据。API主指标若不可用，仍保留其可靠性与费用证据，不改协议、重试或使用条件子集补成五条完整成绩。','', '如某API路线包含传输失败或内容非法，不能声称五路线均有完整可比F1；仅可比较主表中完整500有效的路线，条件子集不得替代。三种子标准差描述训练随机性；样本量很小，不据此宣称统计显著或所有领域均胜出。N0与N1结构不同，也不能把差异全部归因为预训练。','',
        '## 共同500条配对不确定性','',
        '按文档有放回重采样2000次，固定种子20261019；每次所有路线使用完全相同的500个索引，重新汇总实体级Micro F1。区间为2.5%与97.5%百分位，不使用各自有效子集。', '',
        '| 差值方向 | F1差（百分点） | 95%配对区间（百分点） |','|---|---:|---|']
    for name,b in a['bootstrap']['differences'].items():
        if b['status']=='available':lines.append(f"| {name} | {b['difference']*100:+.2f} | [{b['ci95'][0]*100:+.2f}, {b['ci95'][1]*100:+.2f}] |")
        else:lines.append(f'| {name} | N/A | 完整500存在非法输出 |')
    lines+=['','区间仅反映当前语料和固定检查点下的文档抽样不确定性，不包含训练种子方差，未作多重比较校正，也不构成因果解释。','',
        '## 辅助口径与局限','',
        '| 路线（主种子） | 嵌套参考召回 | 未登录词严格F1 | 多标签字符F1 | 同类重叠一对一F1（诊断） |','|---|---:|---:|---:|---:|']
    for route,r in primary.items():
        m=r['main_metrics'];lines.append(f"| {route} | {fmt(m['nested_recall'] if m else None)} | {fmt(m['unseen_surface_strict']['f1'] if m else None)} | {fmt(m['character_type_micro']['f1'] if m else None)} | {fmt(m['relaxed_overlap_same_type_one_to_one']['f1'] if m else None)} |")
    full_n1=results['N1-seed20261019/holdout5000']['main_metrics']
    structure=full_n1.get('structural_reference_recall',{})
    uncovered=len(full_n1['window_uncovered_reference_entities']) if 'window_uncovered_reference_entities' in full_n1 else None
    observed=[f'窗口未完整覆盖参考实体数为{uncovered}' if uncovered is not None else '窗口覆盖统计见独立JSON']
    for name,label in [('coextensive_different_type','同跨度异类'),('crossing','交叉')]:
        if name in structure:observed.append(label+'参考实体数为'+str(structure[name]['gold']))
    lines+=['','冻结数据的适用限制：'+'；'.join(observed)+'。没有观察到窗口漏覆盖时，不将截断虚构为真实失败；零或极小结构分母不能用于能力结论。严格与宽松F1之差只说明匹配口径下的边界现象，不证明错误机制。','', '逐类别、嵌套/交叉/同跨度异类/孤立结构、实体长度及文档长度分层均保存在独立核算JSON中。结构分层允许交叠；字符指标保留多标签；宽松指标使用最大一对一匹配，不能替代严格主指标。未登录只表示训练实体表面词未出现，不证明标注错误。', '',
        'BERT逐字符词表接口保留原文位置，但UNK仍会丢失字符表示信息；许可、近重复审计非穷尽性及数据领域范围沿用已有证据。最终数据上的失配只用于冻结后分析，不能回头调整主成绩。至少三个真实案例、三类有证据的错误归因留到阶段五（阶段四报告生成时的计划快照）；工程故障另列，不冒充抽取错误。', '',
        '## 真实耗时、资源与费用','',
        '| 模型 | 种子 | 完成轮 / 开发最佳轮 | 全部分段训练运行耗时 | 等卡 | 回退更新数 | GPU利用率采样均值 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in v['training']:
        util=r.get('gpu_utilization',{}).get('sample_mean_percent');lines.append(f"| {r['route']} | {r['seed']} | {r['epochs']} / {r['best_epoch']} | {seconds(r['runner_seconds'])} | {seconds(r['wait_seconds']) if r.get('wait_seconds') is not None else '历史复用'} | {r.get('segment_checks',{}).get('discarded_partial_updates',0)} | {f'{util:.2f}%' if util is not None else '见阶段三'} |")
    lines+=['','种子20261019复用阶段三已有检查点，其历史训练耗时不重复算作本阶段新增GPU时长。等待、传输、推理、独立核算及失败复测以各自原始计时为准；GPU利用率为自身训练活跃期间约10秒一次的硬件快照均值，排除检测到外部计算进程的快照，不是时间加权的kernel占用；不因主观期望写成达标。多段等待、失败和回退更新全部列入原始生命周期，不重复算已完成轮。', '',
        '| 本地路线 | 种子 | 加载 | 共同500单条均值 | P50 | P95 |','|---|---:|---:|---:|---:|---:|']
    for r in v['latency']:
        if r['split']!='common500':continue
        t=r['single_sample_latency'];lines.append(f"| {r['route']} | {r['seed'] or '—'} | {seconds(r['load_seconds'])} | {seconds(t['mean_seconds'])} | {seconds(t['p50_seconds'])} | {seconds(t['p95_seconds'])} |")
    lines+=['','D0在QHA服务器CPU运行，N0/N1在QHA GPU7（NVIDIA A40）运行；L0/L3由Mac请求远端API，记录完整网络响应耗时，服务端硬件未知。因此这些数字是各实际部署条件下的工程开销，不能作同一硬件的纯模型速度排名。本页P50/P95按原始逐条时延的(n−1)q线性插值统一重算；API运行器旧摘要采用其原分位规则，不能将二者冒充逐字段相同。单条延迟为真实独立单样本执行；N0/N1完整5000采用固定微批量4，只报告批处理吞吐；D0为CPU逐文匹配并另报单条耗时，二者口径分开，不把批均耗时冒充单次响应时间。单样本与批量形状产生的任何预测差异另存诊断，主成绩始终来自同一次完整5000批量输出。', '',
        '| API路线 | 尝试 / 已结算完整响应 | 响应均值 | P50 | P95 | 内容非法 / 传输失败 | 失败耗时 | 已知usage保守估费 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for route,r in budget['routes'].items():
        t=r['latency'];lines.append(f"| {route} | {r['attempts']} / {r.get('settled_calls',r['attempts'])} | {seconds(t['mean_seconds']) if t['mean_seconds'] is not None else 'N/A'} | {seconds(t['p50_seconds']) if t['p50_seconds'] is not None else 'N/A'} | {seconds(t['p95_seconds']) if t['p95_seconds'] is not None else 'N/A'} | {r['errors']} / {r.get('transport_errors',0)} | {', '.join(seconds(x) for x in r.get('transport_error_seconds',[])) or '—'} | ¥{r['conservative_cny']} |")
    lines += ['',f"本阶段已知usage保守估费¥{budget['stage4_cny']}；累计已知usage保守估费¥{budget.get('known_cumulative_cny',budget['cumulative_cny'])}。未结算的{unknown_count}笔传输未知费用保留为unknown_cost，按用户明确批准的运维规则逐笔预留全上下文最坏上界，合计¥{budget.get('unknown_cost_upper_cny','0')}；累计已知费用加未知预留的预算暴露上界¥{budget.get('exposure_upper_cny',budget['cumulative_cny'])} / ¥100，上界口径余额¥{budget['remaining_cny']}。最坏预留不是实际token用量、实际费用或结算；没有重发未知请求。已知费用也是依据usage的保守估算，并非支付账单。响应时延只统计有原始完整响应的记录（含内容格式非法），传输失败耗时单列，不混入正常响应分位数。QHA按用户提供条件无新增租赁费；GPU时长仍单独记录。API已关闭，本任务GPU训练/推理进程已退出。",'',
        '## 证据与下一阶段','',
        f'- 冻结科学协议：[stage4-frozen-protocol.json](../configs/stage4-frozen-protocol.json)；实际执行/运维补充协议：[执行入口](../{a["execution_manifest"]})。',
        f'- 独立核算：[final-aggregation.json](../{aggregation})；核查：[final-verification.json](../{verification})。',
        '- 原始训练、预测、逐条时延和TensorBoard：[runs/stage4-20261020](../runs/stage4-20261020)。',
        '- 控制台、GPU采样、等待、传输、异常及复测：[evidence/stage4](../evidence/stage4)。',
        '- 工程故障、恢复与分段计时：[阶段四工程与恢复记录.md](阶段四工程与恢复记录.md)。',
        '- 教学角色证据映射：[阶段四角色与AI证据.md](阶段四角色与AI证据.md)。','',
        '阶段四报告生成时记录的阶段五待办：从冻结原始输出中确认至少三个真实失败案例并覆盖三类有依据的原因，区分观察事实、机制推断及未排除的解释；独立保存修复复测，禁止覆盖主结果；制作真实运行的实体可视化。阶段六再完成教师独立CPU复现、真实Windows验收、八节PDF、贡献材料、摘要、录屏与ZIP；以上均为阶段四时点历史快照，不代表当前项目进度。','']
    if budget.get('stage4_unknown_cost_calls'):
        lines+=['',f"运维授权：[实际用户批准记录](../{budget.get('unknown_cost_approval_path','evidence/stage4/unknown-cost-authorization.json')})、[运维补充附件](../{budget.get('continuation_annex_path',api_root+'/'+continuation_version+'/annex-at-call.json')})、[完整尝试日志](../{api_root}/attempts-log-{continuation_version}.jsonl)。原部分结果和失败文件保留；{attempt_count}次尝试为{settled_count}次已结算完整响应加{unknown_count}笔费用未知，二者不可混称。获批传输类中断逐笔预留¥2.129920且不重发；非传输或完整性问题必须停止。",'']
    write_new(root/report_path,'\n'.join(lines))
    # These are existing synthetic validation logs, not invented human work
    # records. Final aggregation/verification are independently required above.
    synthetic_audit_logs=[path for path in (
        'evidence/stage4/api-synthetic-tests.log',
        'evidence/stage4/metrics-synthetic-tests.log',
        'evidence/stage4/metrics-synthetic-tests-retest.log',
        'evidence/stage4/segments-synthetic-tests.log',
        'evidence/stage4/api-integrity-synthetic-tests.log') if (root/path).is_file()]
    privacy_records=[path for path in (
        'runs/stage4-20261020/api-final/privacy-review.json',
        'evidence/stage4/privacy-review-part1.json',
        'evidence/stage4/privacy-review-part1-resolved.json',
        'evidence/stage4/privacy-review-part1-resolution.json',
        'evidence/stage4/privacy-review-part2.json',
        'evidence/stage4/privacy-context-check.json') if (root/path).is_file()]
    continuation_candidates=['evidence/stage4/unknown-cost-authorization.json',budget.get('unknown_cost_approval_path','evidence/stage4/transport-budget-authorization02.json'),budget.get('api_summary_path',api_root+'/summary.json')]
    for version in ('continuation01','continuation02'):
        continuation_candidates += [api_root+'/'+version+'/annex-at-call.json',api_root+'/'+version+'/approval-at-call.json',api_root+'/attempts-log-'+version+'.jsonl',api_root+'/segments-'+version+'.jsonl']
    continuation_records=[path for path in dict.fromkeys(continuation_candidates) if (root/path).is_file()]
    roles=[
        dict(id='E13',requirement='④/⑤ 多种子实现与开发早停',role='吴皓睿',cross_role='周栩乐',planned_weight_percent=21,module='训练集成、固定精度和种子、开发最佳检查点',files=['scripts/train_stage4.py','scripts/train_development.py','runs/stage4-20261020'],run_records=[r['training_run']+'/summary.json' for r in v['training']],report_location='④实现细节、⑤三种子表、附录C',cross_check='对方核对种子、数据范围、dev选择与最终评估隔离'),
        dict(id='E14',requirement='⑤ 最终协议与数据分母',role='周栩乐',cross_role='吴皓睿',planned_weight_percent=10,module='冻结参数、共同500和完整5000关系',files=['configs/stage4-frozen-protocol.json',a['execution_manifest']],run_records=[str(aggregation)],report_location='⑤数据和评估协议、附录D',cross_check='对方核对执行参数与模型接口一致'),
        dict(id='E15',requirement='⑤/⑦ API共同500、公平性与非法处理',role='周栩乐',cross_role='吴皓睿',planned_weight_percent=14,module='L0/L3输入许可审查、完整分母、无自动重试、费用核对',files=['scripts/api_final.py','scripts/api_development_grounded.py','runs/stage4-20261020/api-final/protocol-at-call.json','runs/stage4-20261020/api-final/source-at-call/api_final.py','runs/stage4-20261020/api-final/source-at-call/api_development_grounded.py'],run_records=['runs/stage4-20261020/api-final/calls.jsonl','runs/stage4-20261020/api-final/summary.json']+privacy_records+continuation_records+[str(verification)],report_location='⑤五路线表、⑦费用与隐私、附录一C/D',cross_check='对方核对原始响应到字符跨度的精确映射及非法输出不被替为空集'),
        dict(id='E16',requirement='④/⑤ 本地推理和同源共同子集',role='吴皓睿',cross_role='周栩乐',planned_weight_percent=16,module='GPU7单卡推理、窗口回映和完整输出',files=['runs/stage4-20261020/local-evaluation'],run_records=['runs/stage4-20261020/local-evaluation/manifest.json'],report_location='④实现、⑤完整5000、附录A/C',cross_check='对方抽查ID顺序、Unicode原文及全量到共同500的同源关系'),
        dict(id='E17',requirement='⑤ 严格与辅助指标、配对区间',role='周栩乐',cross_role='吴皓睿',planned_weight_percent=14,module='独立重算、同索引bootstrap、三种子样本标准差',files=['scripts/stage4_metrics.py','scripts/aggregate_stage4.py','tests/test_stage4_metrics.py'],run_records=[str(aggregation),str(verification)],report_location='⑤主指标/辅助/不确定性、附录C',cross_check='对方复核反例测试、分母、ddof=1和不按测试结果挑种子'),
        dict(id='E18',requirement='⑦ 真实延迟与GPU资源',role='吴皓睿',cross_role='周栩乐',planned_weight_percent=10,module='加载/批处理/单条延迟区分、GPU等待与退出',files=['evidence/stage4','runs/stage4-20261020/local-evaluation'],run_records=[str(verification)],report_location='⑦工程开销、附录C',cross_check='对方从逐条计时复算均值、P50、P95并区分错误输出'),
        dict(id='E19',requirement='⑦/附录D 许可隐私与原始证据审计',role='周栩乐',cross_role='吴皓睿',planned_weight_percent=10,module='许可、公共知识文本、预算历史、最终结果冻结',files=['docs/DATA_LICENSE.md','evidence/stage4/api-price.json','runs/stage4-20261020/api-final/price-at-call.json','runs/stage4-20261020/api-final/source-at-call/ner_core.py'],run_records=privacy_records+continuation_records+[str(verification)],report_location='⑦许可隐私、附录一D',cross_check='对方核对无私人身份、密钥、训练外示例泄漏及每笔获批传输未知费用的原状态、逐笔最坏预留和预算总暴露；非传输或完整性问题必须停止'),
        dict(id='E20',requirement='⑧/附录一A/C及附录二 复现和AI来源',role='吴皓睿',cross_role='周栩乐',planned_weight_percent=5,module='可执行核算命令、证据索引、AI制作来源',files=['scripts/verify_stage4.py','scripts/write_stage4_report.py','docs/示范角色与AI制作边界.md']+(['evidence/stage4/ai-production.json'] if (root/'evidence/stage4/ai-production.json').is_file() else []),run_records=synthetic_audit_logs+[str(aggregation),str(verification)],report_location='⑧分工、附录一A/C、附录二',cross_check='对方核对正文与两份贡献材料使用同一编号与比例，模拟确认不当作真实签署')]
    for role in roles:
        role['report_location']=role['report_location'].replace('附录A/C','附录一A/C').replace('附录C','附录一C').replace('附录D','附录一D')
        role['requirement']=role['requirement'].replace('附录D','附录一D')
        role['entry_id']='stage4-'+role['id']
        role['cross_check_status']='simulated_role_responsibility;no_actual_human_confirmation'
        role['actual_validation_evidence']=[path for path in role['run_records'] if (root/path).is_file()]
        role['missing_artifact_paths']=[path for path in dict.fromkeys(role['files']+role['run_records']) if not (root/path).exists()]
        role['contribution_statement_sections']={
            '1':dict(statement_owner=role['role'],entry_id=role['entry_id']+'-module',module=role['module']),
            '2':dict(statement_owner=role['role'],entry_id=role['entry_id']+'-evidence',evidence_ids=[role['id']],files=role['files'],run_records=role['run_records']),
            '3':dict(statement_owner=role['role'],entry_id=role['entry_id']+'-workload',allocation_scope='stage4 teaching plan',planned_weight_percent=role['planned_weight_percent'],simulated_hours=None,main_contribution_selection='deferred to final individual statement;not inferred from GPU time'),
            '4':dict(statement_owner=role['cross_role'],entry_id=role['entry_id']+'-teammate-confirmation',teammate=role['role'],evidence_ids=[role['id']],responsibility=role['cross_check'],status=role['cross_check_status'],confirmation_text=None),
            '5':dict(statement_owner=role['role'],entry_id=role['entry_id']+'-self-confirmation',ai_statement_references=['报告第⑦节','附录二：教师示范制作与AI完整披露'],signature=None,date=None,status='final simulated form pending;no actual human signature or confirmation')}
    mapping=dict(scope='stage4 teaching-role and status snapshot,not real student history',actual_production='AI agents implemented,debugged,ran and independently replayed software;real experiment logs/charges remain immutable evidence',roles=roles,
        planned_totals_percent={name:sum(r['planned_weight_percent'] for r in roles if r['role']==name) for name in ('吴皓睿','周栩乐')},
        simulated_hours=None,confirmations='No actual student confirmations or signatures recorded. Future role-confirmation examples must remain associated with AppendixII disclosure.',
        body_ai_boundary='Exemplary student-led decisions,original-text/metric spot checks,result explanation;AI assistance for scripts,debugging,formatting and language. These role experiences are simulated.',
        appendix_structure={'附录一：复现与证据':{'A':'环境、一条可执行命令与预期输出','B':'目录树与脚本职责','C':'关键真实运行日志、曲线和迭代记录','D':'数据来源与许可'},'附录二：教师示范制作与AI完整披露':'全AI实际制作范围、模拟角色经历/工时/确认、真实实验与验证范围'},
        actual_ai_validation_evidence={'synthetic_tests_not_experiment_results':synthetic_audit_logs,'real_final_output_recalculation':[str(aggregation),str(verification)],'retrospective_factual_production_index_not_original_console':'evidence/stage4/ai-production.json' if (root/'evidence/stage4/ai-production.json').is_file() else None},
        appendix_ii='Disclose whole project demonstration actually made by AI;role histories,hours and confirmations simulated;real dataset/model experiments and cost logs remain real.',
        # Historical Stage 4 snapshot flag; kept for the existing output shape.
        stage5_error_analysis_not_yet_claimed=True)
    if mapping['planned_totals_percent']!={'吴皓睿':52,'周栩乐':48}:raise ValueError('Teaching role weights inconsistent')
    dump_new(root/role_path,mapping)
    role_lines=['# 阶段四角色与AI证据','',
        '这是教师模拟优秀作业的制作台账。正文、⑧分工与两份贡献材料引用同一证据编号；最终PDF附录二集中披露整个项目由AI制作、角色经历与确认属于模拟。实验输出与费用来自真实运行。未记录实际人工工时或签署，不把AI/GPU耗时倒填为人工投入。本文件为阶段四报告生成时的快照，其中阶段五状态不代表当前项目进度。','',
        '| 编号 / 声明条目 | 示范主责 | 具体模块 | 产出证据（文件 / 实验） | 交叉核对职责（非真人已确认） | 报告位置 | 计划阶段工作量 |','|---|---|---|---|---|---|---:|']
    for r in roles:
        linked=[]
        for path in dict.fromkeys(r['files']+r['run_records']):
            linked.append(f'[{path}](../{path})' if (root/path).exists() else f'待补证：`{path}`')
        role_lines.append(f"| {r['id']} / {r['entry_id']} | {r['role']} | {r['module']} | {'<br>'.join(linked)} | {r['cross_role']}：{r['cross_check']}（示范职责，未记录真人确认） | {r['report_location']} | {r['planned_weight_percent']}% |")
    role_lines+=['','本阶段四报告时点的示范责任计划合计吴皓睿52%、周栩乐48%。这是教学分配权重，不代表实际人工时间；已按原条目比例进行整数分配。','',
        '机器可核对入口：[role-evidence.json](../'+str(role_path)+')。每项entry_id固定关联E编号、模块、文件、真实运行记录、报告位置和两名角色。contribution_statement_sections中第1项对应主责模块、第2项对应相同证据、第3项对应本阶段示范权重而不倒填工时、第4项指向另一角色声明中的队友确认位置；第5项关联⑦与附录二，确认文字、签名与日期仍为空。cross_check_status明确为示范职责，不是已发生的真人核验。E01—E12沿用[总证据索引](REPORT_EVIDENCE.md)，E13—E20补充本阶段。','',
        '最终附录统一命名为：附录一《复现与证据》，内含官方模板A复现、B代码、C日志迭代、D来源许可；附录二《教师示范制作与AI完整披露》。两份个人贡献材料第5项同时关联报告第⑦节的示范AI辅助声明与附录二的实际全AI制作范围，避免把附录二误作官方B代码结构。','',
        'E20所列合成测试日志只证明程序安全/指标反例验证，不冒充真实训练或最终评估成绩；真实最终核算与核查另指向aggregate/verify产物。ai-production.json若存在，只是回顾式事实制作索引，不冒充原始控制台日志或真人经历。只链接实际存在的日志，缺失证据明确标待补证；不生成不存在的AI活动日志、人工工时或确认。','',
        '示范情境中，AI可协助数据预处理脚本、代码草案、调试建议和润色；角色负责方案取舍、冻结协议、原文和指标核对、错误解释及结论。实际制作中这些操作由AI执行，附录二必须完整说明，不能把情境职责写成真实人工验证证据。当前不生成正式贡献签署件。','',
        '两份贡献材料仍遵循规范的五项结构：具体模块、对应证据、工作量自评、队友贡献确认、本人确认。各自证据编号与本表一致；工时若后续给出只能是明确教学示例；确认示例须关联附录二，签名处不制作仿真人类签署。','']
    write_new(root/'docs/阶段四角色与AI证据.md','\n'.join(role_lines))
    print(json.dumps(dict(report=str(report_path),role_evidence=str(role_path),teaching_role_weights=mapping['planned_totals_percent']),ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--aggregation',type=Path,default=Path('evidence/stage4/final-aggregation.json'));p.add_argument('--verification',type=Path,default=Path('evidence/stage4/final-verification.json'));p.add_argument('--report',type=Path,default=Path('docs/阶段四核查.md'));p.add_argument('--role-evidence',type=Path,default=Path('evidence/stage4/role-evidence.json'));a=p.parse_args();generate(a.root,a.aggregation,a.verification,a.report,a.role_evidence)

if __name__=='__main__':main()
