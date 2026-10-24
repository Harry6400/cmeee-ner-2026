"""仅从已通过核查的开发证据生成交接说明和阶段四冻结协议；不执行训练/评估。"""
import json,datetime
from pathlib import Path
from ner_core import ROOT,read_rows,dump,score,write_rows,digest


def load(path):return json.loads((ROOT/path).read_text())


def main():
 check=load('evidence/stage3/final-verification.json');cp=load('evidence/stage3/checkpoint-inspection.json')
 if check['status']!='PASS' or cp['status']!='PASS':raise ValueError('尚未完整通过验收')
 base=ROOT/'runs/stage3-20261019';models=check['models'];selected=check['lr_selection']['selected'];on=selected['name'];off='N1-rope-off'
 timing=load('evidence/stage3/runtime-summary.json');lifecycle=models[off].get('interrupted_lifecycle');replay=load('evidence/stage3/resume-replay-prefix-check.json')
 if lifecycle and (lifecycle['status']!='completed' or timing['discarded_partial_updates']!=lifecycle['discarded_partial_updates']):raise ValueError('生命周期与计时汇总未通过')
 chosen=off if models[off]['strict']['f1']>models[on]['strict']['f1'] else on
 baselines=check['baseline']['results'];d0=max(baselines,key=lambda k:baselines[k]['metrics']['micro']['f1'])
 config=json.loads((base/chosen/'config.json').read_text());api_protocol=load('runs/stage3-20261019/api-development-grounded/protocol.json')
 frozen=dict(status='frozen_after_development;stage4_not_executed',frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  task='CMeEE-V2 Unicode character spans;9 types;retain nesting/crossing/coextensive labels',
  data=dict(train='data/processed/train.jsonl',train_samples=13497,dev='data/processed/dev.jsonl',dev_samples=1503,
            final_holdout='data/sealed/holdout.jsonl',final_samples=5000,common_subset='data/sealed/common500.jsonl',common_samples=500,
            data_manifest='data/manifests/readiness.json',same_text_overlap=0,near_overlap_audit='disclosed non-exhaustive;one train/dev near-pair',
            sealed_content_used_for_development=False),
  selected=dict(D0=dict(variant=d0,alternatives_reported=list(baselines),selection='higher internal dev strict F1;both results retained'),
                N0=dict(run='N0',best_epoch=models['N0']['best_epoch'],config=json.loads((base/'N0/config.json').read_text())),
                N1=dict(run=chosen,selected_encoder_lr=config['n1_encoder_lr'],rope=config['rope'],best_epoch=models[chosen]['best_epoch'],config=config,
                        selection='LR by highest dev F1 among3 rope-on trials,tie lower LR;then compare rope on/off at chosen LR,tie retain on',
                        no_claim_exhaustive_lr_rope_interaction=True)),
  training=dict(runtime_selection=check['performance_summary']['selected_runtime'],runtime_evidence='evidence/stage3/performance-summary.json',formal_stage3_runtime_unchanged=True,optimizer=dict(name='AdamW',actual_groups={n:cp['models'][r]['optimizer_groups'] for n,r in [('N0','N0'),('N1',chosen)]},gradient_clip_norm=1.0,scheduler=None),seeds=[20261019,20261020,20261021],seed19_reuse='existing selected development checkpoint;not a best seed search',
                seed20_21='not trained yet;requires stage4 authorization',train_dev_not_merged=True,effective_batch=32,precision='fp32',
                fp32_no_amp_no_tf32=True,epoch_tail='5 repeated shuffled training windows to hold32;no dev/test padding',
                max_epochs=dict(N0=20,N1=10),patience=3,checkpoint_selection='own internal dev F1 for each seed;never final holdout'),
  api=dict(model='deepseek-flash',official_version_at_development='DeepSeek-V4.1-Flash-1017',endpoint='https://api.deepseek.com/chat/completions',
           system=api_protocol['system'],example_ids=api_protocol['example_ids'],examples='runs/stage3-20261019/api-development-grounded/train-examples.jsonl',
           temperature=0,thinking='disabled',max_output_tokens=4096,response_format='json_object',
           wire='text,type,occurrence;exact Unicode substring nth occurrence including overlaps',canonical='start,end,type,text',
           invalid='no fuzzy repair,no normalizing text;retain raw errors;main entity F1 unavailable unless same full sample set valid;conditional valid-only score is diagnostic',
           no_automatic_retries=True,unknown_cost_stops=True,model_and_price_recheck_before_final_calls=True,
           development_trials='2 versions*12calls=24;V1 errors retained;no further tuning on final cases'),
  metrics=dict(primary='strict entity start+exclusive end+type micro P/R/F1;case-sensitive;deduplicate exact tuples;zero denominator=0',
               auxiliary=['class F1','strict nested recall','isolated/crossing/coextensive strata','unseen surface strict F1','multi-hot character(type,position) micro F1','diagnostic overlap one-to-one F1','invalid output rate'],
               invalid_not_silently_dropped=True),
  future_evaluation=dict(common500='same fixed500 for all5 routes;paired resampling differences;no method-specific denominators',
                         local_models='full5000 holdout separately;report common500 and full5000 distinctly',
                         no_official_hidden_test_score_claim=True,multi_seed='report all3 seeds;do not select best seed on holdout',
                         latency='separate loading,single-sample inference/API completion;mean/P50/P95 and errors',
                         known_case_fixes='independent reruns;never overwrite frozen main outputs'),
  budget=dict(limit_cny='100',spent_cny=check['budget']['spent_cny'],remaining_cny=check['budget']['remaining_cny'],
              ledger='private/budget/ledger.json',api_currently_enabled=False,qha_new_rental='0 per user-provided condition;GPU time still measured'))
 dump(ROOT/'configs/stage4-frozen-protocol.json',frozen)
 # 同步可变进度文件的终态，原始逐轮日志不改写。
 for name in models:
  terminal=json.loads((base/name/'summary.json').read_text());progress_path=base/name/'progress.json'
  progress=json.loads(progress_path.read_text());progress.update(status=terminal['status'],finished_at=terminal['finished_at'],terminal_status_synced_at=datetime.datetime.now(datetime.timezone.utc).isoformat());dump(progress_path,progress)
 # 相同6条开发样本的路线检查，不能与1503条成绩混用分母。
 common=read_rows(base/'api-development-grounded/dev.jsonl');ids={r['id'] for r in common};shared={}
 for mode in baselines:
  p=[r for r in read_rows(base/'baselines'/(mode+'-predictions.jsonl')) if r['id'] in ids];shared['D0-'+mode]=score(common,p)
 for name in ('N0',on,off):
  summary=json.loads((base/name/'summary.json').read_text());p=[r for r in read_rows(base/name/f"dev-epoch-{summary['best_epoch']:02d}.jsonl") if r['id'] in ids]
  shared[name]=score(common,p)
 for route in ('L0','L3'):shared[route]=score(common,read_rows(base/'api-development-grounded'/(route+'-valid-predictions.jsonl')))
 dump(base/'development-six-sample-check.json',dict(scope='6 internal dev only;not final common500;not independent test',sample_ids=sorted(ids),results=shared))
 lines=['# 阶段三核查：开发训练与参数冻结','',
  '**阶段三完成；未评估最终留出集，未进入阶段四。**',
  '', '## 先复核与诊断','',
  '- 阶段一二关键证据已复核；历史安装整条命令墙钟等无法补造的缺口见docs/阶段一二复核.md。源码、费用与所有失败按实际来源记录。',
  '- 课程B类允许BERT微调；D0、N0、N1、L0/L3符合信息抽取方向。数据中存在嵌套，采用跨度头有依据；输入已有全文，不需要另做RAG。',
  '- N1原六步全空已独立复现。192条理想标签分数回映929/929实体；手工损失与梯度方向通过；八条训练资料继续30步达到95.38%训练F1。支持“六步训练不足”的解释，未发现主要标签/遮罩/解码实现错误；不是对所有实现的穷尽证明。',
  '- 原六步预测未覆盖，阈值保持0，没有通过调阈值制造改善。',
  '',f"共同接口：原文Unicode字符右开跨度，窗口{config['window_chars']}、重叠{config['overlap_chars']}、解码阈值{config['decode_threshold']}。N0为字符嵌入{config['embedding_dim']}维、BiLSTM每方向{config['lstm_hidden']}维、跨度头{config['head_size']}维，学习率{config['n0_lr']}；N1为固定官方中文BERT及{config['head_size']}维跨度头，头学习率{config['n1_head_lr']}，编码器比较三组学习率。完整外置参数和实际版本见各run的config/metadata与阶段二模型下载清单。",'', '## 完整内部开发集结果（1503条、8247个参考实体）','',
  '| 路线／参数 | 严格P | 严格R | 严格F1 | 字符类型F1 | 嵌套召回 | 最佳轮／实际轮 | 运行耗时 | 峰值张量显存 |',
  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,r in baselines.items():
  m=r['metrics'];s=m['micro'];lines.append(f"| D0 {name} | {s['precision']:.2%} | {s['recall']:.2%} | {s['f1']:.2%} | {m['character_type_micro']['f1']:.2%} | {m['nested_recall']:.2%} | 不训练 | CPU推理{r['inference_seconds']:.3f}秒 | 不用GPU |")
 for name,r in models.items():
  s=r['strict'];lines.append(f"| {name} | {s['precision']:.2%} | {s['recall']:.2%} | {s['f1']:.2%} | {r['character']['f1']:.2%} | {r['nested_recall']:.2%} | {r['best_epoch']}／{r['epochs']} | {r['seconds']/60:.2f}分钟 | {r['peak_memory_mb']:.1f}MiB |")
 lines+=['', '以上全部是选参使用的内部开发成绩，不能称为最终测试或泛化性能。所有候选仅种子20261019；未比较或挑选最终留出上的最好种子。表中D0计CPU推理时间，神经模型计训练作业总时间，不能据此比较推理速度；正式单条延迟阶段四另测，验证批处理耗时不能冒称单条延迟。',
  '',f"选定D0为 **{d0}**；保留另一词典结果。N1先在RoPE开启时选择学习率 **{selected['lr']}**，再比较相同学习率的跨度头RoPE；最终选定 **{chosen}**。未穷尽学习率×RoPE交互，不能声称全局最优。",
  '', '## 对照的实际含义','',
  f"- 词典只取13497条训练资料，共{check['baseline']['dictionary_terms']}词，{check['baseline']['ambiguous_terms']}个同词多类；按训练频次选类，平票采用固定九类顺序，歧义记录保留，不用dev扩词典。",
  '- 分词：同一训练词典仅限制词边界。F1提高伴随召回下降；真实丢失术语与jieba切分保存在baselines/boundary-cases.json。不是“分词总会更好”。',
  '- 结构：单层、不重叠BIO在内部dev最多表示7070/8247个参考实体，约85.73%；这是表示覆盖上限，没有训练平面BIO模型，也不能当作其实际性能。',
  '- 位置：RoPE对照只关闭跨度头旋转位置编码，BERT内部位置编码仍保留；相同上限10轮与连续3轮无改善早停，实际停止轮数如表。另在off启动前记录固定3轮的辅助口径，相同1302次更新的严格F1：'+str({name:round(value['strict']['f1'],6) for name,value in check['fixed_budget_rope'].items()})+'；该固定预算结果与各自最佳轮选择分开解释。',
  '- 指标：主指标起点、右开终点、类别全部匹配。辅助字符指标的计数单位是(Unicode字符位置,类别)，非O、多标签覆盖；同类嵌套可能合并覆盖，不能替代实体指标。宽松重叠采用同类型的一对一最大匹配，仅作诊断。',
  '- 训练末批为固定有效批量32补5个已打乱训练窗口，逐轮记录；验证不补样本、不删尾批。所有参数试验负结果同样保留。',
  '', '## API小规模开发与真实失败','',
  '- 两版各12次，共24次；每版同一6条dev与3条训练示例，L0/L3只改变示例。API样本ID含train字样是官方源文件前缀：六条确在固定内部dev，训练示例与其分离。',
  '- 第一版直接要求数字位置，L0/L3均4/6合法；各两条原文位置错误完整保留，不修正覆盖、不计作正常空实体；没有给缺样本路线伪造完整六条F1。',
  '- 第二版让模型输出实体原文＋出现序号，程序精确映射原文；无规范化、模糊匹配或错误边界修补，12/12合法。定位合法并不代表类别与语义正确，全部预测仍按参考标注评分。',
  '', '| 第二版路线 | 样本数 | 严格P | 严格R | 严格F1 | 平均API完整响应 | 保守费用 |','|---|---:|---:|---:|---:|---:|---:|']
 api=check['api_summary'];routes=api['batches']['api-development-grounded']['routes']
 for name,r in routes.items():
  s=r['strict_on_all_six']['micro'];lines.append(f"| {name} | 6 | {s['precision']:.2%} | {s['recall']:.2%} | {s['f1']:.2%} | {r['latency_mean_seconds']:.3f}秒 | {r['conservative_cny']}元 |")
 lines+=['', '三样本在这六条上没有优于零样本；样本很少且属于开发资料，不能推为通用结论。相同六条的本地模型/词典结果另存development-six-sample-check.json，不把6条与1503条成绩混成公平排名。',
  '',f"本阶段新增API保守估费 **{api['new_cny']}元**，累计 **{check['budget']['spent_cny']}元／100元**，余额 **{check['budget']['remaining_cny']}元**；840条明细包含完整816条历史前缀。API关闭，不自动重试或继续调用。费用按已核查官方峰时未缓存价保守计，不是账户实际扣费账单。缓存用量、原始响应和实际模型名均在原记录。",
  '', '## 运行、问题与证据','',
  '- GPU7 UUID固定，单卡FP32，微批量4、累积8，AMP/TF32关闭，CPU线程4、workers=0。五个训练任务顺序执行，每段小于23小时30分钟；队列记录实际等待与启动状态，未操作其他进程或实例生命周期。',
  (f"- RoPE关闭组共保留{len(lifecycle['segments'])}个真实训练分段，其中{timing['interrupted_training_segments']}段因外部GPU7占用安全停止。恢复从第{lifecycle['restored_epoch']}轮/{lifecycle['restored_global_step']}步完整检查点开始，累计{lifecycle['discarded_partial_updates']}次未完成轮更新回退，原始更新计算{lifecycle['discarded_partial_update_seconds']:.3f}秒仍计入实际资源时间。模型、优化器、RNG恢复准备直接核对；前一次实际恢复的{replay['compared_updates']}步损失、梯度与采样位置逐项一致。原始失败/中断队列保留，派生完成汇总另存。用户主动暂停的整夜间隔不计作GPU训练或外部占用排队。" if lifecycle else '- 本次开发任务无需训练恢复；原始控制台和队列均保留。'),
  '- 每轮保留原始开发预测、训练步JSONL、TensorBoard、配置、数据与源码版本、最佳/最后检查点、吞吐、显存、保存及验证耗时。运行总耗时包含加载和保存，训练/验证分段另存，不把并行的CPU/API时间重复相加。',
  '- best/last已在远端CPU严格加载核查，权重有限，优化器步数与最后完成轮对应；不为核查复制全部大检查点到本机。',
  '- 发现已完成N0的progress.json仍显示running：原因是训练器只在逐轮写进度，终态另存summary。结束验收按summary同步显示状态，旧进度另存progress-before-terminal-sync.json，修改/复测见checkpoint-inspection.json；原始训练与预测不变。',
  '- 工程与抽取错误分开：原空预测、API数字位置失败及第二版接口变化均有前后记录；develop错误候选按边界、类型、未登录词与嵌套初筛，归因仍需结合原文解释，不能把启发式标签当成已证明机制。',
  '- API授权协议原件执行后状态变化，现按账本中原授权版本重建并核对，重建来源明确标注；不把重建副本伪装成历史原始快照。执行时客户端源码另行保留，后续客户端已加不可变快照与跨批次预算锚点。',
  '- 公开语料不等于临床有效性。九条外发段落逐条查看未见明显个人识别信息，不宣称供应商不留存请求；私有病例、组员身份、账本与密钥内容没有外发。数据/模型/环境/private继续不进入最终ZIP。',
  f"- 原RAG业务文件{check['original_rag_files_unchanged']}项与基线一致，未修改原项目、旧账本或历史结果。",
  '', '证据入口：evidence/stage3/final-verification.json、checkpoint-inspection.json、api-audit.json、queue-develop.json；runs/stage3-20261019各模型目录；docs/REPORT_EVIDENCE.md。',
  '', '## 下一阶段与仍未完成','',
  '- configs/stage4-frozen-protocol.json已冻结选择与口径；阶段四仍须用户指示，不能因本轮完成而自动评估。',
  '- 后续五路线在同一固定500条上比较；本地模型另评完整5000条；再训练/核对另外两种子，不挑最终最好种子。API开始前重新核价，已知案例修复独立复测，不覆盖主成绩。',
  '- 三张设计图、正式至少三类失败归因、最终报告/PDF、摘要、个人材料、干净解压复现、Windows实机与≤2分钟录屏均未完成，不提前标通过。',
  '- 教师模拟优秀作业的角色分工以50%／50%为目标，需偏重时统一吴皓睿52%、周栩乐48%。正文与贡献材料演示学生主导的方案选择、核验和解释及合理AI辅助；最终PDF附录二集中披露整个示范的实际AI制作范围和角色经历、工时、确认的模拟性质。贡献材料关联附录二，同一证据映射互相佐证。真实实验日志和数字不以模拟经历替代。','']
 vocabulary=load('evidence/stage3/character-vocabulary-audit.json')['methods']['N1']['splits']
 lines+=['## 字符接口与表示局限','',f"N1逐原文Unicode字符直接查BERT词表，未命中使用UNK，不改写文本位置。这与标准WordPiece分词不同；训练字符出现次数UNK率{vocabulary['train']['unknown_rate']:.4%}，内部dev为{vocabulary['dev']['unknown_rate']:.4%}。它保持坐标接口，但可能损失部分未命中字符的表示信息；本轮没有追加分词优化来改善成绩。见[字符词表审计](../evidence/stage3/character-vocabulary-audit.json)。",'']
 timing=load('evidence/stage3/runtime-summary.json')
 shanghai=datetime.timezone(datetime.timedelta(hours=8))
 interval=' 至 '.join(datetime.datetime.fromisoformat(timing[k]).astimezone(shanghai).strftime('%Y-%m-%d %H:%M:%S') for k in ('queue_started_at','queue_finished_at'))
 lines+=['## 分段耗时与直接证据','',f'开发队列实际起止：北京时间{interval}。原始日志使用UTC，运行目录沿用启动日命名；不因跨日改写原始时间。','',
  '| 任务 | 子进程墙钟秒 | 完整轮训练秒 | 验证秒 | 保存秒 | 回退更新秒 | 已提交训练窗口/秒 |',
  '|---|---:|---:|---:|---:|---:|---:|']
 for r in timing['gpu_development_jobs']:
  lines.append(f"| {r['name']} | {r['child_wall_seconds']:.3f} | {r['training_loop_seconds']:.3f} | {r['validation_seconds']:.3f} | {r['checkpoint_save_seconds']:.3f} | {r['discarded_partial_update_seconds']:.3f} | {r['training_loop_windows_per_second']:.2f} |")
 lines+=['',f"开发训练子进程累计{timing['gpu_development_child_wall_seconds']/60:.2f}分钟，另有诊断子进程{timing['gpu_diagnostic_child_wall_seconds']:.3f}秒。子进程墙钟含加载、训练、验证与保存，不能称为孤立CUDA内核时间。外部占用等待观测{timing['external_busy_observations']}次（观测次数不是时长），各恢复控制器启动前轮询累计{timing['prelaunch_resume_polling_wall_seconds']:.3f}秒（含复查）；用户主动暂停另记，不能把队列跨夜起止差称为连续GPU耗时；本轮复用阶段二BERT，新增模型下载为0。",'',
  '- [逐模型原始控制台最终行摘录](../evidence/stage3/training-console-excerpts.log)',
  '- [分段计时与计时口径](../evidence/stage3/runtime-summary.json)；[实际证据传输计时](../evidence/stage3/final-evidence-transfer.json)',
  '- [原始开发队列及中断](../evidence/stage3/queue-develop.json)；[第一次恢复队列](../evidence/stage3/queue-resume.json)；[第二次恢复队列](../evidence/stage3/queue-resume-02.json)；[派生完成汇总](../evidence/stage3/queue-develop-resolved.json)；[开发控制台日志](../evidence/stage3/development-controller.log)',
  '- [恢复状态核对](../evidence/stage3/resume-preparation.json)；[第二次安全中断归档](../evidence/stage3/resume-attempt-01-archive.json)；[真实重放前缀核查](../evidence/stage3/resume-replay-prefix-check.json)；[用户主动暂停记录](../evidence/stage3/user-requested-pause.json)',
  '- [预测、事件与预算独立核算](../evidence/stage3/final-verification.json)；[远端CPU检查点核查](../evidence/stage3/checkpoint-inspection.json)',
  '- [24次API核算](../evidence/stage3/api-audit.json)；[第一版失败记录](../runs/stage3-20261019/api-development/summary.json)；[第二版记录](../runs/stage3-20261019/api-development-grounded/summary.json)',
  '- [真实开发错误候选](../runs/stage3-20261019/development-error-candidates.json)；[结构分层召回](../runs/stage3-20261019/structural-recall.json)',
  '- [证据与教学角色统一索引](REPORT_EVIDENCE.md)；[下一阶段清单](STAGE4.md)','']
 performance=check['performance_summary']
 lines+=['## GPU吞吐核查','',f"有界预算{performance['tuning_budget_seconds']}秒，实际调优{performance['tuning_total_seconds']:.3f}秒；运行配置选择：{performance['selected_runtime']}。已选配置通过数值等价核查；已完成的正式阶段三对照与权重不替换。利用率、吞吐、失败候选及剩余瓶颈见[真实性能结果](../evidence/stage3/performance-summary.json)，不凭目标值宣称GPU达到85%。",'']
 if (ROOT/'evidence/stage3/performance-setup-issue.json').exists():
  lines+=['- 本次性能测试首次因远端缺少冻结配置文件而在GPU子进程启动前失败；原始失败控制台保留。补齐配置后重试，首次准备耗时计入同一110秒预算，见[现象、依据、修改与复测](../evidence/stage3/performance-setup-issue.json)及[复测结果](../evidence/stage3/performance-setup-retest.json)。','']
 lines+=['| 性能候选 | 测量窗口/秒 | 40步含准备秒 | 全子进程平均GPU利用率 | 核查状态 |','|---|---:|---:|---:|---|']
 for c in performance['candidates']:
  util=c.get('gpu_util_mean_pct',[])
  lines.append(f"| {c['name']} | {c.get('throughput',0):.2f} | {c.get('training_end_to_end_seconds',0):.3f} | {util[0] if util else 0:.1f}% | {c['status']} |")
 lines+=['','吞吐仅对应同一批固定训练窗口、10次预热及30次测量；利用率按包含加载、数值比较与保存的完整子进程采样，不能当作纯训练利用率。CPU缓存准备开销纳入运行方案选择；本测试权重不替换正式开发模型。','']
 (ROOT/'docs/阶段三核查.md').write_text('\n'.join(lines),encoding='utf-8')
 (ROOT/'docs/STAGE4.md').write_text('''# 阶段四清单（尚未执行）

读取configs/stage4-frozen-protocol.json后，只按冻结的模型、示例、参数与严格指标执行。用户尚未授权时不得读取最终留出内容选参或评估。

- 保留已选种子20261019检查点；种子20261020/20261021按相同协议训练和开发早停，不在最终集选种子。
- 所有路线共用固定common500，且本地模型另报完整holdout5000，二者不能当两份独立测试。
- 逐条保存全部输入、原始预测、异常与用量。非法API输出不修正为成功或正常空实体；主F1要求共同完整分母，条件有效子集仅作诊断。
- 延迟分开加载、正常单条推理与完整API响应，报均值/P50/P95及失败数；预算延续原唯一账本、重新核价，未知费用停止。
- 主结果冻结后再做已知案例修复，独立保存复测，不回写主成绩。
- 完成后核对所有汇总与原始输出、报告证据索引、资源退出与费用，再进入阶段五。
''',encoding='utf-8')
 status=f'''# 当前状态

阶段三完成，阶段四未执行。

- 阶段一二已复核，缺口与方向解释见阶段一二复核.md。
- N1空预测诊断通过，五个开发训练任务与参数对照已完成。
- 最终开发选择：D0={d0}；N0最佳第{models['N0']['best_epoch']}轮；N1={chosen}，学习率{config['n1_encoder_lr']}，跨度头RoPE={config['rope']}。
- 训练只用13497条train，选参只用1503条内部dev；最终留出未评估。
- API开发24次已结算，累计保守估费{check['budget']['spent_cny']}元／100元，API关闭。
- 全部原始证据、检查点及冻结协议已保存；原RAG {check['original_rag_files_unchanged']}项业务文件未变。

阅读：阶段三核查.md → REPORT_EVIDENCE.md → STAGE4.md。

最终PDF/ZIP、设计图、正式失败分析、Windows和录屏尚未完成。教师模拟示范的正文、⑧分工表和两份贡献材料共用证据映射；最终附录二披露整个示范的实际AI制作范围与角色经历的模拟性质，贡献材料关联该说明。
本轮结束后不自动启动阶段四或创建定时计划。
'''
 (ROOT/'docs/STATUS.md').write_text(status,encoding='utf-8')
 print(json.dumps(dict(status='stage3_complete',selected_N1=chosen,selected_D0=d0,stage4_executed=False),ensure_ascii=False))

if __name__=='__main__':main()
