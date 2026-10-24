"""从已核查真实结果生成阶段二交接说明，不制作正式课程报告。"""
import json,datetime
from pathlib import Path
from ner_core import ROOT,dump


def load(p):return json.loads((ROOT/p).read_text())


def main():
    result=load('evidence/stage2/final-verification.json')
    if result['status']!='PASS':raise ValueError('尚未验收通过')
    run=ROOT/'runs/stage2-smoke-20261019'
    lines=['# 阶段二核查：最小训练链路','',
           '阶段二通过。这里只验证小规模链路，不代表正式模型效果；没有全量训练、最终评估或付费API调用。','',
           '## 完成内容与范围','',
           '- 固定128条训练、64条内部开发；对应129/64个窗口。种子20261019，开发抽样使用派生种子20261020。样本和源文件记录见runs/stage2-subset/manifest.json。',
           '- D0仅使用128条训练样本；N0字符BiLSTM＋GlobalPointer、N1官方中文BERT＋GlobalPointer。',
           '- GPU7 UUID已核对，A40 48GB；FP32、微批量4、累积8、有效批量32窗口，关闭AMP和TF32。',
           '- 每模型先训练4步，再从完整状态恢复2步；模型、优化器、随机状态和实际采样顺序交叉核对通过。',
           '- 两模型均在Mac本机CPU加载恢复后的权重，对同一64条开发资料推理；逐条保留预测与延迟。',
           '- 全部已处理train/dev仅做窗口映射检查，参考实体未覆盖数为0；不对全量资料执行模型推理，不读取sealed内容。','',
           '## 真实运行数字','',
           '| 模型 | 优化器更新步数 | 训练循环耗时 | GPU子进程墙钟 | 峰值张量显存 | 首步→末步损失 |',
           '|---|---:|---:|---:|---:|---:|']
    summaries={}
    for method in ('N0','N1'):
        segments=[json.loads((run/(method+'-'+s)/'summary.json').read_text()) for s in ('initial','resume')]
        losses=sum([s['losses'] for s in segments],[])
        training=sum(s['train_seconds'] for s in segments)
        wall=sum(s['seconds'] for s in result['gpu_budget']['segments'] if s['name'].startswith(method))
        peak=max(s['peak_memory_mb'] for s in segments)
        summaries[method]=dict(training=training,wall=wall,peak=peak,losses=losses)
        lines.append(f'| {method} | 4＋2 | {training:.3f}秒 | {wall:.3f}秒 | {peak:.2f}MiB | {losses[0]:.4f}→{losses[-1]:.4f} |')
    lines+=['',f"GPU子进程累计墙钟 **{result['gpu_budget']['consumed_seconds']:.3f}秒／600秒**，包括导入、加载、保存和退出，口径比纯训练计算保守。控制器另含模型文件核查等CPU工作，整个控制器耗时见evidence/stage2/gpu-controller.json。四次启动均无需等待外部占用；进程退出后已观察GPU7回到6MiB且无计算PID。",'',
            '显存为PyTorch峰值allocated张量显存，不能与nvidia-smi全进程占用混用。短文本动态填充的烟雾测试没有证明全量训练最大显存。','',
            '| 路线 | 严格P | 严格R | 严格F1 | 预测实体数／参考实体数 | CPU加载 | 64条CPU推理 |','|---|---:|---:|---:|---:|---:|---:|']
    d0=load('runs/stage2-D0/metrics.json')['micro']
    lines.append(f"| D0 | {d0['precision']:.4%} | {d0['recall']:.4%} | {d0['f1']:.4%} | {d0['predicted']}／{d0['gold']} | 不适用 | 见D0计时 |")
    for method in ('N0','N1'):
        c=result['models'][method]['cpu'];m=c['metric']
        lines.append(f"| {method} | {m['precision']:.4%} | {m['recall']:.4%} | {m['f1']:.4%} | {m['predicted']}／{m['gold']} | {c['load_seconds']:.3f}秒 | {c['inference_seconds']:.3f}秒 |")
    lines+=['','两模型只更新6步，以上分数仅用于检查评分和预测链路，不用于模型排名、参数选择或报告正式结果。N0输出10,507个候选，仅29个严格匹配；N1本轮64条全部为空预测、F1为0。都如实保存，不调整阈值或追加训练追求漂亮数字。','',
            'CPU推理延迟包含逐条窗口、前向、解码和合并，不包括加载及首次Python导入；全部命令墙钟另见evidence/stage2/N0-local-cpu.json与N1-local-cpu.json。CPU指标也保存了TensorBoard；验证横轴1代表本次开发验证事件，不冒称完整训练epoch。','',
            '## 环境、下载、传输与费用','',
            '- 本机Python 3.11.15／PyTorch 2.8.0；QHA Python 3.11.13／PyTorch 2.8.0，CUDA发行版由远端实际环境记录核对；Transformers 4.57.6。uv.lock固定实际依赖。',
            '- 本机uv首次安装输出：解析4.03秒、包准备43.85秒、安装273毫秒。这是工具内部阶段计时，没有把它伪装成单独测量的命令墙钟。',
            '- Linux Python官方安装下载37.41秒；来自本次实际uv输出，环境工具与Python通过本机获取后传入项目。',
            '- 官方BERT四个文件下载与版本见evidence/stage2/model-download.json；模型卡许可Apache-2.0。',
            '- QHA不计新增租赁费用是用户给定条件；仍记录GPU时间，不能推广为GPU计算没有成本。',
            '- 新增付费API调用0，新增费用0元；累计保守估费3.792488元，上限100元，余额96.207512元。816条历史请求未变；不是账户实际扣费账单。','']
    for name in ('remote-environment','remote-environment-mirror','model-transfer-to-qha','remote-cpu-tests','gpu-controller','inference-weights-transfer','N0-inference-transfer','N1-inference-transfer','N0-local-cpu','N1-local-cpu'):
        p=ROOT/'evidence/stage2'/f'{name}.json'
        if p.exists():
            d=json.loads(p.read_text());lines.append(f"- {name}：{d['seconds']:.3f}秒，退出码{d['returncode']}；原始控制台同名.log。")
    lines+=['','时间记录保留各机器原始时间；耗时使用各进程单调时钟，不用文件修改时间倒推运行或人工工时。安装、传输可并行，不把分段时间简单相加成总工时。','',
            '## 真实问题、处理与复测','',
            '| 问题 | 处理 | 实际验证与限制 |','|---|---|---|',
            '| QHA直连GitHub的TLS失败；本机curl同入口也超时 | 从PyPI获取官方uv Linux发行包，本机获取官方Python后传入独立目录 | 远端Python 3.11与独立环境创建成功；不修改全局环境 |',
            '| QHA直连files.pythonhosted.org握手失败，依赖安装退出1 | 用锁定清单导出带校验值requirements，经镜像下载并强制校验 | 镜像安装退出0，版本与锁定清单对应，远端18项检查通过；保留失败控制台 |',
            '| 测试时BERT动态切换attention实现产生警告 | 在构造BertConfig前固定eager实现 | 本机复测18项通过且不再出现该警告；不改实验精度 |',
            '| 首次权重传输的rsync过滤顺序会包含初始权重 | 发现后仅停止自己创建的传输进程，改为两个明确的resume/inference.pt路径 | 初次传输退出20被保留，最终两份权重与GPU记录一致；未传优化器 |',
            '| 仅6步训练的预测质量很低 | 原样保留所有候选和空预测，不在当前开发样本上调阈值 | 证明格式、指标和完整样本处理成立；不声称已获得可用抽取性能 |','',
            '这些工程问题与实体抽取错误分开。阶段五仍需至少三个真实失败案例及三类有依据的归因；不能用TLS错误凑抽取失败样例。','',
            '## 复现与证据入口','',
            '- README.md：环境、固定子集、基线、免费audit、CPU推理和阶段二证据核查命令。',
            '- evidence/stage2/final-verification.json：样本顺序、恢复状态、事件标签、CPU指标、权重对应、预算与原RAG文件的一致核查。',
            '- runs/stage2-smoke-20261019：四段真实.log、metrics.jsonl、summary.json、restore-check.json和TensorBoard事件。完整恢复检查点仅留远端，Mac只保留恢复后推理权重。',
            '- runs/stage2-N0-cpu、runs/stage2-N1-cpu：64条真实预测、指标、逐条延迟、张量形状和开发TensorBoard。',
            '- evidence/stage2/immediate-prelaunch-probe.json、post-run-gpu.json：启动前与结束后的真实设备证据。',
            f"- 原RAG业务文件{result['original_rag_files_unchanged']}项与阶段一基线一致；未修改原项目、旧账本或历史结果。",'',
            '## 下一阶段、报告与角色','',
            '- configs/development-experiments.json已登记分词边界、学习率、跨度头RoPE与指标/标签约束对照，均尚未执行。仅开发集选择参数，负结果同样保留。',
            '- docs/REPORT_EVIDENCE.md逐项对应报告①—⑧及附录A—D，并维护吴皓睿／周栩乐同一套角色与证据编号。教学比例目标50%／50%，必要时52%／48%且吴皓睿偏多。实际操作由AI执行；没有填造真实人工工时、相互确认或签名。',
            '- 尚未完成正式训练、最终评估、正式三类失败分析、最终PDF、贡献签署文件、ZIP、Windows验收或录屏。',
            '- 阶段二到此停止。阶段三按docs/STAGE3.md推进，不能自动启动。','',
            '### 仅供规划的线性耗时估算','']
    full_windows=load('runs/stage2-subset/manifest.json')['full_processed_mapping_only']['train']['windows']
    steps=__import__('math').ceil(full_windows/32)
    for method,values in summaries.items():
        seconds=values['training']/6*steps
        lines.append(f'- {method}：按6步训练循环均值×{steps}步，约{seconds/60:.2f}分钟／轮纯训练。')
    lines+=['','这是小样本动态长度下的线性估算，未包含完整开发集验证、不同长文批次、检查点、NFS争用或早停；可信度低，不能作为全量运行上限。阶段三首轮应重新测量后安排训练段。','']
    (ROOT/'docs/阶段二核查.md').write_text('\n'.join(lines),encoding='utf-8')
    status=f'''# 当前状态

更新时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}

**阶段二完成：小规模训练、保存恢复与Mac CPU推理通过。停在阶段三之前。**

- 阶段一冻结数据不变；阶段二仅128条训练、64条内部开发，未读取sealed正文/标注进行开发。
- D0、N0、N1链路通过；两模型各4＋2步，步数/优化器/随机状态/采样位置连续。
- GPU7单卡A40，GPU子进程累计墙钟{result['gpu_budget']['consumed_seconds']:.3f}秒，限额600秒；已退出并释放资源。
- 本机与QHA各18项合成检查通过；两模型Mac CPU各输出64条合法预测，指标可重算。
- 新增API0次，费用0元；累计估费3.792488元，限额100元，余额96.207512元。
- 原RAG业务文件{result['original_rag_files_unchanged']}项未变。实际制作由AI执行，角色映射与真实操作分开记录。

## 阅读入口

1. [阶段二核查](阶段二核查.md)
2. [报告证据与贡献角色](REPORT_EVIDENCE.md)
3. [阶段三清单](STAGE3.md)
4. [阶段一数据核查](阶段一数据核查.md)
5. [QHA资源登记](QHA_REGISTRY.md)

## 尚未完成

全量训练、开发参数实验、L0/L3、最终holdout评估、正式三类失败分析、演示界面、报告PDF、摘要、逐人贡献材料、ZIP、Windows验收、录屏。不能将当前小样本成绩写作正式模型质量。
'''
    (ROOT/'docs/STATUS.md').write_text(status,encoding='utf-8')
    print(json.dumps(dict(report='docs/阶段二核查.md',status='stage2_complete'),ensure_ascii=False))

if __name__=='__main__':main()
