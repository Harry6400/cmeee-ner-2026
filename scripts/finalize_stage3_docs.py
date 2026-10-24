"""阶段三证据验收通过后同步入口和实验状态，不执行任何评估。"""
import json
from pathlib import Path
from ner_core import ROOT,dump


def main():
 check=json.loads((ROOT/'evidence/stage3/final-verification.json').read_text());runtime=json.loads((ROOT/'evidence/stage3/runtime-summary.json').read_text())
 if check['status']!='PASS' or not (ROOT/'docs/阶段三核查.md').is_file():raise ValueError('尚未验收')
 p=ROOT/'configs/development-experiments.json';old=p.read_bytes();archive=ROOT/'evidence/stage3/development-experiments.planned.json'
 if not archive.exists():archive.write_bytes(old)
 d=json.loads(old);d['status']='stage3_executed_internal_dev_only';d['coursework_requirement']='分词边界、结构覆盖、跨度头位置编码、指标口径已绑定真实开发数据；正式报告解释仍在后续阶段'
 links={'A1':['runs/stage3-20261019/baselines/summary.json','runs/stage3-20261019/baselines/boundary-cases.json'],
  'A2':['configs/rope-comparison-plan.json','evidence/stage3/final-verification.json'],
  'A3':['runs/stage3-20261019/lr-selection.json','runs/stage3-20261019/N1-lr-1e-05/summary.json','runs/stage3-20261019/N1-lr-3e-05/summary.json','runs/stage3-20261019/N1-lr-5e-05/summary.json'],
  'A4':['runs/stage3-20261019/baselines/flat-bio-upper-bound.json','runs/stage3-20261019/structural-recall.json','evidence/stage3/final-verification.json']}
 for exp in d['experiments']:
  exp['status']='completed_on_internal_dev';exp['evidence']=links[exp['id']]
  if exp['id']=='A2':
   exp['fixed']='同一选定学习率、种子、窗口、有效批量、最大10轮与patience3；实际停止轮数另报；固定3轮辅助对照'
   exp['budget_interpretation']='best-dev under same max10/patience3 plus fixed3epoch/1302update auxiliary comparison;actual early-stopping epochs reported separately'
 dump(p,d)
 p=ROOT/'docs/STAGE3.md';archive=ROOT/'evidence/stage3/STAGE3.planned.md'
 if not archive.exists():archive.write_bytes(p.read_bytes())
 p.write_text('''# 阶段三：已完成

用户已授权并完成本阶段开发实验，最终留出评估尚未执行。原清单保存在evidence/stage3/STAGE3.planned.md。

- [阶段三核查与真实结果](阶段三核查.md)
- [阶段一二复核及缺口](阶段一二复核.md)
- [报告要求与角色证据索引](REPORT_EVIDENCE.md)
- [下一阶段清单（未执行）](STAGE4.md)

实际执行：N1实现诊断、完整训练词典边界对照、N0、N1三组学习率、选定学习率的RoPE对照，以及获明确授权的两版L0/L3小规模API检查。所有选参只用内部开发集，原始失败不覆盖。冻结协议为configs/stage4-frozen-protocol.json。
''',encoding='utf-8')
 p=ROOT/'docs/REPORT_EVIDENCE.md';s=p.read_text().replace('N0/N1学习率与RoPE运行中','N0/N1学习率与RoPE已完成，结果见docs/阶段三核查.md与evidence/stage3/final-verification.json')
 s=s.replace('分词边界/BIO覆盖已执行，N0和N1三组学习率完成，RoPE关闭组8轮后按用户指示暂停，全部结论尚待最终核查','分词边界/BIO覆盖、N0、N1三组学习率与跨度头RoPE对照已完成，全部原始分段与最终核算见docs/阶段三核查.md及evidence/stage3/final-verification.json')
 s=s.replace('此文件是制作台账，不是最终报告或已签署的贡献声明。实际实现、调试、运行与初核均由AI执行；两名同学为教学角色。当前没有真实人工工时、个人签名或相互确认记录。','此文件是教师模拟优秀作业的制作台账。真实实现、调试、运行与初核由AI执行；正文及贡献材料演示两名角色主导取舍、核验与解释及合理AI辅助，角色经历、示例工时和互相确认的模拟性质在最终PDF附录二集中披露，贡献材料关联该说明。真实实验数字与日志保持独立可核验。')
 if 'runtime-summary.json' not in s:s+='\n阶段三完成证据：evidence/stage3/runtime-summary.json、training-console-excerpts.log、checkpoint-inspection.json、final-verification.json；参数与提示冻结在configs/stage4-frozen-protocol.json。误差候选不是已确认因果，最终报告与贡献材料继续引用E01—E12。\n'
 p.write_text(s,encoding='utf-8')
 p=ROOT/'README.md';s=p.read_text().replace('当前阶段三开发训练与参数对照运行中','阶段三开发训练与参数对照已完成').replace('当前阶段三等待GPU7恢复，审计及排期见docs/阶段三中期审计与GPU排期.md','阶段三开发训练与参数对照已完成，真实结果与分段耗时见docs/阶段三核查.md').replace('18项合成边界与机制检查','26项合成边界、机制、指标与API预算检查')
 s=s.replace('两名同学的模块对应教学角色，实际AI工作与角色映射分开记录；不虚构实际工时、签名或相互确认。','教师模拟示范的两名角色分工、正文⑧节和贡献材料使用同一证据映射；正文展示合理AI辅助，附录二披露整个示范的实际AI制作范围和角色经历、示例工时及确认的模拟性质，贡献材料关联该说明。')
 s=s.replace('[阶段三清单](docs/STAGE3.md)','[阶段三核查](docs/阶段三核查.md)、[下一阶段清单](docs/STAGE4.md)')
 s=s.replace('当前不会启动全量训练、最终评估或付费API。','该历史入口不执行阶段三全量开发训练、最终评估或付费API。')
 marker='- scripts/record_operation.py：保存原始控制台、退出码和单调时钟耗时。'
 if 'scripts/train_development.py：' not in s:
  s=s.replace(marker,marker+'''
- scripts/diagnose_n1.py：独立标签回映、损失/梯度及8条训练诊断。
- scripts/train_development.py、stage3_queue.py：固定协议开发训练、早停、学习率与RoPE单卡队列。
- scripts/stage3_common.py、development_baselines.py：辅助指标、BIO覆盖上限及完整训练词典边界实验。
- scripts/api_development.py、api_development_grounded.py：原位置协议与精确原文定位协议；会真实调用，API默认关闭。
- scripts/audit_api_development.py、verify_stage3.py：免费核算已有输出，依赖制作期本机预算材料。
- scripts/development_structure.py、development_error_candidates.py：已有预测的结构分层与待解释错误候选。
- scripts/inspect_development_checkpoints.py：远端CPU检查best/last与优化器，无训练/评估。
- scripts/summarize_stage3_runtime.py、write_stage3_report.py：分段耗时、原始日志摘录与阶段交接材料。''')
 s+='''
阶段三完整免费核查（制作期材料齐备时）与真实日志入口：

```sh
uv run --locked python scripts/verify_stage3.py
```

见[阶段三核查](docs/阶段三核查.md)、[逐模型原始日志摘录](evidence/stage3/training-console-excerpts.log)、[分段耗时](evidence/stage3/runtime-summary.json)。此命令不训练、不访问最终留出正文、不调用API。阶段四仍需后续指示。
'''
 p.write_text(s,encoding='utf-8')
 p=ROOT/'docs/QHA_REGISTRY.md';s=p.read_text().replace('## 阶段三中期（北京时间2026-10-20）','## 阶段三中期历史（北京时间2026-10-20）').replace('## 用户暂停覆盖（2026-10-20）','## 用户主动暂停历史（2026-10-20；后续已获明确恢复）').replace('等待用户明早明确恢复指令，未授权阶段四或最终留出评估。','该暂停随后由用户2026-10-20明确恢复阶段三的指令解除；阶段四与最终留出评估仍未授权。').replace('当前排队不占卡，不换GPU。','该次恢复排队不占卡、不换GPU，随后再次安全中断；不是当前状态。').replace('最新只读探测：','阶段一历史只读探测：').replace('本阶段未测训练速度','阶段一未测训练速度').replace('阶段二结束；当前没有本任务训练进程，不自动进入阶段三。','阶段二结束时没有本任务训练进程；后续阶段三另获授权，见下方。')
 s+='\n## 阶段三直接验证\n\n- 用户明确授权GPU7单卡开发训练；合同configs/qha-stage3-contract.json，启动前观测、占用检查及PID见evidence/stage3/queue-develop.json。\n- N1训练诊断与五个开发任务顺序完成，训练使用FP32、微批量4、累积8，有效批量32。\n- 原始外部占用中断、用户主动暂停与全部分段恢复生命周期均保留；不把整夜主动暂停归为GPU计算或外部排队。最终统计共'+str(runtime['interrupted_training_segments'])+'次中断、'+str(runtime['discarded_partial_updates'])+'次回退更新、'+f"{runtime['discarded_partial_update_seconds']:.3f}"+'秒回退更新计算。\n- 开发子进程墙钟累计'+f"{runtime['gpu_development_child_wall_seconds']:.3f}"+'秒，另有诊断子进程'+f"{runtime['gpu_diagnostic_child_wall_seconds']:.3f}"+'秒；训练/验证/保存分段和吞吐见runtime-summary.json。\n- 本轮未读取最终留出正文选参或评估。全部完整训练检查点留在本项目远端，远端CPU加载核查通过，未复制优化器到Mac。\n- 结束资源观测见evidence/stage3/post-run-gpu.json；GPU使用权和空闲状态仍以每次新启动前观测为准。不操作外部进程或实例生命周期。\n'
 p.write_text(s,encoding='utf-8')
 print(json.dumps({'status':'stage3_entries_updated','stage4_executed':False}))

if __name__=='__main__':main()
