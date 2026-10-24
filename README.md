# 中文医学实体抽取

比较训练词典 D0、字符 BiLSTM＋GlobalPointer N0、中文 BERT＋GlobalPointer N1，以及零样本／三样本 API 抽取 L0/L3。程序返回九类实体、原文和 Unicode 右开字符区间，支持嵌套实体、方法对比、JSON 导出和固定权重 CPU 复现。

## 环境与独立资源

使用 Python 3.11 和 uv。依赖由 [pyproject.toml](pyproject.toml) 与 [uv.lock](uv.lock) 固定；Node.js 仅用于可选的前端离线检查。源码仓库为公开的 [Harry6400/cmeee-ner-2026](https://github.com/Harry6400/cmeee-ner-2026)。

数据和模型权重通过独立资源包提供。取得对应资源后，先按 [教师资源清单](configs/teacher-assets.json) 或 [Windows 资产清单](configs/windows-assets.json) 核对文件名、大小和 SHA-256，再将资源包解压到源码根目录：

```text
项目根目录/
├── assets/CMeEE-V2.zip
├── models/trained/manifest.json
├── models/trained/N0-seed20261019.pt
├── models/trained/N1-seed20261019.pt
├── pyproject.toml
└── uv.lock
```

`assets/CMeEE-V2.zip`保持ZIP格式。使用合并资源包时，保留同包提供的权重、模型卡、数据来源和许可说明。从课程资源提供者领取正式资源包，按仓库清单核对文件及版本。

原始数据由吴皓睿申请获批后从 [CBLUE 官方数据页](https://tianchi.aliyun.com/dataset/95414)取得，现有传递范围限本项目教师与组员的非商业课程复现。也可按官方渠道及适用条件自行取得数据，并在资源记录中注明来源。详见 [数据许可](docs/DATA_LICENSE.md) 和 [第三方声明](THIRD_PARTY_NOTICES.md)。

## 安装、检查与启动

在源码根目录执行：

```sh
uv sync --locked
uv run --locked python scripts/prepare_data.py --archive assets/CMeEE-V2.zip --output data
uv run --locked python scripts/prepare_data.py --output data --verify
uv run --locked python scripts/reproduce.py check
uv run --locked python scripts/serve_stage5.py --port 8765
```

首次依赖安装需要网络或已准备的包缓存。数据已经准备时执行 `--verify`，核对现有数据与冻结清单。资产缺失或哈希不符时，根据错误提示检查文件、路径及资源版本。

浏览器打开 <http://127.0.0.1:8765/>。终端保持运行，按 Ctrl+C 停止服务。端口被占用时，通过 `--port` 选择其他空闲端口。正式本地模型使用资源包中的权重在CPU上推理。

## 页面操作

1. **实时抽取**：输入一段文本，选择 D0、N0 或 N1 并运行。点击实体查看原文位置及类别；新输入展示抽取结果及本次耗时。
2. **方法对比**：使用同一文本运行所选路线，查看完整结果、共同项与分歧。失败、未检出和未运行分别显示。运行中改变 API 配置会停止尚未发送的路线，已发送请求的真实返回结果仍保留。
3. **错误分析**：查看 F01—F05 的参考、历史预测和独立诊断结果。这些是对应版本的已保存案例，来源随结果显示。
4. **实验结果**：查看共同500、完整5000、多种子、学习率／RoPE和阈值结果。两张 SVG 随源码提供，无需运行绘图脚本。
5. **JSON 导出**：展开或下载结果，核对原文、路线、实体区间、来源与状态。文本或方法改变后重新运行；API 导出记录实际用量和可用的参考估费，密钥保留在页面内存。
6. **复现验证**：启动新的完整任务，查看阶段、进度、日志和输出目录。刷新可恢复已保存任务；取消保留部分输出。历史任务按原时间展示，本次结论以新任务输出为准。

页面根据本地文件显示可用的报告、摘要和贡献声明。对应交付文件可在本机另行放置。

## API 设置

在页面右上角打开“API 设置”，选择提供商或自定义兼容接口，填写自己的密钥和模型，保存后再运行 API 路线。需要时使用页面的模型获取功能核对可选模型；获取模型列表及发送抽取请求都会访问所选服务商。

密钥仅保存在当前页面内存，服务商、模型及自定义地址偏好可以保存。选择L0/L3并点击运行后，程序发送对应API请求。API输入使用操作者获准外发的公开或合成文本。

现场入口记录实际用量和参考费用，账户额度由操作者在服务商平台管理。参考费用按价格目录与折算汇率计算，实际支付以服务商账单为准；用量或价格缺失时记为未知。配置变化后的对比中断会写明状态、未运行路线及原因。

## 固定权重 CPU 复现

资源就位、依赖安装完成后，使用一个尚不存在的新输出目录：

```sh
uv run --locked python scripts/teacher_reproduce.py --archive assets/CMeEE-V2.zip --output output/teacher-001
```

该入口准备或验证固定划分，再新运行 D0/N0/N1 各5000条，核对 [冻结期望](configs/reproduction-expected.json)。主种子为 `20261019`，严格实体 Micro F1 预期为 D0 **38.0192%**、N0 **60.2793%**、N1 **66.8942%**；精确整数和小数以配置为准。以上为固定权重在既定5000条数据上的复现目标；新输入按当前抽取结果展示。

检查新输出中的 `summary.json` 和退出码。每次使用 `teacher-001`、`teacher-002` 等新输出目录。失败时保留输出和错误，按提示检查资源；复现使用正式权重、固定阈值和冻结期望。

同一服务只运行一个完整复现任务，任务运行时可能拒绝其他冲突请求。任务异常退出后，先检查原任务和进程状态，再处理遗留保护锁。

## 文档与运行说明

- [参数、指标与运行限制](docs/实验技术补充.md)
- [数据来源与许可](docs/DATA_LICENSE.md)
- [第三方资源声明](THIRD_PARTY_NOTICES.md)及 [MIT 许可证](LICENSE)

准备好独立资源并安装锁定依赖后，可按上文命令启动页面，或执行 D0/N0/N1 的固定权重 CPU 复现。复现结果、汇总指标与运行日志保存到指定的新输出目录。

仓库提供训练、断点恢复、历史评估和报告构建脚本。使用时按各入口准备对应的数据、模型、实验记录与配置。
