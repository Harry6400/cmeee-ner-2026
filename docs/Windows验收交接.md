# Windows CPU 验收流程

本手册说明如何在已确认的 Windows CPU 验收任务中准备干净源码、运行检查并保留真实记录。开始前明确待验收提交、操作者、浏览器、独立资产和执行范围；本文不把旧合同或旧验证记录延续为新任务授权。

已有记录覆盖开发源码 `3d7da369` 的 Windows 11＋Chrome，接收方于2026-10-22完成独立核查。Edge 未测，真实 API 和后续版本也不在该证明范围。以下浏览器检查 ID 保留采集器原有的 `edge-*` 命名，实际使用的浏览器及版本仍须如实记录。

## 1. 环境、边界与源码绑定

需要 Git、uv、64位 Python 3.11、Node.js 和约定的浏览器。工具从官方渠道取得。检查依赖使用现有 `uv.lock`，安装失败时不重锁依赖；已有工作区、准备目录和输出不覆盖。

本流程只安排本地 CPU 资产检查、离线测试、浏览器操作与固定权重复现。真实 API、训练、GPU、冻结参数／权重变更、发布或合并 PR 均不包含在该流程中。操作者应分别记录本人实际操作与 Codex 执行范围，不冒用他人身份或签署。

以下命令为 PowerShell 示例。在已有仓库克隆的根目录执行；仓库应为私有 `Harry6400/cmeee-ner-2026`。每条外部命令的退出码非0时，先保存输出并排查，不继续假定成功。

```powershell
$Repo = (Get-Location).Path
git remote -v
gh repo view Harry6400/cmeee-ner-2026 --json nameWithOwner,isPrivate,defaultBranchRef
git status --short
git fetch origin
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
git switch -c "codex/windows-acceptance-$Stamp" origin/main
$Commit = (git rev-parse HEAD).Trim()
$Work = Join-Path $env:USERPROFILE "CMeEE Windows验收 $Stamp"
New-Item -ItemType Directory -Path $Work
$Archive = Join-Path $Work '源码.zip'
$Clean = Join-Path $Work '中文 路径源码'
$Session = Join-Path $Work '首轮验收证据'
git archive --format=zip --output="$Archive" $Commit
Expand-Archive -LiteralPath $Archive -DestinationPath $Clean
Set-Location $Clean
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
py -3.11 -X utf8 scripts/windows_acceptance.py init --output "$Session" --source-repo "$Repo" --archive "$Archive" --commit "$Commit" --operator '填写真实操作者及执行范围'
py -3.11 -X utf8 scripts/windows_acceptance.py run --session "$Session" --check install
```

采集器记录提交、分支、初始工作区、源码 ZIP 哈希和解压文件来源，同时采集环境与工具信息。证据目录放在源码目录外；每次修复后的最终验收使用新导出目录和新证据会话。remote 地址不能含访问令牌，公开说明中不粘贴私人路径或完整会话文件。

## 2. 独立资产与离线检查

按 [Windows 资产清单](../configs/windows-assets.json) 核对收到的合并资产包；也可使用 [教师资源清单](../configs/teacher-assets.json) 中的对应资源。核对 SHA-256 后，将整个包解压到 `$Clean`，形成 `assets/CMeEE-V2.zip` 和 `models/trained/`。不要额外解压内部数据 ZIP，不叠加不同旧资源包。

```powershell
# $AssetsArchive 替换为实际收到的、已核对清单的资源包路径
Get-FileHash -Algorithm SHA256 -LiteralPath $AssetsArchive
Expand-Archive -LiteralPath $AssetsArchive -DestinationPath $Clean
py -3.11 -X utf8 scripts/windows_acceptance.py run --session "$Session" --check prepare
py -3.11 -X utf8 scripts/windows_acceptance.py assets --session "$Session"
py -3.11 -X utf8 scripts/windows_acceptance.py run --session "$Session" --check python-tests
py -3.11 -X utf8 scripts/windows_acceptance.py run --session "$Session" --check frontend-tests
```

未收到资源时，按 [数据许可](DATA_LICENSE.md) 和提供者实际渠道取得正式数据与权重。收到同组私下传递的未改原包时，如实记录原始申请者和传递方式，无需冒称本人重新下载；保留来源、模型卡与许可。没有符合条件的资源时记录 BLOCKED，不训练新权重替代。

资产检查核对原数据 ZIP、冻结划分、正式 manifest、N0/N1 权重及四个页面资源。缺失为 BLOCKED，哈希不符为 FAIL，不修改清单绕过。页面 JSON 和两张 SVG 随源码提供，不运行 `build_teacher_view.py` 或绘图工具补齐缺项。

采集器保存完整日志和退出码。某项失败时可继续不依赖它的检查，但不能隐藏失败。测试中的模拟 API 不代表真实服务联通。

## 3. 浏览器实操与任务检查

在另一个 PowerShell 窗口进入 `$Clean`，启动服务并保留终端：

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
uv run --locked python -X utf8 scripts/serve_stage5.py --port 8765
```

使用约定浏览器打开 `http://127.0.0.1:8765/`。端口被占用时先辨明进程或改用空闲端口，不结束其他项目。不能操作真实浏览器时记录 BLOCKED；HTTP 调用和模拟截图不能代替下载、点击或任务管理检查。

| 检查 ID | 操作与依据 |
|---|---|
| `edge-extraction` | 用合成短文“出现发热与咳嗽，使用布洛芬。”运行 N1，核对中文类型、Unicode `[start,end)`、嵌套和 emoji。模型错误保留，不改标注或模型。 |
| `edge-comparison` | 同文运行 D0/N0/N1，核对共同项和分歧；不勾选 API，不为无参考的新文本显示准确率。 |
| `edge-errors` | 加载 F01—F05，核对参考、历史预测和路线来源；切换时旧内容与导出失效。 |
| `edge-experiments` | 检查共同500／完整5000、多种子、学习率／RoPE、曲线和阈值区；两张 SVG 无缺失，API完整主F1保留 N/A。 |
| `json-readback` | 真实下载合成短文 JSON，再从磁盘打开，核对原文、方法、实体、状态与来源，保留文件和读回记录。 |
| `stale-result` | 改文本或路线后旧结果失效、导出禁用；重新运行后可导出；空输入有提示。 |
| `jobs-concurrency` | 完整任务运行时再次启动冲突任务或现场 CPU 抽取，核对拒绝并发响应及日志。 |
| `jobs-cancel` | 使用独立取消任务，核对仅终止本任务进程并保留部分输出。 |
| `jobs-refresh` | 任务运行中刷新页面，能恢复同一任务的状态与日志，不重复启动。 |
| `jobs-interruption` | 使用独立中断任务保留退出、进程与保护锁依据；恢复时不覆盖旧输出、不干预无关进程。 |
| `jobs-complete` | 新完整任务正常结束，三路线输出齐全，页面、摘要及实际输出目录相符。 |

每项分别记录真实状态、观察和证据。下面的 PASS 只说明示例参数用法，不能直接照抄为通过：

```powershell
py -3.11 -X utf8 scripts/windows_acceptance.py record --session "$Session" --check json-readback --status PASS --note '填写实际浏览器版本、文件名、读回字段及观察结果' --evidence 'C:\实际路径\导出截图.png' 'C:\实际路径\合成结果.json' 'C:\实际路径\读回核对.txt'
py -3.11 -X utf8 scripts/windows_acceptance.py record --session "$Session" --check jobs-interruption --status BLOCKED --note '填写真实阻断原因和已保留的现场'
```

采集器不识别截图真假，也不替代接收方审阅。证据使用合成或已允许使用的公开短文，避开密钥、浏览器个人页面和私人目录；不将完整预测改名混入证据。

## 4. CPU复现、独立核算与归档

通过页面完成新的5000条复现后，将实际任务输出目录传给核算器：

```powershell
$Run = Join-Path $Clean 'output\teacher-reproduction\实际完成的job-ID'
py -3.11 -X utf8 scripts/windows_acceptance.py cpu --session "$Session" --output "$Run"
py -3.11 -X utf8 scripts/windows_acceptance.py finalize --session "$Session"
```

核算器独立核对金标哈希、5000个唯一ID、原文／顺序、实体边界与类别，重新计算实体集合计数及 P/R/F1。冻结期望见 [reproduction-expected.json](../configs/reproduction-expected.json)，显示F1为 D0 38.0192%、N0 60.2793%、N1 66.8942%。差异应保留为 FAIL，不改阈值、依赖锁或期望值。

`finalize` 重新核对源码绑定，生成证据 ZIP 和包外哈希；未执行、失败或阻断项不能据此变成通过。一旦归档，不继续写入该会话；新尝试使用新目录。证据包不含数据、权重或逐条预测，完整预测留在验收电脑。

## 5. 失败处理与回传

保留首轮失败、截图、输出和会话。缺工具或资产记 BLOCKED，行为或指标不符记 FAIL，未执行记 NOT_RUN。修复在验收分支进行，不在干净导出目录现场改源码；仅修改已确认的 Windows 阻断问题及必要回归检查。每次最终源码变动后重新导出，以新目录完成复验，不复用旧版本的 PASS。

交回实际修复分支／PR、仓库外的脱敏验收说明、独立证据包及 SHA-256、未完成清单。验收说明写明源码 SHA、环境和实际浏览器、每项检查状态、CPU核算结果、首次失败及修复、操作者与 Codex 的各自执行范围。报告、声明、视频和资产包通过约定的独立渠道提供，不因页面出现材料入口就假定已经齐备，也不自动加入源码仓库。

推送分支、创建 PR 或合并等操作按实际任务授权执行；不自动合并或更新验收结论。接收方核对源码／证据绑定、浏览器 JSON 读回、任务进程归属及三路线独立计数后，才能确认对应版本通过。API联通、视频与正式课程提交分别按真实状态处理。
