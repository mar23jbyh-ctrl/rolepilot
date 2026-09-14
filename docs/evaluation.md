# 评测与证据

RolePilot 的验证重点是 Agent 工作流在真实交互中的可靠性：状态能否恢复、回答能否幂等提交、旧题能否被拒绝、评分能否按固定规则聚合，以及前后端能否完成完整流程。测试结果用于说明当前工程行为，不等同于模型质量、招聘效度或生产环境 SLO。

## 当前回归基线（2026-09-14）

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
npm test --prefix frontend
npm run build --prefix frontend
```

使用 Python 3.11、锁定依赖和合成数据执行公开回归：

| 验证层 | 结果 | 覆盖范围 |
|---|---:|---|
| 公开源 Python 回归 | **684 passed、7 skipped**，0 failed/error，1 条外部弃用警告 | 状态图、评分协议、SQLite 账本、上传边界、匿名身份、OCR 安装路径与公共模型校验、隐私及错误路径 |
| 前端协议与组件测试 | **66 passed** | 请求字段、会话状态、重复提交、环境隔离、失效凭据恢复和主要界面组件 |
| TypeScript/Vite 构建 | **成功** | 类型检查和生产构建 |

测试使用合成简历、JD、回答以及模型/搜索桩。它们验证程序契约和异常处理，不代表真实模型的事实性、面试评分效度或所有简历版式的 OCR 准确率。

公开源不附带运行期会话库，因此 7 项可选历史库回放按设计跳过；核心协议测试使用临时 SQLite 和合成数据，不依赖这些历史记录。唯一警告来自 Starlette/AnyIO 的弃用别名。测试数量不是代码覆盖率或模型准确率。

首次安装需在测试之外准备 `o200k_base` 分词器缓存，之后回归会阻止外部网络请求，具体命令见 [运行指南](run.md)。[GitHub Actions](https://github.com/mar23jbyh-ctrl/rolepilot/actions/workflows/ci.yml) 在 Windows runner 上执行同一套后端回归、前端测试和构建，使用 Python 3.11 与 Node.js 22；最新状态和执行日志可在该页面查看。

另一个 Linux CI 任务实际构建并启动 Docker 交付，通过 `ocr_smoke.py` 上传真实合成中英文图片与扫描 PDF，再保留原数据卷重启并重复检查。这一任务调用真实 Tesseract/PDFium，不替换 OCR 为桩，也不访问云端模型。执行结果以同一 Actions 页面的 `container-ocr` 日志为准。

## 会话可靠性

[`tests/test_release_protocol.py`](../tests/test_release_protocol.py) 对服务层、LangGraph 检查点和 SQLite 事务进行了隔离验证，覆盖：

- 同一 `answer_request_id` 的重复提交只产生一次评估，后续请求返回已保存结果；
- 20 个不同请求 ID 并发竞争同一题时，只有一个请求推进题目版本；
- 同一请求 ID 携带不同正文会触发幂等冲突，旧 `question_version` 会被拒绝；
- 服务重启后可以重放已完成请求；业务投影写入失败时可以在安全边界补写，而不会再次运行评估节点；
- 多进程 SQLite 抢占测试中，4 个进程只有一个获得写入处理权。

这些结果说明本地业务状态具备幂等、版本保护和恢复机制。业务 SQLite 与 LangGraph SQLite 仍是两个独立存储，外部模型请求也不属于本地事务；相关边界见 [`架构说明`](architecture.md)。

## 匿名身份与上传恢复

[`tests/test_release_auth.py`](../tests/test_release_auth.py) 验证旧身份库升级后原凭据与 owner 保持不变、环境标识重启稳定、并发打开同库获得同一标识、有效凭据复用，以及数据库故障不会被误判为身份失效。

[`frontend/tests/anonymous-recovery.test.cjs`](../frontend/tests/anonymous-recovery.test.cjs) 运行实际 TypeScript 客户端与存储模块，使用合成传输验证失效或畸形缓存下的简历上传、有效旧记录迁移、不同数据环境切换与切回、并发初始化/恢复、有限重试和旧会话请求保护。[`frontend/tests/release-protocol.test.cjs`](../frontend/tests/release-protocol.test.cjs) 另验证身份准备完成前不读取恢复记录，环境切换保留编辑材料，以及迟到回答不能覆盖新环境或清除其待确认记录。这些是离线协议测试，不是浏览器端到端测试。

## HTTP 与浏览器流程

除离线回归外，2026-09-14 使用本机 Uvicorn/TCP、完整状态图和 SQLite 验证了端到端流程：失效 Bearer 上传被拒绝、身份重新建立后文字上传成功、有效身份复用、分析、岗位调研、出题、评估前暂停、回答、追问、下一题、重复提交、旧版本拒绝、服务重启恢复、报告读写和删除。重启后数据环境标识保持不变。

该轮使用隔离的模型桩和搜索桩，记录了 14 次 SDK 桩调用和 5 次搜索桩调用；这些调用没有访问真实云模型或真实 Tavily。

可复现命令（输出目录需尚不存在）：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/delivery_smoke.py --execute --output docs/release/evidence/my-smoke
```

此前的人工界面验收在 Edge 浏览器中使用两份合成会话，覆盖 OCR 上传与编辑、开始/暂停、回答和追问、同 ID 重试、旧版本 409、刷新恢复、历史、报告、删除和窄屏布局。该轮使用 28 次 SDK 桩调用和 10 次搜索桩调用，未访问真实云模型或 Tavily；它是一次人工流程检查，不属于每次 CI 自动执行的浏览器测试。本次身份恢复变更的证据为上述离线协议与实际 TCP 验证，不将此前的界面验收作为该变更的浏览器上传证明。

这些流程验证的是产品接缝和协议行为，不是模型回答质量。为保护输入内容和本地环境信息，仓库保留可运行的测试代码，不附带原始运行日志。

## 评分与 OCR

- 评分协议测试覆盖字段类型、量规键、档位范围、回答原话证据、追问合并和程序聚合。固定协议回放覆盖 16 个案例、48 项检查，结果为 48/48；案例没有专家金标准，因此不能推出与专家或招聘结果的一致性。
- OCR 测试覆盖语言包缺失、超时、失败、文件类型、页数和大小边界；独立 benchmark 使用固定开发样本，只能说明该样本上的行为，不能代表所有中文简历、复杂表格和扫描质量。
- 评分和 OCR 的实现入口分别见 [`docs/scoring_contract.md`](scoring_contract.md)、[`tests/test_scoring_contract.py`](../tests/test_scoring_contract.py)、[`tests/test_release_d12_assessment.py`](../tests/test_release_d12_assessment.py)、[`tests/test_ocr.py`](../tests/test_ocr.py) 和 [`scripts/ocr_benchmark.py`](../scripts/ocr_benchmark.py)。

### 公共模型与真实上传检查

[`assets/ocr/models.json`](../assets/ocr/models.json) 固定了官方 `tessdata_best` 来源提交、文件大小与 SHA256；[`tests/test_ocr_installation.py`](../tests/test_ocr_installation.py) 检查随源码提供的两份模型、安装与运行时路径一致性、无下载复用、校验失败拒绝及现有模型保护。

[`scripts/delivery_preflight.py`](../scripts/delivery_preflight.py) 对这两份大型模型执行固定官方 SHA256 校验，包括 Git 历史中的模型 blob；文件名相同但校验不同会被拒绝，未知大型文件仍保留“未扫描”状态。相关边界由 [`tests/test_delivery_preflight.py`](../tests/test_delivery_preflight.py) 验证。

2026-09-14 在 Windows 的隔离临时数据环境中，使用真实 Tesseract、随源码提供的模型与 PDFium，通过本机 Uvicorn/HTTP 检查：

| 输入 | HTTP 结果 | 检查 |
|---|---:|---|
| 合成中英文 JD PNG | 200 | OCR 方法，包含“岗位”、`Python`、`SQL` |
| 合成中英文简历 PNG | 200 | OCR 方法，包含相同关键词 |
| 合成扫描简历 PDF | 200 | PDFium 渲染后 OCR，包含相同关键词 |

三个请求完成后上传原件均已清理，模型与搜索调用为 0。使用空临时 `DATA_DIR` 且不设置 OCR 目录，验证的是下载源码后自动选择公共模型的路径，不依赖开发目录中的历史语言包。

可在已启动的 Docker 服务中复现：

```shell
docker compose exec -T rolepilot python scripts/ocr_smoke.py
```

手动模式使用 `.venv` 的 Python 执行相同脚本。该检查使用清晰、单栏、固定文字样本；扫描 PDF 以 300 DPI、高质量 JPEG 嵌入生成，避免将字体相关的有损压缩伪影作为依赖检查条件。它证明依赖准备与上传链路能工作，不构成复杂版式、低清扫描或所有用户文档的准确率指标。

## 题目守卫

[`scripts/guard_benchmark.py`](../scripts/guard_benchmark.py) 使用 23 类岗位的 103 个固定合成案例，覆盖同岗位题、跨岗位题、来源约束和易混淆主题。离线结果中 77 个确定性案例符合预设判定及阶段，26 个需模型仲裁的案例列为未验证；仲裁模型调用为 0。这是固定规则回归，不是跨岗位通用准确率。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/guard_benchmark.py
```

真实链路验证使用的合成简历、回答模板和简要岗位来源信息位于 [`tests/fixtures/release/`](../tests/fixtures/release/)。开发者可用 `release_run.py` 的预览模式检查输入，再自行决定是否执行付费供应商链路；默认回归不运行该链路。

## 真实供应商链路

历史版本曾完成 3 场真实 DeepSeek/Tavily 链路会话，包含 47 道主问题、36 道追问、217 次业务模型调用和 15 次搜索。这些记录说明过真实供应商接入能够完成端到端链路，但属于历史版本和有限样本，不用于代表当前版本的长期稳定性，也不构成模型质量认证。

## 证据能说明什么

当前验证支持以下结论：

1. 状态图可以在评估前暂停，并沿同一会话恢复；
2. 回答提交在重复、过期、并发、重启和部分失败场景下有明确处理路径；
3. 结构化评分经过程序校验和聚合，缺少有效证据的结果不会被补造成分；
4. 本地 HTTP、SQLite 和浏览器可以完成主要用户流程。

当前验证尚未证明：

- 评分与真实面试官或招聘结果的一致性；
- 任一模型或供应商在长期运行、限流、跨区域和计费异常下的稳定性；
- HTTPS、企业认证、多租户、防滥用和生产多 Worker 运维能力；
- 所有语言、复杂版式和大文件输入的 OCR 质量；
- 外部模型调用的 exactly-once、跨数据库原子提交或供应商账单金额。

如需复现当前回归，先按 [`运行指南`](run.md) 安装依赖，再执行本页的三条命令。真实模型和 Tavily 链路需要自行配置密钥并承担供应商费用；默认测试不会访问外部网络。
