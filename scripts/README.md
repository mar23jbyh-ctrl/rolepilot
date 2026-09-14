# 开发验证工具

这些脚本用于依赖准备、集成验证和评测复现，日常使用只需要启动前后端。运行前可使用 `--help` 查看参数；验证生成的输出目录默认位于被 Git 忽略的 `docs/release/`，不要上传含输入内容的原始结果。

| 工具 | 用途与调用边界 |
|---|---|
| `install_ocr_languages.py` | 为自定义目录准备语言数据，优先复用随源码提供的官方中文模型、保留现有英文模型；缺少中文源时才从固定官方地址下载，不调用模型。默认用户无需运行。 |
| `ocr_smoke.py` | 对已运行服务上传合成中英文 JD 图片、简历图片和扫描 PDF，验证真实 OCR 与 HTTP；需要中文字体，不调用模型或搜索。会创建一份匿名身份，不创建面试会话。 |
| `delivery_smoke.py` | 用隔离临时数据库、合成材料和模型/搜索桩验证本机 TCP、失效凭据后上传恢复、身份复用、完整图、重放、重启与清理；无云端调用。 |
| `delivery_preflight.py` | 核对依赖、Git 文件与发布候选，随源码提供的大型 OCR 模型按固定官方 SHA256 校验；未知大文件仍标记为未扫描。用于开发者的发布检查。 |
| `ocr_benchmark.py` | 固定合成 OCR 样本与指标；依赖本机 OCR/字体，仅验证固定样本。 |
| `guard_benchmark.py` | 复现固定合成岗位/题目守卫案例；只运行确定性规则，不调用模型，需模型仲裁的案例单独计为未验证。 |
| `release_run.py` | 可选的真实供应商链路验证，合成材料位于 `tests/fixtures/release/`；先检查配置和预览，显式执行会产生费用。 |
| `release_score_replay.py` | 对显式指定的本地会话重评失败的动机题；只读原库，需 `--target OWNER_ID:SESSION_ID`，真实执行会产生模型费用。 |
| `release_server.py` | 验证工具使用的隔离服务入口。 |
| `acceptance_finish.py` | 生成源码哈希与文档链接检查报告，无需历史验收记录，不读取运行期数据库或密钥。 |
| `acceptance_cleanup.py` | 只读文件分类清单，区分源码、缓存和本地数据，不执行删除或移动。 |

无需真实密钥的基本验证：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
.\.venv\Scripts\python.exe -X utf8 scripts/guard_benchmark.py
npm test --prefix frontend
npm run build --prefix frontend
.\.venv\Scripts\python.exe -X utf8 scripts/delivery_smoke.py --execute --output docs/release/evidence/my-smoke
.\.venv\Scripts\python.exe -X utf8 scripts/acceptance_finish.py --output docs/release/evidence/my-source-validation.json
```

可选真实供应商验证先预览输入与参数：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/release_server.py --run-id my-provider-check
.\.venv\Scripts\python.exe -X utf8 scripts/release_run.py --run-id my-provider-check --base-url http://127.0.0.1:8001 --answer-mode templates
```

配置有效模型与 Tavily 密钥后，在两个终端分别给上述命令追加 `--execute`，先启动服务，再运行驱动。`templates` 使用固定合成回答；`llm` 会额外调用模型模拟回答。两种模式都会产生面试业务模型和搜索调用。每次使用新的 `run-id`，原始结果仅保存在本地，不应作为模型准确率或生产稳定性的证明。

详情见 [运行指南](../docs/run.md) 与 [评测说明](../docs/evaluation.md)。
