# 开发验证工具

这些脚本用于依赖准备、集成验证和评测复现，日常使用只需要启动前后端。运行前可使用 `--help` 查看参数；验证生成的输出目录默认位于被 Git 忽略的 `docs/release/`，不要上传含输入内容的原始结果。

| 工具 | 用途与调用边界 |
|---|---|
| `install_ocr_languages.py` | 从固定官方来源下载中文语言数据，保留本机英文数据；需要网络，不调用模型。 |
| `delivery_smoke.py` | 用隔离临时数据库、合成材料和模型/搜索桩验证本机 TCP、完整图、重放、重启与清理；无云端调用。 |
| `delivery_preflight.py` | 核对依赖、Git 文件与发布候选；用于开发者的发布检查。 |
| `ocr_benchmark.py` | 固定合成 OCR 样本与指标；依赖本机 OCR/字体，仅验证固定样本。 |
| `guard_benchmark.py` | 复现固定合成岗位/题目守卫案例；只运行确定性规则，不调用模型，需模型仲裁的案例单独计为未验证。 |
| `release_run.py` | 可选的真实供应商链路验证，合成材料位于 `tests/fixtures/release/`；先检查配置和预览，显式执行会产生费用。 |
| `release_score_replay.py` | 对显式指定的本地会话重评失败的动机题；只读原库，需 `--target OWNER_ID:SESSION_ID`，真实执行会产生模型费用。 |
| `release_server.py` | 验证工具使用的隔离服务入口。 |
| `acceptance_finish.py` / `acceptance_cleanup.py` | 汇总验证结果、检查清理范围；供验证工具与回归使用。 |

无需真实密钥的基本验证：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
.\.venv\Scripts\python.exe -X utf8 scripts/guard_benchmark.py
npm test --prefix frontend
npm run build --prefix frontend
.\.venv\Scripts\python.exe -X utf8 scripts/delivery_smoke.py --execute --output docs/release/evidence/my-smoke
```

详情见 [运行指南](../docs/run.md) 与 [评测说明](../docs/evaluation.md)。
