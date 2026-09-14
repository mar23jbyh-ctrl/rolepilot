# 运行指南

本指南面向第一次获取仓库的开发者。完整配置示例见 [`.env.example`](../.env.example)，产品入口见 [`README.md`](../README.md)。

## 环境要求

- Python 3.11；Node.js 20 或更高版本。
- 一个兼容 Chat Completions、支持工具调用且能遵循 JSON 输出要求的模型接口。`BASE_URL` 可指向不同供应商；结构化结果采用提示词约束、JSON 解析和程序校验，未启用原生 `json_schema` 响应模式。不同原生协议需要适配层。
- Tavily API Key。没有搜索密钥或搜索失败时，流程会按代码标记降级，不会把空结果伪装成联网结果。
- 只有上传图片或扫描 PDF 才需要 Tesseract、`chi_sim` 和 `eng`；TXT、DOCX 和文字 PDF 不依赖 OCR。

## Windows PowerShell

在仓库根目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
npm ci --prefix frontend

if (-not (Test-Path -LiteralPath .env)) {
  Copy-Item -LiteralPath .env.example -Destination .env
}
```

编辑 `.env` 至少填写：

```dotenv
MODEL=你的模型ID
BASE_URL=https://你的模型服务/v1
API_KEY=你的模型服务密钥
TAVILY_API_KEY=你的Tavily密钥
```

分词器准备命令首次需要下载公开静态词表，后续使用本机缓存；它不访问模型服务，也不产生模型费用。若测试提示外部请求被阻止，先在测试之外执行该命令。`.env` 未配置时仍可启动健康检查和文字上传；生成面试题需要有效模型配置。

启动开发模式：

```powershell
# 终端一：后端
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# 终端二：前端
npm run dev --prefix frontend
```

浏览器访问 <http://127.0.0.1:5173>。Vite 会把 `/api` 请求代理到 8000 端口。

## 单地址模式

构建前端后，后端会在 `frontend/dist/index.html` 存在时托管 SPA：

```powershell
npm run build --prefix frontend
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

访问 <http://127.0.0.1:8000>；健康检查为 <http://127.0.0.1:8000/api/health>，FastAPI 文档为 <http://127.0.0.1:8000/docs>，需要匿名 Bearer 凭据的自检接口为 `/api/self-check`。

Linux/macOS 将 `.\.venv\Scripts\python.exe` 替换为 `.venv/bin/python`，将 `npm ... --prefix frontend` 换成等价的 npm 命令即可。当前安装示例以 Windows 开发环境为主要验证环境，其他平台需要根据本机 Python、Node.js 和 Tesseract 配置调整。

## OCR（可选）

安装 Tesseract 后，在 Windows 执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/install_ocr_languages.py
```

确认命令可执行且 `chi_sim`、`eng` 语言数据可用，再上传图片或扫描 PDF。OCR 结果会先回填到可编辑文本框，由用户确认后才进入面试材料。

## 验证与开发

安装开发依赖并运行公开回归：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
npm test --prefix frontend
npm run build --prefix frontend
```

详见 [`docs/evaluation.md`](evaluation.md)。测试默认使用合成数据和模型/搜索桩，不需要真实 API 密钥，也不应产生云端费用。

## 常见问题

| 现象 | 排查 |
|---|---|
| `401/403` 或看不到历史会话 | 检查浏览器是否携带当前匿名 Bearer；不要混用 `localhost` 与 `127.0.0.1`、端口或浏览器站点数据。 |
| 上传图片/扫描 PDF 返回 `503` | 运行 OCR 自检，确认 Tesseract、`chi_sim`、`eng` 和 PDF 渲染依赖已安装。 |
| 回答返回 `409 stale_question_version` | 页面持有旧题目；刷新会话并使用当前题目版本，避免重复提交旧题。 |
| 回答返回 `409 session_busy` | 同一会话已有写操作；等待当前请求完成后再提交。 |
| 返回 `recovery_required` | 上一次请求的完成状态无法确认；使用“恢复原回答”查询同一请求。仅在检查点证明已到安全边界时补写业务结果；无法确认时，可保留故障会话后开始新面试。 |
| 搜索降级或报告缺少联网来源 | 检查 `TAVILY_API_KEY`、网络和供应商配额；系统会保留降级状态。 |
| 上下文/整场额度不足 | 调低题量或提高经确认的预算；预算不足时系统会保留已提交回答并安全收尾。 |

## 数据位置与清理

运行期数据默认在 `data/`：业务会话库、LangGraph 检查点库、匿名身份库、上传临时目录和 OCR 数据都在此生成。它们含有用户输入或访问凭据，已被 `.gitignore` 排除，不要提交到 GitHub。删除单个会话应使用界面或对应 API；删除整个本地环境前先停止后端，再备份需要保留的数据库。
