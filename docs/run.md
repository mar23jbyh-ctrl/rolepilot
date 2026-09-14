# 运行指南

推荐通过 Docker 运行：中英文 OCR 模型随源码提供，镜像负责准备 Python、前端、Tesseract 引擎和 PDF 渲染依赖。需要修改代码时可选择手动开发模式。完整配置示例见 [`.env.example`](../.env.example)。

## Docker：推荐给首次使用者

### 1. 准备环境与源码

安装并启动 [Docker Desktop](https://docs.docker.com/get-started/get-docker/)，使用 Linux 容器；Linux 也可使用 Docker Engine 与 Compose 插件。检查：

```shell
docker version
docker compose version
```

在 GitHub 下载 ZIP 并解压，进入含 `compose.yaml` 的目录；或执行：

```shell
git clone https://github.com/mar23jbyh-ctrl/rolepilot.git
cd rolepilot
```

### 2. 填写模型与搜索配置

Windows PowerShell：

```powershell
if (-not (Test-Path -LiteralPath .env)) {
  Copy-Item -LiteralPath .env.example -Destination .env
}
```

Linux/macOS 可在文件管理器中复制 `.env.example` 为 `.env`。编辑以下配置：

```dotenv
MODEL=你的模型ID
BASE_URL=https://你的模型服务/v1
API_KEY=你的模型服务密钥
TAVILY_API_KEY=你的Tavily密钥
```

模型服务需要兼容 Chat Completions、工具调用，并能遵循 JSON 输出要求。结构化结果通过提示词、JSON 解析与程序校验实现，未启用原生 `json_schema` 响应模式。配置地址可指向不同供应商；其他原生协议需要适配层。

Compose 从本机 `.env` 读取模型、搜索和已列出的流程/预算配置，在启动时传入容器；不会把 `.env` 复制进镜像。[`compose.yaml`](../compose.yaml) 列出了支持的变量。手动模式的 Windows 路径、OCR 目录和 `DATA_DIR` 不传入容器，容器固定使用内部数据和 OCR 路径。额外配置可在 Compose 的 `environment` 中加入，不要提交实际密钥。

若模型服务运行在宿主机上，容器中的 `localhost` 指向容器自身。Docker Desktop 可使用 `host.docker.internal` 替代宿主机地址；Linux Engine 需按其网络配置设置宿主机访问。远程模型地址无此区别。参见 [Docker Desktop 网络说明](https://docs.docker.com/desktop/features/networking/)。

### 3. 启动并使用

```shell
docker compose up --build -d --wait
```

首次构建需联网下载基础镜像、Python/npm 和系统依赖。中英文模型已包含在源码中；构建脚本从源码复制模型并检查 OCR 与 PDF 渲染依赖，无需再寻找语言包。构建完成后，打开：

- 应用：<http://127.0.0.1:8000>
- 健康检查：<http://127.0.0.1:8000/api/health>
- API 文档：<http://127.0.0.1:8000/docs>

未配置模型密钥时仍可启动、检查健康状态和上传文件；生成面试题需要有效模型配置。Tavily 缺失或失败会被标记为搜索降级。OCR 结果先回填可编辑文本框，确认后才进入面试材料。

查看状态与日志：

```shell
docker compose ps
docker compose logs --tail 100
```

### 4. 验证真实 OCR

服务运行后执行：

```shell
docker compose exec -T rolepilot python scripts/ocr_smoke.py
```

该命令使用合成中英文材料，实际上传 JD 图片、简历图片和扫描 PDF，检查响应 200、OCR 方法和预设关键词。不调用模型或搜索，不创建面试；为访问接口会创建匿名身份。它验证依赖与上传链路，不代表任意文档的识别准确率。

### 5. 更新与停止

Git 获取方式可先执行 `git pull --ff-only`；ZIP 获取方式下载新版源码。保存 `.env` 后，重新运行启动命令：

```shell
docker compose up --build -d --wait
docker compose down
```

第二条命令用于停止服务，可按需单独执行。普通 `down` 保留命名数据卷；不要添加 `-v`，除非明确需要删除整个数据环境。更新时保持同一项目目录、Compose 项目名和浏览器地址，可继续使用原数据卷与浏览器凭据。Docker 的数据卷与手动模式 `data/` 是独立环境，不会自动迁移历史。

## 手动开发模式

### 环境与安装

需要 Python 3.11、Node.js 22 或更高版本（CI 使用 22）。上传图片或扫描 PDF 还需要 Tesseract 引擎；仅使用 TXT、DOCX 和文字 PDF 时不需要引擎。

在仓库根目录执行 Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
npm ci --prefix frontend

if (-not (Test-Path -LiteralPath .env)) {
  Copy-Item -LiteralPath .env.example -Destination .env
}
```

填写前述四项模型/搜索配置。分词器准备命令首次下载公开静态词表，后续使用缓存；它不调用模型服务或产生模型费用。若回归测试提示外部请求被阻止，先在测试之外执行该准备命令。

启动开发模式：

```powershell
# 终端一：后端
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# 终端二：前端
npm run dev --prefix frontend
```

访问 <http://127.0.0.1:5173>；Vite 将 `/api` 代理到 8000。单地址模式先构建前端，然后启动同一后端：

```powershell
npm run build --prefix frontend
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

访问 <http://127.0.0.1:8000>。Linux/macOS 将 Python 路径替换为 `.venv/bin/python`，创建虚拟环境使用 `python3.11 -m venv .venv`；推荐 Docker 以减少系统依赖差异。

### 手动模式的 OCR

按 [Tesseract 官方安装说明](https://tesseract-ocr.github.io/tessdoc/Installation.html) 安装引擎。Windows 安装后打开新终端，检查 `tesseract --version`；如果不在 PATH，在 `.env` 指定实际可执行文件：

```dotenv
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
OCR_TESSDATA_DIR=
```

`OCR_TESSDATA_DIR` 留空即可自动选语言目录，优先级为：

1. `DATA_DIR/ocr/tessdata`，若有本地安装；
2. 随源码提供的 `assets/ocr/tessdata`；
3. Tesseract 系统默认目录，若前两者均不存在。

默认用户不需要运行语言下载脚本。源码中的中文、英文模型与 [校验清单](../assets/ocr/models.json) 对应，不能只下载代码文件而漏掉模型目录。引擎版本与系统安装包可不同，实际以自检为准。

```powershell
.\.venv\Scripts\python.exe -X utf8 -c "from app.parsers.ocr import dependency_status; print(dependency_status())"
```

图片识别要求 `ready: True`；扫描 PDF 还要求 `pdf.ready: True`。默认 PDFium 已在锁定依赖中，不需要另装 Poppler。若显式设置其他语言或模型目录，必须提供相应的可加载语言文件；缺失时明确报错，不自动降低语言要求。

可选：修复旧的本地模型目录，或向自定义 `OCR_TESSDATA_DIR` 安装模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/install_ocr_languages.py
```

目标目录与运行配置一致；脚本优先复用随源码提供的中文模型，并保留现有英文模型；仅没有可用中文源时才下载固定官方模型并校验 SHA256。它不覆盖不同版本的现有中文模型，也不修改系统目录。需要使用源码模型重建目录时，可指定一个新的 `--dest`，安装后将 `OCR_TESSDATA_DIR` 设置为同一路径。

服务启动后可运行真实上传检查：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/ocr_smoke.py --base-url http://127.0.0.1:8000
```

脚本自动寻找 Windows 微软雅黑、容器文泉驿或 macOS 苹方；其他字体可使用 `--font` 指定。

## 验证与开发

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
npm test --prefix frontend
npm run build --prefix frontend
```

测试默认使用合成数据与模型/搜索桩，不需要真实密钥，也不应产生云端费用。范围见 [评测与证据](evaluation.md)。

## 常见问题

更新手动模式源码后重启后端和 Vite；单地址模式先重新构建前端。浏览器用 `Ctrl+F5` 加载新版页面，保留站点数据以保留原访问身份和恢复记录。

| 现象 | 排查 |
|---|---|
| `docker` 或 `docker compose` 不可用 | 安装并启动 Docker，Windows 使用 Linux 容器；确认上述版本命令成功。 |
| 首次构建下载失败 | 检查 Docker Hub、npm、PyPI、Debian 镜像网络，恢复网络后重跑构建；不是上传接口失败。 |
| 8000 端口占用 | 停止原服务；或修改 Compose 的宿主机端口，并固定浏览器访问地址。 |
| 手动 OCR 提示依赖未准备 | 检查引擎路径、自定义模型目录及自检结果；旧 `data/ocr/tessdata` 缺模型时运行安装脚本。推荐 Docker 避免宿主机依赖。 |
| 图片上传成功但文字识别错误 | 修正回填文字；尝试更清晰、单栏、正向图片，或粘贴原文。 |
| 切换项目实例后上传或历史异常 | 确认前后端都是新版，重启服务并 `Ctrl+F5`；无需清除站点数据。 |
| 看不到原历史会话 | 保持原浏览器地址和原数据环境；Docker 卷与手动数据目录独立。清除浏览器数据后无法仅凭会话 ID 找回访问权。 |
| API 工具返回 `401/403` | 调用 `POST /api/auth/anonymous` 获取当前环境 Bearer；有效旧 Bearer 可以复用原身份。 |
| `409 stale_question_version` | 刷新会话，使用当前题目版本，避免继续提交旧题。 |
| `409 session_busy` | 等待同一会话正在执行的写操作完成。 |
| `recovery_required` | 使用“恢复原回答”查询同一请求；无法确认完成边界时保留故障会话，再开始新面试。 |
| 搜索降级 | 检查 Tavily 密钥、网络和额度，系统保留降级状态。 |
| 上下文或整场额度不足 | 根据模型限制调整预算或题量；预算不足时保留回答并安全收尾。 |

## 数据位置与清理

手动模式默认在 `data/` 保存会话、检查点、匿名身份、上传临时文件和可选本地 OCR 数据；Docker 将会话数据保存在命名卷，将公共模型保存在镜像内。运行数据库可能包含用户输入或访问凭据，不能提交 GitHub；公开 `assets/ocr/` 仅为官方公共模型。

删除单个会话使用界面或对应 API。删除整个环境前停止服务，备份需要保留的数据库；删除匿名身份库会同时失去旧凭据的服务端映射。密钥与完整容器环境输出不要上传到 Issue。
