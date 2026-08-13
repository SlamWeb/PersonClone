# PersonaForge

PersonaForge 是一个 local-first 的创作者数字分身平台：抓取公开创作者内容，构建本地
BGE-M3 混合检索索引，再通过 FastAPI + React 提供带引用、记忆、Trace 和评估的对话体验。

```text
公开内容抓取 -> Markdown 语料 -> parent/child 节点 -> Qdrant 混合检索
-> query understanding/transform -> 作者回答生成 -> Trace 与离线评估
```

项目导航见 [navigation.md](navigation.md)，系统约束见 [SPEC.md](SPEC.md)。

## 已有能力

- 知乎公开回答、文章、想法抓取，必要时使用本地登录态回退。
- 每位作者独立目录与 Qdrant collection，Web 端异步执行 `crawl -> build -> index`。
- BGE-M3 dense/sparse、BM25、query transform 与 RRF 检索链路。
- FastAPI + React 单端口 Web、SSE 流式回答、持久会话和用户长期记忆。
- Chat 全链路 Trace、RAG 评估、生成评估和 Study 1 实验管理。
- SQLite 用户隔离、管理员/协作者账号、部署限流和可恢复后台任务。

## 本地首次启动

要求 Python 3.11、Node.js 20。Windows PowerShell：

```powershell
git clone https://github.com/SlamWeb/PersonaForge-.git
cd PersonaForge-
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[all,dev]"
playwright install chromium
Copy-Item .env.example .env
```

在 `.env` 中至少填写：

```dotenv
DEEPSEEK_API_KEY=你的_Key
```

构建前端并启动：

```powershell
cd web
npm ci
npm run build
cd ..
pf web --port 8000
```

打开 `http://127.0.0.1:8000/`。首次打开会创建第一位管理员；系统不开放公共注册，
管理员可以在成员面板添加协作者。

服务启动时会先打印配置检查。缺少 API Key、本地模型目录错误或数据目录不可写时，
网页仍可启动，但终端与 `GET /health` 会明确说明受影响能力和修复方式。

## 一键创建作者

CLI 全流程：

```powershell
pf forge zhihu <用户名或作者 token> --quality fast
```

也可以登录 Web 后进入作者库，输入知乎用户名或主页 URL。任务会写入 SQLite，由后台
Worker 异步抓取、构建和入库；期间可以继续使用其他作者和对话。

若公开接口无法获得完整内容，再保存本机登录态：

```powershell
pf zhihu-login
```

登录态只保存在 `data/auth/zhihu_storage_state.json`，不会发送给浏览器或提交到 Git。

## Docker 首次启动

Docker 镜像包含前端构建产物、FastAPI、CPU 检索依赖和 Playwright Chromium。

```powershell
Copy-Item .env.example .env
# 编辑 .env，填写 DEEPSEEK_API_KEY
docker compose up -d --build
docker compose logs -f app
```

访问：

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/health
```

新机器没有 BGE-M3 缓存时，启动检查会显示 warning，首次入库或检索时自动下载
`BAAI/bge-m3`；Docker volume 会保留 Hugging Face 缓存。完整部署和排错说明见
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

仓库的 GitHub Actions 会在全新 Linux runner 上完成 Python 测试、React 构建、Docker
镜像构建，以及“空数据、无模型、无 API Key”的首次启动 smoke。这个流程不会占用开发
电脑的内存。

## 评估与实验

时间切分评估集：

```powershell
pf eval prepare <author>
```

Web 的 Evaluate 工作区包含：

- RAG：人工标注、LLM 报告、六路检索指标和可恢复评估任务。
- Generate：人工六维、匿名 AB、LLM Judge 与多次评分稳定性报告。
- Experiment：Study 1 邀请码、实验进度、参与者回放和分析导出。

所有真实语料、索引、会话、API 输出和实验响应都位于 `data/`，默认不进入版本控制。

## 开发验证

```powershell
python scripts/check_no_secrets.py
python -m pytest -q
cd web
npm run build
```

## 数据与秘密边界

以下内容不得提交：

- `.env` 与任何 API Key。
- `data/` 下的真实语料、Cookie/登录态、模型、索引、会话和评估结果。
- 本地数据库、Trace、临时导出和运行日志。

仓库仅保留自制的 `samples/zhihu_mock_md/` 作为格式示例。Docker 构建上下文同样排除
`.env`、`data/` 和 `run.txt`。
