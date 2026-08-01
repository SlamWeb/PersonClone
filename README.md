# PersonaForge Open Source

Local-first creator persona RAG.

This repo is the planned open-source product version split from the research workspace. The MVP goal is simple:

```text
crawl public creator content locally
-> build a local RAG index
-> connect your own LLM API key
-> chat with a local web UI
```

The detailed contract lives in [SPEC.md](SPEC.md).

当前 Web 支持按账号隔离的持久会话与跨会话用户记忆。记忆只从用户消息中异步提取，
不会把作者生成内容写成用户事实；用户可以在侧栏的“我的记忆”里查看、纠正、置顶、
遗忘或关闭自动写入。

## Quick Start

从 GitHub 安装当前开发版：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[all]"
playwright install chromium
pf forge zhihu <author-token> --quality fast
```

`pf forge` 默认抓取所有当前可访问内容，然后依次 build、index 并在
`http://127.0.0.1:8000/` 启动 Web。若公开接口不可用，先保存本地登录态：

```powershell
pf zhihu-login
pf forge zhihu <author-token> --quality fast
```

开发者安装和验证：

```powershell
pip install -e ".[all,dev]"
pf --help
pf init
python -m pytest -q
```

The current MVP contains:

- Zhihu-like crawler output contract.
- Markdown -> parent docs -> title/lead/passage child nodes.
- BGE-M3 dense+sparse local Qdrant indexing.
- Query understanding + query transform + RAG20 generation.
- FastAPI Web backend with SSE streaming.
- React/Vite Web frontend.
- Web 作者库与服务端异步 `crawl -> build -> index` 任务。

## Web MVP

Install backend Web dependencies:

```powershell
pip install -e ".[web,dev]"
```

Start the FastAPI backend:

```powershell
pf web mock-columnist --port 8000
```

Docker 或服务器部署时显式监听所有网卡：

```powershell
pf web mock-columnist --host 0.0.0.0 --port 8000
```

本地默认仍绑定 `127.0.0.1`，不会无意暴露给局域网或公网。

For frontend development:

```powershell
cd web
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173/
```

For a single-port local run, build the frontend first and let FastAPI serve `web/dist`:

```powershell
cd web
npm run build
cd ..
pf web mock-columnist --port 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

首次打开时，网页会要求创建第一位管理员，并自动接管升级前已有的本地历史会话。
以后只显示登录页，不开放公共注册。给协作者增加独立账号：

```powershell
pf user create collaborator
pf user list
```

密码通过终端隐藏输入。作者语料与索引在账号间共享；聊天记录、生成任务和 Trace
按账号隔离。Tailscale 可以限制哪些设备能够访问服务，但多人使用时仍需分别登录。

页面左侧可以切换已就绪作者；“管理作者库”进入 `/authors`。添加作者时输入
知乎用户名或主页 URL，确认后可以关闭弹窗或继续打开其他对话。服务端会把
任务写入 SQLite，再由单 Worker 在后台依次抓取 Markdown、构建节点和创建
Qdrant 索引。

已有服务端登录态会自动作为知乎浏览器 fallback：

```text
data/auth/zhihu_storage_state.json
```

登录态只由后台 Worker 读取，不会发送到网页。

## Docker Deployment

当前 Docker 镜像包含 Web、索引依赖和 Playwright Chromium，因此也能执行
网页发起的服务端作者任务。构建和空数据 smoke：

```powershell
Copy-Item .env.example .env
docker build -t personaforge:local .
docker run --rm --name personaforge -p 8000:8000 personaforge:local
```

访问 `http://127.0.0.1:8000/health` 应返回 `{"status":"ok"}`。
挂载已有本地数据并后台运行：

```powershell
docker compose up -d --build
docker compose ps
```

数据卷、模型缓存、API Key 和线上服务器步骤见
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

## Offline Evaluation

PersonaForge can prepare a strict temporal holdout without rebuilding the local index. It keeps the newest valid answers out of retrieval, including all later articles and pins, then dynamically excludes those parent IDs in every dense/sparse query.

```powershell
pf eval prepare <author>
pf eval run <author> --dataset data/eval/<dataset>/dataset.jsonl --split dev --run-name baseline
```

For a low-cost smoke run, add `--limit 1`. Each run writes a local manifest, machine-readable `runs.jsonl`, and one Markdown review file per question under `data/eval/`. Evaluation outputs are intentionally ignored by git. LLM judging and rewrite loops are a later stage; v0 starts with reproducible generation and human review.

## Current Decisions

- MVP 是 local-first CLI + Web；连接该 Web 的用户可以把作者任务提交给服务
  宿主机执行。
- Sample corpus will use self-made Zhihu-like Markdown under `samples/zhihu_mock_md/`.
- `--quality fast` is the default build path and does not call an LLM for preprocessing.
- `--quality full` may add document summaries, but does not create hypothetical questions.
- Query transform happens at query time.
- LLM providers will be abstracted for DeepSeek, OpenAI, and OpenRouter.
- Embedding stays local with BGE-M3 in the first version.
- Web uses FastAPI + React/Vite. Streamlit/Gradio are not the main architecture.
- Web 作者库复用 CLI 的 crawl/build/index 实现，通过 SQLite 队列异步编排，
  不在 API 层重复实现抓取和入库逻辑。
- `pf forge` is the one-command CLI orchestration for crawl -> build -> index -> Web.

No real crawled corpus, auth state, local index, model files, eval output, or API keys should be committed.

## Notes For Contributors

Implementation notes are tracked in [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md). Each module should be explainable enough for an interview, not just runnable.
