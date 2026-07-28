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

## Offline Evaluation

PersonaForge can prepare a strict temporal holdout without rebuilding the local index. It keeps the newest valid answers out of retrieval, including all later articles and pins, then dynamically excludes those parent IDs in every dense/sparse query.

```powershell
pf eval prepare <author>
pf eval run <author> --dataset data/eval/<dataset>/dataset.jsonl --split dev --run-name baseline
```

For a low-cost smoke run, add `--limit 1`. Each run writes a local manifest, machine-readable `runs.jsonl`, and one Markdown review file per question under `data/eval/`. Evaluation outputs are intentionally ignored by git. LLM judging and rewrite loops are a later stage; v0 starts with reproducible generation and human review.

## Current Decisions

- MVP is a local CLI + local Web app, not a hosted crawler service.
- Sample corpus will use self-made Zhihu-like Markdown under `samples/zhihu_mock_md/`.
- `--quality fast` is the default build path and does not call an LLM for preprocessing.
- `--quality full` may add document summaries, but does not create hypothetical questions.
- Query transform happens at query time.
- LLM providers will be abstracted for DeepSeek, OpenAI, and OpenRouter.
- Embedding stays local with BGE-M3 in the first version.
- Web uses FastAPI + React/Vite. Streamlit/Gradio are not the main architecture.
- Web v0 supports existing local indexes only; crawl/build/index stay in CLI.
- `pf forge` is the one-command CLI orchestration for crawl -> build -> index -> Web.

No real crawled corpus, auth state, local index, model files, eval output, or API keys should be committed.

## Notes For Contributors

Implementation notes are tracked in [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md). Each module should be explainable enough for an interview, not just runnable.
