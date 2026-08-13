# PersonaForge 部署与首次启动

本文面向没有本地 Python/Node 开发环境的新机器。推荐用 Docker Compose 运行单实例，
再用 Tailscale 或 Cloudflare Tunnel 与协作者共享。

## 1. 当前架构

```text
浏览器
  -> FastAPI :8000
     -> React 静态文件
     -> SQLite 用户/会话/任务
     -> 本地 Qdrant 作者索引
     -> BGE-M3 CPU embedding
     -> DeepSeek/Tavily 外部 API
```

当前只启动一个 FastAPI 进程。Qdrant local 和 SQLite 都依赖单机文件，不要通过增加
多个 Uvicorn worker 横向扩容；需要多实例时再迁移到独立 Qdrant/PostgreSQL/队列。

## 2. Docker 镜像包含什么

Dockerfile 使用两阶段构建：

1. Node.js 20 执行 `npm ci && npm run build`，生成 `web/dist`。
2. Python 3.11 安装 CPU PyTorch、crawler/index/web 依赖和 Playwright Chromium。

最终容器以非 root 用户运行。`/app/data` 保存运行数据，`/app/models` 保存 Hugging Face
模型缓存。源码、前端产物和依赖在镜像中，API Key 与真实数据不在镜像中。

## 3. 新机器首次启动

只需要安装 Git 和 Docker Desktop：

```powershell
git clone https://github.com/SlamWeb/PersonaForge-.git
cd PersonaForge-
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=你的_Key
TAVILY_API_KEY=
PERSONAFORGE_EMBEDDING_MODEL=BAAI/bge-m3
PERSONAFORGE_DOCKER_MODEL_SOURCE=personaforge-models
PERSONAFORGE_PORT=8000
```

启动：

```powershell
docker compose up -d --build
docker compose ps
docker compose logs -f app
```

打开 `http://127.0.0.1:8000/`。第一次没有作者数据是正常状态，可先创建管理员，再从
作者库提交异步抓取和入库任务。

停止但保留数据：

```powershell
docker compose down
```

不要执行 `docker compose down -v`，除非确定要删除 Docker 管理的模型缓存卷。

## 4. 启动检查

每次 `pf web` 启动都会检查：

| 检查项 | 缺失时行为 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 服务继续启动，Chat、query understanding 和 LLM Judge 不可用 |
| BGE-M3 | 模型 ID 未缓存时提示首次使用将下载；显式本地路径不存在时报告 error |
| 数据目录 | 不可写时报告具体路径与系统错误 |
| `TAVILY_API_KEY` | 仅禁用联网背景搜索，本地 RAG 可继续使用 |

终端会打印 `[PersonaForge startup check]`。机器可读信息位于：

```text
GET /health
```

`status=degraded` 表示 Web 进程正常，但某些能力尚未配置；它不是容器崩溃。响应中的
`preflight.checks[].action` 给出修复动作。启动检查不会打印 Key 的值，也不会联网验证
Key，从而避免启动时产生费用。

## 5. 模型缓存

默认模型是 `BAAI/bge-m3`。第一次执行作者入库或检索时，FlagEmbedding 会下载权重到
容器的 `HF_HOME=/app/models/huggingface`，Compose 用命名卷持久化 `/app/models`。

已有宿主机模型时，可在 `.env` 指向宿主机目录并使用容器内路径：

```dotenv
PERSONAFORGE_DOCKER_MODEL_SOURCE=D:/PersonaForgeCache
PERSONAFORGE_EMBEDDING_MODEL=/app/models/bge-m3
```

如果容器内该目录不存在或为空，启动检查会直接显示完整路径。Windows 路径不能直接
作为容器内 `--model-name`，必须先通过 volume 映射到 `/app/models/...`。

## 6. 数据迁移与备份

Compose 把仓库的 `./data` 挂载到 `/app/data`。迁移到另一台机器时只需复制 `data/`
和本机 `.env`，但两者都不得提交到公开仓库。

`data/` 可能包含：

- 抓取语料和知乎登录态。
- Qdrant 索引与 BGE-M3 相关产物。
- 用户、会话、记忆、Trace 和后台任务数据库。
- RAG/生成评估结果与实验参与者响应。

备份前先停止容器，避免复制到写入一半的 SQLite/Qdrant 文件。

## 7. 分享给协作者

临时展示可以使用：

```powershell
cloudflared tunnel --url http://localhost:8000
```

这会生成临时 HTTPS 地址。它不等于生产部署：链接泄露后任何人都能访问登录页，且服务
仍依赖你的电脑在线。正式长期共享优先使用 Tailscale 私网或 Cloudflare Named Tunnel。

应用层仍需独立账号。管理员在成员面板创建协作者，不要共享管理员密码。部署保护默认
限制单用户和全局同时生成数，离线评估 Worker 不受 Chat 限流影响。

## 8. 全新环境验证

`.github/workflows/ci.yml` 的 `docker-first-start` job 在 GitHub 托管的全新 Linux runner：

1. 从 clean checkout 构建镜像。
2. 使用空数据、空模型缓存、无 API Key 启动容器。
3. 验证 `/health` 返回明确的 degraded 配置诊断。
4. 验证 React 首页可访问。

因此首次启动验证不会使用开发电脑内存，也不会读取本机 `.env`、Cookie 或真实语料。

## 9. 常见故障

### `DEEPSEEK_API_KEY is missing`

检查 `.env` 是否位于仓库根目录，修改后执行：

```powershell
docker compose up -d --force-recreate
```

### 本地模型目录不存在

确认 `.env` 中写的是容器内路径 `/app/models/...`，同时
`PERSONAFORGE_DOCKER_MODEL_SOURCE` 指向真正包含模型的宿主机目录。

### 页面能打开但没有作者

空 `data/` 不会自带真实 persona。登录后从作者库添加，或在停服后把已有 `data/` 复制
到新机器。

### 容器反复重启

```powershell
docker compose ps
docker compose logs --tail 200 app
```

先看启动检查，再检查端口占用、数据目录权限和 SQLite/Qdrant 文件是否来自正在运行的
另一个实例。
