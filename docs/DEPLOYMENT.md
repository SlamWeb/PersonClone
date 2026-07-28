# PersonaForge 部署入门

这份文档先解决一个具体目标：把当前本地 Web 打包成一个可重复运行的 Docker 镜像。
它不包含线上任意作者抓取，也暂不引入 PostgreSQL、Redis 或任务队列。

## 1. 先分清五个名词

| 名词 | 在 PersonaForge 里的含义 |
| --- | --- |
| Dockerfile | 构建镜像的配方，描述前端如何编译、Python 依赖如何安装、服务如何启动 |
| Image（镜像） | 按 Dockerfile 构建出的只读应用快照 |
| Container（容器） | 镜像的一次运行实例 |
| Volume（数据卷） | 独立于容器保存的数据目录，避免删除容器时丢失语料、索引和模型缓存 |
| Port mapping（端口映射） | 把电脑的 `8000` 端口转发到容器的 `8000` 端口 |

可以把它们记成：

```text
Dockerfile 是菜谱
Image 是做好的预制菜
Container 是把预制菜加热后端上桌
Volume 是不会随餐桌撤掉的冰箱
```

## 2. 为什么使用两阶段构建

PersonaForge 有两个运行环境：

```text
React/TypeScript -> 需要 Node.js 编译
FastAPI/Python   -> 需要 Python 运行
```

`Dockerfile` 先用 Node.js 构建 `web/dist`，再只把构建结果复制进 Python 镜像。
最终运行镜像不需要 Node.js、TypeScript 编译器或前端源码，这叫 multi-stage build。

## 3. Dockerfile 逐段解释

### 前端构建阶段

```dockerfile
FROM node:20-bookworm-slim AS web-builder
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build
```

- `FROM`：选择带 Node.js 20 的基础镜像。
- 标签后固定了镜像 digest；即使远端同名标签以后更新，当前构建仍使用同一份基础镜像。
- `WORKDIR`：后续命令默认在 `/build/web` 运行。
- 先复制依赖清单再执行 `npm ci`：只要依赖清单没变，Docker 就能复用缓存层。
- `npm run build`：把 React/TypeScript 编译成浏览器可以直接加载的静态文件。

### Python 运行阶段

```dockerfile
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN ... pip install ".[index,web]"
COPY --from=web-builder /build/web/dist ./web/dist
```

- 第二个 `FROM` 开启全新的运行阶段。
- 这里只安装检索和 Web 依赖，不安装 Playwright；线上聊天服务不负责爬取。
- PyTorch 使用 CPU wheel，避免默认镜像携带数 GB 的 CUDA 运行库。
- 第三方依赖只由 `pyproject.toml` 决定并单独形成缓存层；修改业务源码时，
  Docker 只重装很小的 PersonaForge wheel，不会重新安装 PyTorch。
- `COPY --from=web-builder` 只拿上一阶段编译好的前端文件。

### 安全和启动

```dockerfile
USER personaforge
VOLUME ["/app/data", "/app/models"]
EXPOSE 8000
HEALTHCHECK ...
CMD ["pf", "web", ...]
```

- 服务使用普通用户运行，降低进程被利用后的权限。
- `/app/data` 保存作者语料、索引、会话和 trace。
- `/app/models` 保存 BGE-M3 下载缓存。
- `EXPOSE 8000` 是镜像文档，真正开放端口仍需 `docker run -p`。
- `HEALTHCHECK` 定期访问 `/health`，判断服务是否活着。
- `CMD` 是容器启动时默认执行的命令。

## 4. `.dockerignore` 为什么重要

执行 `docker build .` 时，最后的 `.` 叫 build context。Docker 会先读取这个目录中的文件，
再按 `COPY` 指令放入镜像。`.dockerignore` 在这一步排除：

- `.env` 和 API Key。
- `data/` 下的真实作者语料、登录态、索引、会话和 trace。
- 本机虚拟环境、缓存和 `node_modules`。
- 测试、研究产物和临时文件。

它既能减少构建体积，也是防止秘密进入镜像的第一道边界。

## 5. 本机构建和运行

安装并启动 Docker Desktop 后，在仓库根目录运行：

```powershell
docker build -t personaforge:local .
```

这条命令把当前目录作为构建上下文，按照 `Dockerfile` 生成名为
`personaforge:local` 的镜像。

第一次只验证空数据服务：

```powershell
docker run --rm --name personaforge `
  -p 8000:8000 `
  personaforge:local
```

- `--rm`：停止后自动删除容器，不删除镜像。
- `--name`：给运行实例起一个可读名字。
- `-p 8000:8000`：电脑端口映射到容器端口。

然后访问：

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/health
```

空数据服务可以打开网页和健康检查，但没有可聊天的作者。
下一步会用 Compose 挂载本地 `data/` 和模型缓存，并通过 `.env` 注入 API Key。

## 6. 用 Compose 保存运行参数

单条 `docker run` 适合 smoke，但真实运行还需要：

- 挂载 `data/`，让容器读取已有 persona，并把 session/trace 写回宿主机。
- 注入 `.env` 中的 LLM 和 Tavily API Key。
- 持久化 BGE-M3 模型缓存。
- 设置健康检查和异常退出后的重启策略。

这些参数放在 `compose.yaml`。新 clone 的仓库先创建本地配置：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填写：

```dotenv
DEEPSEEK_API_KEY=...
TAVILY_API_KEY=...
PERSONAFORGE_DOCKER_MODEL_SOURCE=personaforge-models
PERSONAFORGE_PORT=8000
```

`personaforge-models` 是 Docker 管理的命名卷。如果电脑已经下载过模型，可以改成：

```dotenv
PERSONAFORGE_DOCKER_MODEL_SOURCE=D:/PersonaForgeCache
```

然后运行：

```powershell
docker compose up -d --build
docker compose ps
docker compose logs -f app
```

- `up`：按 Compose 配置创建并启动服务。
- `-d`：后台运行，终端可以继续使用。
- `--build`：启动前按当前源码更新镜像。
- `ps`：查看容器状态和健康检查。
- `logs -f`：持续查看服务日志，按 `Ctrl+C` 只退出日志查看，不会停止容器。

停止服务：

```powershell
docker compose down
```

`down` 删除容器和网络，但不会删除命名卷。只有显式执行
`docker compose down -v` 才会删除 Compose 管理的模型卷。

## 7. 当前部署边界

第一版线上 demo 使用预先构建且可以公开展示的 persona。任意作者的
`crawl -> build -> index` 仍由开源用户在自己的电脑上执行，避免把知乎登录态、
反爬风险和长时间 GPU 建库任务塞进公开 Web 请求。

第一版只运行一个 FastAPI 进程。Qdrant local 使用本地文件存储，同一个索引目录
不能被多个 Web worker 同时打开；需要水平扩容时再改成独立 Qdrant 服务。
