# PersonaForge 项目导航

这份文档是给第一次打开仓库的自己、合作者和面试时的“项目地图”。它回答四个问题：代码在哪里、一次请求怎么走、数据产物在哪里、评估和实验结果在哪里。

当前唯一工作目录是 `C:\PersonaForge-OpenSource`。旧目录 `C:\PersonaForge` 已弃用，里面的内容不再作为开发、研究或运行命令的来源。

## 1. 三条主线

```text
内容入库主线：知乎内容 -> Markdown -> parent/child 节点 -> BGE-M3 -> Qdrant
产品对话主线：用户问题 -> query understanding/transform -> 检索 -> Narrative Schema -> Writer -> 流式回答
质量验证主线：冻结问题 -> 检索候选池/生成结果 -> 人工或 LLM 评分 -> 可复现结果
```

代码、数据和页面的关系是：

```text
web/src/                 React 页面
        | HTTP / SSE
src/personaforge/web/    FastAPI 路由、鉴权、会话、业务编排
        | 调用领域模块
crawler / ingest / persona / eval / studies
        | 读写本地数据
data/                    作者语料、索引、运行库、评估和实验产物
```

## 2. 根目录地图

| 路径 | 作用 | 是否应该提交到 Git |
|---|---|---|
| `src/personaforge/` | Python 后端和 AI/RAG 核心代码 | 是 |
| `web/src/` | React + TypeScript 前端源代码 | 是 |
| `tests/` | Python 后端、RAG、Web、实验工具测试 | 是 |
| `samples/` | 不含真实作者内容的演示语料 | 是 |
| `docs/` | 部署说明、进度记录、研究协议和架构图 | 是 |
| `scripts/` | 少量可复用的离线评估脚本 | 是 |
| `data/` | 本机真实语料、索引、模型、账号和实验数据 | 否，已被 Git 忽略 |
| `.tmp/`、`.pytest_cache/` | 临时文件和测试缓存 | 否 |
| `pyproject.toml` | Python 包、依赖、`pf` 命令和测试配置 | 是 |
| `Dockerfile`、`compose.yaml` | 容器运行和部署 | 是 |
| `run.txt` | 本机常用启动命令记录，可能包含机器专属绝对路径 | 不作为部署说明；公开命令以 README 为准 |
| `README.md` | 面向新用户的快速开始 | 是 |
| `SPEC.md` | 工程总边界和产品契约 | 是 |
| `AGENTS.md` | 给自动化编码协作者的工作约束 | 是 |

注意：`web/dist/` 是前端构建后的静态产物，不是前端源代码，默认不提交。真正需要读和改的是 `web/src/`。

## 3. Python 后端模块

### 3.1 命令入口

| 文件 | 主要职责 |
|---|---|
| `src/personaforge/cli.py` | 定义 `pf` 命令树并把命令分发到 crawler、build、index、retrieve、ask、eval、web 等模块 |
| `src/personaforge/env.py` | 读取数据目录、模型目录和环境变量配置 |
| `src/personaforge/llm.py` | LLM provider 的统一调用接口和配置 |
| `src/personaforge/SPEC.md` | Python 包入口、CLI 和整体包边界 |

常用命令的职责：

```text
pf crawl       抓取并保存 Markdown
pf build       Markdown -> parent/child 节点和构建清单
pf index       节点 -> embedding -> Qdrant collection
pf retrieve    只测试召回，不生成回答
pf ask         召回并调用 Writer 生成一条回答
pf eval        准备冻结数据集、生成评测结果、建立候选池、运行 judge
pf web         启动 FastAPI 和已构建的 React 页面
pf forge       按顺序执行抓取、构建、索引和启动 Web
```

### 3.2 爬虫：`src/personaforge/crawler/`

职责是把平台内容变成本地、稳定、可复用的 Markdown 来源格式。

| 文件 | 主要职责 |
|---|---|
| `zhihu.py` | 公开接口和内容抓取流程 |
| `zhihu_browser.py` | Playwright 浏览器、公开访问和本地登录态 fallback |
| `markdown.py` | 回答、文章、想法和 profile 的 Markdown 解析/清洗 |
| `models.py` | 爬取结果、作者 profile、manifest 等数据结构 |
| `storage.py` | 文件落盘、作者/内容类型目录管理 |
| `exceptions.py` | 爬虫错误类型 |
| `SPEC.md` | 爬取输入、输出目录、作者和内容类型契约 |

标准作者数据入口是：

```text
data/authors/zhihu/<author-token>/raw/
  answer/
  article/
  pin/
  assets/
  profile.json
  manifest.jsonl
```

### 3.3 入库和检索：`src/personaforge/ingest/`

这是 RAG 的核心模块，负责把 Markdown 变成可召回的 parent/child 结构，并在本地建立 Qdrant 索引。

| 文件 | 主要职责 |
|---|---|
| `loader.py` | 读取 crawler 产出的 Markdown 和 manifest |
| `build.py` | 组织完整构建流程并写 `build_manifest.json` |
| `chunking.py` | 长文切分策略 |
| `nodes.py` | 生成 parent、title child、lead child、passage child |
| `models.py` | parent、child、检索结果等数据结构 |
| `embeddings.py` | 加载和调用 BGE-M3，生成 dense/sparse 表征 |
| `qdrant_index.py` | Qdrant collection 建立、写入和查询底层实现 |
| `index.py` | 组织 embedding、批量写入和索引清单 |
| `retrieve.py` | dense/sparse 双路召回、RRF、parent 折叠和最终排序 |
| `query_understanding.py` | 判断是否需要联网、生成搜索和背景理解、query transform |
| `SPEC.md` | 入库、节点、权重、召回和 Qdrant 契约 |

当前作者索引的标准位置：

```text
data/authors/zhihu/<author-token>/index/
  parents.jsonl
  nodes.jsonl
  build_manifest.json
  qdrant_manifest.json
  qdrant/collection/<collection-name>/storage.sqlite
```

`parents.jsonl` 是可读的完整材料库，`nodes.jsonl` 是召回单位，Qdrant 目录是向量数据库的本地存储。三者是同一次入库的不同层次产物。

### 3.4 Persona 和生成：`src/personaforge/persona/`

职责是把检索到的内容组织成“像这个作者说话”的生成上下文。

| 文件 | 主要职责 |
|---|---|
| `writer.py` | 生成 prompt、组装上下文、调用 LLM、执行回答策略和 MRPrompt 流程 |
| `narrative.py` | 读取 Narrative Schema，并提供作者叙事信息 |
| `pack.py` | 兼容旧版 Persona Pack |
| `suggestions.py` | 生成产品页面上的建议问题 |
| `SPEC.md` | Persona Pack、Narrative Schema、writer 和 RAG 上下文边界 |

当前生成大致是：

```text
当前问题和多轮上下文
-> query understanding / transform
-> dense+sparse+RRF 召回
-> 截取 writer context top-k（可做 RAG20/RAG5 对比）
-> Narrative Schema 或旧 Persona Pack
-> 作者身份、边界、Magic-If 等写作指令
-> LLM provider
-> 回答和 trace
```

在线生成的完整架构已经拆成三张可读图，包含请求 JSON、Turn Planner 输出、四路
Query Transform、两级 Parent RRF、MRPrompt 消息数组、流式生成和回答后的记忆维护：

- [`docs/architecture/generation/README.md`](docs/architecture/generation/README.md)：生成链路总说明。
- [`generation-overview.png`](docs/architecture/generation/generation-overview.png)：从 React query 到最终回答。
- [`retrieval-detail.png`](docs/architecture/generation/retrieval-detail.png)：四路 Dense/Sparse 与 Parent 聚合。
- [`mrprompt-context.png`](docs/architecture/generation/mrprompt-context.png)：最终 Writer 上下文窗口。

RAG 与 Generate 的完整评估口径也已整理成可读图，包含候选池分支、双轴标签、
Hit/MRR/nDCG/Precision/Recall/MAP、六维 Gold Judge、三次稳定性和人工 AB：

- [`docs/architecture/evaluation/README.md`](docs/architecture/evaluation/README.md)：评估链路总说明与变量解释。
- [`rag-evaluation.png`](docs/architecture/evaluation/rag-evaluation.png)：RAG 候选池、标注与指标。
- [`generation-evaluation.png`](docs/architecture/evaluation/generation-evaluation.png)：生成质量与 Judge 稳定性。

## 4. Web 后端与前端

### 4.1 后端入口

| 文件 | 主要职责 |
|---|---|
| `src/personaforge/web/app.py` | FastAPI 应用、鉴权依赖、HTTP/SSE 路由、静态文件挂载 |
| `src/personaforge/web/schemas.py` | 前后端共享的请求/响应校验模型 |
| `src/personaforge/web/service.py` | Chat 的主要业务编排：作者、会话、记忆、检索、生成、trace |
| `src/personaforge/web/streaming.py` | SSE 流式事件格式和错误处理 |
| `src/personaforge/web/conversations.py` | SQLite 会话、消息、turn 和断点恢复 |
| `src/personaforge/web/multiturn.py` | 多轮历史选择、摘要和 planner |
| `src/personaforge/web/user_memory.py` | 用户长期记忆的写入、读取、纠正、遗忘和空闲处理 |
| `src/personaforge/web/auth.py` | 管理员、协作者、登录状态和账号隔离 |
| `src/personaforge/web/trace.py` | 保存和读取每次生成的 trace |
| `src/personaforge/web/startup_checks.py` | 启动前检查 API Key、模型配置和数据目录，并向终端与 `/health` 输出修复建议 |
| `src/personaforge/web/author_jobs.py` | 异步作者抓取、build、index 任务和任务状态 |
| `src/personaforge/web/chat_tasks.py` | 非流式 turn 任务、重试和后台任务状态 |
| `src/personaforge/web/retrieval_evaluation.py` | RAG 人工标注、双轴 LLM Qrels 报告、分切分读取和指标展示 |
| `src/personaforge/web/retrieval_eval_jobs.py` | 多作者检索评估初始化任务：时间切分、全量候选池、API/Codex 标注、预算与断点恢复 |
| `src/personaforge/web/generation_evaluation.py` | Generate 评估、人工 rubric、AB 对比和异步 LLM judge |
| `src/personaforge/studies/study1_service.py` | Study 1 参与者会话、材料快照、保存、提交和管理员回放 |
| `src/personaforge/studies/study1_materials.py` | Study 1 刺激材料准备、来源分配和审计 |
| `src/personaforge/studies/study1_analysis.py` | Study 1 数据分析和完整性检查 |

### 4.2 三个前端工作区

| 页面文件 | 用户看到的内容 | 主要后端接口 |
|---|---|---|
| `web/src/App.tsx` | 顶层登录、Chat/Evaluate/Experiment 模式、路由和布局骨架 | 顶层状态与模式入口 |
| `web/src/PersonaWorkspace.tsx` | 选作者、历史会话、Chat、流式回答、trace、记忆入口 | `/api/personas/*`、`/api/chat/*`、`/api/memories*` |
| `web/src/EvaluationWorkspace.tsx` | RAG 人工标注、LLM 报告和评估任务三个视图的总入口 | `/api/evaluations/retrieval/*` |
| `web/src/RetrievalLlmReport.tsx` | 双轴 Qrels 指标、逐题排序、Gold 与候选证据 | `/api/evaluations/retrieval/pools/*/llm-labels/*` |
| `web/src/RetrievalEvalJobs.tsx` | 创建、观察和恢复作者评估任务，下载/导入 Codex handoff | `/api/evaluations/retrieval/jobs/*` |
| `web/src/GenerationEvaluationWorkspace.tsx` | Generate 系统选择、六维度评分、AB 对比、LLM judge | `/api/evaluations/generation/*` |
| `web/src/StudyWorkspace.tsx` | Study 1 参与者实验和管理员进度/回放 | `/api/studies/study1/*` |
| `web/src/AuthScreen.tsx` | 首次创建管理员和登录 | `/api/auth/*` |
| `web/src/AuthorManager.tsx` | 作者库、作者切换和异步添加作者 | `/api/personas/preview`、`/api/author-jobs/*` |
| `web/src/MemberManagement.tsx` | 协作者账号、管理员和成员管理 | `/api/admin/users` |
| `web/src/api.ts` | 前端 API 类型、请求封装、SSE 解析和错误处理 | 所有 `/api/*` |
| `web/src/styles.css` | 全局布局、Chat、Evaluate、Experiment 视觉样式 | 不直接访问后端 |
| `web/src/main.tsx` | React 挂载入口 | 不直接访问后端 |

部署入口：

- `Dockerfile`：两阶段构建 React 与 Python/Playwright 运行镜像。
- `compose.yaml`：挂载 `data/` 和模型缓存、注入 `.env`、维持单实例服务。
- `docs/DEPLOYMENT.md`：新机器首次启动、模型缓存、分享和故障排查。
- `.github/workflows/ci.yml`：在 GitHub clean runner 上测试、构建并验证 Docker 首次启动。
- `scripts/check_no_secrets.py`：阻止真实 `data/`、Cookie 文件和常见 API Key 进入提交。

### 4.3 一次 Chat 请求的实际路径

```text
web/src/PersonaWorkspace.tsx
  -> web/src/api.ts
  -> POST /api/chat/stream
  -> src/personaforge/web/app.py
  -> src/personaforge/web/service.py
  -> conversations.py 读取当前用户/会话上下文
  -> user_memory.py 读取用户记忆
  -> ingest/query_understanding.py 处理 query
  -> ingest/retrieve.py 召回作者 parent
  -> persona/narrative.py 或 persona/pack.py 读取作者画像
  -> persona/writer.py 组装最终 prompt 并调用 LLM
  -> streaming.py 以 SSE 返回状态、token、来源和完成事件
  -> trace.py 保存 trace；conversations.py 保存消息和 turn
```

### 4.4 重要 API 分组

| API 前缀 | 功能 |
|---|---|
| `/api/auth/*` | 登录、退出、首次管理员初始化 |
| `/api/admin/users` | 创建和查看协作者/管理员 |
| `/api/personas/*` | 作者库、头像、会话、建议问题和 trace |
| `/api/author-jobs/*` | 抓取、build、index 的异步任务 |
| `/api/chat/*` | 流式聊天、turn 状态、重试和事件 |
| `/api/memories*` | 用户记忆查看和维护 |
| `/api/evaluations/retrieval/*` | RAG 候选池、人工标签、双轴 LLM 报告、作者评估异步任务和 Codex handoff |
| `/api/evaluations/generation/*` | 生成系统、六维 rubric、AB、LLM judge |
| `/api/studies/study1/*` | 参与者入口、Study 1 保存、提交、管理员统计和导出 |

Evaluate 的作者作用域由左侧头像控制。选择某个作者后，RAG 和 Generate 只请求该作者
的评估资产；点击“全部作者”只进入汇总观察范围。没有作者元数据的旧资产会标记为
“未归属/旧数据”，不会被猜测归属，也不能参与跨作者 Generate AB。跨作者汇总使用宏
平均：先按作者平均，再对作者平均；不同冻结数据集、切分、Top K 或标签/Judge 版本不混合。

## 5. 本地数据地图

`data/` 已写入 `.gitignore`。这里是运行产物，不是源代码；换机器、换作者或清理环境前要先确认是否需要备份。

### 5.1 作者产品数据

```text
data/authors/zhihu/<author-token>/
  raw/         爬到的 answer/article/pin Markdown 和头像资源
  staging/     build/index 前的中间产物
  index/       parents、nodes、build 清单和 Qdrant collection
  sessions/    该作者的聊天会话文件（当前兼容层仍保留）
  traces/      该作者的生成 trace JSON
  *.json       profile、Persona Pack、Narrative Schema、建议问题等
```

当前 `wu-ren-jun-28` 的聊天权威索引在：

```text
data/authors/zhihu/wu-ren-jun-28/index/
```

### 5.2 全局运行库

```text
data/system/personaforge.sqlite3
```

这是 Web 的 SQLite 运行库，保存管理员/协作者、账号隔离后的会话消息、turn、异步任务、评估标注和 Study 1 参与者状态。SQLite 是“结构化业务状态”的存储；作者 Markdown 和 Qdrant 不放进 SQLite。

其他目录：

| 路径 | 内容 |
|---|---|
| `data/auth/` | 知乎浏览器登录态；绝不能提交或分享 |
| `data/models/` | BGE-M3 等本地模型缓存 |
| `data/raw/` | crawler 的临时/历史原始输出；正式多作者产品数据以 `data/authors/.../raw` 为准 |
| `data/index/` | 早期共享索引兼容目录；新增作者优先使用作者级 `data/authors/.../index` |
| `data/runtime/` | 运行时临时产物，具体命令产生什么以文件名为准 |
| `data/tmp/` | 临时中间文件 |

### 5.3 评测集、候选池和评估结果

每个作者和语料快照都有独立的离线评测根目录：

```text
data/eval/<author>-temporal-<snapshot>/
```

| 子目录/文件 | 含义 |
|---|---|
| `dataset.jsonl` | 冻结问题和时间切分元数据 |
| `dataset_manifest.json` | 数据集版本、哈希和切分说明 |
| `retrieval_pool/dev/pool.jsonl` | RAG 候选池；当前用于人工相关性标注 |
| `retrieval_pool/dev_core_top3/pool.jsonl` | 核心小候选池，减少人工标注量 |
| `retrieval_pool/all30_exhaustive_qrels_v2/` | cutoff 前全部可见 parent 的全量候选池 |
| `retrieval_pool/.../llm_labels/<label-set>/manifest.json` | 标注范围、模型、prompt、usage、缓存、成本和稳定性清单 |
| `retrieval_pool/.../llm_labels/<label-set>/labels.jsonl` | 每个 `item_id + parent_id` 的内容支撑与作者表达支撑双轴标签 |
| `retrieval_pool/.../llm_labels/<label-set>/metrics.json` | 六路检索在两个效用轴上的多 K 指标 |
| `runs/<run-id>/` | 某个生成系统在冻结问题上的完整生成结果 |
| `runs/<run-id>/items/` | 每道 dev 题的回答、上下文或单题产物 |
| `runs/<run-id>/summary.md` | 该次运行的人工可读摘要 |
| `judge_inputs/` | 发给 LLM judge 的标准化输入 |
| `judge_runs/<judge-id>/` | judge 配置、原始调用、逐题分数和汇总 |
| `manual_pairwise_*/` | 人工 AB 选择、人工与 API judge 对照结果 |

当前已经存在的生成 run 包括 `baseline-dev-v0`、`persona-pack-grounded-dev-20260728`、`persona-pack-response-strategy-v2-grounded-dev-20260728` 和 `persona-pack-response-strategy-v3-writer-replay-dev-20260728`。它们是不同系统版本的实验记录，不要只看文件夹名判断质量，先读各自 `manifest.json` 和 `summary.md`。

### 5.4 Study 1 研究实验产物

```text
data/studies/<study-id>/
  material_bank.json       冻结的盲化材料库
  audit.json / AUDIT.md    材料质量和来源审计
  researcher_review/       研究者审核记录
  other_human_candidates.jsonl 其他真人候选回答池
```

当前 Study 1 目录：

```text
data/studies/wu-ren-jun-28-study1-dev10/
data/studies/wu-ren-jun-28-study1-dev10-v2/
```

参与者的会话、评分、划线、配对理由和管理员回放状态主要进入 `data/system/personaforge.sqlite3`；材料冻结文件进入 `data/studies/<study-id>/`。这是一种“文件保存刺激材料，SQLite 保存交互状态”的拆分。

研究协议和统计边界见：

```text
docs/research/STUDY1_PROTOCOL_V2.md
src/personaforge/studies/SPEC.md
```

## 6. 测试代码在哪里

```text
tests/test_crawler_*.py       爬虫和 Markdown 输出
tests/test_ingest.py          build、节点、embedding、索引
tests/test_index.py           Qdrant/索引契约
tests/test_retrieve.py        dense/sparse/RRF/parent 召回
tests/test_query_understanding.py  联网判断、背景和 query transform
tests/test_writer.py          Writer 和生成 prompt
tests/test_persona_pack.py    Persona Pack
tests/test_narrative_schema.py Narrative Schema
tests/test_eval*.py           离线评测 runner、replay 和候选池
tests/test_retrieval_*.py     RAG 候选池与评估 Web 后端
tests/test_generation_evaluation.py Generate 评估与 judge 任务
tests/test_web.py              Web 总体接口
tests/test_auth.py             登录、管理员和协作者
tests/test_conversations.py   会话、消息和持久化
tests/test_multiturn*.py      多轮上下文、planner 和验收
tests/test_user_memory.py     用户记忆
tests/test_trace.py            trace 保存、读取和清理
tests/test_author_jobs.py      异步抓取/build/index 任务
tests/test_study1_*.py        Study 1 材料和 Web 实验流程
```

全量回归：

```powershell
chcp 65001 > $null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "src"
python -m pytest -q
```

前端构建检查：

```powershell
cd web
npm run build
cd ..
```

## 7. 最短启动路径

### 第一次看代码

按这个顺序读，十分钟可以建立整体心智模型：

1. `README.md`：用户怎么安装和启动。
2. `navigation.md`：本文件，确认目录和数据位置。
3. `SPEC.md`：产品边界和不该做什么。
4. `src/personaforge/cli.py`：命令从哪里进入。
5. `src/personaforge/web/app.py`：HTTP 路由在哪里。
6. `src/personaforge/web/service.py`：Chat 业务如何串起来。
7. `src/personaforge/ingest/retrieve.py`：RAG 如何召回。
8. `src/personaforge/persona/writer.py`：最终 prompt 如何生成。
9. `web/src/PersonaWorkspace.tsx`：Chat 页面如何调用后端。
10. `tests/`：系统真正保证了哪些行为。

### 启动本地 Web

```powershell
$env:PYTHONPATH = "src"
$env:PERSONAFORGE_BGE_M3_USE_FP16 = "1"
python -m personaforge.cli web wu-ren-jun-28 `
  --port 8000 `
  --embedding-device cuda `
  --model-name C:\PersonaForge-OpenSource\data\models\bge-m3
```

如果前端源代码刚改过，先在另一个终端执行：

```powershell
cd web
npm run build
cd ..
```

然后重新启动 FastAPI；FastAPI 服务的是 `web/dist`，不是实时读取 `web/src`。

## 8. 结构审查结论

### 目前合理的地方

- crawler、ingest、persona、web、eval、studies 已经按领域拆开，后续扩展有边界。
- 前端已经按 Chat、RAG Evaluate、Generate Evaluate、Study 1 工作区拆分，没有把所有页面塞进一个组件。
- 业务状态统一进入 `data/system/personaforge.sqlite3`，作者知识库仍然是作者级文件和 Qdrant，职责是清楚的。
- 真实语料、模型、索引、评估结果和实验数据都在 Git 忽略范围内，公开仓库不会默认泄露本地数据。
- `SPEC.md`、模块 `SPEC.md`、研究协议和测试已经形成了“契约 + 实现 + 验证”的基本闭环。

### 目前容易让人迷路的地方

1. `src/personaforge/eval/` 是离线评测核心，`src/personaforge/web/retrieval_evaluation.py` 和 `generation_evaluation.py` 是 Web 适配层；名字相近，但不是重复实现。后续可以在不改变 API 的情况下把离线逻辑再明确分成离线层和 Web 层。
2. `data/raw/`、`data/index/` 是早期兼容目录；现在多作者正式路径是 `data/authors/zhihu/<author>/raw` 和 `.../index`。新人如果只看目录名，确实容易误把旧目录当权威数据源。
3. `docs/IMPLEMENTATION_NOTES.md` 更像历史实现笔记，当前模块合同应优先看对应 `SPEC.md`。以后新增决定不要只写在实现笔记里。
4. `src/personaforge/web/app.py` 仍然承担了较多路由装配工作，这是当前 MVP 可接受的集中入口，但随着评估和实验继续增长，下一步应把路由按领域拆成 router 文件。
5. `src/personaforge/studies/` 已有独立 SPEC，但研究数据仍然全部依赖本地 `data/studies/`；没有这些本地产物时，代码测试能跑，正式实验页面不能凭空生成真实材料。

### 当前不建议做的事

在开始 Evaluate 之前，不建议为了“看起来整齐”移动大量文件或重写路由。现在最重要的是冻结导航、冻结评测目录和运行 ID，然后开始 RAG/Generate 评估。目录重构会改变 import、数据路径和前端接口，收益小于风险。

## 9. 下一步建议

1. 以本文件为入口，先让你和合作者能独立找到代码和产物。
2. 调检索方法时只看 `dev10`；需要完整 Recall 分母时，从“评估任务”创建全量 Qrels。`test20` 只用于最终冻结，不参与调参。
3. 每次改生成链路都创建新的 `runs/<run-id>`，不要覆盖旧结果。
4. 每次改实验协议都创建新的 `study_id`，不要混用旧 `2+3` 和当前 `study1-v2` 数据。
5. 评估页面出现问题时，沿着 `web/src -> app.py -> evaluation store -> data/system/personaforge.sqlite3` 排查，不要直接改 SQLite 文件。

## 10. 以后改功能应该改哪里

下面这张表是后续开发的变更路由。不要只改页面或只改一个 Python 文件就结束；凡是改变数据合同或接口的功能，都要一起检查 SPEC、前端类型和测试。

| 想改的东西 | 首先看 | 通常需要同步 |
|---|---|---|
| 知乎抓取、作者 profile、answer/article/pin 分类 | `crawler/SPEC.md`、`crawler/zhihu.py`、`crawler/storage.py` | `crawler/models.py`、作者 `data/.../raw`、crawler 测试、`author_jobs.py` |
| 作者新增、作者切换、作者是否已就绪 | `web/author_jobs.py`、`web/service.py`、`AuthorManager.tsx` | `web/app.py` 路由、`web/api.ts` 类型、作者数据目录、Web 测试 |
| Markdown 入库、parent/child、切片 | `ingest/SPEC.md`、`ingest/build.py`、`ingest/nodes.py` | `ingest/models.py`、`parents.jsonl`/`nodes.jsonl`、入库和节点测试 |
| BGE-M3、dense/sparse、Qdrant collection | `ingest/embeddings.py`、`ingest/qdrant_index.py`、`ingest/index.py` | `qdrant_manifest.json`、模型运行配置、索引测试、评估候选池 |
| query understanding、联网背景、query transform | `ingest/query_understanding.py`、`ingest/retrieve.py` | `web/service.py`、trace 字段、Writer 输入、query/retrieve 测试 |
| RAG 召回、RRF、parent 聚合、RAG5/RAG20 | `ingest/retrieve.py`、`eval/SPEC.md`、`eval/runner.py` | 离线 run、候选池、`retrieval_evaluation.py`、RAG 页面和评估测试 |
| Persona Pack、Narrative Schema、MRPrompt、Magic-If | `persona/SPEC.md`、`persona/writer.py`、`persona/narrative.py` | 作者画像文件、生成 run、生成 prompt、trace、Writer 测试 |
| 生成质量、六维 rubric、人工评分、LLM judge | `eval/gold_judge.py`、`web/generation_evaluation.py`、`GenerationEvaluationWorkspace.tsx` | `data/eval/<dataset>/judge_*`、API schema、前端类型、judge 测试 |
| Chat 多轮上下文、planner、摘要 | `web/SPEC.md`、`web/multiturn.py`、`web/conversations.py` | `service.py`、SQLite schema、trace、multiturn 测试 |
| 用户长期记忆 | `web/user_memory.py`、`web/SPEC.md` | SQLite、Chat 上下文选择、记忆页面、用户记忆测试 |
| trace、耗时、token、检索过程 | `web/trace.py`、`web/streaming.py`、`web/SPEC.md` | `service.py`、`api.ts`、开发者模式页面、trace 测试 |
| Study 1 协议、材料、参与者流程 | `studies/SPEC.md`、`docs/research/STUDY1_PROTOCOL_V2.md` | `studies/study1_materials.py`、`study1_service.py`、`StudyWorkspace.tsx`、材料审计和实验测试 |
| Study 1 统计字段、导出、回放 | `studies/study1_analysis.py`、`StudyWorkspace.tsx` | SQLite 导出、分析包、管理员回放、研究文档 |
| 顶部模式、侧栏、Chat/Evaluate/Experiment 布局 | `web/SPEC.md`、`App.tsx`、对应 Workspace | `styles.css`、`api.ts`（如接口变化）、前端构建检查 |

### 每次改动的最小检查清单

```text
1. 先读根 SPEC、最近模块 SPEC 和本导航表。
2. 明确这次改动影响的是代码、接口、数据文件、页面还是统计口径。
3. 先改后端数据合同，再改前端 API 类型和页面；不要让前端猜后端字段。
4. 保留旧数据兼容逻辑，除非明确新建版本或迁移脚本。
5. 为新行为补一个 focused test；跨模块改动再跑全量测试。
6. 如果目录、接口、字段或运行命令改变，同步更新 SPEC 和本导航。
7. 检查 `.env`、auth state、raw corpus、Qdrant、评估结果没有被提交。
```
