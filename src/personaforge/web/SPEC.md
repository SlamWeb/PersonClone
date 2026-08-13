# Web MVP 规格

`src/personaforge/web` 负责本地 Web 服务和前端静态资源托管。

## 当前目标

Web MVP 面向面试展示，目标是把现有 CLI RAG 链路变成一个可交互产品：

```text
选择本地 persona
-> 输入问题
-> FastAPI 调用当前 RAG + Narrative Schema writer 链路
-> SSE 真流式返回回答
-> 回答完成后展示检索来源
```

## 不改动的边界

本阶段只做 Web，不改：

- crawler 的平台抓取实现。
- build/index 的文档与索引语义。
- query understanding / query transform 策略。
- dense+sparse retrieval 和 parent RRF 聚合。
- writer 的生成策略本身；Web 只增加 `mrprompt` 与 Narrative Schema 的接线，旧
  `persona_pack` 仍保留为兼容对照。

Web 只编排已有 crawler、build 和 index 能力，不在 Web 模块里重写这些逻辑。

## 技术选择

后端：

```text
FastAPI + Uvicorn
```

原因：

- 更符合“后端 + AI 应用 / RAG 工程”的求职叙事。
- 能讲清楚 API schema、服务层、SSE streaming、错误处理。
- 后续接 LangGraph trace 和评估接口更自然。

前端：

```text
React + Vite + TypeScript + 原生 CSS
```

原因：

- 比 Streamlit/Gradio 更像真实产品。
- 比 Next.js 轻。
- 暂不上 shadcn/Tailwind/Ant Design，避免第一版被 UI 配置拖住。

聊天页使用受视口约束的三段布局：

- 桌面端作者与历史会话侧栏固定占满视口高度，只有历史会话列表内部滚动。
- 右侧消息区独立滚动，输入区固定在聊天面板底部，不随消息历史离开视口。
- 打开会话或发送消息时自动贴近最新消息；用户主动上翻后，流式更新不得强制拉回
  底部。
- 当消息区距离底部至少 120 px 时，在输入区上方显示“回到最新消息”按钮；点击后平滑
  滚动到底部，到达底部后按钮立即消失。
- 输入区使用单一圆角容器。文本框随内容从 42 px 自动增高，桌面端最高 220 px、移动端
  最高 180 px，超过后只在文本框内部滚动；`Enter` 发送，`Shift+Enter` 换行。
- 窄屏将作者与历史区放在顶部受限区域，消息区和输入区继续占用剩余视口，页面
  本身不得产生需要寻找输入框的纵向滚动。

## 目录规划

Python 后端：

```text
src/personaforge/web/
  app.py          FastAPI create_app 和路由
  schemas.py      Pydantic request/response schema
  service.py      Web 调用当前 RAG + writer 的服务层
  streaming.py    SSE 序列化工具
```

检索评估任务由 Web 进程共享同一套 SQLite 状态；需要脱离 Web 长时间运行时，使用
`scripts/run_retrieval_eval_worker.py` 启动独立 worker，并显式指向当前仓库的 `data/`。
worker 只消费 `queued/running` 任务，完成后自行退出，不应与旧目录的服务混用数据库。

React 前端：

```text
web/
  package.json
  index.html
  src/
    App.tsx
    PersonaWorkspace.tsx
    api.ts
    main.tsx
    styles.css
```

## API v0

### `GET /health`

返回服务状态、启动前检查和本地 embedding 模型的预热状态。

启动前检查必须是非破坏性的：不加载 BGE-M3、不调用外部 API、不打印秘密。它检查
`DEEPSEEK_API_KEY`、BGE-M3 配置、数据目录可写性和可选的 `TAVILY_API_KEY`，每项返回
稳定的 `check_id/status/message/action`。缺少必要配置时 Web 仍应启动，顶层状态为
`degraded`；配置齐全时为 `ok`。

模型 ID（如 `BAAI/bge-m3`）未命中本地 Hugging Face 缓存只记为 warning，因为首次
入库或检索可以自动下载。只有显式本地模型路径不存在或为空时才记为 error。

embedding 模型预热状态：

- `idle`：尚未开始加载；
- `loading`：正在后台加载；
- `ready`：可以直接执行检索；
- `failed`：预热失败，保留错误摘要供本地排查。

当启动时已经存在至少一个可用作者索引，Web 服务会在后台预热共享的 BGE-M3
encoder。FastAPI 不等待模型加载即可先提供页面和 API，但首个聊天请求不再承担正常
情况下约几十秒的模型冷启动成本。预热只改变加载时机，不改变 query transform、召回
路线、排序或生成 prompt。

Windows 下 `FlagEmbedding -> datasets -> pyarrow` 的首次原生依赖导入必须发生在
Uvicorn 和应用 worker 启动之前，随后才允许后台加载模型权重。当前环境约束
`pyarrow>=21,<24`；已验证 PyArrow 24 在本项目 Windows 进程中可能触发无法被 Python
异常捕获的 access violation，表现为服务没有 shutdown 日志便直接退出。

`pf web` 启动时必须把同一份检查报告打印到终端。Docker 健康检查只判断 Web 进程是否
可访问，因此 `degraded` 仍返回 HTTP 200；具体能力是否可用由 `preflight` 字段表达。

### `GET /api/personas`

扫描本地：

```text
data/authors/zhihu/<author>/index/
```

只返回已存在 `parents.jsonl` 且有 `qdrant/` 的作者。

### `POST /api/chat/stream`

请求：

```json
{
  "author": "wu-ren-jun-28",
  "query": "如何看待女生常说的配得感",
  "query_mode": "grounded",
  "writer_prompt": "mrprompt",
  "parent_top_k": 20
}
```

`mrprompt` 需要作者目录中的 `narrative_schema.json`。没有该文件的作者仍可选择
`strong_identity` 或兼容的 `persona_pack`。现有会话摘要、最近对话和用户记忆继续由
会话层提供；Web 不额外实现一套 STM。`parent_top_k=20` 与 `parent_top_k=5` 可用于
RAG20/RAG5 对照，但这两个版本必须在相同的 query understanding、query transform 和
writer prompt 下比较。

响应为 SSE：

```text
event: meta
data: {"author":"...","retrieval_queries":[...]}

event: status
data: {"stage":"retrieval","label":"正在检索历史表达"}

event: token
data: {"text":"..."}

event: done
data: {"answer":"...","sources":[...]}
```

错误：

```text
event: error
data: {"error":"..."}
```

### `GET /api/personas/{author}/traces/{trace_id}`

返回某次 assistant 回答对应的 trace v0。trace 是本地运行档案，路径为：

```text
data/authors/zhihu/<author>/traces/<trace_id>.json
```

它包含输入与运行配置、query understanding 和联网背景、每一路 dense/sparse child 检索、parent RRF 聚合、writer 输入摘要、耗时和最终状态。它不保存 API key、cookie、登录态，也默认不重复保存完整 writer prompt 与 parent 正文。

## 流式策略

采用真流式：

```text
DeepSeek stream
-> FastAPI StreamingResponse
-> React fetch ReadableStream 解析 SSE
-> 前端实时追加 token
```

为了支持 Web，LLM provider 层需要新增：

```text
stream_text(messages, options) -> iterator[str]
```

这只是 provider 能力扩展，不改变现有非流式 `complete_text`。

## 缓存

FastAPI 进程内缓存：

- BGE-M3 encoder。
- DeepSeek client。

原因：Web 是长进程，不能每个问题都冷启动 embedding model。

## CLI

```powershell
pf web <author-token> --host 127.0.0.1 --port 8000 --embedding-device cuda
```

`author-token` 作为本地开发默认 persona。若不传，则 Web 扫描本地 persona 并选择第一个。
本地默认只监听 `127.0.0.1`；容器或服务器部署必须显式传
`--host 0.0.0.0`，不能为了部署而改变本地安全默认值。

## MVP 展示范围

第一版页面展示：

- 左侧自定义作者切换器，只显示当前作者头像与昵称，不使用浏览器原生
  `select`。
- 作者切换器弹层列出全部本地作者和正在构建的作者；作者数量较少时不提供
  搜索框。
- 独立作者库页面管理已就绪作者和后台构建任务。
- 添加作者三步向导：输入知乎用户名或主页 URL、确认作者、查看后台进度。
- 左侧按作者隔离的历史会话列表。
- 右侧聊天软件式消息流，用户和 persona 都显示头像。
- 流式回答。
- 普通 sources 折叠区只展示可读标题，并链接到知乎原回答；不显示本地 Markdown 文件名、parent ID、分数或 child 路由。
- query mode、parent topK 收进“高级设置”，默认不打扰普通使用。
- Writer 变体属于实验时的高频选择，显示在输入框上方的三段式“回答模式”控件中：`定向提示 / 强身份 / Persona Pack`。浏览器记住最近选择；没有 Pack 的作者禁用 Persona Pack。

## 视觉方向

Web 主界面采用“分身空间”而不是通用 AI Chat 的信息架构：

- 桌面端使用三栏：`64px` 分身头像栏、`248px` 当前分身会话栏、纯白聊天区。
- 分身头像栏用于直接切换作者，添加作者与进入分身库使用独立图标命令，不再使用下拉框。
- 当前作者名称、就绪状态、新对话和按时间分组的会话只出现在第二栏；会话行不展示时间，只在运行时显示“正在生成”。
- 顶部不放大标题，不显示“正在以某作者回答”这类说明文案。
- 右侧空状态只使用作者头像和一句开场白，不展示推荐问题或功能介绍。
- 作者回答采用无框 answer block，用户消息使用无头像、无描边的右侧浅灰气泡；复制等消息操作放在正文或气泡下方，不能挤压正文内边距。
- 作者主题色由稳定的身份色板映射产生，只用于头像选中态、状态点和少量交互反馈，不大面积铺色。
- 每条正式消息都提供复制按钮；普通引用只显示可读标题与原文链接。
- 回答模式、RAG、TopK、开发者模式与 trace 记录统一收进会话栏底部的“实验台”。
- 手机端把分身头像栏压缩为顶部横栏，会话栏通过抽屉进入，聊天和输入区继续占满剩余视口。
- 主聊天区保持纯白，侧栏使用中性灰，不使用渐变、装饰色块或层层卡片。
- 全局优先使用平台系统字体，只使用 Regular、Medium、Semibold/Bold 等可用语义字重，不使用 `650/720/760` 一类任意数字制造合成粗体。
- 按钮使用克制的触觉微交互：悬停和选中状态在约 140-180 ms 内连续过渡，按下时在
  60 ms 内轻微下沉并缩小，释放后平滑复位；图标按钮和发送按钮的按压反馈可以略强。
  必须尊重 `prefers-reduced-motion`，不能用涟漪或明显弹跳干扰阅读。

## Chat 与 Evaluate 工作区

右侧主区域顶部使用 `Chat / Evaluate / Experiment` 工作区控件。`Chat` 与 `Evaluate`
继续保存在产品壳内；产品中的 `Experiment` 是仅管理员可见的研究工作台，跳转到
`/experiment/admin` 查看实验作者、参与码、进行中/已完成进度、匿名提交回放和独立导出。
它不是参与者入口，也不放进左侧实验台。参与者只能通过研究者分发的干净作者专属
`/experiment/<study_id>` 链接进入，避免聊天侧栏、作者库和开发者评估暗示被试。

Study 1 支持多作者并以 `study_id` 隔离。`/experiment/admin` 提供实验作者选择、参与码
生成、专属链接复制和独立导出；参与者页面不提供作者选择。旧 `/experiment` 只用于兼容
首份材料库，不作为正式招募链接。相关 API 包括：

参与者首次提交邀请码后，前端在浏览器本地保存 `{session_id, resume_token}`，后续每个
`/api/studies/study1/sessions/{session_id}/...` 请求通过
`X-Study-Session-Token` 发送该凭据。参与码恢复会轮换凭据；仅保存旧 `session_id` 的早期
浏览器记录会回到邀请码页，而不会绕过会话校验。

```text
GET  /api/studies/study1/admin/studies
GET  /api/studies/study1/studies/{study_id}
POST /api/studies/study1/studies/{study_id}/sessions
GET  /api/studies/study1/admin/overview?study_id={study_id}
GET  /api/studies/study1/admin/export?study_id={study_id}&format=jsonl|csv
```

`Evaluate` 只消费 `pf eval retrieval-pool` 已冻结的候选池，不在 Web 请求中调用 LLM、
embedding 或 Qdrant。页面包含：

- 总进度、每题进度和“只看未完成”。
- 问题列表与前后导航。
- 当前问题、候选标题、完整 parent 正文和知乎原文链接。
- `0 无用 / 1 有一定帮助 / 2 明显有用 / 暂时跳过`。
- `0/1/2` 键盘快捷键和前后方向键。
- JSONL、CSV 导出。

Evaluate 必须在固定高度工作区内运行：RAG 和 Generate 共用相同宽度的左侧评估栏与
顶部留白；材料正文只在中间阅读区滚动，底部评分操作始终可见。RAG 候选材料不重复写
“候选材料”这类无信息标签；`pin` 使用“想法”作为标题，article/answer 使用原题目。
Generate 的“作者原回答”“系统回答”“候选 A/B”必须采用黑色、可快速扫描的明确标题，
并通过边框与留白区分比较对象。评估与实验表单的小字号以 100% 浏览器缩放可读为底线，
不依赖用户放大页面才能辨认。

每次点击有效分数后立即写入 `data/system/personaforge.sqlite3`。数据库主键为
`pool_id + item_id + parent_id + user_id`，因此用户之间互相隔离，重复评分为修订而非
新增。候选顺序根据用户、池、问题和 parent ID 做稳定哈希排序，刷新后不变。

为了降低标注暗示，未评分候选的 API 响应不返回 route/rank 详情；评分成功后才返回并
允许展开。网页发现不到候选池时只显示生成命令，不隐式创建或修改实验数据。

新增 API：

```text
GET /api/evaluations/retrieval/pools
GET /api/evaluations/retrieval/pools/{pool_id}
GET /api/evaluations/retrieval/pools/{pool_id}/queries/{item_id}
PUT /api/evaluations/retrieval/pools/{pool_id}/queries/{item_id}/candidates/{parent_id}
GET /api/evaluations/retrieval/pools/{pool_id}/export?format=jsonl|csv
```

### Generate 评估工作区

`Evaluate` 内部再使用 `RAG / Generate` 两段式控件切换评估对象。RAG 保留当前候选材料
标注；Generate 自动发现冻结且与 `temporal_dev10_v0` 哈希一致的完整系统 run。切换
不会改变 Chat 工作区或当前作者。

Generate 提供三个相互独立的测量入口和一个总览：

- 人工六维：选择一个系统，逐题同时查看 question、Gold 和 Candidate；D1--D6 分别
  选择 1--5，理由可选。允许逐维即时保存，但六维齐全才算该题完成。
- 匿名 AB：选择两个完整系统，逐题查看 Gold 与匿名 A/B，强制二选一，不提供平局；
  系统名在作答后才揭示。
- LLM Judge：展示 Gold Judge V1 六维均分、三组汇总、稳定性和逐题证据；未运行时可
  创建后台任务，用户可以离开页面或切换工作区。
- 总览：按系统显示六维结果、人工进度、AB 胜率和 Judge 状态。六维不合成排行榜总分，
  D6 单独展示。

人工记录按登录用户隔离并写入现有 SQLite；系统 run 和自动 Judge 结果为所有登录用户
共享的实验资产。Generate API 至少包括：

```text
GET  /api/evaluations/generation/systems
GET  /api/evaluations/generation/systems/{system_id}
GET  /api/evaluations/generation/systems/{system_id}/items/{item_id}
PUT  /api/evaluations/generation/systems/{system_id}/items/{item_id}/rubric
GET  /api/evaluations/generation/comparisons/{left_id}/{right_id}
GET  /api/evaluations/generation/comparisons/{left_id}/{right_id}/items/{item_id}
PUT  /api/evaluations/generation/comparisons/{left_id}/{right_id}/items/{item_id}
POST /api/evaluations/generation/judge-jobs
GET  /api/evaluations/generation/judge-jobs/{job_id}
```

Generate 页面不能现场重新生成 dev10 Candidate。生成仍由可复跑 eval runner 完成；新
run 写入后刷新系统列表即可出现。Judge 后端服务同时允许 CLI 调用，但网页是普通用户
的主要入口。

同一系统的 Judge 任务使用持久化结果目录。若任务因瞬时模型或网络错误失败，再次发起
时应复用原任务和已写入的 `result.json`，从已完成题目继续，而不是重新创建一套重复
的 Judge 结果。

Study 1 的配对题应直接告诉被试：拖动选中文字可标注“像”或“不像”的依据，且 A/B
两边均可标注；不要把这项发现性交互藏在实现里。

## Persona Metadata

Web 优先读取：

```text
data/authors/zhihu/<author>/profile.json
data/authors/zhihu/<author>/raw/profile.json
```

支持字段：

```json
{
  "nickname": "你的ZombieMan",
  "avatar_url": "https://...",
  "headline": ""
}
```

如果没有 profile，前端用 author token 和 initials 兜底。头像不应成为使用 Web 的硬依赖。

抓取成功后，服务端应尝试把头像缓存到作者 raw 目录，并优先通过本地 API
提供头像；远程 `avatar_url` 继续保留为失败回退：

```text
data/authors/zhihu/<author>/raw/assets/avatar.*
```

## 作者库与添加作者

作者库是独立页面 `/authors`，不是聊天页里的大型下拉框。桌面端使用紧凑列表，
移动端改为紧凑信息块。每位作者展示：

- 头像、昵称和知乎用户名。
- 已抓取内容数量。
- 最后完成时间。
- 当前状态。
- 进入聊天和管理操作。

添加作者入口同时存在于作者切换器底部和作者库右上角。输入支持用户名与
`https://www.zhihu.com/people/<token>` 主页 URL，并复用 crawler 的
`parse_user_token()` 归一化。

添加向导：

```text
输入用户名或主页 URL
-> 服务端读取公开作者资料，展示头像、昵称、用户名、简介和主页链接
-> 用户确认默认抓取全部 answer/article/pin
-> 创建后台任务
-> 弹窗可关闭，任务继续
-> Markdown、build、index 全部成功后作者变为可聊天
```

已有作者再次添加时视为同步请求。旧索引继续服务，新材料和索引在 staging
目录构建成功后才替换正式目录。当前聊天不能被新作者任务强制切换。

## 作者后台任务

添加和同步作者是服务端持久化异步任务，不能在一个 HTTP 请求内串行完成。

第一版采用：

```text
FastAPI
-> SQLite 任务表
-> 单 Worker 队列
-> 复用现有 pf crawl / pf build / pf index
```

SQLite 路径：

```text
data/system/personaforge.sqlite3
```

同一时间只执行一个作者构建任务，避免多个 BGE-M3 索引任务争抢显存。聊天
服务保持可用；后台任务可能降低运行期间的吞吐，但不能阻塞页面导航和已有
会话。

任务阶段只报告真实状态和已完成数量，不伪造百分比：

```text
queued
resolving_profile
crawling
building
indexing
activating
ready
failed
cancelled
interrupted
```

网页刷新、关闭添加弹窗或切换作者不会终止任务。服务进程异常退出时，运行中
任务在下次启动后标记为 `interrupted`，由用户点击重试；第一版不伪装成逐篇
断点续传。

任务工作目录：

```text
data/authors/zhihu/<author>/staging/<job-id>/
```

新 raw 和 index 均先写入 staging。成功后再替换正式目录；失败时保留旧
ready 版本。成功任务清理 staging，失败任务保留有限排错信息。

抓取默认先走公开策略，公开入口受限时由 Worker 读取服务端本地登录态：

```text
data/auth/zhihu_storage_state.json
```

登录态不得通过 API 返回，不得进入 trace、任务日志、前端状态或 Git。登录态
失效时任务显示“等待管理员重新登录”或明确失败信息，由服务宿主机管理员重新
执行 `pf zhihu-login`。

作者材料、索引、Persona Pack 和建议问题是所有已登录用户共享的服务端资源；
会话、消息、生成任务和 Trace 必须按登录用户隔离。作者首次就绪不依赖 Persona
Pack 和建议问题。

### 作者任务 API

```text
POST /api/personas/preview
POST /api/author-jobs
GET  /api/author-jobs
GET  /api/author-jobs/{job_id}
POST /api/author-jobs/{job_id}/cancel
POST /api/author-jobs/{job_id}/retry
```

作者任务 API 只接收平台、用户名/主页 URL、抓取类型和可选数量限制，不接受
Cookie 内容或任意本地输出路径。

## 会话存储

当前会话、消息和生成任务统一保存在本地 SQLite：

```text
data/system/personaforge.sqlite3
```

旧版 `data/authors/zhihu/<author>/sessions/*.json` 在启动时幂等导入 SQLite，
原文件保留为只读备份。assistant 消息保存 `trace_id` 和 `turn_id`，因此历史回答
既能打开对应运行过程，也能恢复或重试未完成任务。

## 多轮会话 v1

本节既是已经冻结的多轮会话契约，也是当前实现说明。主要实现位于
`conversations.py`、`multiturn.py`、`chat_tasks.py`、`service.py` 和 `app.py`；
前端恢复与 Trace 展示位于 `web/src/App.tsx`。

### 总体边界

- 一个会话永久绑定一个作者，切换作者时切换到该作者自己的会话。
- 用户消息、生成消息和作者原始材料是三种不同数据，生成消息绝不能写回作者索引。
- 对话历史用于理解当前指代和保持交流连续性，不能证明作者观点或外部事实。
- 现有四路 Query Transform、BGE-M3 dense/sparse、两级 RRF、Parent top20 和
  Writer prompt variant 保持不变；多轮层只决定检索输入、是否执行或复用检索，
  以及哪些历史消息进入 Writer。
- 前端保持一种聊天形态，不增加“问答模式/对话模式”开关。回答深度由当前话语
  自动判断，用户明确要求“一句话”或“详细展开”时优先服从用户。

### 完整运行链路

```text
保存当前 user message 和 turn_run
-> 读取会话摘要与最近 3 个完整 Turn
-> Conversation-aware Turn Planner
-> 用 resolved_question 召回最多 2 个更早的相关 Turn
-> 根据 retrieval_policy 选择 new / reuse / none
-> new：进入现有 Tavily + 四路 Query Transform + RAG20
-> reuse：按 evidence_source_turn_id 重新装载该轮 Parent 全文
-> none：跳过联网、Query Transform 和作者材料检索
-> Writer Context Builder
-> 流式生成并保存 assistant message
-> 异步更新会话摘要和 Turn 向量
```

不同会话可以并行运行，同一个会话内严格串行。生成任务先持久化再执行，用户切换
会话或刷新页面后任务仍继续；进程重启时，未完成任务标为 `interrupted`，允许
在不重复插入 user message 的前提下重新生成。

生成完成后的会话回读必须继续携带 `owner_id`，不能退回 `local-user`。一旦 Turn 和
assistant 消息已经原子写为 `completed`，后续会话回读、Trace 或记忆维护失败均不得
把该 Turn 降级为 `failed`，更不能用收尾错误覆盖已经保存的完整回答。

### SQLite 数据模型

会话从作者目录下的 JSON 文件迁入现有数据库：

```text
data/system/personaforge.sqlite3
```

第一版继续复用项目已有的 `sqlite3 + Repository` 模式，不引入 ORM。核心实体：

- `conversations`：`id`、`owner_id`、`author`、标题、摘要、摘要覆盖位置和时间。
- `messages`：`id`、`conversation_id`、顺序、角色、正文、状态、`trace_id` 和时间。
- `turn_runs`：当前 user message、选中的 assistant attempt、Planner 结果、
  Parent IDs、运行状态和时间。
- `generation_attempts`：同一 user message 的多次生成结果及对应 `trace_id`。
- `turn_embeddings`：完整 Turn 的 BGE-M3 向量、模型名和模型版本。

`owner_id` 来自服务端 Session 对应的 `users.id`，不能接受前端自行传入的用户 ID。
旧 Session JSON 通过幂等迁移先导入 `local-user`；首次创建管理员时，系统在同一
SQLite 中把这些历史会话一次性认领给首位管理员。相同 `session_id` 不重复导入，
迁移验证完成前保留只读备份。

## 登录与用户隔离 v1

账号模块采用邀请制，不开放公共注册：

```text
数据库没有用户 -> Web 首屏创建第一位管理员并认领 local-user 历史
数据库已有用户 -> Web 首屏只允许登录
已有管理员 -> 侧栏“成员”抽屉创建协作者或管理员账号
```

成员抽屉仅对管理员显示。`协作者` 可共同使用 Chat、作者库和 Evaluate；`管理员`
额外可管理 Study 1 的实验、参与码、导出与成员账号。`pf user create <username>` 仍保留
为服务端恢复和自动化入口，但不再是正常协作流程的唯一入口。

管理 API 只允许管理员调用：

```text
GET  /api/admin/users
POST /api/admin/users
```

接口只返回用户名、显示名、角色和创建时间，绝不返回密码或密码哈希。新账号的会话与
长期记忆从创建时起独立；作者材料、索引和实验配置仍是团队共享资源。

密码只保存 Argon2 哈希。登录成功后由 FastAPI 生成高熵随机 Session Token，浏览器
通过 `HttpOnly + SameSite=Lax` Cookie 自动携带；SQLite 只保存 Token 的 SHA-256
摘要和过期时间。Session 默认 30 天、可由服务端撤销；HTTPS 下 Cookie 自动启用
`Secure`。同源 React + FastAPI 不采用 JWT，避免引入无法即时撤销的客户端身份状态。

SQLite 新增：

```text
users          用户名、显示名、角色、密码哈希
auth_sessions  用户、Token 摘要、创建/过期/最近访问时间
```

访问边界：

- `/health`、React 静态资源和登录状态接口公开。
- 作者库、添加作者、建议问题和作者任务仅对已登录用户开放，但内容在用户间共享。
- `conversations.owner_id` 隔离会话及其级联消息、Turn、摘要和向量。
- 读取、重试或订阅生成任务时，必须通过 `turn_runs -> conversations.owner_id` 校验。
- Trace 文件仍按作者落盘，但 API 只有在当前用户消息确实引用该 `trace_id` 时才返回。
- 越权资源统一返回 404，避免泄露其他用户的会话或任务是否存在。

Tailscale 只控制哪些设备可以访问服务，不能替代应用身份。多人共用同一个服务时，
仍必须登录，不能继续把所有访问者映射为 `local-user`。

删除会话时级联删除消息、Turn、摘要、向量和普通 Web trace。已经显式复制到
`data/eval/` 的冻结实验记录不受影响。

## 用户长期记忆 v1

用户长期记忆与会话摘要、作者 Persona Pack 是三种不同状态：

- 会话摘要只压缩当前会话，用于短期指代和连续性。
- 用户长期记忆跨会话、跨作者共享，但按登录用户 `owner_id` 严格隔离。
- Persona Pack 描述创作者稳定表达，只属于作者，不能写入用户记忆。

第一版不引入 LangGraph、Redis 或新的向量数据库。记忆正文、版本、来源和设置保存在
现有 `data/system/personaforge.sqlite3`，少量活跃记忆使用现有 BGE-M3 在进程内执行
dense+sparse 双路召回和 RRF 融合。表结构包括：

- `user_memories`：semantic / episodic / procedural 三类记忆、敏感级别、重要度、
  置信度、来源 message IDs、版本关系和软删除状态。
- `user_memory_embeddings`：BGE-M3 dense 与 sparse 表征缓存。
- `user_memory_settings`：每个用户的总开关和自动写入开关。
- `user_memory_checkpoints`：每个用户、每个会话已经完成长期记忆审查的消息序号；只有
  checkpoint 之后的新 Turn 会进入下一批窗口。

读取链路仅在 `grounded` 模式启用：

```text
当前 query
-> BGE-M3 从 active memories 召回最多 8 条候选
-> 现有 Turn Planner 在同一次 LLM 调用中选择最多 4 条 memory_ids
-> 只把选中内容作为“用户上下文”交给 Writer
```

`raw` 模式保持无长期记忆基线。用户记忆不能作为作者观点或外部事实；当前消息与记忆
冲突时当前消息优先。Query Transform 使用 Planner 完成必要指代消解后的问题，不把
全部用户记忆直接拼进作者 RAG query。

写入在回答保存完成后进入独立后台维护队列，不占用聊天 worker，不影响下一条回答，
也不使用定时轮询。每个会话维护
最多 3 个尚未审查的完整 Turn：未达到阈值时只返回 `deferred`，不调用记忆 LLM；累计
3 Turn，或 Planner 判断当前消息明确值得长期记忆、用户主动要求记住/忘记/纠正时，
立即刷新整个窗口。无新 Turn 时永远不会重复执行。

```text
checkpoint 之后最多 3 个完整 user + assistant Turn
-> 候选提取 LLM
-> 无候选：推进 checkpoint，跳过 Critic
-> 有候选：保守审查 LLM
-> 确定性证据、主语、疑问、敏感信息和 schema 校验
-> create / extend / replace / reject
-> 推进 checkpoint，窗口清零
```

关键安全边界：

- assistant 和创作者生成内容可以作为代词与对话动作的 `context_messages`，但永远不能
  出现在可信证据白名单中，也不能独立支持一条用户记忆；最终证据只能引用 user message。
- 疑问、假设和担忧不得提升为用户信念；担忧本身可以作为 episodic 事件保存。
- 必须区分用户与哥哥、朋友等第三方，不得把第三方经历归到用户本人。
- API key、密码、Cookie、token 永不进入候选；第三方财务事件只存必要概括，不保存
  金额、比例、余额和杠杆倍数。
- 自动修订以新版本 supersede 旧版本，保留审计链；兼容的新证据使用 `extend`，发生
  冲突、状态变化或用户纠正时使用 `replace`，以时间更近的用户证据为准，避免旧错误
  被拼回新版本。
- 删除单个会话不会自动删除已经独立形成的长期记忆；用户可在“我的记忆”中纠正、
  置顶、遗忘或全部清空。遗忘是软删除，后续召回只读取 active 状态。

记忆 API：

```text
GET    /api/memories
PATCH  /api/memories/{memory_id}
DELETE /api/memories/{memory_id}
DELETE /api/memories
GET    /api/memory-settings
PATCH  /api/memory-settings
```

Trace 新增 `user_memory_recall` 和异步 `memory_update.user_memory`。普通摘要 Trace 只保存
候选/采用 memory ID、操作类型、拒绝原因和耗时，不重复保存敏感正文；完整 Trace 也不得
保存凭据或原始精确敏感数值。

### 会话上下文选择

短会话不做无意义压缩：前 6 个已完成问答 Turn 全量进入上下文。超过该范围后，
Writer 上下文由三部分组成：

```text
结构化会话摘要
+ 最近 3 个完整 Turn
+ 最多 2 个语义相关的更早 Turn
```

Planner 先读取“摘要 + 最近 3 Turn + 当前消息”并生成 `resolved_question`，再用
`resolved_question` 对更早 Turn 做语义召回。语义单位是完整的
`user message + assistant message`，命中后把整轮原始角色消息交给 Writer。
BGE-M3 向量缓存在 SQLite，不为会话记忆新增 Qdrant collection。

满足以下任一条件后异步生成摘要：

```text
已完成超过 6 个问答 Turn
或尚未摘要的历史超过约 8000 tokens
```

摘要只覆盖最近 3 个完整 Turn 以前的内容，不把仍会以原文进入 Writer 的三轮重复压入
摘要。以后每累计 3 个可摘要的旧 Turn 再更新一次。摘要使用 DeepSeek V4 Flash、
`temperature=0` 和结构化 JSON，至少包含：

```json
{
  "topics": [],
  "entities": [],
  "user_requests": [],
  "assistant_previous_claims": [],
  "unresolved_references": []
}
```

`assistant_previous_claims` 必须表达为“此前 assistant 曾表示”，不得把生成观点
提升为作者真实立场。摘要是可丢弃的派生缓存；生成失败时退回原始历史。

### Conversation-aware Turn Planner

Turn Planner 扩展并取代当前单轮 Search Planner，因此普通 grounded 新问题仍是
“Planner + Background/Transform + Writer”三次 LLM 调用，不增加额外串行调用。
Planner 只能看对话上下文和当前输入，不能看 Persona Pack、作者材料或预测作者
立场。固定输出：

```json
{
  "turn_type": "new_topic|follow_up|explain_previous|casual|unclear",
  "resolved_question": "...",
  "retrieval_policy": "new|reuse|none",
  "evidence_source_turn_id": null,
  "needs_web": false,
  "search_queries": [],
  "response_depth": "brief|normal|deep",
  "clarification_focus": "",
  "memory_ids": [],
  "memory_write_policy": "defer|flush"
}
```

路由约束：

- `needs_web=true` 必须同时为 `retrieval_policy=new`。
- `reuse` 和 `none` 不执行 Tavily 与 Query Transform。
- `reuse` 必须给出当前会话中存在的 `evidence_source_turn_id`；不能验证时降级为
  `new`。
- `unclear` 仅在不同理解会改变检索和回答时触发，此时不执行检索，由 Writer
  根据 `clarification_focus` 用作者语气提出一句简短澄清。
- `follow_up` 表示同一主题中的新内容问题，包括新的判断、建议、原因和解决办法，
  必须使用 `new`；话题连续不等于材料仍然适用。
- `reuse` 仅用于解释、展开或论证已有 assistant 回答中的明确内容。
- `memory_write_policy` 默认 `defer`；只有明确的记忆操作，或当前消息披露了未来跨会话
  仍有价值的稳定偏好、个人事实和持续事件时才为 `flush`。普通问题不会触发提前写入。
- 压缩、改写、翻译、字数或格式纠错只需要已有对话，使用 `none`；闲聊也使用
  `none`。

### 多轮 Query Transform

`retrieval_policy=new` 时，现有 Background + Query Transform 节点只增加两个
输入：当前用户原话和 Planner 生成的 `resolved_question`。它还可以看到 Tavily
结果，但不再读取完整历史。四路含义保持不变：

- `literal_question`：最小补全后的独立问题，不再机械使用“那男的呢”等残缺原话。
- `event_background`：保留联网确认的实体、事件和客观事实。
- `mechanism_scene`：转成具体关系机制、行为动机和冲突场景。
- `colloquial_surface`：转成口语词、网络表达和短词组合。

Writer 不得看到 `resolved_question`、搜索 query 或四路 retrieval query，避免
技术改写隐含地替作者决定回答角度。

### Writer 消息契约

Writer 使用真实 Chat roles，而不是把历史拼成一段“对话材料”：

```text
system：身份 Prompt + Persona Pack + 信息优先级
system：会话摘要 + 客观背景 + 本轮作者 Parent 全文
user/assistant：选中的历史消息，保持原始顺序
user：当前用户原话，永远是最后一条
```

如果 provider 不接受多个 `system` message，adapter 可以在发送前按同样顺序合并，
但上层数据结构继续保持“稳定身份指令”和“动态参考上下文”分离。

信息冲突优先级：

```text
当前用户明确要求
> 当前题目的客观背景
> 当前 RAG20 作者原文
> Persona Pack
> 历史 assistant 回答
> 模型自身常识
```

历史 assistant 回答与新 RAG 冲突时，以新 RAG 为准。用户没有追问矛盾时直接给出
当前回答；用户明确询问“你之前不是说……”时才承认并解释变化。

`response_depth` 控制 prompt 与 `max_tokens`，但具体长度参考当前 RAG 排名前几篇
相似作者原文的长度分布，而不是给所有作者设置同一字符数。`brief` 明显短于相似
原文中位数，`normal` 接近中位数，`deep` 接近上四分位，同时设置防失控上限。

`none` 路由仍使用 Persona Pack 与最近对话保持作者感。没有 Persona Pack 时可以
装载最近一次成功 Turn 的 Parent 作为表达身份参考，但不能把它当作当前事实证据。

### Trace v2 扩展

多轮能力扩展现有 Trace v1，不另建平行 trace。保留现有字段并升级
`schema_version`，前端必须兼容旧记录缺少新增字段的情况：

```text
input
conversation_context
turn_planner
history_recall
query_understanding
retrieval
writer
generation
memory_update
```

新增节点记录摘要版本、最近消息 IDs、语义召回 Turn IDs 与分数、Planner JSON、
复用证据来源、回答深度和最终进入 Writer 的历史消息。它们只进入开发者模式，
不向普通用户展示。摘要更新在回答后异步执行，可以稍后补写 `memory_update`，
但不能改写已经冻结的生成输入和输出。

### 功能验收

已冻结并实现 12 条脚本化多轮用例，覆盖指代补全、继续展开、解释上一段、实质性
新问题、主动换话题、旧话题找回、需要联网的追问、模糊澄清、闲聊、长短控制、
刷新恢复和失败重试，测试入口为 `tests/test_multiturn_acceptance.py`。

第一轮验收不依赖 LLM Judge，优先验证可客观判断的行为：路由、保义改写、历史
选择、联网决策、证据复用、话题隔离、数据库恢复和任务幂等。作者相似性继续使用
现有生成评估体系单独衡量，不能用它掩盖多轮链路错误。

## Trace v0

Web 负责产生统一的 trace v0，而不是复用 CLI 的一次性 `--trace-path` 输出。每次 Web 回答有一个稳定的 `trace_id`：

```text
prepare_chat
-> 写入 status=prepared 的 trace
-> SSE meta 返回 trace_id
-> stream_answer
-> 写入 status=completed 或 failed 的 trace
```

Trace 结构按阶段分组：

- `input`：问题、persona、会话、query mode、writer variant、检索参数。
- `query_understanding`：路由、搜索词、来源、客观背景、4 路 retrieval query。
- `retrieval`：每一路 child hit 与最终 parent 聚合；只保存 parent 元数据和命中节点，不保存 parent 正文副本。
- `writer`：模板 variant、参与上下文的 parent 标题/ID、消息角色及长度；选择 `persona_pack` 时额外记录 Pack ID、SHA-256 和 claim 数量。
- `generation`：provider 名称、temperature、max tokens、耗时、输出字符数、状态和错误。

前端第一版不做独立评测后台。每条作者回答下面放一个低干扰的“查看过程”入口，打开后按阶段展示 trace；Parent 标题可在新标签页打开知乎原文，技术细节默认折叠。完整 prompt 临时预览、judge/rewrite trace 与跨运行对比属于后续阶段。

### 实时运行状态

回答开始前不能只显示省略号。后端在真正进入每个阶段前通过 SSE `status` 事件通知前端，普通界面只显示面向用户的客观动作，不展示模型推理过程：

```text
正在理解问题
正在查询相关背景       # 仅 Search Planner 判断确实需要 Tavily 时出现
正在整理检索线索
正在检索历史表达
正在准备回答
已完成检索，正在生成回答
```

等待状态使用独立于正式回答的作者行，带低干扰文字流光动画，不使用转圈加载器，也不显示实时秒数。首个生成 token 到达时，状态行消失，正式作者回答另起一行。回答完成后不保留普通用户可见的阶段摘要，只保留“查看过程”入口。

若 Tavily 失败，系统应记录错误到 trace，并展示“未获得额外背景，继续检索作者历史表达”，随后以无联网背景的链路继续回答；不能因为辅助背景服务失败而直接终止整题。

## Trace v1

Trace v1 是 Web 运行记录的统一事实来源。它不是模型思维链，也不保存 API Key、cookie 或登录态；它记录的是可复核的系统节点、输入输出摘要、资源消耗和降级结果。

```text
Search Planner
-> 可选 Tavily
-> Query Transform
-> Embedding
-> Dense / Sparse 召回
-> Parent RRF
-> Writer 上下文组装
-> 流式生成
```

每个节点有稳定字段：

```json
{
  "id": "generation",
  "label": "流式生成回答",
  "status": "completed",
  "order": 7,
  "started_offset_ms": 0,
  "duration_ms": 5690,
  "details": {},
  "usage": {
    "source": "provider",
    "prompt_tokens": 16813,
    "completion_tokens": 345,
    "total_tokens": 17158
  }
}
```

### Token 规则

- DeepSeek 的非流式和流式调用优先读取接口返回的 `usage`。
- 流式调用启用 `stream_options.include_usage`，记录输入、输出、总 token 和缓存命中/未命中 token。
- 没有 usage 的 provider 只能保存显式标注的 `estimated` 估算；前端不得把它显示成真实用量。
- 生成记录额外包含 `time_to_first_token_ms`、总耗时、输出字符数；不记录逐 token 的内部时间线。

### 保存等级与留存

- 默认 `summary`：保存 query、联网背景、检索路线、最终 Parent 元数据、writer 长度和节点指标；不重复写入完整 Parent 正文或完整 prompt。
- 开发者模式可以选择 `full`：仅保存在用户本机的 trace 中，额外保存 writer 完整 messages 和最终 Parent 全文，供后续 Judge、人工评审或可复现实验使用。
- 每位作者的普通 Web trace 最多保留 200 条，超过后删除最旧记录。`data/eval/` 下的评测产物不归这条规则管理，保持不可变。

### Web 展示与后续评测

- Trace 始终生成；开发者模式只控制“查看过程”入口和完整记录开关，不影响回答质量。
- 开发者抽屉按节点时间线展示阶段、耗时、token 来源、降级或错误，child hit 仍然折叠。
- 未来的 LLM-as-Judge、人工打分和 rewrite 不能覆盖原 trace；它们应以 `trace_id` 引用这次运行，并把评分、标注与派生回答放进独立评测记录。

## 建议问题

空状态建议问题不直接使用作者历史原题，避免点击后 RAG 直接召回原回答而变成复读。

当前策略：

```text
pf suggest <author>
-> 读取 index/parents.jsonl 的历史问题标题
-> 调 LLM 生成新的知乎式问题候选
-> 过滤完全重复、共享长短语、字符重合过高的问题
-> 写入 data/authors/zhihu/<author>/profile_suggestions.json
```

Web 只读取本地 `profile_suggestions.json`，不会在打开页面时偷偷调用 LLM。

API：

```text
GET /api/personas/{author}/suggestions
```

前端展示为开场白下方的轻量 chips。点击 chip 只填入输入框，不自动发送，避免误触触发生成成本。

下一阶段再做：

- LangGraph/trace 时间线。
- eval 面板。
- 多模型对比。
- session memory。

## 工作区与侧栏

顶部 `Chat / Evaluate` 是同一应用内的工作区切换，不共享同一套上下文侧栏：

- `Chat` 展示作者身份、历史会话、记忆和实验设置侧栏。
- `Evaluate` 展示评估集、完成进度和问题列表侧栏，不显示历史会话。
- 两种侧栏都可独立收起和恢复，状态分别保存在浏览器本地；切换工作区不会覆盖另一种侧栏状态。
- 作者头像窄栏始终保留，用于切换分身和进入作者库。

Evaluate 主区使用连续白色画布：顶部问题区和底部评分区只通过细边界分隔，中间的单篇待评内容使用克制的阅读框承载，不使用灰白相间的大色块。评分区必须明确展示当前判断问题，正文、标签和评分说明不得依赖低对比度小字传达关键信息。

候选内容不显示“候选材料”等自解释标签。`answer` 和 `article` 展示原始标题；`pin` 不把正文首句伪装成标题，统一显示“想法”。Chat 会话侧栏只显示当前作者名称和状态，不重复展示作者头像；全局作者头像栏负责切换身份，主区空状态头像负责角色呈现。

检索评估集选择器同时展示完整候选池和派生核心池，并在名称后显示候选总数。核心池与
完整池共享人工标签，但完成进度分别按各自候选集合计算；默认优先展示新生成的核心池，
避免把宽候选池误当成人工任务单。

### Gold-aware 检索报告

检索机器报告兼容两代标注。旧 `query_only_relevance_v1` 继续显示单一 `score`；
`gold_aware_dual_axis_v2` 必须提供“内容支撑”和“作者表达支撑”切换。切换后，指标、
候选相关性顺序、当前分数、证据和 Gold unit 映射都必须来自同一个轴，不能用内容轴
排序却展示表达轴分数。

V2 报告显示作者真实回答、冻结 Gold units、候选全文、两轴分数、候选证据、映射的
unit ID、复判次数和一致性状态。完整 Qrels 的 Recall 文案必须明确分母为时间切分前
全部可用作者语料；旧候选池只能称为六路候选并集。指标包括 Hit、MRR、graded nDCG、
Precision、Recall 和 MAP，支持 `K=1/3/5/10/20/30` 及 all/dev/test 视图。若存在
`comparison_v1_vs_v2.json`，报告展示改判总数和旧 0 分转为 V2 1/2 分的数量。

这些 Gold-aware 数据只允许从登录后的 Evaluate 接口读取，不得加入 Chat 请求、
在线 reranker、writer 上下文或参与者 Study 页面。部分完成的标签集必须显示
`completed/total`，其中 completed 只统计完成所需稳定性复评的标签。页面同时显示已有
首遍、待首遍和待复评数量，不得把只有第一遍结果的样本计入稳定 Qrels，也不得将
partial Test20 指标包装成正式结果。

候选池与 LLM 报告是两个独立资产。候选池选择器可以列出尚未建立 LLM 标签集的冻结池，
但必须明确显示“未标注”；选中此类候选池后仍保留报告侧栏和候选池切换入口，不能因为
没有标签集而卸载整个报告工作区。无标签池只显示空状态，不得伪造指标或逐题结果。

当前全量报告的正式完成状态为 `12690/12690`，30 道题均可按相关性顺序查看 423 篇
候选材料；前端必须读取标签目录内冻结的多 K 指标，而不是按当前页面可见候选重新计算。
两个轴均应显示 `eligible_author_corpus_before_cutoff` 的完整语料 Recall 范围，并提供
all/dev/test 与 `K=1/3/5/10/20/30` 切换。

Evaluate 顶部只保留 `RAG / Generate` 一级对象切换。RAG 内的“人工标注 / LLM 报告”
属于当前 RAG 工作区，固定放在左侧栏标题下方，不能与顶部一级控件争抢空间。两种视图
共用相同的侧栏宽度、高度、收起入口和内部滚动边界；切换视图时产品外壳不得跳动。

LLM 报告再拆成“指标总览”和“逐题标注”。指标总览只展示当前 all/dev/test 划分内的
逐题平均指标，必须明确写出“不是某一道单题”；逐题标注只展示当前问题、Gold 锚点和
全部候选材料，不重复显示聚合指标。问题列表和候选材料各自具有真实滚动容器，不能依赖
浏览器整页滚动，也不能因为父容器 `overflow: hidden` 而裁掉内容。

RAG 工作区新增第三个同级视图“评估任务”，用于初始化和维护多作者检索评估资产。任务
创建表单选择作者、labeler、dev/test 范围和预算；管理员可以创建 DeepSeek API 任务，
协作者可以创建 Codex handoff、下载任务包、导入结果和查看报告。普通 Study 参与者不能
访问该入口。

任务列表至少显示阶段、完成量、缓存命中/未命中 token、输出 token、估算成本、预算、
错误和恢复动作。状态包括 `queued / running / awaiting_codex / paused_budget / completed /
failed / cancelled / interrupted`。离开页面不影响任务；刷新或服务重启后从 SQLite 恢复。
Codex 导入由浏览器读取本地结果 JSON 后提交结构化请求，不要求用户在后端终端输入命令。
评估任务列表跟随左侧表单当前选择的作者展示；创建任务后立即显示任务卡和后台准备提示，
不能因为全局作者范围与表单作者不同而把刚创建的任务过滤掉。

任务完成后只发布不可变 dataset/pool/label 资产，现有“LLM 报告”继续从这些资产读取；
任务面板本身不直接重算指标，也不能把 partial 标签包装为正式结果。
若同一全量 pool 上存在 dev/test 两套 label set，报告的问题列表必须按 label manifest 的
`selected_splits` 过滤；例如 dev 任务完成后只显示 10 道 dev 题，不能把 test20 显示为
未完成。逐题报告按所选效用轴先展示 2 分、再展示 1 分和 0 分材料，并保留检索顺序供核对。

### Generate 方法对比总览

Generate 的“总览”不是把不同 dev 集或不同 Judge 版本混成一张榜单。前端先按
`dataset_id + dataset_sha256` 选择冻结数据集，再按 `prompt_version` 选择评估体系。
同一选择下展示所有自动发现的生成系统：已完成 Judge 的系统展示内容/风格/自然度分组
均值和 D1--D6 矩阵，未运行该版本 Judge 的系统保留在列表中并明确标记为“未运行”。
缺失值不得当作 0 分，也不得参与方法排序。

### 多作者评估作用域

评估资产必须按作者隔离。顶部作者头像在 Evaluate 工作区中改变当前作者作用域；人工
检索、LLM 检索报告、评估任务和 Generate 系统列表都只能读取该作者的资产。页面不得
把不同作者的候选池、Gold 回答或生成回答拼成一个逐题任务。没有作者元数据的历史资产
继续保留，但显示为“未归属/旧数据”，不能猜测归属，也不能参与跨作者 AB 对比。

作者头像栏提供“全部作者”入口。该入口只用于管理者查看跨作者汇总，不用于人工标注、
逐题 Generate 评分或跨作者 AB。跨作者汇总的主口径是宏平均：先在每个作者内部按兼容
的冻结数据集、切分、Top K 和 Judge/标签版本计算平均，再对作者平均；查询数加权的
微平均只能作为补充。不同协议、数据集哈希、标签版本或 Judge 版本不得直接平均。

后端列表接口支持可选的 `author` 查询参数，前端切换作者时重新请求而不是仅隐藏已加载
的卡片。Generate 的 `_compatible_pair` 必须同时检查冻结数据集、题目集合和作者身份；
任一系统缺失作者身份，或两者作者不同，都返回明确错误。这样作者切换不仅是视觉筛选，
也是数据边界和实验有效性的约束。

生成方法的显示名由 `manifest.writer_prompt` 映射为 MRPrompt、Persona Pack、Strong
Identity、Current 或 Writer Replay；未知方法回退到 manifest 的运行名。Judge 结果公开
携带 `prompt_version`，旧结果没有该字段时由后端回退到当前 Gold Judge 版本，以便旧
dev10 运行继续进入对比页面。

## Docker 静态资源契约

容器部署设置：

```text
PERSONAFORGE_WEB_DIST=/app/web/dist
```

`app.py` 优先读取该环境变量定位编译后的 React 静态资源；本地开发未设置时，
继续使用仓库内的 `web/dist`。这样生产镜像可以使用普通 wheel 安装，不依赖
editable install 的源码路径。
# 2026-08 生成报告交互补充

Generate 总览使用同一冻结 dev 集和 Judge 版本横向比较不同生成 run。每个 run 的方法
名称、方法标识、父版本、Prompt 版本、Writer 上下文篇数、温度、模型和代码 revision
来自 manifest 的非敏感元数据，不能只依赖本地运行目录名。

点击某个方法的 Judge 详情后，页面先显示方法概览和稳定性表，再显示 10 个独立题目链接。
每个链接带 `generationView`、`generationSystem`、`generationItem` 查询参数，刷新或复制
链接可以直接打开对应题目的详情，不要求从第一题滚动到目标题。单题详情显示作者原回答、
系统回答、六个维度的最终分、三次原始评分、证据和 Judge 理由，并保留上一题/下一题导航。
### RAG LLM 标注报告

RAG 评估保留“人工标注”和“LLM 报告”两种并列视图。人工视图继续使用按用户
稳定打乱、评分前隐藏 route/rank 的流程；机器视图读取候选池旁的共享
`llm_labels/<label_set>/` 资产，不在 Web 请求中调用 LLM。

“指标总览”支持 all30、dev10、test20 切换，并在 `K=1/3/5/10/20/30` 上展示六路
Hit、MRR、分级 nDCG、Precision、Recall 和 MAP 的逐题平均值。完整 Qrels 的 Recall
范围写为时间切分前全部可用作者语料；旧候选池报告才写为冻结六路候选并集。

“逐题标注”将当前问题的候选 parent 按 2 分、1 分、0 分和最佳路由排名排序，每张材料
卡显示完整正文、0/1/2 标签、短证据、理由、知乎链接和六路 rank；同时可以切换为
“检索顺序”核对各路召回位置。切换问题只改变单题标题、Gold 锚点和候选列表，不应出现
不随问题变化的聚合指标卡。

候选池 V0 与六路 V1 都允许被 Web 自动发现；问题列表随 split 切换，逐题报告始终按
相关性展示最有用材料。这个排序只服务于观察，不会修改冻结候选池、标签或生成链路。

新增 API：

```text
GET /api/evaluations/retrieval/pools/{pool_id}/llm-labels
GET /api/evaluations/retrieval/pools/{pool_id}/llm-labels/{label_set}
GET /api/evaluations/retrieval/pools/{pool_id}/llm-labels/{label_set}/queries/{item_id}
```

### 部署保护与输入边界

默认启动时开启轻量部署保护，目标是保护通过 Cloudflare Tunnel 或私有网络分享的
单进程实例，不把它当作分布式安全系统。普通 Chat 请求按登录用户限流：每 10 分钟最多
10 次、每个用户同时只能有 1 个排队或运行中的回答、全局最多 2 个活动回答；超限返回
`429` 和 `Retry-After`。活动数读取持久化的 `turn_runs` 状态，因此完成、失败、重启恢复
后都会自然释放容量。

登录失败按直接连接地址在 5 分钟内最多记录 8 次；成功登录会清除该地址的失败计数。
Chat 输入长度限制为 4000 个字符。保护层只包 Chat 和登录，不包离线评估、Judge、RAG
标注任务或作者入库任务，它们继续使用各自的队列、预算和并发控制。

`pf web` 和 `pf forge` 默认开启保护；仅在本机受控测试时使用
`--no-deployment-guards` 临时关闭。当前实现是进程内锁和内存时间窗，若以后扩展为多进程
或多实例，需要迁移到 Redis 等共享限流存储，不能直接复用这一版。

安全审计约束：SQLite 的用户输入必须通过参数绑定传入；动态 SQL 只能使用代码内固定的
字段白名单，不能把用户名、作者名、路径或请求参数直接拼进 SQL。返回统一安全响应头，
包括 `X-Content-Type-Options`、`X-Frame-Options`、`Referrer-Policy` 和受限的
`Permissions-Policy`；没有为了不破坏现有 React 静态资源而强行加入 CSP。
