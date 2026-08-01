# Web MVP 规格

`src/personaforge/web` 负责本地 Web 服务和前端静态资源托管。

## 当前目标

Web MVP 面向面试展示，目标是把现有 CLI RAG 链路变成一个可交互产品：

```text
选择本地 persona
-> 输入问题
-> FastAPI 调用当前 RAG20 + writer 链路
-> SSE 真流式返回回答
-> 回答完成后展示检索来源
```

## 不改动的边界

本阶段只做 Web，不改：

- crawler 的平台抓取实现。
- build/index 的文档与索引语义。
- query understanding / query transform 策略。
- dense+sparse retrieval 和 parent RRF 聚合。
- writer prompt 和生成策略。

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

React 前端：

```text
web/
  package.json
  index.html
  src/
    App.tsx
    api.ts
    main.tsx
    styles.css
```

## API v0

### `GET /health`

返回服务状态，并暴露本地 embedding 模型的预热状态：

- `idle`：尚未开始加载；
- `loading`：正在后台加载；
- `ready`：可以直接执行检索；
- `failed`：预热失败，保留错误摘要供本地排查。

当启动时已经存在至少一个可用作者索引，Web 服务会在后台预热共享的 BGE-M3
encoder。FastAPI 不等待模型加载即可先提供页面和 API，但首个聊天请求不再承担正常
情况下约几十秒的模型冷启动成本。预热只改变加载时机，不改变 query transform、召回
路线、排序或生成 prompt。

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
  "writer_prompt": "strong_identity",
  "parent_top_k": 20
}
```

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

Web 主界面采用“Claude 式轻聊天 + 左侧作者会话列表”的产品气质：

- 顶部不放大标题，不显示“正在以某作者回答”这类说明文案。
- 左侧顶部直接显示当前作者，应用品牌弱化到底部。
- 右侧空状态使用作者头像和一句开场白，形成轻 NPC 感。
- 作者回答采用 answer block，不用厚重聊天气泡承载长文。
- 用户消息保留右侧轻气泡。
- 每条正式消息都提供复制按钮。
- 高级参数默认折叠，后续 developer mode 再承载 trace、eval、中间过程。
- 色彩选择暖白/纸张感 + 墨色，避免通用 AI SaaS 蓝紫渐变模板感。

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
新增协作者 -> pf user create <username>
```

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

写入在回答保存完成后异步执行，不影响首 token 延迟：

```text
最近用户原话（不含 assistant）
-> 候选提取 LLM
-> 保守审查 LLM
-> 确定性证据、主语、疑问、敏感信息和 schema 校验
-> append / supersede / reject
```

关键安全边界：

- assistant 和创作者生成内容永远不是用户记忆来源。
- 疑问、假设和担忧不得提升为用户信念；担忧本身可以作为 episodic 事件保存。
- 必须区分用户与哥哥、朋友等第三方，不得把第三方经历归到用户本人。
- API key、密码、Cookie、token 永不进入候选；第三方财务事件只存必要概括，不保存
  金额、比例、余额和杠杆倍数。
- 自动修订以新版本 supersede 旧版本，保留审计链；人工纠正直接替换语义，避免旧错误
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

以后每新增 3 个完整 Turn 再更新一次。摘要使用 DeepSeek V4 Flash、
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
  "clarification_focus": ""
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

## Docker 静态资源契约

容器部署设置：

```text
PERSONAFORGE_WEB_DIST=/app/web/dist
```

`app.py` 优先读取该环境变量定位编译后的 React 静态资源；本地开发未设置时，
继续使用仓库内的 `web/dist`。这样生产镜像可以使用普通 wheel 安装，不依赖
editable install 的源码路径。
