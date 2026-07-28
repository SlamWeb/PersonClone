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

- crawler 抓取逻辑。
- build/index 入库逻辑。
- query understanding / query transform 策略。
- dense+sparse retrieval 和 parent RRF 聚合。
- writer prompt 和生成策略。

Web 只复用这些能力，不重新设计。

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

返回服务状态。

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

- 左侧 persona 选择，显示头像、知乎昵称、用户名、内容数量。
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

## 会话存储

第一版不用数据库，采用 local-first JSON 文件：

```text
data/authors/zhihu/<author>/sessions/<session_id>.json
```

会话只负责产品层历史记录，不改变当前 writer 的上下文策略。也就是说，继续同一个会话时，当前生成链路仍以“当前问题 + RAG 材料”为主；长期记忆和多轮上下文注入留到后续 developer mode / memory 模块。

assistant 消息额外保存 `trace_id`。这样历史会话里的任意一轮生成也可以打开对应运行过程。

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
