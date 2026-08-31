# `/api/chat/stream` 并发与取消审计

审计日期：2026-08-31。范围仅限 PersonClone/PersonaForge 当前仓库，不读取 CreatorOS
内部状态。

## 结论

改造前并非完全串行：`ChatTaskManager` 有两个后台线程，因此两个 turn 可以并行运行；但
`/api/chat/stream` 使用同步 SSE 生成器轮询 SQLite，worker 数硬编码为 2，客户端断开后
后台生成仍继续。DeepSeek 使用 `urllib` 同步流，无法由 ASGI cancellation 立即关闭。

改造后，交互式流使用 async generator。默认 DeepSeek JSON/stream 和 Tavily 请求走
`httpx.AsyncClient`；同步领域编排、SQLite、本地 BGE 和 embedded Qdrant 离开 FastAPI
event loop。一个进程级并发闸门同时约束交互式流和持久化 turn worker，超额请求排队。

HTTP 方法、URL、Cookie 登录态、请求字段和既有 SSE payload 未变。`meta`、`token`、
`error`、`done` 继续作为核心事件；当前前端已消费的 `accepted`、`status` 兼容事件也保留。

## 实际调用链与执行分类

| 环节 | 实际实现 | 分类 | 并发/资源边界 |
|---|---|---|---|
| 路由与登录态 | FastAPI 同步依赖读取 Cookie，创建持久化 turn | sync I/O（FastAPI threadpool） | SQLite 每次操作独立连接，WAL；同一 conversation 仍拒绝重叠 turn |
| SSE | `async_chat_stream_events()` 交给 `StreamingResponse` | native async generator | 每个请求的 answer、sources、trace_id、author 都是局部变量 |
| 并发准入 | `ChatConcurrencyLimiter` | async queue + thread semaphore | `PERSONAFORGE_CHAT_MAX_CONCURRENCY`，默认 2；超限排队，不返回 429 |
| 会话/记忆读取 | `ConversationStore` / `UserMemoryStore` | sync I/O via preparation worker | 不占用 event loop；账号/作者边界由数据库查询保持 |
| Turn planner / query transform | 既有同步解析函数 + `_LoopJsonClient` | native async HTTP，解析在 worker | provider usage 放在 request-local proxy，不读取共享 `last_usage` |
| Tavily（需要时） | `TavilySearchClient.asearch_many()` | native async HTTP | 逐请求 client；取消会 cancel in-flight future 并退出 client context |
| BGE-M3 | `BGEM3FlagModel.encode()` | CPU/GPU intensive in worker | 单个共享模型由 `_SynchronizedEncoder` 的 `RLock` 串行保护；安全但会成为吞吐瓶颈 |
| Qdrant 检索 | embedded `QdrantClient(path=...)` | sync I/O/计算 in worker | 每次检索创建独立 client，并在 `finally` 关闭；不同作者不共享 client |
| Parent 读取/RRF | JSONL 读取与 Python 聚合 | sync I/O + CPU in worker | request-local ranking/result；不共享 sources 列表 |
| Prompt 构建 | `build_writer_messages()` | sync CPU（轻量，随 preparation 在 worker） | messages 与 persona/narrative 对象只属于当前请求 |
| Writer stream | `DeepSeekJsonClient.astream_text_with_usage()` | native async HTTP stream | `async with` 管理 client/response；取消时关闭连接；usage callback 为请求局部 |
| token/turn/trace 落盘 | `ConversationStore`、trace writer | sync I/O via `asyncio.to_thread` | trace 文件按随机 trace_id 隔离；SQLite 写入按 turn_id 隔离 |
| 完成后记忆维护 | `ChatTaskManager.schedule_maintenance()` | 独立 maintenance thread | 在 `done` 持久化后执行，不改变 SSE 完成语义 |

没有使用 `AsyncQdrantClient`：当前索引是本地 embedded path，检索函数把同步 BGE、Qdrant
查询、RRF 和 parent JSON 装成一个领域操作。把整个操作放入受限 worker 可以保证 event
loop 不被 GPU/CPU 或文件 I/O 阻塞；仅替换 Qdrant client 不会让 BGE 编码异步，反而会
复制检索编排。远程 Qdrant 部署时再拆为 async client 是合理的后续边界。

## 取消与错误隔离

- ASGI 取消会进入 async generator 的 `CancelledError` 分支。
- 原生 async DeepSeek/Tavily response context 被关闭；preparation worker 收到 cooperative
  cancellation，正在执行的 async HTTP future 被 cancel。
- 已生成的 partial answer 写回 turn，turn 标记为 `interrupted`，不会伪造 `done` 或向已断开的
  socket 发送 `error`。
- 成功请求先持久化 answer、sources、trace，再发送 `done`。
- 一个请求的异常只把自己的 turn/trace 标记失败并发送自己的 `error`，不会取消同批其他请求。

同步第三方 provider（没有 `astream_text*`）仍有兼容 fallback，会用线程桥接；默认 DeepSeek
路径已是原生 async。自定义同步 provider 的网络调用只能在其下一次返回控制权后关闭，这是
该扩展点的已知限制。

## 真实串行/并发基准

环境：本地 RTX 2060、已预热 BGE-M3、两个真实本地作者索引、真实 DeepSeek stream；请求均为
`query_mode=raw`、`writer_prompt=strong_identity`、`parent_top_k=5`、`max_tokens=200`。
索引复制到隔离临时目录，基准未写入现有作者的会话或 trace。时间为各轮开始后的相对秒数。

### 串行

| 作者 | 开始 | 首 token | 完成 | 字符数 | trace_id |
|---|---:|---:|---:|---:|---|
| `an-ling-91` | 0.000 | 2.934 | 5.748 | 368 | `trace-20260830-220005-ecbd1818` |
| `mr-dang-77` | 5.797 | 8.164 | 10.969 | 309 | `trace-20260830-220011-e937956b` |

串行总耗时：`11.027 s`。

### 并发

| 作者 | 开始 | 首 token | 完成 | 字符数 | trace_id |
|---|---:|---:|---:|---:|---|
| `an-ling-91` | 0.000 | 2.056 | 5.041 | 363 | `trace-20260830-220016-d00052f3` |
| `mr-dang-77` | 0.039 | 2.051 | 4.851 | 340 | `trace-20260830-220016-4775444a` |

并发总耗时：`5.091 s`；节省比例：`53.84%`。四个请求均收到 `done`；同一并发轮中的
author、answer、trace_id 未串线。

## 已知瓶颈

1. BGE-M3 单实例加锁，多个请求的 embedding 阶段串行。这是当前显存安全边界。
2. embedded Qdrant 和 parent JSON 仍是同步本地操作，但已离开 event loop。
3. SQLite token checkpoint 会产生写放大，因此继续按字符数/时间批量 flush，而 SSE token
   仍逐块发送。
4. 同步自定义 LLM provider 只能 best-effort 取消；默认 DeepSeek 不受此限制。
