# PersonaForge Open Source SPEC

这是从 `C:\PersonaForge` research 工作区拆出来的开源产品版规格。research 版继续保留实验、真实语料、消融和论文探索；开源版只保留可交付、可运行、可面试讲清楚的工程闭环。

## 0. 仓库分工

本仓库 `C:\PersonaForge-OpenSource` 是默认工程主线，承载面向简历、面试和开源交付的前后端、RAG、记忆、用户系统、部署、性能与工程质量建设。`C:\PersonaForge` 只用于论文、实验设计、评估研究和研究原型。用户未特别说明且任务偏工程时，默认在本仓库工作。

## 1. 产品定位

PersonaForge Open Source 是一个 local-first 的创作者风格 RAG 工程项目：

```text
输入一个公开创作者主页
-> 本地抓取公开内容为 Markdown
-> 本地切片、向量化、建索引
-> 用户填自己的 LLM API key
-> 本地 Web 里向这个创作者风格提问或生成回答
```

第一版不是中央托管网站。中央网站可以作为未来展示层，但不是 MVP。

选择 local-first 的原因：

- 数据、登录态、API key 都留在用户电脑上，开源边界更干净。
- 不需要一开始承担平台反爬、账号风控、在线成本和隐私责任。
- 面试时可以完整展示后端、RAG、Agent graph、前端、缓存、CLI、工程化和评估。
- 以后仍然可以在此基础上做 hosted demo，但 hosted demo 只作为产品化扩展。

## 2. 非目标

开源版第一阶段不做这些事：

- 不开源真实知乎作者语料、真实索引、auth state、`.env` 或实验数据。
- 不把当前 research repo 里的全部消融脚本搬过去。
- 不在中央服务器代用户长期保存爬取数据。
- 不做 hypothetical question 入库。
- 不把 Web 做成“在线输入任意知乎用户名后由服务器代抓”的托管服务。
- 不追求论文级作者相似度研究，两周内先工程收口。

## 3. 目标用户

第一用户是有一点技术背景的 AI/RAG 项目使用者：

- 会 clone repo。
- 会配置 Python 环境。
- 会填自己的 LLM API key。
- 想把某个公开创作者的内容变成本地可问答、可生成的风格助手。

第二用户是面试官和开源浏览者：

- 能快速跑 mock demo。
- 能看懂工程架构。
- 能看到真实 RAG/Agent/后端项目的边界和取舍。

## 4. MVP 形态

MVP 由 CLI 和本地 Web 组成。

CLI 负责：

- 初始化项目配置。
- 抓取公开创作者内容。
- 把 raw Markdown 构造成本地 index。
- 下载或检查 embedding model。
- 启动本地 Web 服务。

Web 负责：

- 选择已有本地 persona index。
- 聊天或生成回答。
- 展示检索来源。
- 展示基础运行状态。

Web 第一版不负责输入用户名和触发爬取。用户名、抓取和 build 都先在 CLI 完成。

Web 技术栈采用：

```text
FastAPI + SSE 后端
React/Vite + TypeScript 前端
```

选择原因：项目面向“后端 + AI 应用 / LLM/RAG 工程”求职叙事，FastAPI 更适合展示 API schema、服务层、流式响应和后续 trace/eval 扩展；React/Vite 让 MVP 更像真实前后端产品，但第一版不引入 UI 组件库。

## 5. 建议命令形态

最终希望支持：

```powershell
pf init
pf crawl zhihu wu-ren-jun-28
pf build wu-ren-jun-28 --quality fast
pf index wu-ren-jun-28 --embedding-device cuda
pf ask wu-ren-jun-28 "如何看待女生常说的配得感" --query-mode grounded --embedding-device cuda
pf prompt-pack wu-ren-jun-28 "如何看待女生常说的配得感" --writer-prompt strong_identity --out .tmp/chatgpt_prompt.md
pf web wu-ren-jun-28 --host 127.0.0.1 --port 8000
```

也提供一键命令：

```powershell
pf forge zhihu wu-ren-jun-28 --quality fast --port 8000
```

一键命令顺序调用 crawl/build/index/web，底层分步命令继续保留，方便排错和
面试讲解。默认抓取所有当前可访问内容；`--max-items` 用于 smoke，
`--skip-crawl/--skip-build/--skip-index` 用于显式复用已有产物，`--no-web`
用于只完成索引。任何 skip 对应的产物不存在时必须立即失败，不能假装完成。

`pf forge` 自动寻找 `<data-dir>/auth/zhihu_storage_state.json` 作为登录态 fallback。
登录态只留在用户本机。默认 Web host 为 `127.0.0.1`；Docker 或服务器部署时显式
使用 `--host 0.0.0.0`。

## 6. Sample Corpus

sample corpus 放在：

```text
samples/zhihu_mock_md/
```

格式必须最大限度贴近 crawler 输出，方便复用 ingest：

```text
samples/zhihu_mock_md/
  profile.json
  answer-100001-示例问题一.md
  answer-100002-示例问题二.md
  article-200001-示例文章一.md
  pin-300001-示例想法一.md
```

所有 sample 文本必须是自造文本，不使用真实博主语料，也不使用版权状态不清晰的大段文本。

正式 crawler 默认输出使用作者级目录：

```text
data/authors/zhihu/<author-token>/raw/
  profile.json
  manifest.jsonl
  answer/
  article/
  pin/
```

sample corpus 可以继续保持扁平目录，ingest 必须兼容它，方便仓库里放一个更轻的 mock demo。

Markdown 结构贴近当前 crawler 输出：

```markdown
---
source: "zhihu"
kind: "answer"
id: "100001"
title: "示例问题"
url: "https://example.local/answers/100001"
author_token: "mock-author"
created_at: "2026-01-01T00:00:00+00:00"
updated_at: "2026-01-01T00:00:00+00:00"
fetched_at: "2026-01-01T00:00:00+00:00"
question_id: "q-100001"
comment_count: 0
excerpt: "短摘录。"
---

# 示例问题

回答正文。
```

mock demo 仍需要 LLM key，因为 MVP 要展示真实生成链路。后续可以加 `--dry-run` 或固定假回答用于 CI，但不作为主要 demo。

## 7. Crawler 策略

知乎抓取策略：

- 默认游客公开抓取。
- 抓不到或内容不足时，提示用户配置本地登录态。
- 登录态只读取用户本机配置，不上传、不写入仓库。
- crawler 输出只落到用户本地 `data/authors/zhihu/{author_token}/raw/`。
- 必须有速率限制、失败重试、断点续爬和最小化日志。

第一版只支持 CLI 输入用户名：

```powershell
pf crawl zhihu <author-token>
```

Web 不直接做爬取入口。

## 8. Ingest 与索引

开源版第一阶段采用当前 research 里验证过更稳定的方向：

```text
raw Markdown
-> parent docs
-> title / lead / passage child nodes
-> BGE-M3 dense + sparse lexical weights
-> Qdrant local index
```

不做 hypothetical question 入库。原因：

- 现有实验显示数据源端生成 hypothetical question 收益不稳定。
- 它增加预处理成本和 LLM 依赖。
- query transform 放在查询端更灵活，也更容易调试。

`--quality` 设计：

- `fast`：默认。只用标题、开头、正文 passage 建索引，不调用 LLM 生成 representations。
- `full`：可选。允许生成 document summary，但仍不生成 hypothetical question。summary 只能作为补充 node，不能替代 passage。

passage 切片原则：

- 优先自然段。
- 自然段太短时合并。
- 超过目标长度时按长度切分。
- child node 只用于检索，最终上下文回填 parent 或 parent 片段。

## 9. Query 与 RAG

MVP 默认 RAG 路线：

```text
用户问题
-> Search Planner 判断是否需要 Tavily 补题目背景
-> Background + Query Transform 生成 4 路 retrieval query
-> 4 路 query 分别检索
-> BGE-M3 dense+sparse hybrid search
-> child_top_k = 100
-> parent 聚合
-> parent_top_k = 20
-> context-search prompt
-> LLM 生成
```

query transform 在查询端完成，不写回索引。

聚合时不能只看单个 child 的最高分，也要考虑同一个 parent 多个 child 频繁命中。默认使用 parent-level RRF 或等价稳定聚合策略。

生成 prompt 必须明确：

- 检索材料是作者历史材料，不是事实来源大全。
- 模型要先从材料里找对当前问题有用的线索。
- 无关材料只可作为语气和行文参考，不能强行纳入观点。
- 最终回答不能提“材料1/材料2”。

CLI 第一版生成入口：

```powershell
pf ask <author-token> "<question>" --query-mode grounded --embedding-device cuda
```

`pf ask` 做完整单轮链路：

```text
query understanding
-> optional Tavily
-> 4-way query transform
-> dense+sparse RRF retrieve
-> top20 parent context pack
-> writer
```

默认只打印回答正文。调试时用 `--trace-path` 保存 query、搜索、检索和 writer 输出摘要。

为了测试不同模型的“模型底色”差异，CLI 还提供 prompt pack 导出：

```powershell
pf prompt-pack <author-token> "<question>" --query-mode grounded --writer-prompt strong_identity --out .tmp/chatgpt_prompt.md
```

`prompt-pack` 复用同一套 query understanding、query transform、dense+sparse RRF 和 top20 parent 上下文，但不调用 writer LLM。它只把最终 writer messages 渲染成一份可复制到 ChatGPT 网页或其他模型网页的 Markdown。这个功能用于手动对比“相同 RAG + 相同 prompt，不同模型输出是否更像作者”，不是线上产品路径。

## 10. LLM Provider 抽象

第一版只抽象 chat completion，不抽象 embedding。

Provider 接口：

```text
generate(messages, options) -> text
stream(messages, options) -> chunks
json(messages, schema, options) -> object
```

首批 provider：

- DeepSeek
- OpenAI
- OpenRouter

Embedding 第一阶段固定使用本地 BGE-M3，避免 provider 维度过早膨胀。

配置来自 `.env` 或本地 config 文件，不能写入 git。

## 10.1 依赖与环境策略

`pyproject.toml` 是 Python 包、CLI 入口和依赖声明的主入口。

默认安装方式：

```powershell
pip install -e ".[dev]"
```

后续可以为了传统部署习惯补充 `requirements.txt`，但依赖源头必须仍然是 `pyproject.toml`，避免两份依赖长期不同步。

## 11. Graph v0

做 graph_v0，但只包当前 best 链路，不重写逻辑。

目标不是为了炫 LangGraph，而是为了可观测、可插拔、可面试讲解。

graph_v0 节点：

```text
Input
-> QueryTransform
-> Retrieve
-> ParentAggregate
-> ContextPack
-> Generate
-> StreamResponse
```

每个节点输出结构化 trace：

- 输入摘要
- 输出摘要
- 耗时
- token 估计
- 检索来源
- 错误信息

后续可以扩展节点：

- QueryUnderstanding
- WebGrounding
- EvidenceSelect
- Judge
- Rewrite
- SessionMemory

这些不是 MVP 必做。

## 12. 记忆设计

开源版要把“记忆”讲清楚，但不要第一版做复杂。

三层记忆：

- Corpus memory：作者历史内容索引，是主要长期记忆。
- Session memory：当前聊天历史，限制长度，避免污染作者风格。
- User config memory：用户配置、provider、index 路径、质量模式。

第一版先实现 corpus memory 和最小 session memory。

跨会话用户记忆开始前，Web 必须先完成账号隔离。当前采用首次网页创建管理员、
后续 `pf user create` 邀请建号的方式；作者语料和索引共享，会话、消息、生成任务
与 Trace 绑定服务端 Session 对应的 `user_id`。不开放公共注册，不接受前端传入的
`owner_id`。后续用户语义记忆必须沿用同一隔离边界。

## 13. Web MVP

Web 目标是可用，不是营销首页。

第一屏就是聊天界面：

- 左侧或顶部选择本地 persona。
- 中间是对话流。
- 输入框附近选择模型、RAG 参数、长度。
- 来源默认折叠。
- 可以展开查看检索 parent、child 命中和排名。

Web 不展示 raw corpus 全文，除非用户明确展开来源。

Web v0 只接已有本地 index，不触发 crawl/build/index。流式输出使用真 SSE：

```text
DeepSeek stream
-> FastAPI StreamingResponse
-> React fetch ReadableStream
```

每次回答生成本地 trace 档案；普通聊天界面只提供轻量“查看过程”入口，复杂的 trace 时间线和评估面板放到下一阶段。

## 13.1 Trace v0

Trace v0 是一次生成运行的结构化本地档案，不是零散终端日志。它服务于三件事：定位回答失败原因、为离线评估保留可复盘证据、为 Web 开发者模式提供数据。

每次 Web 回答对应一个 `trace_id`，写入：

```text
data/authors/zhihu/<author>/traces/<trace_id>.json
```

trace v0 记录：

- 输入问题、会话 ID、query mode、writer prompt variant 和检索参数。
- query understanding / Tavily grounding 的判断、搜索词、客观背景和 query transform 结果。
- 每一路 dense/sparse child 命中、最终 parent RRF 聚合结果及其命中来源。
- writer 的模板版本、上下文 parent 摘要、消息角色与长度统计；默认不重复保存完整 writer prompt 和 parent 正文。
- 模型名称、各阶段耗时、输出长度、完成或失败状态与错误信息。

trace 不记录 API key、cookie、登录态。会话的 assistant 消息仅保存 `trace_id`，让历史对话也能回到对应运行过程。

第一版 API：

```text
GET /api/personas/{author}/traces/{trace_id}
```

Web 在每条作者回答下提供低干扰的“查看过程”入口。普通用户只需看来源；开发者可展开看到理解、检索、聚合、writer 配置和耗时。离线 eval 后续复用相同 trace schema，而不是另起一份不兼容格式。

## 13.2 离线评测 v0

离线评测与正常 Web 知识库分开：Web 继续使用全量索引；eval 不重建索引，而是在每一路 Qdrant 检索中动态排除 holdout 之后的全部 parent。

第一版固定为严格时间切分：

```text
较早内容 -> train（可被检索）
接下来的 10 个合格 answer -> dev
最后的 20 个合格 answer -> test
```

合格 answer 必须有原问题标题、创建时间和至少 200 个字符正文。cutoff 从 dev 的第一条开始；所有时间不早于 cutoff 的 answer/article/pin 都进入排除名单，保证评测时不会用未来表达回答过去未知的问题。

第一版命令：

```powershell
pf eval prepare <author> --dev-size 10 --test-size 20
pf eval run <author> --dataset data/eval/<dataset>/dataset.jsonl --split dev --run-name baseline
```

每次 run 写入 `data/eval/<dataset>/runs/<run-name>/`，包含：

- `manifest.json`：数据集 hash、排除名单 hash、git revision、所有模型和检索参数。
- `runs.jsonl`：每题生成答案、query understanding、检索摘要和 trace。
- `items/01.md` 等：供人工阅读与填写评分。
- `summary.md`：本次运行的可读摘要。

v0 不接 LLM Judge，也不做 rewrite。开发期只跑 dev；只有候选方案稳定后才跑冻结的 test。每题开发期生成一次，最终候选再做 3 次重复并报告波动。

## 14. 安全与开源边界

`.gitignore` 必须覆盖：

```text
.env
data/raw/
data/index/
data/auth/
data/models/
data/eval/
*.sqlite
*.db
```

仓库允许提交：

```text
samples/
src/
tests/
docs/
README.md
SPEC.md
```

任何日志、trace、eval 输出都不能包含 API key、cookie、登录态或真实私有数据。

## 15. 两周工程路线

### Week 1: 开源骨架收口

1. 建立干净 repo 结构。
2. 写 sample corpus。
3. 移植最小 crawler raw format contract。
4. 移植 ingest fast path：title/lead/passage -> Qdrant。
5. 建立 provider 抽象：DeepSeek/OpenAI/OpenRouter。
6. 建立 CLI：`init/crawl/build/web/forge`。
7. 建立 `.gitignore` 和基础 tests。

### Week 2: 可演示闭环

1. 接入 graph_v0 trace。
2. 接入当前 best RAG20 context-search 生成链路。
3. Web 接已有本地 index 聊天。
4. README 写真实快速开始。
5. 加 mock demo smoke test。
6. 加检索来源展示。
7. 准备简历项目描述和面试讲解文档。

## 16. 面试叙事

这个项目面试时主打：

- Local-first RAG 产品工程。
- BGE-M3 dense+sparse hybrid retrieval。
- passage-level child nodes + parent aggregation。
- query-time transform 取代数据源端 hypothetical question。
- RAG20 context-search prompt 解决语义相关不等于立场有用的问题。
- LLM provider abstraction。
- LangGraph trace 化而不是为复杂而复杂。
- CLI + Web + crawler + ingest + generation + eval 的完整闭环。
- 人工评估发现系统优劣，并用 eval 反推工程决策。

## 17. 待定问题

这些问题暂不阻塞 MVP：

- 是否做 hosted demo。
- 是否引入 BM25 与 BGE-M3 lexical weights 做三路融合。
- 是否做 evidence selection。
- 是否做 judge/rewrite 在线闭环。
- 是否做作者相似度 HCI 论文实验。

原则：两周内不继续钻相似度研究，先把开源工程跑通。

## 18. Docker Web 运行时

第一版 Docker 镜像只承担已有 persona 索引的 Web 推理，不在容器中执行
知乎抓取，因此运行依赖安装 `index + web`，不安装 Playwright。

镜像采用两阶段构建：

```text
Node.js builder -> npm ci + React production build
Python runtime  -> 缓存第三方依赖层 + 安装当前源码 wheel + web/dist
```

运行契约：

- FastAPI 监听 `0.0.0.0:8000`，宿主机通过端口映射访问。
- `/app/data` 保存 persona、Qdrant、session 和 trace，必须使用外部数据卷。
- `/app/models` 保存 Hugging Face/BGE-M3 缓存，必须跨容器复用。
- `.env` 仅在容器启动时注入，绝不复制进镜像。
- 首版使用一个 FastAPI 进程，避免多个进程同时打开同一个 Qdrant local 目录。
- Web 静态资源通过 `PERSONAFORGE_WEB_DIST` 明确指定，避免依赖 editable install 的源码目录布局。
- Python 第三方依赖层仅由 `pyproject.toml` 失效；README 或业务源码变化只重装
  无依赖的 PersonaForge wheel，不能反复下载或安装 PyTorch。
- Node/Python 基础镜像固定 digest，避免浮动标签导致不可复现的构建变化。
- `compose.yaml` 把宿主机 `data/` 绑定到 `/app/data`；模型默认使用命名卷，
  也允许通过 `PERSONAFORGE_DOCKER_MODEL_SOURCE` 复用宿主机已有缓存。
- Compose 使用 `restart: unless-stopped`，但不会替代应用级健康检查和日志。
