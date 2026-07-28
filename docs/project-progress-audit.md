# PersonaForge 项目进展审计

审计日期：2026-07-27

## 一句话结论

PersonaForge 已经不是“只有想法或脚手架”的项目：核心的 local-first 创作者 Persona RAG 闭环、Web 聊天、可观测 Trace 和无泄漏离线评测都已有真实实现，并通过当前测试与前端构建验证。

如果把“核心 MVP 能力”看作 100%，当前大约完成 80%；如果按“可以放心放到简历并让陌生人顺利复现”的发布标准衡量，当前大约完成 65%。差距主要不在 RAG 主链，而在多 Provider、graph_v0 封装、一键命令、README 复现路径和最终展示材料。

## 审计依据

- Git 历史共有 3 个已提交里程碑：
  - `6429e24`（2026-07-10）：Initial PersonaForge open-source scaffold
  - `c082c4c`（2026-07-13）：Build local web chat MVP
  - `529c151`（2026-07-17）：Add trace timeline and leak-safe evaluation
- 文件创建时间显示，实际开发从 2026-07-05 开始；核心 RAG 文件集中形成于 7 月 6 日至 10 日，Web 文件集中形成于 7 月 12 日，Trace/Eval 集中形成于 7 月 13 日至 14 日。
- 当前工作区有 9 个已跟踪文件被修改，改动集中在 writer prompt、来源公开链接和 Web 展示；另有未跟踪的 `run.txt`。
- 2026-07-27 验证结果：
  - Python：`55 passed in 8.66s`
  - Web：`tsc && vite build` 成功，1582 个模块完成构建

## 当前完成度

| 模块 | 状态 | 实际做到的程度 |
|---|---|---|
| 产品边界与规格 | 已完成 | 根级和模块级 `SPEC.md` 已明确 local-first、开源边界、输入输出合同和非目标。 |
| Sample corpus | 已完成 | 有自造的 answer/article/pin 示例语料与 profile，可用于离线开发。 |
| Zhihu crawler | 已完成 v0 | 支持公开 API、浏览器 fallback、本地登录态、Markdown/profile/manifest 输出和多作者目录。 |
| Ingest | 已完成 v0 | Markdown 可解析为 parent docs，再构造 title/lead/passage child nodes，并产出可审计 JSONL。 |
| 向量索引 | 已完成 v0 | BGE-M3 dense+sparse 表征与本地 Qdrant collection 已实现，测试使用 fake encoder/client 避免 CI 下载模型。 |
| Query/Retrieval | 已完成 v0 | 支持 query understanding、可选 Tavily、4 路 query transform、dense+sparse 检索和 parent RRF 聚合。 |
| Persona writer | 已完成 v0 | 支持 `current` 与 `strong_identity` 两套 prompt、RAG20 上下文和 prompt-pack 导出。 |
| CLI 主路径 | 基本完成 | `init/crawl/build/index/retrieve/ask/prompt-pack/suggest/eval/web` 已接入；`forge` 仍是占位。 |
| Local Web chat | 已完成 v0 | FastAPI + SSE 真流式、React/Vite UI、persona 选择、本地会话、来源展示与高级设置已实现。 |
| Trace | 已完成 v1 | Web 运行记录包含阶段、耗时、token usage、检索命中、降级和错误，并有前端时间线抽屉。 |
| Offline eval | 已完成 v0 | 严格时间切分、动态排除未来 parent、运行 manifest、逐题产物和泄漏审计已实现。 |
| Provider abstraction | 部分完成 | 已有 `JsonChatClient` 协议和 DeepSeek 实现；OpenAI、OpenRouter 实现尚未落地。 |
| graph_v0 | 部分完成 | 有结构化 Trace 和阶段式服务编排，但没有独立 graph 模块，也没有实际 LangGraph/StateGraph 封装。 |
| README quickstart | 部分完成 | 能说明安装、Web 和 eval，但没有一条从 sample corpus 到可聊天结果的完整、低摩擦复现路径。 |
| 简历/面试材料 | 未完成 | 规格里计划了简历描述与面试讲解文档，仓库里尚未形成最终版本。 |

## 真实时间线

### 2026-07-05：产品规格与开源骨架

- 建立根级 `SPEC.md`、`AGENTS.md`、`pyproject.toml`、`.gitignore`。
- 明确 research workspace 与 open-source product repo 的边界。
- 创建 sample corpus、Python package、CLI 入口和 crawler 模块。

### 2026-07-06 至 07-10：核心 RAG 链路

- 完成 raw Markdown loader、parent/child 数据模型和自然段切片。
- 完成 BGE-M3 dense+sparse 适配、本地 Qdrant 索引和检索。
- 加入 query understanding、可选联网背景、4 路 query transform 和 parent RRF。
- 完成 persona writer、`ask` 与 `prompt-pack`。
- 7 月 10 日形成第一个提交：开源工程骨架已经包含大部分后端主链，而非空壳。

### 2026-07-12 至 07-13：Local Web MVP

- 新增 FastAPI 服务、SSE 流式协议和 React/Vite 前端。
- 支持 persona 选择、本地会话、suggested questions、来源折叠与开发/生产启动方式。
- 7 月 13 日形成第二个提交：从 CLI/RAG 工程升级为可交互产品。

### 2026-07-13 至 07-17：可观测性与评测

- Trace 从终端日志升级为结构化本地运行档案。
- 前端增加运行阶段、耗时、token usage、检索 parent/child 和错误降级展示。
- 增加严格时间 holdout 的 leak-safe eval，动态排除 cutoff 之后的 answer/article/pin。
- 7 月 17 日形成第三个提交：项目具备“定位问题—保留证据—离线复盘”的工程闭环。

### 2026-07-17：未提交的体验收口

- writer prompt 增加引号使用约束，用于减少模型生成的标签化和 AI 味表达。
- Web 来源从内部 path/parent ID 改成可点击的知乎公开链接。
- 为旧 session/trace 增加读取时 URL 补全，避免破坏已有本地数据。
- 这些改动已有测试覆盖，但仍在工作区中，尚未提交。

### 2026-07-18 至 07-27：开发暂停 / 当前复盘点

- 已跟踪源码的最后修改时间停在 7 月 17 日。
- 当前最合适的动作不是继续堆新 RAG 实验，而是完成发布收口和简历叙事。

## 与原两周路线的对照

### Week 1：开源骨架收口

- 已完成：repo 结构、sample corpus、crawler、ingest fast path、Qdrant、CLI 基础命令、gitignore、tests。
- 部分完成：Provider 有协议和 DeepSeek，但没有 OpenAI/OpenRouter。
- 未完成：`pf forge` 一键链路仍是占位。

### Week 2：可演示闭环

- 已完成：RAG20 生成链路、Web chat、来源展示、Trace、mock 数据、基础 README。
- 超出原计划：严格时间切分的 leak-safe eval 已落地。
- 部分完成：graph_v0 有 Trace/阶段编排，没有独立图封装。
- 未完成：简历项目描述、面试讲解文档、陌生用户端到端 smoke quickstart。

## 简历发布前的建议顺序

1. 清理并提交当前 7 月 17 日的 prompt/source URL 改动；确认 `run.txt` 不进入 Git。
2. 增加一条真正可复制的 sample quickstart：sample build → index → ask/Web，并说明模型下载与 API key。
3. 在“实现 graph_v0”与“把简历措辞改成 traceable staged pipeline”之间做明确选择，避免把 Trace 编排说成 LangGraph。
4. 补 OpenAI/OpenRouter provider，或把 README/SPEC 的首批 Provider 承诺收缩到 DeepSeek。
5. 实现 `pf forge`，或从对外命令和文档中移除占位承诺。
6. 增加一份 `docs/INTERVIEW_GUIDE.md`：项目问题、关键取舍、工程难点、指标与下一步。
7. 准备 2 至 3 张截图：聊天主界面、Trace 时间线、Eval 产物；这三张最能支撑简历叙事。

## 当前风险

- `run.txt` 含本机 Conda 路径、研究仓库模型路径和真实作者 token，不应提交到开源仓库。
- 默认 Conda 环境里仍装着旧版 PersonaForge；不设置 `PYTHONPATH=src` 时，CLI 可能运行旧代码。发布前应重新 `pip install -e ".[web,index,dev]"` 并验证 `pf --help`。
- README 的 `pf web mock-columnist` 不是完整的首次运行路径；陌生用户仍需要自行猜测如何从 sample 生成 Qdrant index。
- “多 Provider”和“graph_v0”当前规格强于实现，简历表述需要与代码对齐。

