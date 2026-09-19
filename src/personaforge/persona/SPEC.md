# Persona 生成规格

`src/personaforge/persona` 负责把检索到的作者历史表达和题目背景组织成 writer prompt，并调用 LLM 生成回答。

本模块不负责爬取、切片、向量化和 Qdrant 检索。

## v0 职责

```text
用户问题
客观题目背景（可为空）
检索候选 parent Top20（默认）
-> writer context Top20 或实验性 Top5
-> context pack
-> writer prompt
-> LLM answer
```

## 设计边界

- writer 可以看到题目背景，但题目背景只解释事件/梗/概念，不代表作者立场。
- writer 可以看到作者历史表达全文，用于判断观点、切入角度、论证方式和语言风格。
- writer 不应该在最终回答里提“材料1/材料2/样本/检索结果/历史表达”。
- writer 不做引用回答，不输出来源列表。
- 来源、排名、child 命中只进入 trace，不进入最终文本。

## 当前 prompt 策略

当前保留四套 writer prompt，方便自用和实验对比：

- `current`：当前调过的反 AI / 反 advice / 反契约训诫 prompt，默认使用，保证现有自用效果不被覆盖。
- `strong_identity`：通用强身份沉浸 prompt，不写任何特定作者词汇，测试“RAG20 + 强模型是否能自行归纳作者表达身份”。
- `persona_pack`：在 `strong_identity` 的同一基础提示词上增加证据化作者画像；RAG20、query understanding、query transform 和生成参数均保持不变，用于隔离检验 Persona Pack 的增益。
- `mrprompt`：独立的 Narrative Schema 分支；按 Anchoring、Selecting、Bounding、Enacting 四步使用场景化长期记忆。它不改变 RAG、query understanding、query transform 和生成参数。当前 Web 对具备已验证 Narrative Schema 的作者默认优先选择该分支；缺少 Schema 时回退到 Persona Pack 或 Strong Identity。
- `rag_magic_if`：Magic If v1，保留旧版提示词，作为历史运行的可复现基线。
- `rag_magic_if_v2`：通用的样本条件式 Magic If，不写作者特定口癖或固定结尾规则；只要求模型从当前相关历史回答中迁移表达分布，作为新一轮生成质量对照方法。

CLI 切换方式：

```powershell
pf ask <author> "<question>" --writer-prompt current
pf ask <author> "<question>" --writer-prompt strong_identity
pf ask <author> "<question>" --writer-prompt persona_pack
pf prompt-pack <author> "<question>" --writer-prompt strong_identity --out .tmp/chatgpt_prompt.md
pf ask <author> "<question>" --writer-prompt mrprompt --narrative-schema-path <author>/narrative_schema.json
```

`prompt-pack` 用于模型差异手测。它复用检索和上下文打包，但不调用 writer LLM，只把 `build_writer_messages(...)` 的结果渲染成一份可粘贴到 ChatGPT 网页的 Markdown。这样可以比较“同样 RAG20 + 同样 prompt”下，不同模型的表达底色差异。

RAG 数量是运行配置，不复制成两套 prompt：检索仍先保留 parent Top20，随后由
`writer_context_top_k` 决定送给 writer 的数量。默认是 20；RAG5 对照使用 5。
因此 RAG5/RAG20 只改变 writer 可见的作者历史表达数量，不改变召回、query transform、
Narrative Schema 或生成参数。作者当前对话的最近消息和更早摘要继续由现有会话记忆负责，
本模块不再额外创建一个 STM 子系统。

`current` 策略：

- 像该创作者回答当前知乎问题。
- 不要写成 AI 分析文、课堂讲解、总分总作文。
- 不要写成情感课、行动建议、人生指导或契约训诫。
- 允许使用“你”做口语化推演，但不要进入 advice mode。
- 优先解释现象背后的机制，不要把回答写成道德审判或解决方案。
- 允许短句、跳跃、突然判断、口语化表达。
- 优先学习历史表达里的观点结构和切入方式。
- 无关历史表达只作为语气参考，不能强行塞进答案。
- 不要说“根据材料”“材料里”“历史表达中”。

反例约束：

- 错误类型：把回答写成“交易、合同、条款、甲乙方、谁该承担后果”的契约训诫。
- 错误原因：这会把创作者写成情感导师或契约论老师。
- 更好的方向：解释为什么当事人会产生这种感觉，以及这种感觉背后的关系机制。
- 不要复用反例里的说法。

`strong_identity` 策略：

- 任务不是“模仿文风”或“总结风格”，而是接管创作者的公开表达身份。
- 从 RAG20 中内部判断该创作者通常抓什么矛盾、采取什么表达形态、句子和段落节奏如何。
- 如果历史表达显示创作者常给建议，就给建议；常吐槽，就吐槽；常短评，就短评；常长文，就长文。
- 保留创作者表达中的不平衡、偏执、跳跃、重复、粗糙、尖锐或突然判断，不自动修成更礼貌、更中立、更完整、更有条理的 AI 文。
- 标点同样从历史表达中归纳。除非多篇历史表达明确显示引号是作者的高频特征，否则回答默认不用、最多使用一处；禁止用引号强调普通概念、制造标签、代替解释、改写题意或模拟人物内心话，必要的原话引用除外。
- 只输出最终回答正文，不描述风格，不输出分析过程。

## 证据化 Persona Pack

每位作者可以在作者目录保存一份本地资产：

```text
data/authors/zhihu/<author>/persona_pack.json
```

Pack 分为四层：

- `response_strategy`（可选但推荐）：作者把问题当作待完成任务还是表达入口，以及常见的开篇转向、回答动作和收尾动作。
- `worldview`：跨问题较稳定的价值判断、因果归因和观察框架。
- `reasoning`：常见切入、机制抽象、举例、推进和收尾动作。
- `voice`：词汇语域、人格姿态、句法节奏和口语表达倾向。

每条 claim 必须包含：

- 稳定 `claim_id`、置信度和适用主题。
- `activation_condition`：什么问题下才值得激活。
- `avoid_overapplication`：如何避免把画像机械套到无关问题。
- 至少一条 `doc_id + excerpt` 原文证据。

`pack.py` 加载 Pack 时会把每条摘录与当前作者 `index/parents.jsonl` 做逐字核验。缺失文档、改写证据、重复 claim ID 或非法置信度均直接失败，不允许把模型臆测静默送入 writer。

生成时的优先级是：

```text
当前问题与客观背景
-> 当前 RAG20 作者原文
-> Persona Pack 中有证据的稳定倾向
-> 模型自身常识
```

Pack 是概率性身份先验，不是写作清单。每次只应激活与问题相关的少量倾向，不能拼贴证据句、堆口癖或把两性主题立场迁移到无关领域。

`response_strategy` 会在其他画像前渲染，但只提供一条紧凑边界：是否直接回答、
是否转向、是否给建议以及何时结束，都优先跟随当前 RAG 中最相似的作者原文。
它不要求 Writer 枚举或分类回应动作，也不展示证据摘录，避免把自然写作变成元规划
任务。作者原文若借题发挥可以照做，但 Pack 不能授权模型凭自身常识发明新转向。

2026-07-28 的首次回应策略消融将四条长 claim 与显式动作分类加入 Pack，
在冻结 `temporal_dev10_v0` 上六维 Gold Judge 均未提高，且 D6 从 4.3
降至 3.8。该 Pack 仅保留为实验资产，没有晋升为作者默认 Pack。代码继续支持
可选 `response_strategy`。

同日的 V3 把它缩成以同类 RAG 回应形态为准的一条短约束，并通过 Writer replay
冻结了原问题、客观背景和 20 篇 parent 顺序。V3 的六维均分仍从 V1 的 3.63
降至 3.48，D6 从 4.3 降至 3.7，因此同样不晋升。当前不再继续增强全局回应策略
prompt；如要研究该构念，应改做由题目级同类 RAG 证据触发的局部控制。

Writer replay 必须复用基线每题原有的客观背景、20 篇 parent、parent 顺序和生成
参数，不重新调用 query understanding、query transform、embedding 或 Qdrant。
因此回应策略实验只改变 Persona Pack 和一次 Writer 采样；生成温度带来的随机性
仍需在结论中单独声明。

时间切分评测使用 `persona_pack` 时，Runner 会额外检查 Pack 的所有证据文档都不在 `excluded_parent_ids` 中。这样检索和画像两条路径都不能偷看 holdout。

如果评测使用的 `--index-dir` 不在作者目录下，可以显式传入：

```powershell
pf eval run <author> --dataset <dataset.jsonl> --run-name <name> `
  --writer-prompt persona_pack --persona-pack-path <persona_pack.json>
```

## Narrative Schema（新增实验分支）

每位作者可以在作者目录保存一份：

```text
data/authors/zhihu/<author>/narrative_schema.json
```

首版 schema 参考 Memory-Driven Role-Playing 的叙事记忆组织方式，但不复制原论文的评测集或模型。它把长期记忆拆成：

- `identity`：公开身份锚点和知识边界。
- `global_summary`：跨主题的简短叙事概括。
- `core_traits`：少量稳定的观察姿态，不是口癖清单。
- `scene_facets`：按触发线索组织的场景记忆；每个 facet 记录情境、思考方式、行为动作、表达信号和边界。
- `source_evidence`：审计用的 `claim_id + doc_id + excerpt`，必须能在当前 `parents.jsonl` 中逐字找到。
- `generation_policy`：选择、边界和表现规则。

`narrative.py` 负责解析、哈希、证据核验和 writer 渲染。渲染时不会把文档 ID 或证据摘录送给 writer，避免模型拼贴原句；证据只用于审计和时间切分防泄漏。

当前 `wu-ren-jun-28/narrative_schema.json` 是从已审核的 `persona_pack.json` 迁移出的 `evidence_backed_bootstrap_v1`，Persona Pack 只是迁移输入，不是 `mrprompt` 的运行时依赖。新作者可以通过 `pf routing-profile <author>` 或 Web rebuild 自动生成同一格式的 `evidence_backed_representative_v1`：先对 answer/article 标题做 BGE-M3 + cosine agglomerative clustering，再从每个领域簇挑选最多 5 篇中心性优先、互动数据辅助的完整 parent，默认全局最多 72 篇。完整正文只在生成请求期间存在，落盘 Schema 只保留短摘录和 doc_id，并在写入前逐字核验。标题聚类只决定取材范围，不直接推断人格；facet 是一次作者级结构化提炼的场景视角，不要求每个领域对应一个 facet。

`mrprompt` 的运行时优先级为：

```text
当前用户问题 > 题目客观背景 > 本轮 RAG（Top5 或 Top20） > Narrative Schema > 用户记忆/历史回答 > 模型常识
```

模型内部执行四步：

1. Anchoring：根据当前问题、本轮原文和身份锚点确定此刻的观察位置。
2. Selecting：完整 Schema 会注入上下文，再按当前情境选择有帮助的场景记忆，不平均融合所有 facet。
3. Bounding：遵守适用主题、时间和知识边界，不补写私人经历或实时事实。
4. Enacting：把判断动作和表达信号自然写进回答，不解释 schema 或生成过程。

旧的 `persona_pack` 变体继续保留用于兼容和对照；前端允许显式切换 Writer，默认值只决定首次进入和没有本地选择记录时的优先分支。

## 上下文打包

`pack_author_context(...)` 接收已经按 `writer_context_top_k` 截断的 parent hits，输出给 writer 的紧凑上下文。

每个 parent 保留：

- 标题
- 正文全文

不保留：

- 检索排名
- dense/sparse 分数
- child node 命中信息
- URL、ID、时间等元数据

原因：writer 不需要知道检索过程，检索过程只用于 trace。评测 trace 同时记录
`retrieved_parent_top_k` 和 `context_parent_top_k`，保证 RAG5/RAG20 的比较可审计。

## 文件说明

### `writer.py`

- `build_writer_messages(...)`：构造 writer messages。
- `build_prompt_pack(...)`：构造可粘贴到 ChatGPT 网页的 Markdown prompt pack。
- `render_prompt_pack(...)`：把 chat messages 渲染为单段 Markdown。
- `pack_author_context(...)`：把 top parent 全文打包为创作者历史表达上下文。
- `generate_answer(...)`：调用 LLM 生成回答。
- `AnswerResult`：保存 answer、messages 和进入 writer 的 parent 标题，便于 CLI trace。

### `pack.py`

- `load_persona_pack(...)`：解析并验证 Persona Pack。
- `load_persona_pack_for_index(...)`：从作者目录加载与当前索引配套的 Pack。
- `verify_persona_pack_evidence(...)`：逐字核验 claim evidence。
- `render_persona_pack_prompt(...)`：把经过验证的 Pack 渲染为非清单式 writer 上下文。

### `narrative.py`

- `load_narrative_schema(...)`：加载并校验版本、字段和 SHA-256。
- `load_narrative_schema_for_index(...)`：从作者目录加载 schema，并绑定 `parents.jsonl`。
- `verify_narrative_schema_evidence(...)`：逐字核验 facet 的训练期证据。
- `render_narrative_schema_prompt(...)`：只渲染可用于写作的场景记忆，不渲染审计摘录。

### `narrative_builder.py`

- `NarrativeSchemaBuilder`：读取 `index/parents.jsonl`，复用标题清洗、BGE-M3 和
  AgglomerativeClustering，按每簇最多 5 篇、全局最多 72 篇选择代表 parent，并调用一次
  结构化 LLM 生成作者级 Schema。
- 全文只进入临时请求；写盘前通过 `load_narrative_schema(...,
  verify_evidence=True)` 核验，JSON 只保存字段、短 excerpt、doc_id 和 corpus/config 元数据。
- 画像缓存同时检查 `corpus_version` 和 builder config hash；Schema 失败不覆盖旧文件。

## 后续不进入当前版本的能力

- judge/rewrite 在线闭环。
- 多轮 session memory。
- 长度控制前端选项。
- 多 provider 完整抽象。
- Narrative Schema 人工审核工作台（自动构建器已支持 CLI/Web rebuild，人工审核 UI 暂缓）。

## 强效双上下文实验分支

`strong_style_v1` 和 `strong_style_2pass_v1` 只用于离线生成评估，不覆盖默认 Chat
链路，也不替换旧的 `rag_magic_if_v2` 运行。两者都保留通用 Magic If v2，只改变
Writer 看到历史表达的组织方式：

```text
RAG20 parent 候选
-> 内容参考 Top5
-> 表达示范候选（第 6 名以后）经 LLM 选择 Top3
-> strong_style_v1：直接生成
-> strong_style_2pass_v1：先生成内容规划，再生成最终回答
```

内容参考用于当前问题的事实、观点和论证材料；表达示范只用于观察句式、节奏、
口语程度和停顿等表达形态。表达选择器最多返回三个候选 ID，选择失败时按候选池
顺序回退。双阶段版本的内容规划只负责当前问题的核心判断和展开方向，不得变成
固定写作提纲，也不输出到最终回答。

这两个分支由评估 Runner 通过 `content_context_top_k=5` 和
`style_context_top_k=3` 控制，并把内容 parent、表达选择和内容规划写入 trace。
Web 通过扫描已完成的 run manifest 自动发现它们；新增实验必须创建新的不可变 run，
不能覆盖旧答案或直接改变默认 Chat。

`pure_role_rag10_v1` 是额外的最少中间层对照。它只把原始检索得到的十篇作者内容
整体交给一条身份化 Prompt，不做 Persona Pack、Narrative Schema、表达选择或内容
规划；正式评测使用 `query_mode=raw` 以保持变量边界清楚。它只通过评估 Runner
注册为独立 run，不改变默认 Chat。

## AuthorRoutingProfile（CreatorOS 边界）

`routing_profile.py` 负责把作者索引转换成供外部 CreatorOS 查询的作者路由画像，
不参与现有问答、RAG 排名或 writer prompt。画像分为两类：

- `domain_prototypes`：从 `parents.jsonl` 中 answer/article 的问题或文章标题清洗、去重后，
  使用现有 BGE-M3 dense 向量和 `AgglomerativeClustering(metric="cosine", linkage="average")`
  聚类；小簇合并进 `long_tail`，代表标题和 `doc_id` 保留为证据。
- `perspective_prototypes`：优先读取经过证据核验的 Narrative Schema，其次读取 Persona Pack；
  两者都不存在时才从不同领域代表标题分层采样并尝试一次结构化 LLM 抽取。没有 LLM 时画像仍可
  处于 `domain_ready`/`perspective_pending`，不会丢失领域向量。

画像写入 `data/authors/zhihu/<author>/routing_profile.json`，只保存 embedding 元数据和
Qdrant point reference，不保存 float 向量或原文。所有 domain/perspective 向量进入全局
`creator_routing_profiles` collection，按 `corpus_version` 先写新版本、再清理该作者旧版本。
`corpus_version` 由排序后的 `doc_id + updated_at` 计算，未变化且配置未变化时直接复用 JSON。

CLI：

```powershell
pf routing-profile <author> --no-llm
```

API：`GET /api/personas/{author}/routing-profile` 查询，管理员通过
`POST /api/personas/{author}/routing-profile/rebuild` 独立重建。CreatorOS 只能通过这些
稳定 API 使用画像，不读取 PersonClone 内部作者目录；热点发现、候选召回、最终决策和内容
生成仍属于 CreatorOS，暂不在本模块实现。


## Context V2 Writer budget

`build_writer_messages` 保持原有 prompt 与未超预算的消息结构，增加 provider-neutral 输入预算。
Web 配置 `PERSONAFORGE_WRITER_CONTEXT_BUDGET`，扣除最大输出后传入；直接调用默认输入 48000。
分项估算、25% 余量、裁减顺序与明确约束保护见
`docs/architecture/generation/context-memory-v2.md`。预算保护覆盖单轮及多轮构建入口。
不改变 Narrative Schema、RAG 排序或作者生成策略；过大的受保护输入显式失败，不静默截断。
