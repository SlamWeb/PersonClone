# 离线评测规格

`src/personaforge/eval` 负责把“作者过去表达能否支撑对未来问题的回答”变成无泄漏、可复跑的本地实验。它不负责 Web UI 或 rewrite；Gold Judge 的评分核心也放在本模块，由 CLI 和 Web 后台任务共享。

## 本轮目标

核心 CLI 入口包括：

```powershell
pf eval prepare <author>
pf eval run <author> --dataset <dataset.jsonl> --split dev
pf eval retrieval-pool <author> --dataset <dataset.jsonl>
pf eval retrieval-core --source-manifest <manifest.json>
pf eval judge <system_id>
```

`prepare` 从已有 `index/parents.jsonl` 构造严格时间切分数据集；`run` 在既有全量 Qdrant 索引上动态排除未来 parent，运行当前 RAG + writer 链路并写出可读结果。`retrieval-pool` 与 `retrieval-core` 冻结人工相关性标注池，`judge` 和 Web 后台任务共享同一套六维 Gold Judge 实现。

## 数据切分

- 标准协议只把 `kind=answer`、有标题、有创建时间、正文至少 200 字的 parent 作为候选题。
- 按 `created_at` 升序排列，尾部依次分为 `dev=10` 和 `test=20`。
- temporal cutoff 是 dev 的第一题创建时间。
- 所有创建时间不早于 cutoff 的 parent，不论 answer/article/pin，都会写入 `excluded_parent_ids`。
- `dataset.jsonl` 保存原问题、gold 原回答、目标 parent_id、创建时间和 split；它只落本地 `data/eval/`，不进入 git。

作者回答少于 30 篇时，不能用文章补齐问题，也不能伪造 Dev。可以显式使用
`test-only` 稀疏作者协议：把该作者所有有标题、有正文、有创建时间的 `kind=answer`
parent 放入 Test，允许短回答进入，并在 manifest 中写入
`selection.protocol=sparse_author_test_only`、`test_only=true` 和实际的题目数量。
该协议没有 Dev 集，只能作为该作者的独立 Test 报告；短回答应在后续敏感性分析中单独报告。
对于无法稳定拆出完整 Gold 类别的极短回答，Gold 单元提取保留原回答作为缺失类别的最小锚点，
并在后续报告中标记该降级记录；这不是把短回答扩写成普通长回答。
由于稀疏作者可能没有足够早期材料，协议采用回溯语料：非评估文章可以进入检索语料，
但所有评估回答都排除，防止直接命中 Gold。manifest 会将
`strict_future_exclusion=false`、`corpus_policy=exclude_eval_answers_only` 和
`recall_scope=eligible_author_corpus_excluding_eval_answers` 写清楚，不能与标准时间协议
的指标混为一谈。Web 任务在 Test 任务遇到不足 30 篇标准回答时自动选择该协议，在 Dev
任务中明确失败并提示改建 Test-only 任务。文章和想法仍不能成为 query。

因此，test 原回答虽然仍物理存在于本地全量索引中，但 dense/sparse 的任何一路都不能检索到其 child node。

## 冻结工程开发集

日常生成质量迭代固定使用：

```text
dataset_id: temporal_dev10_v0
path: data/eval/wu-ren-jun-28-temporal-v0/dataset.jsonl
split: dev
dataset_sha256: 2e854b03da188d54dd320f319264dd4574855ef826e27290accdaa2ec39f5553
excluded_parent_ids_sha256: b7f36686846f5ef7b5e0515c5d0e8f7addbcb718a02b1d569f901d6c1808ca8b
```

实验记录和口头讨论都必须使用 `temporal_dev10_v0` 这个完整 ID，不能只写
`dev10`。此前人工填写过的十题属于另一套数据集，不允许与本集按题号、均分或
运行名称直接比较。

任何方法变体都必须复用同一份 `dataset.jsonl`、同一排除名单和同一
Gold Judge 配置；如文件哈希变化，应视为新数据集并分配新的 dataset ID。

## Runner 合同

Runner 默认固定当前 baseline：

```text
grounded
+ strong_identity
+ BGE-M3 dense+sparse
+ 4 路 query transform
+ child_top_k=100
+ per_query_parent_k=30
+ parent_top_k=20
+ writer_context_top_k=20
+ DeepSeek Flash
```

`parent_top_k` 是检索与 trace 的保留深度，`writer_context_top_k` 是最终送进 writer
的 parent 数量。两者不能混为一谈：RAG5 对照运行时保持 `parent_top_k=20`，只把
`writer_context_top_k` 改为 5；RAG20 则两者都为 20。这样不会把“检索变差”和“writer
看到的材料变少”混在一个变量里。

例如：

```powershell
pf eval run wu-ren-jun-28 --dataset <dataset.jsonl> --split dev `
  --run-name mrprompt-rag20 --writer-prompt mrprompt `
  --writer-context-top-k 20

pf eval run wu-ren-jun-28 --dataset <dataset.jsonl> --split dev `
  --run-name mrprompt-rag5 --writer-prompt mrprompt `
  --writer-context-top-k 5
```

每题输出：

- generated answer 和 gold answer。
- query understanding / Tavily trace。
- 每路检索摘要、最终 parent、writer variant 和参数。
- 排除名单审计：任何 excluded parent 出现在 route 或 parent context 时，run 立即失败。

每次 run 还必须写 manifest，包括 dataset hash、excluded-parent hash、git revision、模型名、参数、时间和 split。它不写 API key、cookie、完整 writer prompt 或 parent 正文副本。

## 检索候选池与人工标注

检索评估先对冻结的 `temporal_dev10_v0` 开发集构造候选池，不直接在全语料上逐篇
标注。候选池由显式离线命令生成；网页只读取冻结文件，不能在标注过程中重新检索：

```powershell
pf eval retrieval-pool wu-ren-jun-28 `
  --dataset data/eval/wu-ren-jun-28-temporal-v0/dataset.jsonl `
  --dataset-id temporal_dev10_v0 `
  --split dev `
  --embedding-device cuda `
  --model-name C:\PersonaForge-OpenSource\data\models\bge-m3
```

每道题的候选来自四个独立来源：

1. 原问题 BGE-M3 dense，折叠后的 parent Top30。
2. 原问题 BGE-M3 sparse，折叠后的 parent Top30。
3. 现有四路 query transform；每路内部 dense+sparse parent RRF，再跨 query 做
   parent RRF，最终 Top30。
4. 原问题 BM25，折叠后的 parent Top30。

四路先各自从 child 命中按首次出现折叠为 parent，再取各路 Top30，最后对 parent ID
做并集去重；并集不再按跨模型分数截断。这样 dense、sparse、BM25 的分数尺度不会
互相污染，也不会因同一长文出现多个 child 而获得额外票数。

BM25 直接读取现有 `nodes.jsonl`，不重新切片。中文使用 `jieba.cut_for_search`，保留
数字、英文缩写和网络词；采用 Okapi BM25，默认 `k1=1.2`、`b=0.75`。它只接收原问题，
避免把 query transform 的收益算进词法基线。

输出目录默认为：

```text
<dataset-dir>/retrieval_pool/<split>/
  pool.jsonl
  manifest.json
```

候选文件冻结完整 parent 正文、原文 URL 和各路 rank，manifest 记录数据集哈希、排除
名单哈希、配置和 Git revision。生成时任何 `excluded_parent_ids` 出现在候选中都立即
失败。已存在的池默认拒绝覆盖，只有明确传入 `--force` 才允许重建。

完整 Top30 并集用于保留召回覆盖，不直接等同于人工工作量。人工核心集从冻结完整池
派生，不重新运行 embedding、Qdrant、BM25、联网或 LLM：

```powershell
pf eval retrieval-core `
  --source-manifest data/eval/wu-ren-jun-28-temporal-v0/retrieval_pool/dev/manifest.json `
  --route-depth 3
```

核心集保留“任意一路 rank <= 3”的 parent，再按 parent ID 去重。当前
`temporal_dev10_v0` 核心集为 10 题、96 个 query-parent 对。完整池与核心池共享同一
标签命名空间：在任一视图打分都会被另一视图复用，但每个视图的完成度只统计自身包含
的候选，不能把完整池尾部标签算进核心集进度。

人工效用等级固定为：

- `0 无用`：不能帮助回答该题。
- `1 有一定帮助`：提供部分观点、例子或表达参考。
- `2 明显有用`：直接支撑高质量回答。
- `暂时跳过`：只移动到下一篇，不作为标签写入。

标注阶段隐藏候选来自哪一路及其 rank，打分后才允许查看技术详情。每位登录用户的
标签独立保存到 SQLite，刷新和服务重启后继续；同一标签允许修订。后续 `Recall@K`、
`nDCG@K`、`MRR` 和无有用结果率均从这份冻结标注派生，K 与候选池大小分开设置。

正式多作者报告以分级 `nDCG`、Useful/Strong Precision 和 Useful/Strong Recall 为主：

- `nDCG@K` 直接使用 0/1/2 相关等级，gain 分别为 0/1/3。
- Useful 指标签至少为 1；Strong 指标签等于 2。
- Precision 的常用观察深度为 1/3/5/10/20/30。
- Recall 使用更深的 10/20/30/50/100；Qrels 候选池大小与待评估方法的排名深度相互独立。只要该方法另外冻结了 Parent Top K 排名，就可以在既有 Qrels 上计算 Recall@K，无需重新标注候选池。
- 大语料作者假设六路 Top50 并集覆盖主要相关材料，Recall 分母为该候选池内全部已标注相关材料；因此它是候选池边界内的 Recall，不外推为未标注全语料的绝对 Recall。
- 前端根据待评估方法排名快照的各路实际可用深度自动提供 K，并同时展示“请求深度”和“实际深度”。当前排名快照只有 Top30 时不能直接显示 Recall@100；应在不改变 Qrels 的前提下，本地重新生成该方法的 Parent Top100 排名快照。

独立排名快照使用本地索引生成，不调用 LLM API，也不会修改已有 Qrels：

```powershell
$env:PYTHONPATH = "src"
python -m personaforge.cli eval retrieval-rankings `
  --pool-manifest <冻结候选池 manifest.json> `
  --index-dir <作者 index 目录> `
  --model-name <本地 BGE-M3 路径> `
  --embedding-device cuda `
  --depth 100
```

快照会写入候选池下的 `rankings/<ranking_id>/`，包括每道题六条路线的有序
Parent 列表和配置清单。前端只展示已经完成的快照；没有快照的作者或方法不显示
Recall@100，实际语料不足时也不提供超过实际深度的 K，避免把 Top30 结果误当成深度检索结果。

### 卢诗翰离线评估

卢诗翰使用独立数据集 `lu-shi-han-89-temporal-v0`，固定为 Dev 10 + Test 20。候选池
不是全库标注，而是六条检索路线各取 parent Top50 后并集去重：

```text
data/eval/lu-shi-han-89-temporal-v0/retrieval_pool/
  all30_six_route_top50_v1/
```

该池包含 30 道题、4030 个 query-parent 对、1313 个唯一 parent。六条路线为原问题
Dense、原问题 Sparse、原问题 Dense+Sparse RRF、Query Transform RRF、原问题 BM25、
Query Transform Dense+BM25 RRF。候选池只定义评估边界；指标仍按每条路线自己的排序计算。

本批次离线标注文件位于：

```text
all30_six_route_top50_v1/
  codex_handoffs/lu-shi-han-codex-v1/codex_review.json
  codex_handoffs/lu-shi-han-codex-v1/offline_review_provenance.json
  llm_labels/lu_shi_han_offline_local_v1/
```

本批次不调用付费 API。22 道题复用已有的本地 Codex handoff，另外 8 道题使用本机
`Qwen2.5-1.5B-Instruct` 离线补齐；本地模型无法返回完整证据时保守记为两个轴均为 0，
不从截断文本推断正向标签。因此这是一份“完整覆盖的离线初标”，不是人工金标准，不能
用来声称人工一致性或作为最终论文标注。正式报告必须保留
`offline_review_provenance.json`，并把本批次与人工/高质量 Judge 标签分开命名。

## Writer replay

Persona Pack 或 Writer prompt 的消融使用 `personaforge.eval.replay`。它从已完成
run 的 trace 还原每题最终 parent，并从同一 `parents.jsonl` 读取全文，只重放
Writer。以下输入必须冻结：

- 原问题与客观背景。
- 最终 20 篇 parent 的 ID、全文和顺序。
- Writer 的 temperature 与 max_tokens。

replay 不调用 query understanding、query transform、embedding 或 Qdrant。它会
复核数据集哈希、排除名单、Persona Pack 证据和 parent 顺序，并在 manifest 记录
来源 run 哈希。这样 Persona Pack 消融不会被检索波动混淆；但非零 temperature
仍会引入一次生成采样噪声。

## 生成质量评估

生成评估固定使用 `temporal_dev10_v0` 的同一问题、Gold 和未来文档排除名单。一个
“系统”不是 LLM provider，而是一份不可变的方法运行版本：

```text
<dataset-dir>/runs/<run-name>/
  manifest.json
  runs.jsonl
```

Web 自动发现 `personaforge.eval.run.v0` 和 `personaforge.eval.writer-replay.v0`。只有
同时满足以下条件的 run 才能进入正式系统选择器：

- manifest 状态为 completed，且 dataset SHA-256 与 dataset manifest 一致。
- split 为 dev，十个 `dev-*` item 完整、互不重复且全部生成成功。
- 每题都冻结 question、gold_answer 和非空 candidate answer。

新增模块时必须创建新的 run-name 和 run SHA-256，不能覆盖旧系统答案。显示名称、变更
说明和父系统可以作为 manifest 的可选展示元数据；旧 run 缺少这些字段时以目录名和
config 兜底。

Gold Judge 使用 `gold-judge-v1.0`，输入严格限定为同一题的 question、作者真实 Gold
和一个冻结 Candidate。六个 1--5 分维度为：

1. D1 核心立场与价值取向一致性。
2. D2 论证方式与推理组织相似性。
3. D3 词汇与语域一致性。
4. D4 语气与人格姿态一致性。
5. D5 句法与节奏一致性。
6. D6 自然表达与生成痕迹，分数越高表示通用生成痕迹越弱。

每个维度独立返回分数、Gold/Candidate 短证据和理由。默认每题重复三次，最终分数取
中位数，同时保留原始评分以计算完全一致率、正负一分一致率和极差。D1/D2、D3--D5
和 D6 可以分别汇总为“内容忠实度、语言表达相似度、自然表达”，但不把六维压成一个
总分，也不能用 D6 抬高作者相似性。

LLM 系统比较保留两种互补视角：六维 Gold Judge 继续用于单篇诊断；新增证据驱动的
Profile Pairwise Judge 用于直接比较两个生成方法。后者输入严格固定为：当前问题、
同题作者 Gold、作者历史证据 Profile、候选 A、候选 B。Profile 只能作为带原文出处的
评审证据，不能当作写作规范；候选身份和方法名不进入给 Judge 看的候选正文。

Pairwise Judge 必须对每道 Test 题生成 forward 和 swapped 两份请求。每份请求都要求
在 A/B 中选择一篇，并返回低/中/高信心、最多若干条 Profile 证据 ID、Gold 方向证据
和一句比较理由；不提供平局选项。两次选择映射到同一个系统时记为 position-consistent，
不一致时该题不计入胜率，只进入不稳定样本复核。Pairwise 不是把六维分数重新压成总分，
它回答的是“在这道题上哪种方法更像作者”。

当前实现使用离线 handoff：`generation-profile-pack` 可把已有 train-only Persona
Pack 转成评审专用档案，`generation-profile-corpus` 在没有总结档案时从时间边界内的
历史 parent 抽取确定性的原文证据，`generation-pairwise-export` 生成绑定哈希的正反
任务，`generation-pairwise-import` 校验并汇总回填结果。上述命令都不调用 LLM API；
Profile 和候选内容的来源、哈希、题目数和 Prompt 哈希都写入 manifest。

生成评估中的人工 AB 仍是独立构念：在展示 Gold 的前提下强制选择整体更像作者的 A 或 B，
系统身份隐藏，A/B 顺序按用户、系统对和题目稳定随机，刷新后不变。自动 Pairwise Judge
和人工 AB 的数据不能混为同一标签，但可以用一致性、换位稳定性和人工抽样校准来共同解释。

Web 发起 Judge 时只创建持久化异步任务。任务可以离开页面继续执行，服务重启后未完成
任务允许恢复；相同 system SHA、Judge prompt hash、模型和重复次数的完成结果应复用。
Judge 结果写入对应 run 下的 `judges/gold-v1/<job-id>/`，不覆盖生成 run。

## 不做什么

- 不重爬、不重切片、不重 embedding、不重建 Qdrant collection。
- 不把真实语料或 eval 输出提交到仓库。
- 不把 Profile Pairwise Judge 当成六维总分，也不把不一致的换位结果强行计入胜率。
- 不调用付费 API 生成离线 Profile 或 Codex handoff；只有明确启动 API Judge 任务时才允许联网。
- 不把 test 当作日常调参集。常规实验跑 dev，候选最终方案才跑 test。

## 验收

- 临时语料单测能验证时间切分和排除名单。
- 检索单测能验证 excluded parent 不会传入 Qdrant query。
- `pf eval prepare` 能对现有作者索引写出本地数据集。
- `pf eval run --split dev --limit 1` 能写出 manifest、runs.jsonl、单题 Markdown，并通过排除审计。
- Web 能自动发现同一数据哈希下的完整 dev10 系统，人工六维和 AB 标签即时持久化；`pf eval judge <system_id>` 与 Web 异步任务产生相同结构的结果。

## 最近验证

- 临时语料测试覆盖：严格时间切分会排除 cutoff 后的 answer/article/pin；Runner 会写出 manifest、JSONL 和单题 Markdown。
- 本地全量索引 smoke 已验证：dynamic excluded-parent filter 同时作用于 4 路 query 的 dense/sparse 检索；最终 parent context 与各 route child hit 均为零泄漏。
- 第一轮完整 dev baseline 只写入本地 `data/eval/`；它不构成冻结 test 结果，也不包含 LLM Judge。
- `temporal_dev10_v0` 的检索候选池已完成：10 题、739 个去重 query-parent 对；每题
  四路各 30 个 parent，合并后为 60--81 篇，70 个未来 parent 的泄漏数为 0。

## 强效双上下文实验分支

`strong_style_v1` 和 `strong_style_2pass_v1` 是生成质量实验分支，不覆盖默认 Chat
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
口语程度和停顿等表达形态。表达选择器不得按主题相似度重新挑材料，最多返回三个
候选 ID；选择失败时按候选池顺序回退。双阶段版本的内容规划只负责当前问题的
核心判断和展开方向，不得变成固定写作提纲或输出到最终回答。

两套方法必须使用新的 `run_name`、`method_id` 和 manifest，固定同一份
`temporal_dev10_v0`、同一排除名单、同一模型和温度后再比较。当前已生成的本地运行是：

```text
strong-style-rag20-grounded-dev-20260807-v1
strong-style-2pass-rag20-grounded-dev-20260807-v1
```

Judge 输出偶发出现模型多写一个括号或最外层数组的问题。Gold Judge 重试时会追加
严格的 JSON 结构提醒；这只提高任务完成率，不改变 rubric、评分值或聚合方式。若服务
进程在长任务中被强制终止，SQLite 中的运行中任务需要由下一次启动恢复，不能手工补写
评分。

## 纯身份化 RAG10 对照

`pure_role_rag10_v1` 是“最少中间层”对照：使用原始问题、原始检索、Writer 可见
Top10 parent 和一条通用身份化 Prompt；不使用 query understanding、表达选择器、
内容规划、Persona Pack 或 Narrative Schema。它的目标不是直接成为默认方案，而是
验证复杂的上下文拆分是否反而损害自然表达。

该方法使用新的不可变 run-name。为了保持对照含义，运行时采用 `query_mode=raw`、
`writer_context_top_k=10`，并把同一冻结 dev10 的 Gold Judge 结果与其他方法并列展示。
# 2026-08 生成方法对比补充

纯 RAG + Magic If 是一个可复现的 Writer 变体，使用 `writer_prompt=rag_magic_if`。
它不依赖 Persona Pack、Narrative Schema 或 Style Profile；历史 parent 只通过检索结果
进入 Writer。RAG5/RAG20 的差别只在 `writer_context_top_k`，检索保留深度仍为
`parent_top_k=20`。

`writer_prompt=rag_magic_if_v2` 是 v1 的通用化提示词对照。它不改变数据集、检索、
query transform、模型或生成参数，只改变 Writer system prompt；v1 运行继续保留，
不能被 v2 覆盖。这样可以单独评估提示词是否诱导了固定行文结构。

生成质量评估支持同一冻结数据集的两个切分：`dev10` 用于日常调参与人工检查，
`test20` 只用于冻结后的泛化比较。两者使用完全相同的生成链路、模型、温度和 Gold
Judge；Test20 结果不能反过来驱动 Prompt 或检索参数调优。每个 run 的 manifest 保存
`split` 和实际题目数，不能把 Test20 运行伪装成 Dev10，也不能把不同切分的系统放进
同一组 AB 对比。

每个生成 run 的 `manifest.json` 必须保存 `method_id`、`display_name`、`description`、
`parent_method`、`prompt_version`、`prompt_sha256` 和完整的非敏感运行参数。新增方法或
参数时必须新建不可变 run，不能覆盖旧结果。Judge 结果仍固定三次重复，并保存每题每个
维度的三次原始评分、最终中位数、95% CI、完全一致率、相差不超过 1 的比例和极差。
## 2026-08 六路检索相关性评估

检索评估使用时间切分后的 dev10 与 test20，共 30 道题。`retrieval_pool.py` 为每道题
冻结六条诊断路由的 parent Top30，再按 `parent_id` 取并集。当前冻结池共有 2291 个
“题目-候选 parent 对”，不是 2291 篇独立文章；未来材料泄漏数必须为 0。

六条路由固定为：

1. `raw_dense`：原问题 BGE-M3 Dense。
2. `raw_sparse`：原问题 BGE-M3 Sparse。
3. `raw_hybrid_rrf`：原问题 Dense 与 Sparse 的 parent RRF。
4. `transformed_rrf`：四路 Query Transform，各自 Dense+Sparse 后再做 parent RRF。
5. `raw_bm25`：原问题 BM25。
6. `transformed_dense_bm25_rrf`：四路 Query Transform，各自 Dense+BM25 后做 parent RRF。

第 3、4、6 路是可比较的融合方案，第 1、2、5 路保留为分支诊断基线。各路先在自己
的分数体系内排序，跨路不混用原始分数。冻结 Query Transform 计划保存在本地
`query_plans/`，重跑候选池不得重新调用 LLM 或联网。

`retrieval_judge.py` 支持 API Judge，也支持 Codex 离线直接审阅。Codex 审阅文件只列
非零判断，但必须对每道题显式写入 `review_complete=true`；只有通过候选 ID、pool ID
和 SHA-256 完整性校验后，省略项才会物化成显式 0 标签。这样未审完的题不会被误当成
全负例。输出仍放在候选池目录旁：

```text
retrieval_pool/all30_six_route_v1/
├── manifest.json
├── pool.jsonl
├── codex_review_v1.json
└── llm_labels/codex_relevance_v1/
    ├── manifest.json
    ├── labels.jsonl
    └── metrics.json
```

标签文件不复制正文，只保存 `item_id + parent_id + score + evidence + reason` 等审计字段。
离线审阅的物化命令为：

```text
pf eval retrieval-codex-label \
  --pool-manifest data/eval/.../all30_six_route_v1/manifest.json \
  --review-file data/eval/.../all30_six_route_v1/codex_review_v1.json \
  --label-set codex_relevance_v1
```

`retrieval_metrics.py` 在 `K=1/3/5/10/20/30` 上分别计算 Hit、MRR、分级 nDCG、
Precision 和 Recall，并独立输出 all30、dev10、test20。Recall 的分母是“冻结六路候选
并集中所有被判为 1 或 2 的材料”，不是语料库全局相关文档，因此前端和报告必须明确
显示 `recall_scope=six_route_candidate_union`。只有一题候选并集全部有标签时才计算该题
的 Recall 与理想 DCG；覆盖率单独报告，避免半成品标签池产生虚高指标。

0/1/2 分别表示无用、有一定帮助、明显有用。机器标签用于快速诊断召回与排序，不冒充
人工金标准，也不能与 Study 1 的作者相似性判断混为同一构念。

当前二值指标固定使用 `relevance_threshold=1`，即 1 分和 2 分都视为相关；分级
`nDCG` 继续用 `2^score-1` 区分两个正等级。后续允许基于同一份冻结标签增加
`relevance_threshold=2` 的严格敏感性报告，但它必须作为并列观察口径保存，不能静默
替换历史默认结果。RAG 与 Generate 的架构图和指标释义位于
`docs/architecture/evaluation/`。

### 解释边界

Top3 指标低而 Top20/Top30 Recall 明显上升，表示候选池中存在有用材料但浅层排序不稳；
不能把它解释成“完全没有召回”。相反，如果各 K 的 Recall 都低，才优先怀疑 query
理解、语义覆盖或候选路由缺失。只有 dev 用于调参与选择方案，test 用于冻结后的最终
报告；前端可以并列查看，但工程迭代不得反复根据 test 修改参数。

## 2026-08 Gold-aware 检索效用 V2

`query_only_relevance_v1`（现有目录名仍保留 `codex_relevance_v1` 以保证历史文件可读）
只回答“材料对这个问题是否一般相关”。V2 改为离线、参考答案感知的作者条件评估：
Judge 同时看到问题、作者真实回答、从真实回答冻结出的原子单元和候选历史材料。真实
回答只允许进入离线评估，绝不能传给在线 query understanding、retriever、reranker、
writer 或 Chat API。

V2 不把“有用”压成一个含混总分，而是独立保存两个 `0/1/2` 轴：

1. `content_support`：候选能否帮助重建 Gold 的核心立场、因果机制、事实或例子。
2. `persona_expression_support`：候选能否帮助重建 Gold 中实际出现的论证动作、语气、
   节奏或表达习惯。仅仅“同一作者写过”不能得分，必须对本题 Gold 的表达实现可迁移。

两个轴不加权、不相乘，也不生成单一总分。与 V1 的 `3×3` 转移矩阵只比较语义上较
接近的 `V1 score -> V2 content_support`；表达支撑单独报告。

Gold 原子单元按题冻结并哈希，至少包含 `stance / reasoning / example / expression`。
Judge 保存候选短证据、映射的 Gold unit ID、简短理由和置信度。大规模标注采用：

- 全部候选先做一次温度为零的确定性判断。
- 任一轴为 1、低置信候选和各分层的稳定抽样样本做第二次独立判断。
- 两次在任一轴冲突时做第三次判断；最终逐轴取中位数，原始 attempts 永久保留。
- 批量请求只复用同一题的 Gold 上下文，每个候选仍必须独立返回结果；任务可恢复。
- `labels.jsonl` 和正式指标只包含已经完成其所需 1/2/3 遍稳定性流程的样本。只有第一遍
  成功但复评尚未完成的样本保留在 `attempts.jsonl`，manifest 分开报告 `missing_pass1`、
  `pending_pass2`、`pending_pass3` 与 `stability_completed`，不得提前计入 Qrels。
- 断点续跑必须从 attempts 重建每一遍状态；已经开始的分层抽样复评在后续运行中继续保留。
  作为 seed 输入的标签集必须整体为 `completed`，partial 标签集不得作为已冻结种子复用。
- 批量 Judge 若因漏回 candidate、缺少非零分映射或其他结构校验失败，在耗尽原批重试后
  自动二分为更小批次，最小退化到单条；单个坏响应不得让同批其他候选永久缺失。

### 候选前缀与成本合同

Gold-aware API 标注固定使用 `candidate_first_v1` 请求布局。每个请求的稳定候选批次必须
出现在 user JSON 的最前部，问题、Gold、Gold units 等随题变化的内容放在后部；system
prompt 保持不变。全量 Qrels 按“候选批次优先”调度，同一批候选跨题复用：每个批次前
两道题串行预热，后续题目才允许有限并发。该布局只优化 DeepSeek 自动前缀缓存，不改变
任何候选、标签定义、Gold 信息或稳定性复评规则。

每次请求必须独立记录 provider usage，包括 prompt cache hit/miss、output tokens、耗时、
请求批次和 pass。任务根据冻结在 manifest 中的非敏感价格表估算人民币成本；价格允许由
环境变量覆盖，报告必须写明“估算”而非账户实际扣费。默认 smoke 预算为 2 元，完整 dev10
预算为 5 元；达到预算后在批次边界自动进入 `paused_budget`，已经落盘的 attempts 与 usage
不得丢失，调高预算后从断点继续。

缓存命中是 best effort，不能作为正确性前提。即使缓存完全未命中，输出标签、完整性校验
和恢复语义也必须保持一致。

### 双轴 Codex handoff

Codex 离线审阅与 DeepSeek API 使用完全相同的双轴 rubric、Gold units、候选集合和完整性
约束。任务导出包含冻结 pool/dataset/unit 哈希的 handoff 包；导入结果必须覆盖每个
`item_id + parent_id`，且任一非零轴都提供候选证据、理由和对应 Gold unit ID。只有通过
SHA-256、覆盖率、分值范围和证据校验后，结果才能物化为正式 Qrels。

Codex 直接标注暂时作为独立 label set 接受，不要求先与人工校准；报告必须保留
`labeler=codex_handoff`，不能伪装为人工 Gold。旧单轴 Codex review 继续只读兼容，不自动
升级或与双轴结果混合。

### 作者评估初始化任务

新增持久化、可恢复的离线任务，顺序固定为：语料预检与快照、时间切分、候选池、全量可见
parent 池、Gold units、标注、指标与发布。任务支持 `deepseek_api / codex_handoff /
manual_import` 三种 labeler；第一版前端主要完成 retrieval-only 初始化，不触发 Writer。

Gold-unit 缓存只有在 item ID 完整覆盖当前 split 且四个必需类别都存在时才可复用；旧的
部分缓存会自动重建，避免候选池、数据集和 Gold 单元错位。

每个作者独立生成 dataset ID、corpus snapshot、pool、label set 和 job 目录。服务重启后
`running` 任务转回可恢复状态；Codex 任务在导出后进入 `awaiting_codex`，API 任务可因预算
进入 `paused_budget`。任何阶段都不能覆盖已有完成资产；输入哈希变化必须创建新版本。
同一全量 pool 上的 dev/test 标注必须使用不同 label set，并在 manifest 的
`selected_splits` 中显式冻结范围。Web 报告只能列出该 label set 实际覆盖的题目，不能把
同池未运行的另一切分显示成“未完成”。

第一阶段在冻结六路并集的 2291 个题目-候选对上运行 V2，用于和 V1 比较。第二阶段
构造全量 Qrels：对 30 道时间切分题分别枚举线上检索在该 cutoff 下真正可见的全部
423 个 parent，共 12690 个题目-parent 对。目标答案和 cutoff 之后的 answer/article/pin
全部排除，六路原始 rank 原样附着到相应 parent，未被六路召回的 parent 仍接受离线
判断但没有伪造 rank。

全量 Qrels 完成后，六路方法都在 `K=1/3/5/10/20/30` 上按两个轴分别计算：Hit、MRR、
graded nDCG、Precision、Recall 和 MAP。此时 Recall 分母是 cutoff 前完整可用作者语料中
所有相关 parent，不再是六路候选并集。报告必须同时给出覆盖率、无相关材料题数、
all/dev/test 三个视图；日常调参只看 dev10，test20 只作为最终冻结结果。

前端 V2 报告只出现在登录后的评估管理界面，支持切换两个效用轴，查看 Gold、候选、
映射证据、完整候选排序、V1/V2 转移和各路多 K 指标。普通 Chat 用户和 Study 参与者
不得获得 Gold 或 source/system 元数据。

当前本地冻结结果：六路并集 2291 对的 V2 已完整完成，V1 与 V2 内容轴共有 924 对
发生改判，其中 668 对为旧 0 分转成 V2 的 1/2 分。全量 Qrels 已从同一 label set
断点续跑并稳定完成：30 题、每题 423 个 parent、共 12690 对；其中 7445 对完成至少
一次独立复评，完全一致率为 64.57%，内容轴和作者表达轴的 `±1` 一致率分别为
99.70% 和 99.56%。最终标签中，内容支撑 `0/1/2` 分别为 `10399/1557/734`，
作者表达支撑分别为 `5457/6161/1072`。

最终六路指标均覆盖 30 道题，且不存在无相关材料的题。独立 Test20 上，内容轴
`nDCG@3` 最优为 Query Transform Dense+BM25 RRF 的 `0.568`，作者表达轴
`nDCG@3` 最优为 Query Transform Dense+Sparse RRF 的 `0.592`；两条 Query
Transform 融合路线整体显著优于原问题单路检索，纯 BM25 整体最弱。正式多 K 结果
保存在标签目录的 `metrics.json`，本地可读汇总为 `final_metrics_report.md`，逐项明细为
`final_route_metrics.csv`。这些结果用于冻结报告；后续方法选择不得再依据 Test20 调参。

第二作者 `ban-ma-ban-ma-30-2` 已按相同合同验证多作者初始化：冻结快照包含 212 个 parent，
时间切分为 dev10/test20；dev10 的 cutoff 前全量候选为每题 173 个，共 1730 个题目-parent
对。`candidate_first_v1` 的 DeepSeek 双轴标注完整结束，正式 label set 为
`gold_aware_candidate_first_deepseek_v1_dev`。该记录只用于验证多作者、缓存、预算、恢复和
前端发现链路，不把 dev 指标解释为最终泛化结论。
