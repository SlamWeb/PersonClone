# 入库与检索规格

`src/personaforge/ingest` 负责把爬虫产出的本地 Markdown 语料转成可审计的中间产物，并为后续向量索引和检索提供稳定输入。

本模块先做清楚“原始语料到父文档、子节点”的边界，再接向量库。不要一上来把解析、切片、向量化、检索、生成混成一团。

## 职责

- 读取 crawler 产出的作者级 raw corpus。
- 解析 front matter。
- 把每个 Markdown 文件转换成一个父文档。
- 从父文档构造多个子节点。
- 写出可审计的中间产物：
  - `parents.jsonl`
  - `nodes.jsonl`
  - `build_manifest.json`

## 非职责

- 不负责爬取。
- 不负责调用 LLM。
- 不负责生成回答。
- 第一阶段不直接依赖 Qdrant。
- 第一阶段不生成 hypothetical question。
- 第一阶段不做摘要节点，除非后续 `--quality full` 明确打开。

## 输入合同

输入目录来自爬虫。默认新结构是：

```text
data/authors/zhihu/<author-token>/raw/
  profile.json
  manifest.jsonl
  answer/
    answer-*.md
  article/
    article-*.md
  pin/
    pin-*.md
```

`manifest.jsonl` 每行必须包含相对 raw 根目录的 `path`：

```json
{"kind": "answer", "path": "answer/answer-xxx.md"}
```

ingest 优先按 manifest 的 `path` 读取文件。这样后续目录怎么分组，不会影响入库逻辑。

为了兼容迁移前的历史语料格式，第一版 ingest 也允许旧扁平结构：

```text
data/raw/<author>_md/
  profile.json
  manifest.jsonl
  answer-*.md
  article-*.md
  pin-*.md
```

旧结构只作为迁移兼容，不作为新 crawler 的默认输出。

每个 Markdown 文件的结构：

```markdown
---
source: "zhihu"
kind: "answer"
id: "..."
title: "..."
url: "..."
author_token: "..."
created_at: "..."
updated_at: "..."
fetched_at: "..."
question_id: "..."
comment_count: 0
excerpt: "..."
---

# 标题

正文
```

`answer`、`article`、`pin` 都是一等文档类型。尤其是 `pin`，虽然没有问题标题，但它是作者短观点和世界观碎片的重要来源，必须正常入库。

## 父文档设计

一个 Markdown 文件就是一个父文档。

父文档编号：

```text
<source>:<kind>:<id>
```

例如：

```text
zhihu:answer:11429724228
zhihu:article:1919836380621706563
zhihu:pin:1806398148295413760
```

设计理由：

- 父文档保留作者的一篇完整表达。
- 检索时可以搜更短的子节点。
- 生成时可以回填完整父文档，保留语气、节奏、跳跃和上下文。

## 子节点设计

第一版只做三类子节点：

```text
title
lead
passage
```

不做：

```text
hypothetical_question
summary
evidence
```

原因：

- 之前实验里，数据源端生成 hypothetical question 收益不稳定。
- query transform 放在查询端更灵活，错了也不会污染索引。
- 当前阶段要保持入库链路可解释、可调试、可面试讲清楚。

## lead 节点

`lead` 是正文开头部分，用来抓作者回答时的第一刀、立场和切入角度。

生成方式：

```text
去掉 front matter
-> 去掉第一行 # 标题
-> 按空行切自然段
-> 跳过纯图片段
-> 从第一段开始合并
-> 合并到接近 800 个中文字符停止
```

如果正文整体很短，`lead` 和 `passage` 完全重复，则可以不单独生成 `lead`。

## passage 节点

`passage` 按自然段合并生成。

推荐参数：

```text
目标长度：约 900 个中文字符
最大长度：约 1400 个中文字符
最小长度：约 250 个中文字符
```

规则：

- 优先按自然段合并。
- 单段过长时，再按中文句号、问号、感叹号等句末标点切分。
- 尾部过短时，尝试并入前一个 passage。
- 不把纯图片段作为检索节点。

## 第一版 build 输出

命令目标：

```powershell
pf build wu-ren-jun-28 --raw-dir data/raw/wu-ren-jun-28_md --index-dir data/index/wu-ren-jun-28 --quality fast
```

新目录下推荐：

```powershell
pf build wu-ren-jun-28 --raw-dir data/authors/zhihu/wu-ren-jun-28/raw --index-dir data/authors/zhihu/wu-ren-jun-28/index --quality fast
```

输出：

```text
data/index/wu-ren-jun-28/
  parents.jsonl
  nodes.jsonl
  build_manifest.json
```

第一版 `build` 只生成中间产物，不建 Qdrant。

## 文件说明

### `models.py`

定义 ingest 中间产物的数据结构：

- `ParentDocument`：一篇原始 Markdown 对应一个父文档，保留标题、正文、来源、类型、路径和元数据。
- `ChildNode`：检索用子节点，记录 `title`、`lead` 或 `passage` 文本，并保留 `parent_id` 方便回填父文档。

### `loader.py`

负责把 crawler raw corpus 读成父文档：

- `load_parent_documents(raw_dir)`：读取一个 raw 目录，优先按 `manifest.jsonl` 的 `path` 找文件；如果没有 manifest 或旧 manifest 没有 `path`，则兼容扫描旧扁平目录。
- `load_parent_document(path, raw_root=...)`：解析单个 Markdown，生成 `ParentDocument`。
- `split_front_matter(text)`：拆出 YAML-like front matter 和正文。
- `parse_front_matter(value)`：解析 crawler 写出的简单 front matter。
- `split_top_heading(markdown_body)`：去掉正文开头的一级标题，把标题和正文分离。

### `chunking.py`

负责纯文本切片，不理解平台和作者：

- `split_paragraphs(text)`：按空行切自然段，跳过纯图片段。
- `build_lead(text)`：合并正文开头若干自然段，默认接近 800 个中文字符停止。
- `build_passages(text)`：按自然段合并 passage，默认目标约 900 字，最大约 1400 字，尾部过短则并入前一个 passage。
- `normalize_for_compare(text)`：用于判断 `lead` 是否和唯一 passage 完全重复。

### `nodes.py`

负责把一个父文档转成检索节点：

- `build_nodes_for_parent(parent)`：为单篇父文档生成 `title`、可选 `lead`、若干 `passage`。
- `build_nodes(parents)`：批量生成节点。

如果正文很短，`lead` 和唯一 `passage` 完全重复，则不生成 `lead`，避免索引里出现重复节点。

### `build.py`

负责 build 编排和落盘：

- `build_corpus(raw_dir, index_dir, quality="fast")`：读取父文档，生成子节点，写出 `parents.jsonl`、`nodes.jsonl`、`build_manifest.json`。
- `build_result_to_dict(result)`：把 build 结果转成普通字典，方便后续 CLI 或 Web 展示。

当前只实现 `quality="fast"`。`quality="full"` 预留给后续 summary node，不在这一版假装支持。

## Web 异步入库

网页添加作者只负责编排，不改变本模块的父文档、子节点和 Qdrant collection
合同。后台任务依次执行：

```text
staging/raw
-> build_corpus(staging/raw, staging/index)
-> index_corpus(staging/index, staging/index/qdrant)
-> 验证 parents.jsonl、nodes.jsonl、qdrant_manifest.json 和 qdrant/
-> 激活为作者正式 raw/ 与 index/
```

一个作者同一时间只能有一个构建任务。不同作者仍使用各自 collection。

同步已有作者时，旧 `index/` 在新索引完成前继续提供检索。新索引失败时不得
删除或覆盖旧索引；只有 staged index 验证成功后才执行目录切换。第一版对
几百篇规模的作者语料采用“raw 增量合并、索引完整重建”，不做逐点向量更新。

### `embeddings.py`

负责 embedding 适配：

- `BgeM3Encoder`：封装 `FlagEmbedding` 的 `BGEM3FlagModel`。
- `encode_texts(texts)`：对一批 node 文本同时输出 dense embedding 和 sparse lexical weights。
- `TextEmbedding`：一条文本的 dense+sparse 表征。
- `SparseEmbedding`：BGE-M3 lexical weights 转成 Qdrant sparse vector 需要的 `indices/values`。

当前只做 BGE-M3，不抽象多种 embedding provider。原因是第一版要先把当前验证过的 dense+sparse 路线跑通。

### `qdrant_index.py`

负责 Qdrant collection 和 point 转换：

- `collection_name_for_author(source, author_token)`：生成每个作者独立的 collection 名。
- `point_id_for_node(node_id)`：把内部 `node_id` 转成稳定 UUID。
- `create_local_client(path)`：创建本地 Qdrant client。
- `recreate_collection(client, collection_name, dense_size=...)`：重建 collection。
- `make_point(node, embedding)`：把一个 child node 和 embedding 转成 Qdrant point。
- `upload_points(...)`：批量 upsert points。

注意：Qdrant point id 不能依赖任意字符串。内部 `node_id` 形如：

```text
zhihu:answer:123:passage:0
```

这种字符串会通过 `uuid5` 转成稳定 UUID，原始 `node_id` 保留在 payload 里，方便 trace 和回填。

### `index.py`

负责把 `nodes.jsonl` 写入 Qdrant：

- `load_nodes(path)`：读取 `nodes.jsonl`。
- `index_corpus(index_dir, author=...)`：读取 nodes，调用 BGE-M3 编码，创建每作者 collection，写入 Qdrant，并输出 `qdrant_manifest.json`。
- `IndexResult`：记录 collection 名、节点数、dense 维度、Qdrant 本地路径和 manifest 路径。

默认 Qdrant 位于 `index/qdrant/` 时，manifest 中使用相对路径 `qdrant`。这样网页后台任务先在 staging 建库、再原子切换到正式作者目录后，清单不会残留 staging 的绝对路径。显式指定 index 目录外的 Qdrant 路径时才保留调用方传入的路径。

单元测试通过 fake encoder 和 fake client 验证编排逻辑，不要求 CI 下载 BGE-M3 或启动真实 Qdrant。

### `cli.py`

`pf build` 已接入 ingest：

```powershell
pf build <author-token>
```

默认读取：

```text
data/authors/zhihu/<author-token>/raw/
```

默认写入：

```text
data/authors/zhihu/<author-token>/index/
```

也可以显式传入：

```powershell
pf build mock-columnist --raw-dir samples/zhihu_mock_md --index-dir .tmp/sample-index --quality fast
```

`pf index` 已接入 Qdrant 入库：

```powershell
pf index <author-token>
```

默认读取：

```text
data/authors/zhihu/<author-token>/index/nodes.jsonl
```

默认写入本地 Qdrant：

```text
data/authors/zhihu/<author-token>/index/qdrant/
```

可显式指定：

```powershell
pf index mock-columnist --index-dir .tmp/sample-index --qdrant-path .tmp/sample-index/qdrant --embedding-device auto --batch-size 12
```

需要安装可选依赖：

```powershell
pip install -e ".[index]"
```

## Qdrant collection 设计 v0

第一版采用“每个作者一个 collection”：

```text
personaforge__<source>__<author-token>
```

例如：

```text
personaforge__zhihu__wu-ren-jun-28
```

选择这个方案的原因：

- 本地项目里作者数量有限，按作者隔离最简单。
- 删除某个作者时可以直接删除 collection。
- 不容易出现跨作者串库。
- 面试解释清楚，后续要做 hosted 多租户时再考虑统一 collection + `author_token` filter。

每个 Qdrant point 对应一个 child node，而不是 parent：

```text
point = title / lead / passage child node
```

vector：

```text
dense  = BGE-M3 dense embedding
sparse = BGE-M3 lexical weights
```

payload：

```text
node_id
parent_id
source
author_token
kind
source_id
node_type
title
path
index
```

Qdrant 只负责 child 检索，不保存 parent 全文。生成阶段需要的完整父文档从 `parents.jsonl` 回填。

## 入库路线 v0

```text
nodes.jsonl
-> BGE-M3 encode node.text
-> dense vector + sparse lexical weights
-> 每个 child node 转成 Qdrant point
-> 写入作者独立 collection
-> 输出 qdrant_manifest.json
```

当前 `pf index` 默认重建 collection。增量入库、删除同步、node hash 跳过重算都不进入 v0。

## 检索设计 v0

检索单位是子节点，生成上下文回填父文档全文。

完整路线：

```text
用户问题
-> Search Planner 判断是否需要联网，并生成 Tavily 搜索 query
-> 如果需要联网，Tavily 只检索题目事件/梗/人物的客观背景
-> Background + Query Transform 节点生成客观 background 和 4 路 retrieval query
-> 4 路 retrieval query 分别检索
-> 每个查询走 dense 和 sparse 两路
-> 每一路得到 child 排名列表
-> child 排名列表先折叠成 parent 排名列表
-> 每条 query 内 dense/sparse parent 排名先用 RRF 融合
-> 4 条 query 的 parent 排名再用 RRF 融合
-> 取 parent top20
-> 给 writer top20 父文档全文
```

### Query Understanding 与 Query Transform

第一版采用 2 次 LLM 调用，不把所有决策混成一个大 prompt。

#### 第一次 LLM：Search Planner

输入：

```text
原始知乎问题
```

输出：

```json
{
  "needs_web": true,
  "search_queries": [
    "用于 Tavily 的客观搜索词"
  ]
}
```

职责边界：

- 只判断是否需要联网。
- 只生成 Tavily search query。
- 普通词在题目中用法反常、缺少上下文，可能是近期黑话、标签或旧词新义时，必须联网消歧，不能直接采用字典第一义。
- 不生成 background。
- 不生成 retrieval query。
- 不预测作者立场。
- 不给 writer angle。

选择这样拆分的原因：

- Tavily 还没有返回结果前，不应该凭空写 background。
- Router 越小越稳定，越容易测试。
- 避免 query understanding 节点提前介入作者立场，污染后续 writer。

#### Tavily

Tavily 只用于补齐题目的客观背景：

```text
人物是谁
事件发生了什么
某个梗/外文词/热搜词是什么意思
一个普通词在当前网络语境中是否产生了新含义
当事人原话或争议点是什么
```

Tavily 不用于搜索“如何评价”类立场文章。搜索 query 应抽出实体、事件和关键词，例如：

```text
武亮 直播 大一 不需要买电脑 男生生活费1500 女生2000
```

而不是：

```text
如何评价武亮直播言论
```

#### 第二次 LLM：Background + Query Transform

输入：

```text
原始知乎问题
Tavily 搜索结果（如果 Search Planner 判定需要联网）
```

输出：

```json
{
  "objective_background": "题目涉及的词义、事件、人物或梗的客观解释。",
  "retrieval_queries": [
    {"route": "literal_question", "query": "..."},
    {"route": "event_background", "query": "..."},
    {"route": "mechanism_scene", "query": "..."},
    {"route": "colloquial_surface", "query": "..."}
  ]
}
```

`objective_background` 约束：

- `needs_web=false` 且没有搜索结果时，必须为空字符串。
- 有搜索结果时，最多 1-2 句。
- 只解释题目背景，不评价、不站队、不预测作者立场。
- 只回答“这题在说什么”，不能扩展成“涉及哪些社会问题/权力结构/现实困境”。
- 搜索确认旧词新义时，明确写出“此题中的 X 不是通常所指的 Y，而是指 Z”。

4 路 retrieval query：

- `literal_question`：保留题目字面意思，不扩展，不抽象。
- `event_background`：如果有联网背景，保留事件实体、关键词和关键事实；没有背景时接近 `literal_question`。
- `mechanism_scene`：把题目转成具体关系机制、行为动机、冲突场景和日常动作，不写成公共价值框架。
- `colloquial_surface`：换成知乎常见口语表达、网络表达和短词组合，利于 sparse lexical 命中。
- 经过联网消歧后，除 `literal_question` 可保留题面原词，其余路线都必须带入新义对应的具体人物和行为，不能继续围绕已排除的字面义检索。

4 路 query 不是最终 80 篇上下文。每路先独立召回和聚合，然后跨路 RRF 融合，最终仍只取 `parent_top_k=20`。

反例约束：

```text
问题：为什么很多女明星嫁入豪门后，都觉得自己上当了？
错误 objective_background：该问题涉及豪门婚姻中的权力不对等、家庭压力、女性自主权和经济控制等现实问题。
错误 retrieval query：豪门婚姻 权力不对等 女性自主权 经济控制
```

错误原因：这些不是背景解释，而是额外引入公共议题框架，会污染后续检索和生成。更合适的 query 应贴近“嫁豪门、婚后、觉得上当、女人、有钱男人、不满足”等具体场景和关系机制。

旧词新义回归要求：

- “为什么大多数人在嘲笑力工？”必须先搜索“力工”的网络语境，不得直接解释为体力劳动者。
- background 应明确区分字面义与题目采用的新义。
- 后续 retrieval query 应围绕“辛苦攒钱、彩礼、住房、结婚梭哈”等消歧后的行为，不得继续召回“体力劳动歧视”。

## child 到 parent 的聚合

第一版采用“首命中折叠”。

也就是：在每一条检索排名列表里，同一个父文档只保留第一次出现的位置。

例子：

```text
child 排名：
rank 1   parent A / passage 2
rank 10  parent B / passage 1
rank 20  parent B / lead
rank 30  parent A / passage 5
rank 40  parent C / title
```

折叠成：

```text
parent 排名：
rank 1   parent A
rank 10  parent B
rank 40  parent C
```

选择这个方案的原因：

- 不依赖 dense 或 sparse 的原始分数。
- 不会因为长文 child 多就天然占优势。
- 排名逻辑简单，便于调试和面试解释。
- 多个 child 命中的信息先记录到 trace，不参与 v0 排序。

暂时不采用“同一父文档多个 child 命中累加”的原因：

- 这会奖励长文。
- `rank10 + rank20` 可能超过 `rank1`，但这不一定代表更相关。
- 当前阶段更需要稳定和可解释，而不是复杂调参。

## RRF 融合

RRF 只看排名，不看原始分数。

公式：

```text
RRF(parent) = sum(1 / (k + rank))
```

第一版使用：

```text
k = 60
```

使用 RRF 的原因：

- dense 和 sparse 的分数尺度不可直接比较。
- 不同 query variant 的分数也不可直接比较。
- 排名融合比手写分数权重更稳定。

## top_k 约定

第一版默认：

```text
每个查询变体 dense child top100
每个查询变体 sparse child top100
最终 parent top20
```

选择 `parent top20` 的原因：

- 之前实验里，RAG20 是当前效果最好的方向。
- 给 writer 父文档全文，能保留作者完整表达方式。
- 当前目标优先效果和可解释，不优先省 token。

## 给 writer 的上下文

给 writer 的不是命中片段，而是 top20 父文档全文。

不写：

```text
材料1
材料2
命中第几名
```

因为 writer 应该把这些内容当成作者自己的历史表达来吸收，而不是在回答里显式提“材料里说”。

trace 中可以记录命中片段和排名，但不直接暴露给 writer。

## trace 设计

检索 trace 用于调试和评估，不默认写进普通聊天输出。

trace 应记录：

```text
原问题
query transform 结果
每一路 child 命中
child 到 parent 的首命中折叠过程
parent RRF 排名
最终进入上下文的 parent 列表
```

这样后续效果不好时，可以判断问题出在：

- query transform
- child 检索
- child 到 parent 聚合
- parent 融合
- writer 生成

## retrieve 实现状态

已实现基础检索链路：

```text
原始 query
-> BGE-M3 query embedding
-> Qdrant dense child topK
-> Qdrant sparse child topK
-> 每一路 child 首命中折叠 parent
-> parent RRF 融合
-> 回填 parents.jsonl
```

文件：

- `retrieve.py`
  - `retrieve_parents(...)`：检索入口。
  - `retrieve_parents_for_queries(...)`：多 query 检索入口，用于 Query Transform 后的 4 路 retrieval query。
  - `query_child_nodes(...)`：调用 Qdrant `query_points`，分别走 `using="dense"` 和 `using="sparse"`。
- `fuse_parent_hits(...)`：按首命中折叠 parent，再用 RRF 融合。
- `fuse_parent_rankings(...)`：跨 query 的 parent ranking RRF 融合。
- `fuse_transformed_dense_parent_rankings(...)`：只取四个 Query Transform 的 Dense
  child 排名，先按每个变换查询折叠 Parent，再跨四个查询做 Parent RRF；这是检验
  “Query Transform 本身”收益的离线对照路线，不改变在线默认的 Dense+Sparse 链路。
  - `load_parents(...)`：从 `parents.jsonl` 回填 parent 全文。
- `query_understanding.py`
  - `plan_web_search(...)`：第一次 LLM，判断是否需要联网和生成 Tavily query。
  - `TavilySearchClient`：调用 Tavily Search API。
  - `build_background_and_retrieval_queries(...)`：第二次 LLM，生成客观 background 和 4 路 retrieval query。
  - `build_grounded_query_plan(...)`：串行编排 Search Planner -> Tavily -> Background + Query Transform。

CLI：

```powershell
pf retrieve wu-ren-jun-28 "如何看待女生常说的配得感" --embedding-device cuda
```

默认 `query-mode=raw`，只跑原始 query 的 dense+sparse 两路，方便调试基础检索。

开启 query understanding + query transform：

```text
pf retrieve wu-ren-jun-28 "如何看待女生常说的配得感" --query-mode grounded --embedding-device cuda
```

`grounded` 模式会先用 DeepSeek 生成 Search Planner；如需要联网，则调用 Tavily；然后生成 4 路 retrieval query，并执行两层 RRF。可以用 `--trace-path` 写出完整 trace，方便检查问题出在 background、query transform、召回还是 parent 融合。

注意：当前 CLI 每次 retrieve 都会重新加载 BGE-M3。Web 服务里必须做全局 encoder 缓存，避免每个问题都产生模型冷启动。

## 后续可选实验

这些不进入第一版：

- BM25 三路融合。
- 对 title、lead、passage 人工加权。
- 同一 parent 多 child 弱加分。
- evidence selection。
- judge/rewrite 在线闭环。

这些都可以后续做消融，但不能现在把 v0 搞复杂。
