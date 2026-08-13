# PersonaForge RAG 链路说明

这份文档区分两个容易混淆的概念：

1. **生产 Chat 链路**：用户输入问题后，系统怎样得到最终给 Writer 的 20 篇历史材料。
2. **RAG 评估候选池**：为了比较不同检索分支，我们怎样保存每条路由的候选，并给候选做标注。

## 一、从用户输入开始

用户输入一条问题后，Web 服务先处理多轮上下文。对于“这个呢”“那男的呢”这类追问，Turn Planner 会结合当前会话摘要和最近消息，把它解析成一个可以独立检索的问题 `resolved_question`。

然后有两种查询模式：

- `raw`：直接使用原问题或解析后的独立问题检索，不调用联网判断和 Query Transform。
- `grounded`：默认 Chat 使用的模式，先做联网判断和 Query Transform。

## 二、Query Understanding 不是三条检索问题

当前代码里有两个不同的“最多 3 个”：

### 1. Tavily 搜索词最多 3 个

Search Planner 是第一次 LLM 调用。它只回答：当前问题是否需要联网，以及交给 Tavily 的搜索词是什么。

例如：

```text
原问题：为什么大多数人在嘲笑力工？

Tavily 搜索词：
1. "力工" 网络梗 是什么意思
2. "力工梭哈" 彩礼 婚恋
```

这是为了查清“力工”在当前问题里的含义，不是给作者历史材料做向量检索。

### 2. 本地历史材料的 Query Transform 固定输出 4 个变体

第二次 LLM 调用会读取原问题和可选的 Tavily 背景，输出：

1. `literal_question`：问题的字面表达。
2. `event_background`：带入必要的事件或热点背景。
3. `mechanism_scene`：转成具体人物、行为、动机和冲突场景。
4. `colloquial_surface`：转成更适合中文词法检索的口语和网络表达。

即使不需要联网，也会输出这 4 个变体；此时 `objective_background` 为空。

所以：**Tavily 最多 3 个搜索词，不等于 Query Transform 只有 3 个本地检索问题。**

## 三、一个检索问题怎样变成材料

无论是原问题，还是四个变体中的任意一个，都会进入同一套本地检索：

```text
一个检索问题
  -> BGE-M3 编码
       -> dense 向量
       -> sparse 权重
  -> Qdrant dense child Top100
  -> Qdrant sparse child Top100
  -> 每一路 child 命中折叠为 parent
  -> 该检索问题内部做 Dense+Sparse RRF
  -> 得到该检索问题的 parent Top30
```

这里的 child 是文章切出来的片段，parent 是完整的知乎回答、文章或想法。检索发生在 child 层，因为长文切片更容易命中；最终喂给 Writer 的是 parent 全文。

同一 parent 如果命中了多个 child，在同一路由里只保留它第一次出现的位置，不因为长文切出了更多片段就重复获得票数。

## 四、四个 Query Transform 怎样得到最终 Top20

四个变体会并行执行上面的检索流程：

```text
literal_question
  -> dense+sparse -> query 内 RRF Top30

event_background
  -> dense+sparse -> query 内 RRF Top30

mechanism_scene
  -> dense+sparse -> query 内 RRF Top30

colloquial_surface
  -> dense+sparse -> query 内 RRF Top30
```

然后再做第二层 RRF：

```text
四个变体各自的 parent Top30
  -> 按 parent_id 合并
  -> 跨变体 RRF
  -> 最终 parent Top20
```

因此它不是“4 个问题简单拼接成 120 篇”，而是每个变体先完成自己的 Dense+Sparse 融合，再通过 parent 的跨变体排名融合，最后只保留 20 篇。

## 五、原问题 raw 模式怎样检索

生产 Chat 的 `raw` 模式只有一个问题，但仍然会做 Dense+Sparse：

```text
原问题
  -> BGE-M3 dense child Top100
  -> BGE-M3 sparse child Top100
  -> parent 折叠
  -> Dense+Sparse RRF
  -> parent Top20
```

这就是当前系统中的“原问题混合检索”。它存在于生产 Chat 链路，但此前没有作为 RAG 评估池的一条独立 route 保存。

## 六、为什么评估池现在看起来是四路

当前评估池为了分析各个分支，保存的是：

1. `raw_dense`：原问题 Dense parent Top30。
2. `raw_sparse`：原问题 Sparse parent Top30。
3. `transformed_rrf`：四个变体经过两层 RRF 后的 parent Top30。
4. `raw_bm25`：原问题经过中文分词 BM25 后的 parent Top30。

这四条路是**诊断路由**，不是四个完全对称的生产方案。它们能回答：

- Dense 单独表现怎样？
- Sparse 单独表现怎样？
- Query Transform 加多路 RRF 后表现怎样？
- BM25 作为词法基线表现怎样？

但它不能直接回答：

```text
原问题 Dense+Sparse RRF
vs
Query Transform + Dense+Sparse + 跨变体 RRF
```

因此正式比较时建议新增第五条：

```text
raw_hybrid_rrf
```

它就是原问题 Dense+Sparse parent RRF Top30。这样既保留 `raw_dense` 和 `raw_sparse` 的诊断价值，又有一个真正的原问题混合检索基线。现有四路结果不需要删除或重算含义，只需要增加这一条并在前端标成“原问题混合基线”。

## 七、RAG 评估怎样形成候选池

对于每一道 dev 题，评估器保存各路 Top30 的 parent，并按 `(item_id, parent_id)` 去重：

```text
raw_dense Top30
raw_sparse Top30
transformed_rrf Top30
raw_bm25 Top30
[未来 raw_hybrid_rrf Top30]
        |
        v
按 parent_id 合并，不再按分数截断
        |
        v
该问题的候选 parent 池
```

候选池不是“最终 RAG 给 Writer 的 20 篇”。它是为了评估召回覆盖而故意保留得更宽。对候选池中的每个“问题-材料”组合，LLM 或人工给出：

- `0`：无用
- `1`：有一定帮助
- `2`：明显有用

然后指标只回看某条路自己的 Top3：

- `Hit@3`：Top3 中有没有至少一篇有用材料。
- `MRR@3`：第一篇有用材料排得越靠前越高。
- `nDCG@3`：同时考虑 0/1/2 的等级和排序。
- `Precision@3`：Top3 中有用材料的比例。

所以候选池宽，和指标的 `K=3`，是两件不同的事。候选池负责尽量不漏掉可能有用的材料；指标负责检查每条路把有用材料排到前几名的能力。

## 八、给 Writer 的最终上下文

默认 grounded Chat 的最终输入大致是：

```text
Writer LLM
  system prompt
  + 当前问题
  + 必要的 objective_background
  + Query Transform 跨变体 RRF 得到的 parent Top20 全文
  + 当前会话摘要
  + 最近三轮对话
  + 用户记忆
  -> 最终回答
```

RAG 评估页面显示的候选池材料，不等于每次实际生成都把 739 条材料塞给 Writer。739 是 10 道题所有路由候选的总 query-parent 对；单次生成仍然只使用最终选出的 Top20 或具体实验配置中的 Top5/Top10。

## 九、当前结论

你对现有命名的疑惑是正确的。当前代码的真实情况是：

- Query Transform 有 4 个本地检索问题。
- `max_search_queries=3` 只限制 Tavily 搜索词。
- 生产 Chat 已有原问题 Dense+Sparse RRF，但 RAG 评估池没有把它作为单独 route。
- 后续应新增 `raw_hybrid_rrf`，形成“原问题混合基线 vs Query Transform 混合检索”的直接对照。

相关实现：

- `src/personaforge/ingest/query_understanding.py`
- `src/personaforge/ingest/retrieve.py`
- `src/personaforge/eval/retrieval_pool.py`
- `src/personaforge/eval/retrieval_metrics.py`
- `src/personaforge/web/service.py`
