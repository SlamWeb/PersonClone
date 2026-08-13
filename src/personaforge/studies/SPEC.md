# 人类实验工具规格

`src/personaforge/studies` 承载论文实验需要、但仍值得工程化复用的人类实验工具。
它与 `src/personaforge/eval` 分开：后者评估系统，前者管理盲化材料、参与者任务和实验数据。

## Study 1 目标

研究协议与产品实现合同分开维护。最终冻结协议见
[`docs/research/STUDY1_PROTOCOL_V2.md`](../../../docs/research/STUDY1_PROTOCOL_V2.md)。
协议版本为 `study1-v2`：两篇单篇、阶段过渡、两组配对。旧 `2 + 3` 数据只作 legacy 回放与导出，正式招募不得继续使用旧流程。

Study 1 研究“What Makes LLM Sound Like the Author?”。正式参与者是熟悉目标作者的长期读者，
每人完成：

```text
2 个匿名单篇判断
2 个同题匿名配对判断
```

两组配对固定覆盖：

1. 目标作者 Gold vs AI：总体作者相似性差距；
2. AI vs AI：生成方法的相对作者相似性。

单篇与配对不能使用同一道问题。正式实验前必须 pilot 阅读时长，目标控制在 30 分钟内。

## Study 1 Web 实验合同

Study 1 集成到产品 Web，但与系统质量评估保持独立边界：

```text
Chat / Evaluate / Experiment
```

`/experiment/<study_id>` 是无需产品账号的作者专属匿名参与者入口；产品顶部的
`Experiment` 只跳转 `/experiment/admin`，复用现有管理员账号并查看研究进度与提交。
旧的 `/experiment` 保留为单作者兼容入口，固定读取排序后的第一份材料库，
正式招募不得使用这个不显式指定作者的地址。参与者使用研究者预先生成的匿名参与码进入，
同一参与码只能用于其所属 `study_id`，并稳定绑定同一份材料分配、
草稿和进度。真实材料继续来自 Git 忽略的 `data/studies/**/material_bank.json`，创建会话
时把完整分配快照写入 SQLite，源文件变化不能改写已经开始的实验。

浏览器按 `study_id` 在本地保存最近一次 session，用于参与者刷新页面或关闭后继续。
完成页必须提供“使用其他参与码”，清除的只是当前浏览器恢复指针，不删除已提交数据。
管理员后台的“打开新参与者入口”使用 `?new=1` 在新标签打开，进入后立即清掉该参数和
当前浏览器旧指针；管理员复制给正式被试的普通链接不带该参数，仍支持断点恢复。

### 会话安全、冻结与并发合同

- 参与码只用于首次进入或跨设备恢复。服务端会为该次恢复签发 256-bit 随机会话凭据，
  数据库只保存其 SHA-256；之后读取状态、保存草稿、提交、返回上一题和自由体验均必须携带
  该凭据。旧浏览器指针或旧凭据失效后，重新输入同一参与码即可轮换到新凭据，不会新建或丢失
  已有 session。
- 公开入口对“开始实验”和“自由体验”实施进程内 IP 限流。它是小规模本机实验的防护层，
  不替代正式公网部署时的 Cloudflare Access / WAF 限流。
- 每次保存当前题目时，校验当前 `current_index`、写入 response、以及提交后的
  `current_index + 1` 必须处于同一个 SQLite `BEGIN IMMEDIATE` 事务中。过期草稿不能覆盖
  已提交答案或推进后的下一题。
- 第一次生成参与码或开始实验时，系统冻结 canonical `material_bank.json` 的 SHA-256。
  同一 `study_id` 的材料此后发生任何变化，系统拒绝继续生成邀请码或开始新 session；研究者必须
  新建 `study_id`。session assignment 与导出记录都保存该材料指纹。

每位参与者完成 `2 + 2`：

1. 两个匿名单篇判断：整体分 `-2～+2`，并自由标注 `1～6` 处文字；
2. 两个匿名同题强制配对：`A 更像 / B 更像`，不提供平局；
3. 配对额外记录 `两篇很接近 / 比较确定 / 非常确定`；
4. 两个阶段之间必须确认过渡页；四个 trial 完成后只进行一次整体既往接触检查，再统一提交正式任务。

单篇和配对使用四个不重复问题。配对固定覆盖 `Gold vs AI` 与 `AI vs AI`；三种 AI 来源按稳定 cohort
轮换，使每位参与者在第二阶段恰好接触三个系统。A/B 位置和 trial 顺序稳定随机，单篇来源在五来源之间平衡轮换。

单篇至少一处、最多六处不重叠划线，不强制正负数量。每处划线记录稳定 ID、字符区间、原文、
`-2～+2` 影响分和简短理由。不展示预定义 feature 类别。配对不再划线，必须分别填写选择理由与不选择理由。
已保存的划线在提交前必须可以直接撤销、修改影响分或修改理由。单篇提交必须同时具备
整体分、1～6 处带理由划线和整篇关键理由；正式任务统一提交前允许通过“上一题”返回修改。统一提交后
全部锁定，不能返回修改。

参与者资料只收集关注时长、阅读频率、自评熟悉度和生成式 AI 使用频率，不收姓名、
手机号、年龄、性别或学历。页面自动保存草稿，刷新和关闭浏览器后可恢复。正式任务结束
后可选择与目标分身自由交流最多三轮并填写开放反馈；这些记录明确属于探索性体验数据，
不进入 Study 1 的主要 feature 分析。

管理员功能包括选择作者实验、批量生成该实验的参与码、复制作者专属参与链接、查看该实验
进行中/已完成状态、回放整体分、带影响分划线及配对理由，并按 `study_id` 导出分析包 ZIP。
回放界面必须把保存的字符区间重新渲染到匿名回答正文中，并清楚显示单篇的整体判断、
配对的 A/B 选择和确定程度、每处正负证据及其理由；原始 JSON 只属于导出数据，不能作为
管理员日常回放界面。
作者 A 的参与码不能进入作者 B 的实验，两个作者的进度和导出记录不能串联。已提交参与者
答案不能从后台修改。

每个参与码在 SQLite 中绑定独立 session，导出同时包含 `study_id`、`participant_code`
和 `session_id`。研究者在系统外维护“参与码 -> 招募对象”对应表；产品不收姓名。不同设备、
不同浏览器或同一设备依次使用不同参与码均不会共享答案。SQLite 使用 WAL、30 秒 busy
timeout 和短事务，支持小规模多人同时填写；同一参与码不得发给两个人。

## 多作者目录与上线状态

聊天产品中的“作者已入库”和研究实验中的“作者可招募”是两个不同状态：

```text
作者已入库：crawl -> build -> index -> 可聊天
作者可招募：时间切分数据集 -> 五来源材料 -> audit -> 人工冻结 -> 参与码
```

每位作者只能有自己独立的研究目录和唯一 `study_id`：

```text
data/studies/<study-directory>/material_bank.json
```

Web 启动时扫描所有 `data/studies/*/material_bank.json`。材料少于五题、任一道题缺少
五来源正文、JSON 损坏或 `study_id` 重复的实验会显示在管理员目录中，但不可生成参与码。
前端不允许参与者自己
选择作者；研究者应把 `/experiment/<study_id>` 专属链接发给对应作者的读者。

## 五来源材料池

每道冻结问题最多包含五种回答：

- `gold`：目标作者原回答；
- `rag_identity`：RAG20 + 强身份 Prompt，无 Persona Pack；
- `persona_pack`：当前冻结 Persona Pack 系统；
- `codex`：隔离 Codex 会话，输入同一题的 RAG20、Persona Pack 与通用身份指令；
- `other_human`：同一知乎问题下另一位真人的公开回答。

材料正文、知乎作者标识和真实实验配置只能写入 `data/studies/`，该目录必须被 Git 忽略。
代码、公开 schema 和自造 sample 可以提交。

## 材料构建约束

- 时间切分后的 Gold 不能进入任何 AI 生成上下文。
- Codex 每题使用新的 `--ephemeral` 进程，并在只含允许输入的隔离目录运行。
- 记录 Codex 模型、CLI 版本、Prompt SHA-256、参考 Parent IDs 和输出 SHA-256。
- 其他真人回答先保存候选池，再按稳定随机种子冻结；不能看完后挑最像或最不像的一篇。
- 只排除目标作者本人、空文本、纯图片/纯玩梗和低于最低可读长度的回答。
- 不按目标作者 Gold 长度匹配其他真人，也不截断或扩写；字数、段落数和句子数进入审计元数据。
- 对极端长度差异只做风险标记，不自动删除，让研究者在冻结正式刺激前统一审计。
- 知乎抓取复用本机 `data/auth/zhihu_storage_state.json`，不输出 Cookie。

## 新增作者标准流程

下面以 `new-author` 为例。这个流程不要求为每位作者复制 Python 文件，只需要准备该作者
自己的时间切分数据集、两个冻结系统 run、Parent store 和有证据来源的 Persona Pack：

```powershell
$AUTHOR = "new-author"
$STUDY = "new-author-study1-dev10-v2"
$DATASET = "data/eval/new-author-temporal-v0"
$OUT = "data/studies/new-author-study1-dev10"
$PARENTS = "data/authors/zhihu/new-author/index/parents.jsonl"
$PACK = "data/authors/zhihu/new-author/persona_pack.json"

python -m personaforge.studies.study1_materials prepare `
  --author $AUTHOR `
  --author-label "作者显示名" `
  --study-id $STUDY `
  --dataset-dir $DATASET `
  --out-dir $OUT `
  --parent-store $PARENTS `
  --rag-run "baseline-dev-v0" `
  --persona-run "persona-pack-dev-v0" `
  --persona-pack $PACK

python -m personaforge.studies.study1_materials collect-humans `
  --out-dir $OUT

python -m personaforge.studies.study1_materials generate-codex `
  --dataset-dir $DATASET `
  --out-dir $OUT `
  --parent-store $PARENTS

python -m personaforge.studies.study1_materials audit `
  --out-dir $OUT
```

已经完成五来源审计、只需要从旧协议迁移刺激材料时，不重新生成正文：

```powershell
python -m personaforge.studies.study1_materials clone-v2 `
  --source data/studies/old-study/material_bank.json `
  --out-dir data/studies/new-study-v2 `
  --study-id new-study-v2
```

该命令只复制刺激文本并写入来源哈希；新目录必须使用从未招募过参与者的新 `study_id`。

`collect-humans` 会从材料库读取目标作者 token，`generate-codex` 会从材料库读取 Persona
Pack 路径，避免后续步骤误用另一位作者配置。审计通过并完成人工冻结后，刷新
`/experiment/admin`，选择该作者，生成参与码并复制专属链接；服务会在请求时重新扫描材料库，
无需为新增作者修改或重启后端。

## 首位作者兼容命令

```powershell
python -m personaforge.studies.study1_materials prepare
python -m personaforge.studies.study1_materials collect-humans
python -m personaforge.studies.study1_materials generate-codex
python -m personaforge.studies.study1_materials audit
```

无参数调用仍默认使用 `wu-ren-jun-28-temporal-v0` 的冻结 Dev10，便于复现现有实验。
默认新建的协议标识为 `wu-ren-jun-28-study1-dev10-v2`。
默认输出为：

```text
data/studies/wu-ren-jun-28-study1-dev10/
  material_bank.json
  other_human_candidates.jsonl
  codex_inputs/
  researcher_review/
  audit.json
  AUDIT.md
```

## 验收

- 10 道题与两个已有系统 run 可按 `item_id` 严格对齐。
- 每题恢复知乎 `question_id`，且 Gold answer ID 不会进入其他真人候选。
- 候选随机选择与输入顺序无关，重复运行保持一致。
- Codex 输入不包含 `gold_answer`，参考 Parent 不包含当前 Gold `parent_id`。
- 审计明确报告每题五来源完整性、长度统计、泄漏风险和待人工复核项。
- 两个 `study_id` 的参与码、会话、进度、回放和导出严格隔离。
- 新增作者只需要新增私有数据目录和运行参数，不需要修改 Study 1 Python 或 React 代码。
- 管理员下载分析包后可运行
  `python -m personaforge.studies.study1_analysis <analysis.zip> --coding <已编码 CSV>`，
  输出完整性、来源平衡、描述统计、Spearman 探索性相关和参与者聚类 bootstrap 95% CI。
