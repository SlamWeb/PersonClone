# 爬虫规格

`src/personaforge/crawler` 负责把公开创作者内容抓取成本地 Markdown corpus。

## 职责

- 接收平台用户名或主页 URL。
- 默认尝试 zero-touch public crawl。
- 失败时提供登录态 fallback，而不是默认要求登录。
- 输出作者级 raw corpus：`profile.json`、`manifest.jsonl` 和按类型分目录的 Markdown。
- 保持 crawler 输出格式可被 ingest 直接读取。

## 非职责

- 不做 embedding。
- 不建 Qdrant。
- 不调用 LLM。
- 不绕过验证码、风控或私有内容限制。
- 不把真实抓取内容提交到 git。

## 当前策略链

```text
PublicApiStrategy
-> BrowserPublicStrategy
-> LoggedInBrowserStrategy
```

第一版实现为：

- `ZhihuPublicCrawler`：普通公开 API 尝试，最无感知。
- `ZhihuBrowserCrawler`：Playwright fallback，可带 `storage_state`。

如果知乎要求登录、验证或限流，CLI 应该解释失败原因，并提示用户可选运行 `pf zhihu-login`。

## 文件说明

### `models.py`

定义 crawler 的统一数据模型：

- `ContentItem`：一篇 answer/article/pin。
- `CreatorProfile`：作者基础信息。
- `utc_now_iso()`：统一生成 UTC ISO 时间。

### `markdown.py`

把 `ContentItem` 渲染为 Markdown：

- `html_to_markdown()`
- `render_item_markdown()`
- `write_markdown_corpus()`
- `write_profile()`
- `item_filename()`
- `slugify_filename()`

### `storage.py`

小型存储工具：

- `write_jsonl()`

目前用于写 `manifest.jsonl`。

### `zhihu.py`

知乎公开 API 和 payload 解析：

- `parse_user_token()`
- `answer_payload_to_item()`
- `article_payload_to_item()`
- `pin_payload_to_item()`
- `profile_payload_to_profile()`
- `ZhihuPublicCrawler`

### `zhihu_browser.py`

Playwright fallback：

- `ZhihuBrowserCrawler`
- `save_zhihu_session()`
- `load_storage_state()`
- `normalize_zhihu_link()`
- `link_matches_kind()`
- `content_id_from_url()`

## CLI 入口

```powershell
pf crawl zhihu <author> --all
pf zhihu-login
pf forge zhihu <author> --quality fast
```

`pf forge` 是产品级一键入口，会复用本模块的 crawler 后继续 build、index 和 Web；
它不在 CLI 中另写一套抓取逻辑。若 `<data-dir>/auth/zhihu_storage_state.json`
已经存在，forge 会自动作为浏览器 fallback 使用。

默认输出：

```text
data/authors/zhihu/<author>/raw/
  profile.json
  manifest.jsonl
  answer/
    answer-*.md
  article/
    article-*.md
  pin/
    pin-*.md
```

## Raw Corpus 目录规范

第一版按“平台/作者/raw”组织本地语料：

```text
data/
  authors/
    zhihu/
      <author-token>/
        raw/
          profile.json
          manifest.jsonl
          answer/
          article/
          pin/
        index/
        cache/
        eval/
```

设计原因：

- 前端未来要支持多个作者，作者目录必须先隔离。
- `answer`、`article`、`pin` 的内容形态不同，人类排查和增量爬取时应该分开管理。
- 入库时仍然统一读 `manifest.jsonl`，不要把三类内容拆成三个互不相干的知识库。
- `profile.json` 放在作者 raw 根目录，前端可以直接拿昵称、头像和主页信息。

`manifest.jsonl` 是 raw corpus 的索引，每行对应一篇内容。除内容元数据外，必须包含：

```json
{"kind": "answer", "path": "answer/answer-xxx.md"}
```

`path` 是相对 raw 根目录的路径。后续 ingest 应优先按 manifest 的 `path` 读取文件，而不是靠扫描文件名猜类型。

为了迁移旧数据，ingest 可以兼容旧扁平结构：

```text
data/raw/<author>_md/
  profile.json
  manifest.jsonl
  answer-*.md
  article-*.md
  pin-*.md
```

但 crawler 新输出不再使用这个结构。

## Web 后台任务集成

Web 添加作者不另写一套知乎抓取器。后台 Worker 仍调用本模块公开入口，并把
输出定向到作者 staging 目录：

```text
data/authors/zhihu/<author>/staging/<job-id>/raw/
```

默认行为：

- 接受知乎用户名或主页 URL。
- 抓取全部可达的 answer、article 和 pin。
- 先使用公开策略。
- 对 answer、article、pin 分别判断公开策略结果；某一类型为空时，不能因为另一类型
  已经有结果而跳过 fallback。
- 如果任一请求类型没有获得内容，再读取服务端本地
  `data/auth/zhihu_storage_state.json` 作为浏览器 fallback。
- Playwright 自带 Chromium 未安装时，优先复用宿主机已有 Chrome，其次 Edge；都不可用
  才提示运行 `playwright install chromium`。
- Cookie 只存在于服务宿主机，不能由前端上传、读取或下载。
- 抓取任务失败时不能破坏作者当前正式 `raw/`。

已有作者同步时，staging 中的新 manifest 按 `(kind, id)` 与旧 manifest
合并，新记录覆盖旧记录，未重新抓到但仍存在的旧记录暂时保留。完整索引由
ingest 在合并后的 raw corpus 上重新构建。

作者头像属于 profile 资源。抓取成功后，编排层可以把远程头像缓存为：

```text
raw/assets/avatar.*
```

`profile.json` 继续保留原始 `avatar_url`，保证本地缓存失败时仍有回退。

## 验证入口

```powershell
python -m pytest tests/test_crawler_markdown.py tests/test_zhihu.py tests/test_zhihu_browser.py -q
```

测试默认离线，不依赖知乎网络状态或用户登录。

## 面试解释

可以这样讲：

> crawler 被设计成策略链：先尝试完全无感的 public API，再尝试浏览器公开抓取，最后才提示用户提供本地登录态。输出不是直接进向量库，而是先落成可审计 Markdown、profile 和 manifest，这样后续 ingest 可以复用同一套 raw corpus contract。
## 2026-07-05 实现笔记

已实现文件：

- `exceptions.py`：crawler 专用异常，区分普通错误、平台阻挡、源格式变化。
- `models.py`：`ContentItem` 和 `CreatorProfile`，作为所有 crawler 策略的统一输出。
- `markdown.py`：把富文本 HTML 渲染成可审计 Markdown，并写出 `manifest.jsonl`。
- `storage.py`：小型 JSONL 写入工具。
- `zhihu.py`：知乎公开 API 尝试与 payload 解析。
- `zhihu_browser.py`：Playwright fallback，支持 public page 和本地 `storage_state`。

当前 CLI：

```powershell
$env:PYTHONPATH='src'
python -m personaforge.cli crawl zhihu <author-token> --max-items 20
python -m personaforge.cli zhihu-login --storage-state data/auth/zhihu_storage_state.json
```

真实网络 smoke 结果：

- `--no-browser --max-items 3` 在当前环境下可以免登录抓到 3 条公开想法。
- 该结果只说明当前公开端点可用，不作为单元测试前提。

验证：

```powershell
python -m pytest -q
```

当前覆盖：

- Markdown 渲染和 manifest 落盘。
- Zhihu payload 到 `ContentItem` 的转换。
- 浏览器 fallback 的 URL 标准化、类型匹配、storage state 解析。
- CLI crawl 写出 `profile.json`、`manifest.jsonl` 和 `.md` 文件。

## 2026-07-05 与历史语料格式的兼容性检查

已对照迁移前的历史 crawler 和 raw corpus：

- 旧 raw corpus 包含 `answer-*`、`article-*`、`pin-*` 三类 Markdown。
- 三类文件共享同一个 raw contract：
  - YAML-like front matter
  - `source`
  - `kind`
  - `id`
  - `title`
  - `url`
  - `author_token`
  - `created_at`
  - `updated_at`
  - `fetched_at`
  - 平台统计字段，如 `question_id`、`voteup_count`、`comment_count`、`like_count`
  - 正文为 `# title` 加 Markdown body
- 开源版 `markdown.py` 当前输出与旧 contract 保持兼容。
- 开源版 `zhihu.py` 已补回旧版的 `js-initialData` 解析能力，用于页面 HTML fallback。
- 开源版 `zhihu_browser.py` 已补回 answer/article/pin 分类型 DOM selector。

旧 ingest loader 验证结果：

```text
zhihu:answer:<id>  answer   ok
zhihu:article:<id> article  ok
zhihu:pin:<id>     pin      ok
```

后续开源版 ingest 应显式把 `pin` 纳入 document kind，而不是只写 `answer/article`。原因是知乎“想法”没有问题标题，但仍然是重要作者语料，应作为 parent document 正常入库。

## 2026-07-06 多作者目录调整

当前 crawler 默认输出已改为：

```text
data/authors/zhihu/<author-token>/raw/
```

Markdown 默认按内容类型写入 `answer/`、`article/`、`pin/` 子目录。`manifest.jsonl` 每行包含相对路径 `path`，用于后续 ingest 稳定读取。

如果用户显式传入 `--out-dir`，仍然使用用户指定目录作为 raw 根目录，但其内部结构仍按类型分目录。
