"""Command line entrypoint for PersonaForge."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path
from typing import Iterable

from personaforge import __version__
from personaforge.crawler.exceptions import CrawlError
from personaforge.crawler.markdown import write_markdown_corpus, write_profile
from personaforge.crawler.models import ContentItem, ContentKind, CreatorProfile
from personaforge.crawler.zhihu import ZhihuPublicCrawler, fallback_profile, parse_user_token
from personaforge.crawler.zhihu_browser import ZhihuBrowserCrawler, save_zhihu_session
from personaforge.ingest.embeddings import BgeM3Encoder
from personaforge.ingest.build import build_corpus
from personaforge.ingest.index import index_corpus
from personaforge.ingest.query_understanding import build_grounded_query_plan, plan_to_trace
from personaforge.ingest.retrieve import retrieve_parents, retrieve_parents_for_queries
from personaforge.llm import DeepSeekJsonClient
from personaforge.persona.pack import load_persona_pack_for_index
from personaforge.persona.suggestions import generate_suggestions
from personaforge.persona.writer import WRITER_PROMPT_CHOICES, build_prompt_pack, generate_answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pf",
        description="PersonaForge: local-first creator persona RAG.",
    )
    parser.add_argument("--version", action="version", version=f"personaforge {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Create local data directories.")
    init_parser.add_argument("--data-dir", default="data", help="Local data root.")

    user_parser = subparsers.add_parser("user", help="Manage local PersonaForge users.")
    user_subparsers = user_parser.add_subparsers(dest="user_command", required=True)
    user_create_parser = user_subparsers.add_parser("create", help="Create an invited local user.")
    user_create_parser.add_argument("username", help="Login username.")
    user_create_parser.add_argument("--display-name", help="Name shown in the Web UI.")
    user_create_parser.add_argument("--admin", action="store_true", help="Create an administrator.")
    user_create_parser.add_argument("--data-dir", default="data", help="Local data root.")
    user_list_parser = user_subparsers.add_parser("list", help="List local users.")
    user_list_parser.add_argument("--data-dir", default="data", help="Local data root.")

    crawl_parser = subparsers.add_parser("crawl", help="Crawl a creator into local Markdown.")
    crawl_parser.add_argument("platform", choices=["zhihu"], help="Content platform.")
    crawl_parser.add_argument("author", help="Creator token or username.")
    crawl_parser.add_argument("--out-dir", help="Output raw Markdown directory.")
    crawl_parser.add_argument("--all", action="store_true", help="Crawl all reachable items.")
    crawl_parser.add_argument("--max-items", type=int, default=100, help="Maximum items to save unless --all is set.")
    crawl_parser.add_argument(
        "--kind",
        action="append",
        choices=["answer", "article", "pin"],
        help="Content kind to crawl. Can be repeated. Defaults to answer/article/pin.",
    )
    crawl_parser.add_argument("--delay-seconds", type=float, default=1.5, help="Delay between requests/scrolls.")
    crawl_parser.add_argument("--max-api-pages", type=int, default=10, help="Maximum API pages per kind.")
    crawl_parser.add_argument(
        "--storage-state",
        type=Path,
        help="Optional Playwright storage_state JSON for logged-in fallback.",
    )
    crawl_parser.add_argument("--headed", action="store_true", help="Open a visible browser for fallback crawling.")
    crawl_parser.add_argument("--no-api", action="store_true", help="Skip API strategies and use browser page crawling.")
    crawl_parser.add_argument("--no-browser", action="store_true", help="Do not use Playwright fallback.")
    crawl_parser.add_argument("--quiet", action="store_true", help="Hide crawl progress messages.")

    login_parser = subparsers.add_parser("zhihu-login", help="Save a local Zhihu browser login state.")
    login_parser.add_argument(
        "--storage-state",
        type=Path,
        default=Path("data/auth/zhihu_storage_state.json"),
        help="Where to save Playwright storage_state JSON.",
    )
    login_parser.add_argument("--timeout-seconds", type=float, default=300.0)

    build_index_parser = subparsers.add_parser("build", help="Build a local index from Markdown.")
    build_index_parser.add_argument("author", nargs="?", help="Creator token.")
    build_index_parser.add_argument("--raw-dir", help="Existing raw Markdown directory.")
    build_index_parser.add_argument("--index-dir", help="Output index directory.")
    build_index_parser.add_argument(
        "--quality",
        choices=["fast", "full"],
        default="fast",
        help="Build quality. fast avoids LLM preprocessing.",
    )

    index_parser = subparsers.add_parser("index", help="Embed nodes and write a local Qdrant collection.")
    index_parser.add_argument("author", help="Creator token.")
    index_parser.add_argument("--index-dir", help="Directory containing nodes.jsonl.")
    index_parser.add_argument("--qdrant-path", help="Local Qdrant storage path.")
    index_parser.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model name.")
    index_parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for BGE-M3 embedding.",
    )
    index_parser.add_argument("--batch-size", type=int, default=12, help="Embedding batch size.")
    index_parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 when loading BGE-M3.")

    retrieve_parser = subparsers.add_parser("retrieve", help="Run retrieval against a local Qdrant index.")
    retrieve_parser.add_argument("author", help="Creator token.")
    retrieve_parser.add_argument("query", help="User query.")
    retrieve_parser.add_argument("--index-dir", help="Directory containing parents.jsonl and nodes.jsonl.")
    retrieve_parser.add_argument("--qdrant-path", help="Local Qdrant storage path.")
    retrieve_parser.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model name.")
    retrieve_parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for query embedding.",
    )
    retrieve_parser.add_argument("--child-top-k", type=int, default=100)
    retrieve_parser.add_argument("--per-query-parent-k", type=int, default=30)
    retrieve_parser.add_argument("--parent-top-k", type=int, default=20)
    retrieve_parser.add_argument(
        "--query-mode",
        choices=["raw", "grounded"],
        default="raw",
        help="raw uses the original query only; grounded runs search planning, optional Tavily, and 4-way query transform.",
    )
    retrieve_parser.add_argument("--max-search-results", type=int, default=5)
    retrieve_parser.add_argument("--trace-path", help="Optional JSON file for query understanding and retrieval trace.")
    retrieve_parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 when loading BGE-M3.")

    ask_parser = subparsers.add_parser("ask", help="Retrieve context and generate a persona-style answer.")
    ask_parser.add_argument("author", help="Creator token.")
    ask_parser.add_argument("query", help="User query.")
    ask_parser.add_argument("--index-dir", help="Directory containing parents.jsonl and nodes.jsonl.")
    ask_parser.add_argument("--qdrant-path", help="Local Qdrant storage path.")
    ask_parser.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model name.")
    ask_parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for query embedding.",
    )
    ask_parser.add_argument("--child-top-k", type=int, default=100)
    ask_parser.add_argument("--per-query-parent-k", type=int, default=30)
    ask_parser.add_argument("--parent-top-k", type=int, default=20)
    ask_parser.add_argument(
        "--query-mode",
        choices=["raw", "grounded"],
        default="grounded",
        help="grounded runs search planning, optional Tavily, and 4-way query transform before generation.",
    )
    ask_parser.add_argument("--max-search-results", type=int, default=5)
    ask_parser.add_argument("--temperature", type=float, default=0.85)
    ask_parser.add_argument("--max-tokens", type=int, default=1600)
    ask_parser.add_argument(
        "--writer-prompt",
        choices=WRITER_PROMPT_CHOICES,
        default="current",
        help="Writer prompt variant. current keeps the tuned anti-AI prompt; strong_identity tests a generic identity-immersion prompt.",
    )
    ask_parser.add_argument("--trace-path", help="Optional JSON file for query, retrieval, and answer trace.")
    ask_parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 when loading BGE-M3.")

    prompt_pack_parser = subparsers.add_parser(
        "prompt-pack",
        help="Retrieve context and export a pasteable ChatGPT prompt pack without calling the writer LLM.",
    )
    prompt_pack_parser.add_argument("author", help="Creator token.")
    prompt_pack_parser.add_argument("query", help="User query.")
    prompt_pack_parser.add_argument("--index-dir", help="Directory containing parents.jsonl and nodes.jsonl.")
    prompt_pack_parser.add_argument("--qdrant-path", help="Local Qdrant storage path.")
    prompt_pack_parser.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model name.")
    prompt_pack_parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for query embedding.",
    )
    prompt_pack_parser.add_argument("--child-top-k", type=int, default=100)
    prompt_pack_parser.add_argument("--per-query-parent-k", type=int, default=30)
    prompt_pack_parser.add_argument("--parent-top-k", type=int, default=20)
    prompt_pack_parser.add_argument(
        "--query-mode",
        choices=["raw", "grounded"],
        default="grounded",
        help="grounded runs search planning, optional Tavily, and 4-way query transform before prompt export.",
    )
    prompt_pack_parser.add_argument("--max-search-results", type=int, default=5)
    prompt_pack_parser.add_argument(
        "--writer-prompt",
        choices=WRITER_PROMPT_CHOICES,
        default="strong_identity",
        help="Writer prompt variant to export.",
    )
    prompt_pack_parser.add_argument("--out", help="Output Markdown file. Prints to stdout when omitted.")
    prompt_pack_parser.add_argument("--trace-path", help="Optional JSON file for query and retrieval trace.")
    prompt_pack_parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 when loading BGE-M3.")

    suggest_parser = subparsers.add_parser("suggest", help="Generate product-facing suggested questions for a persona.")
    suggest_parser.add_argument("author", help="Creator token.")
    suggest_parser.add_argument("--index-dir", help="Directory containing parents.jsonl.")
    suggest_parser.add_argument("--out", help="Output suggestions JSON path.")
    suggest_parser.add_argument("--count", type=int, default=6, help="Number of suggestions to keep.")
    suggest_parser.add_argument("--source-limit", type=int, default=80, help="Number of history titles to send to the LLM.")

    eval_parser = subparsers.add_parser("eval", help="Prepare and run leak-safe temporal evaluation.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_prepare_parser = eval_subparsers.add_parser("prepare", help="Build temporal dev/test holdouts from parents.jsonl.")
    eval_prepare_parser.add_argument("author", help="Creator token.")
    eval_prepare_parser.add_argument("--index-dir", help="Directory containing parents.jsonl.")
    eval_prepare_parser.add_argument("--out-dir", help="Output dataset directory under data/eval by default.")
    eval_prepare_parser.add_argument("--dev-size", type=int, default=10)
    eval_prepare_parser.add_argument("--test-size", type=int, default=20)
    eval_prepare_parser.add_argument("--min-answer-characters", type=int, default=200)

    eval_run_parser = eval_subparsers.add_parser("run", help="Generate answers for one prepared eval split.")
    eval_run_parser.add_argument("author", help="Creator token.")
    eval_run_parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl from pf eval prepare.")
    eval_run_parser.add_argument("--index-dir", help="Directory containing parents.jsonl and nodes.jsonl.")
    eval_run_parser.add_argument("--qdrant-path", help="Local Qdrant storage path.")
    eval_run_parser.add_argument(
        "--persona-pack-path",
        help="Optional Persona Pack path when the eval index is outside the author directory.",
    )
    eval_run_parser.add_argument("--out-dir", help="Dataset directory. Defaults to the dataset parent directory.")
    eval_run_parser.add_argument("--run-name", required=True, help="Unique local name for this experiment run.")
    eval_run_parser.add_argument("--split", choices=["dev", "test"], default="dev")
    eval_run_parser.add_argument("--limit", type=int, help="Optional item limit for smoke testing.")
    eval_run_parser.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model name.")
    eval_run_parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for query embedding.",
    )
    eval_run_parser.add_argument("--child-top-k", type=int, default=100)
    eval_run_parser.add_argument("--per-query-parent-k", type=int, default=30)
    eval_run_parser.add_argument("--parent-top-k", type=int, default=20)
    eval_run_parser.add_argument("--query-mode", choices=["raw", "grounded"], default="grounded")
    eval_run_parser.add_argument("--max-search-results", type=int, default=5)
    eval_run_parser.add_argument("--temperature", type=float, default=0.85)
    eval_run_parser.add_argument("--max-tokens", type=int, default=1600)
    eval_run_parser.add_argument(
        "--writer-prompt",
        choices=WRITER_PROMPT_CHOICES,
        default="strong_identity",
        help="Writer prompt variant. Eval defaults to the current strong identity baseline.",
    )
    eval_run_parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 when loading BGE-M3.")

    web_parser = subparsers.add_parser("web", help="Start the local Web UI.")
    web_parser.add_argument("author", nargs="?", help="Creator token.")
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host. Use 0.0.0.0 inside Docker or on a server.",
    )
    web_parser.add_argument("--port", type=int, default=8000)
    web_parser.add_argument("--data-dir", default="data", help="Local data root.")
    web_parser.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model name.")
    web_parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for query embedding.",
    )
    web_parser.add_argument("--child-top-k", type=int, default=100)
    web_parser.add_argument("--per-query-parent-k", type=int, default=30)
    web_parser.add_argument("--parent-top-k", type=int, default=20)
    web_parser.add_argument("--max-search-results", type=int, default=5)
    web_parser.add_argument("--temperature", type=float, default=0.85)
    web_parser.add_argument("--max-tokens", type=int, default=1600)
    web_parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 when loading BGE-M3.")

    forge_parser = subparsers.add_parser(
        "forge",
        help="Crawl, build, index, and start the local Web UI.",
    )
    forge_parser.add_argument("platform", choices=["zhihu"])
    forge_parser.add_argument("author")
    forge_parser.add_argument("--quality", choices=["fast", "full"], default="fast")
    forge_parser.add_argument("--data-dir", default="data", help="Local data root.")
    forge_parser.add_argument("--model-name", default="BAAI/bge-m3")
    forge_parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    forge_parser.add_argument("--batch-size", type=int, default=12)
    forge_parser.add_argument("--no-fp16", action="store_true")
    forge_parser.add_argument(
        "--max-items",
        type=int,
        help="Limit crawled items for a smoke run. Defaults to all reachable items.",
    )
    forge_parser.add_argument("--delay-seconds", type=float, default=1.5)
    forge_parser.add_argument(
        "--max-api-pages",
        type=int,
        default=100,
        help="Safety cap per content kind; crawling still stops when the source is exhausted.",
    )
    forge_parser.add_argument("--storage-state", type=Path)
    forge_parser.add_argument("--headed", action="store_true")
    forge_parser.add_argument("--no-api", action="store_true")
    forge_parser.add_argument("--no-browser", action="store_true")
    forge_parser.add_argument("--quiet", action="store_true")
    forge_parser.add_argument("--skip-crawl", action="store_true")
    forge_parser.add_argument("--skip-build", action="store_true")
    forge_parser.add_argument("--skip-index", action="store_true")
    forge_parser.add_argument(
        "--no-web",
        action="store_true",
        help="Stop after indexing instead of starting the Web UI.",
    )
    forge_parser.add_argument("--host", default="127.0.0.1")
    forge_parser.add_argument("--port", type=int, default=8000)
    forge_parser.add_argument("--child-top-k", type=int, default=100)
    forge_parser.add_argument("--per-query-parent-k", type=int, default=30)
    forge_parser.add_argument("--parent-top-k", type=int, default=20)
    forge_parser.add_argument("--max-search-results", type=int, default=5)
    forge_parser.add_argument("--temperature", type=float, default=0.85)
    forge_parser.add_argument("--max-tokens", type=int, default=1600)

    return parser


def _ensure_data_dirs(data_dir: Path) -> list[Path]:
    paths = [
        data_dir / "authors",
        data_dir / "raw",
        data_dir / "index",
        data_dir / "auth",
        data_dir / "models",
        data_dir / "eval",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        paths = _ensure_data_dirs(Path(args.data_dir))
        print("Created local data directories:")
        for path in paths:
            print(f"- {path}")
        return 0

    if args.command == "user":
        return _run_user(args)

    if args.command == "crawl":
        return _run_crawl(args)

    if args.command == "zhihu-login":
        return _run_zhihu_login(args)

    if args.command == "build":
        return _run_build(args)

    if args.command == "index":
        return _run_index(args)

    if args.command == "retrieve":
        return _run_retrieve(args)

    if args.command == "ask":
        return _run_ask(args)

    if args.command == "prompt-pack":
        return _run_prompt_pack(args)

    if args.command == "suggest":
        return _run_suggest(args)

    if args.command == "eval":
        return _run_eval(args)

    if args.command == "web":
        return _run_web(args)

    if args.command == "forge":
        return _run_forge(args)

    parser.print_help()
    return 0


def _run_user(args: argparse.Namespace) -> int:
    try:
        from personaforge.web.auth import AuthStore
    except ImportError as exc:
        raise RuntimeError('User management requires Web dependencies: pip install -e ".[web]"') from exc

    store = AuthStore(Path(args.data_dir))
    if args.user_command == "list":
        users = store.list_users()
        if not users:
            print("No local users. Start the Web UI to create the first administrator.")
            return 0
        for user in users:
            print(f"{user.username}\t{user.role}\t{user.display_name}")
        return 0

    password = getpass.getpass("Password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise ValueError("Passwords do not match.")
    first_user = not store.has_users()
    user = store.create_user(
        username=args.username,
        password=password,
        display_name=args.display_name,
        role="admin" if args.admin or first_user else "member",
        claim_local_data=first_user,
    )
    print(f"Created {user.role}: {user.username}")
    if first_user:
        print("Existing local-user conversations were assigned to this account.")
    return 0


def _run_crawl(args: argparse.Namespace) -> int:
    if args.platform != "zhihu":
        raise ValueError(f"Unsupported platform: {args.platform}")

    token = parse_user_token(args.author)
    out_dir = Path(args.out_dir) if args.out_dir else Path("data/authors") / "zhihu" / token / "raw"
    kinds = tuple(args.kind or ("answer", "article", "pin"))
    max_items = None if args.all else args.max_items
    progress = None if args.quiet else print

    profile: CreatorProfile | None = None
    items: list[ContentItem] = []
    errors: list[str] = []

    if not args.no_api:
        public = ZhihuPublicCrawler(
            delay_seconds=args.delay_seconds,
            max_api_pages=args.max_api_pages,
            progress=progress,
        )
        try:
            profile = public.crawl_profile(token)
        except CrawlError as exc:
            errors.append(f"public profile: {exc}")
        try:
            items.extend(public.crawl_user(token, kinds=_content_kinds(kinds), max_items=max_items))
        except CrawlError as exc:
            errors.append(f"public crawl: {exc}")

    if not items and not args.no_browser:
        browser = ZhihuBrowserCrawler(
            headless=not args.headed,
            storage_state=args.storage_state,
            delay_seconds=args.delay_seconds,
            use_api=not args.no_api,
            max_api_pages=args.max_api_pages,
            progress=progress,
        )
        try:
            profile = profile or browser.crawl_profile(token)
        except CrawlError as exc:
            errors.append(f"browser profile: {exc}")
        try:
            items.extend(browser.crawl_user(token, kinds=_content_kinds(kinds), max_items=max_items))
        except CrawlError as exc:
            errors.append(f"browser crawl: {exc}")

    if not items:
        print("No items were crawled.")
        if errors:
            print("Attempts:")
            for error in errors:
                print(f"- {error}")
        print("If the public route is blocked, run:")
        print("  pf zhihu-login --storage-state data/auth/zhihu_storage_state.json")
        print(
            "Then retry with "
            "--storage-state data/auth/zhihu_storage_state.json "
            "(use --headed if you want to see the fallback browser)."
        )
        return 2

    profile = profile or fallback_profile(token)
    write_profile(profile, out_dir)
    paths = write_markdown_corpus(items, out_dir)

    print(f"Saved {len(paths)} item(s) to {out_dir}")
    print(f"Profile: {out_dir / 'profile.json'}")
    print(f"Manifest: {out_dir / 'manifest.jsonl'}")
    return 0


def _run_zhihu_login(args: argparse.Namespace) -> int:
    save_zhihu_session(args.storage_state, timeout_seconds=args.timeout_seconds)
    print(f"Saved Zhihu storage state to {args.storage_state}")
    return 0


def _run_build(args: argparse.Namespace) -> int:
    if not args.author and not args.raw_dir:
        raise ValueError("`pf build` needs an author token or --raw-dir.")

    author = args.author or Path(args.raw_dir).name
    raw_dir = Path(args.raw_dir) if args.raw_dir else Path("data/authors") / "zhihu" / author / "raw"
    index_dir = Path(args.index_dir) if args.index_dir else Path("data/authors") / "zhihu" / author / "index"

    result = build_corpus(raw_dir, index_dir, quality=args.quality)

    print(f"Built ingest artifacts for {author}:")
    print(f"- parents: {result.parent_count} -> {result.parents_path}")
    print(f"- nodes: {result.node_count} -> {result.nodes_path}")
    print(f"- manifest: {result.manifest_path}")
    return 0


def _run_index(args: argparse.Namespace) -> int:
    index_dir = Path(args.index_dir) if args.index_dir else Path("data/authors") / "zhihu" / args.author / "index"
    qdrant_path = Path(args.qdrant_path) if args.qdrant_path else index_dir / "qdrant"
    encoder = BgeM3Encoder(
        args.model_name,
        device=args.embedding_device,
        use_fp16=not args.no_fp16,
    )
    result = index_corpus(
        index_dir,
        author=args.author,
        qdrant_path=qdrant_path,
        encoder=encoder,
        batch_size=args.batch_size,
    )

    print(f"Indexed {result.node_count} node(s) for {args.author}:")
    print(f"- collection: {result.collection_name}")
    print(f"- dense size: {result.dense_size}")
    print(f"- qdrant: {result.qdrant_path}")
    print(f"- manifest: {result.manifest_path}")
    return 0


def _run_retrieve(args: argparse.Namespace) -> int:
    index_dir = Path(args.index_dir) if args.index_dir else Path("data/authors") / "zhihu" / args.author / "index"
    qdrant_path = Path(args.qdrant_path) if args.qdrant_path else index_dir / "qdrant"
    encoder = BgeM3Encoder(
        args.model_name,
        device=args.embedding_device,
        use_fp16=not args.no_fp16,
    )
    query_trace = None
    if args.query_mode == "grounded":
        llm = DeepSeekJsonClient.from_env()
        plan = build_grounded_query_plan(
            args.query,
            llm=llm,
            max_results_per_query=args.max_search_results,
        )
        query_trace = plan_to_trace(plan)
        result = retrieve_parents_for_queries(
            args.query,
            plan.transform.retrieval_queries,
            author=args.author,
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            child_top_k=args.child_top_k,
            per_query_parent_k=args.per_query_parent_k,
            parent_top_k=args.parent_top_k,
        )
        print(f"Needs web: {plan.search_plan.needs_web}")
        if plan.search_plan.search_queries:
            print("Search queries:")
            for item in plan.search_plan.search_queries:
                print(f"- {item}")
        if plan.transform.objective_background:
            print(f"Objective background: {plan.transform.objective_background}")
        print("Retrieval queries:")
        for item in plan.transform.retrieval_queries:
            print(f"- {item.route}: {item.query}")
    else:
        result = retrieve_parents(
            args.query,
            author=args.author,
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            child_top_k=args.child_top_k,
            parent_top_k=args.parent_top_k,
        )

    if args.trace_path:
        _write_retrieve_trace(Path(args.trace_path), query_trace=query_trace, result=result)

    print(f"Query: {result.query}")
    print(f"Collection: {result.collection_name}")
    print("Top parents:")
    for hit in result.parents:
        routes = ", ".join(
            f"{child.route}#{child.rank}:{child.node_type}:{child.score:.4f}"
            for child in hit.first_hits
        )
        print(f"{hit.rank}. {hit.parent_id} | {hit.score:.6f} | {hit.title}")
        print(f"   path: {hit.path}")
        print(f"   first hits: {routes}")
    return 0


def _retrieve_for_generation(args: argparse.Namespace):
    index_dir = _generation_index_dir(args)
    qdrant_path = Path(args.qdrant_path) if args.qdrant_path else index_dir / "qdrant"
    encoder = BgeM3Encoder(
        args.model_name,
        device=args.embedding_device,
        use_fp16=not args.no_fp16,
    )
    query_trace = None
    objective_background = ""

    if args.query_mode == "grounded":
        llm = DeepSeekJsonClient.from_env()
        plan = build_grounded_query_plan(
            args.query,
            llm=llm,
            max_results_per_query=args.max_search_results,
        )
        query_trace = plan_to_trace(plan)
        objective_background = plan.transform.objective_background
        retrieve_result = retrieve_parents_for_queries(
            args.query,
            plan.transform.retrieval_queries,
            author=args.author,
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            child_top_k=args.child_top_k,
            per_query_parent_k=args.per_query_parent_k,
            parent_top_k=args.parent_top_k,
        )
    else:
        retrieve_result = retrieve_parents(
            args.query,
            author=args.author,
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            child_top_k=args.child_top_k,
            parent_top_k=args.parent_top_k,
        )

    return retrieve_result, query_trace, objective_background


def _generation_index_dir(args: argparse.Namespace) -> Path:
    return (
        Path(args.index_dir)
        if args.index_dir
        else Path("data/authors") / "zhihu" / args.author / "index"
    )


def _run_ask(args: argparse.Namespace) -> int:
    retrieve_result, query_trace, objective_background = _retrieve_for_generation(args)
    index_dir = _generation_index_dir(args)
    persona_pack = load_persona_pack_for_index(
        index_dir,
        required=args.writer_prompt == "persona_pack",
    ) if args.writer_prompt == "persona_pack" else None
    llm = DeepSeekJsonClient.from_env()
    answer = generate_answer(
        query=args.query,
        parent_hits=retrieve_result.parents,
        llm=llm,
        objective_background=objective_background,
        writer_prompt=args.writer_prompt,
        persona_pack=persona_pack,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.trace_path:
        _write_ask_trace(
            Path(args.trace_path),
            query_trace=query_trace,
            retrieve_result=retrieve_result,
            answer=answer,
            objective_background=objective_background,
        )

    print(answer.answer)
    return 0


def _run_prompt_pack(args: argparse.Namespace) -> int:
    retrieve_result, query_trace, objective_background = _retrieve_for_generation(args)
    index_dir = _generation_index_dir(args)
    persona_pack = load_persona_pack_for_index(
        index_dir,
        required=args.writer_prompt == "persona_pack",
    ) if args.writer_prompt == "persona_pack" else None
    prompt_pack = build_prompt_pack(
        query=args.query,
        parent_hits=retrieve_result.parents,
        objective_background=objective_background,
        writer_prompt=args.writer_prompt,
        persona_pack=persona_pack,
    )

    if args.trace_path:
        _write_prompt_pack_trace(
            Path(args.trace_path),
            query_trace=query_trace,
            retrieve_result=retrieve_result,
            objective_background=objective_background,
            writer_prompt=args.writer_prompt,
            persona_pack_id=persona_pack.pack_id if persona_pack else None,
            persona_pack_sha256=persona_pack.sha256 if persona_pack else None,
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(prompt_pack, encoding="utf-8", newline="\n")
        print(f"Wrote prompt pack: {out_path}")
    else:
        print(prompt_pack, end="")
    return 0


def _run_suggest(args: argparse.Namespace) -> int:
    index_dir = Path(args.index_dir) if args.index_dir else Path("data/authors") / "zhihu" / args.author / "index"
    out_path = (
        Path(args.out)
        if args.out
        else Path("data/authors") / "zhihu" / args.author / "profile_suggestions.json"
    )
    llm = DeepSeekJsonClient.from_env()
    result = generate_suggestions(
        author=args.author,
        index_dir=index_dir,
        out_path=out_path,
        llm=llm,
        count=args.count,
        source_limit=args.source_limit,
    )
    print(f"Generated {len(result.suggestions)} suggestion(s) from {result.source_title_count} title(s):")
    for idx, item in enumerate(result.suggestions, start=1):
        print(f"{idx}. {item}")
    print(f"Wrote: {result.path}")
    return 0


def _run_eval(args: argparse.Namespace) -> int:
    if args.eval_command == "prepare":
        from personaforge.eval.dataset import prepare_temporal_dataset

        index_dir = Path(args.index_dir) if args.index_dir else Path("data/authors") / "zhihu" / args.author / "index"
        out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else Path("data/eval") / f"{args.author}-temporal-dev{args.dev_size}-test{args.test_size}"
        )
        result = prepare_temporal_dataset(
            author=args.author,
            index_dir=index_dir,
            out_dir=out_dir,
            dev_size=args.dev_size,
            test_size=args.test_size,
            min_answer_characters=args.min_answer_characters,
        )
        print(f"Prepared temporal dataset for {args.author}:")
        print(f"- dev/test: {result.dev_count}/{result.test_count}")
        print(f"- cutoff: {result.cutoff}")
        print(f"- excluded parent docs: {result.excluded_parent_count}")
        print(f"- dataset: {result.dataset_path}")
        print(f"- manifest: {result.manifest_path}")
        return 0

    if args.eval_command == "run":
        from personaforge.eval.runner import EvalRunConfig, run_temporal_eval

        dataset_path = Path(args.dataset)
        index_dir = Path(args.index_dir) if args.index_dir else Path("data/authors") / "zhihu" / args.author / "index"
        qdrant_path = Path(args.qdrant_path) if args.qdrant_path else index_dir / "qdrant"
        out_dir = Path(args.out_dir) if args.out_dir else dataset_path.parent
        config = EvalRunConfig(
            author=args.author,
            dataset_path=dataset_path,
            split=args.split,
            run_name=args.run_name,
            out_dir=out_dir,
            query_mode=args.query_mode,
            writer_prompt=args.writer_prompt,
            child_top_k=args.child_top_k,
            per_query_parent_k=args.per_query_parent_k,
            parent_top_k=args.parent_top_k,
            max_search_results=args.max_search_results,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            limit=args.limit,
            persona_pack_path=Path(args.persona_pack_path) if args.persona_pack_path else None,
        )
        encoder = BgeM3Encoder(args.model_name, device=args.embedding_device, use_fp16=not args.no_fp16)
        result = run_temporal_eval(
            config,
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            llm=DeepSeekJsonClient.from_env(),
        )
        print(f"Completed {result.item_count} {args.split} eval item(s):")
        print(f"- run: {result.run_dir}")
        print(f"- manifest: {result.manifest_path}")
        print(f"- results: {result.runs_path}")
        print(f"- summary: {result.summary_path}")
        return 0

    raise ValueError(f"Unknown eval command: {args.eval_command}")


def _write_retrieve_trace(path: Path, *, query_trace: dict | None, result) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query_understanding": query_trace,
        "retrieval_queries": [
            {"route": item.route, "query": item.query} for item in result.retrieval_queries
        ],
        "routes": {
            name: [
                {
                    "rank": hit.rank,
                    "score": hit.score,
                    "node_id": hit.node_id,
                    "parent_id": hit.parent_id,
                    "node_type": hit.node_type,
                    "title": hit.title,
                    "path": hit.path,
                    "route": hit.route,
                }
                for hit in hits
            ]
            for name, hits in result.routes.items()
        },
        "parents": [
            {
                "rank": hit.rank,
                "parent_id": hit.parent_id,
                "score": hit.score,
                "title": hit.title,
                "path": hit.path,
                "first_hits": [
                    {
                        "rank": child.rank,
                        "score": child.score,
                        "node_id": child.node_id,
                        "node_type": child.node_type,
                        "route": child.route,
                    }
                    for child in hit.first_hits
                ],
            }
            for hit in result.parents
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def _write_ask_trace(path: Path, *, query_trace: dict | None, retrieve_result, answer, objective_background: str) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query_understanding": query_trace,
        "objective_background": objective_background,
        "retrieval_queries": [
            {"route": item.route, "query": item.query} for item in retrieve_result.retrieval_queries
        ],
        "parents": [
            {
                "rank": hit.rank,
                "parent_id": hit.parent_id,
                "score": hit.score,
                "title": hit.title,
                "path": hit.path,
                "first_hits": [
                    {
                        "rank": child.rank,
                        "score": child.score,
                        "node_id": child.node_id,
                        "node_type": child.node_type,
                        "route": child.route,
                    }
                    for child in hit.first_hits
                ],
            }
            for hit in retrieve_result.parents
        ],
        "writer_parent_titles": answer.parent_titles,
        "writer_prompt": answer.writer_prompt,
        "persona_pack_id": answer.persona_pack_id,
        "persona_pack_sha256": answer.persona_pack_sha256,
        "answer": answer.answer,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def _write_prompt_pack_trace(
    path: Path,
    *,
    query_trace: dict | None,
    retrieve_result,
    objective_background: str,
    writer_prompt: str,
    persona_pack_id: str | None,
    persona_pack_sha256: str | None,
) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query_understanding": query_trace,
        "objective_background": objective_background,
        "retrieval_queries": [
            {"route": item.route, "query": item.query} for item in retrieve_result.retrieval_queries
        ],
        "parents": [
            {
                "rank": hit.rank,
                "parent_id": hit.parent_id,
                "score": hit.score,
                "title": hit.title,
                "path": hit.path,
                "first_hits": [
                    {
                        "rank": child.rank,
                        "score": child.score,
                        "node_id": child.node_id,
                        "node_type": child.node_type,
                        "route": child.route,
                    }
                    for child in hit.first_hits
                ],
            }
            for hit in retrieve_result.parents
        ],
        "writer_parent_titles": [hit.title for hit in retrieve_result.parents],
        "writer_prompt": writer_prompt,
        "persona_pack_id": persona_pack_id,
        "persona_pack_sha256": persona_pack_sha256,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def _run_web(args: argparse.Namespace) -> int:
    from personaforge.web.app import run_web
    from personaforge.web.service import WebConfig

    config = WebConfig(
        author=args.author,
        data_dir=Path(args.data_dir),
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        embedding_device=args.embedding_device,
        use_fp16=not args.no_fp16,
        child_top_k=args.child_top_k,
        per_query_parent_k=args.per_query_parent_k,
        parent_top_k=args.parent_top_k,
        max_search_results=args.max_search_results,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    run_web(config)
    return 0


def _run_forge(args: argparse.Namespace) -> int:
    """Run the existing local pipeline stages without duplicating their logic."""

    if args.platform != "zhihu":
        raise ValueError(f"Unsupported platform: {args.platform}")
    author = parse_user_token(args.author)
    data_dir = Path(args.data_dir)
    author_dir = data_dir / "authors" / "zhihu" / author
    raw_dir = author_dir / "raw"
    index_dir = author_dir / "index"
    qdrant_path = index_dir / "qdrant"
    _ensure_data_dirs(data_dir)

    storage_state = args.storage_state
    default_storage_state = data_dir / "auth" / "zhihu_storage_state.json"
    if storage_state is None and default_storage_state.exists():
        storage_state = default_storage_state

    print(f"Forging local persona: zhihu/{author}")
    if args.skip_crawl:
        _require_forge_artifact(raw_dir, "crawl", "--skip-crawl")
        print(f"[1/4] Reusing raw Markdown: {raw_dir}")
    else:
        print("[1/4] Crawling public creator content")
        code = _run_crawl(
            argparse.Namespace(
                platform="zhihu",
                author=author,
                out_dir=str(raw_dir),
                all=args.max_items is None,
                max_items=args.max_items or 100,
                kind=None,
                delay_seconds=args.delay_seconds,
                max_api_pages=args.max_api_pages,
                storage_state=storage_state,
                headed=args.headed,
                no_api=args.no_api,
                no_browser=args.no_browser,
                quiet=args.quiet,
            )
        )
        if code:
            return code

    if args.skip_build:
        _require_forge_artifact(index_dir / "parents.jsonl", "build", "--skip-build")
        _require_forge_artifact(index_dir / "nodes.jsonl", "build", "--skip-build")
        print(f"[2/4] Reusing ingest artifacts: {index_dir}")
    else:
        print("[2/4] Building parent and child documents")
        code = _run_build(
            argparse.Namespace(
                author=author,
                raw_dir=str(raw_dir),
                index_dir=str(index_dir),
                quality=args.quality,
            )
        )
        if code:
            return code

    if args.skip_index:
        _require_forge_artifact(
            index_dir / "qdrant_manifest.json",
            "index",
            "--skip-index",
        )
        _require_forge_artifact(qdrant_path, "index", "--skip-index")
        print(f"[3/4] Reusing Qdrant index: {qdrant_path}")
    else:
        print("[3/4] Embedding child nodes and writing Qdrant index")
        code = _run_index(
            argparse.Namespace(
                author=author,
                index_dir=str(index_dir),
                qdrant_path=str(qdrant_path),
                model_name=args.model_name,
                embedding_device=args.embedding_device,
                batch_size=args.batch_size,
                no_fp16=args.no_fp16,
            )
        )
        if code:
            return code

    if args.no_web:
        print("[4/4] Web startup skipped (--no-web)")
        print(f"Persona ready: {author_dir}")
        return 0

    print(f"[4/4] Starting Web UI at http://{args.host}:{args.port}/")
    return _run_web(
        argparse.Namespace(
            author=author,
            data_dir=str(data_dir),
            host=args.host,
            port=args.port,
            model_name=args.model_name,
            embedding_device=args.embedding_device,
            no_fp16=args.no_fp16,
            child_top_k=args.child_top_k,
            per_query_parent_k=args.per_query_parent_k,
            parent_top_k=args.parent_top_k,
            max_search_results=args.max_search_results,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )


def _require_forge_artifact(path: Path, stage: str, flag: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot use {flag}: {stage} artifact does not exist: {path}"
        )


def _content_kinds(values: Iterable[str]) -> tuple[ContentKind, ...]:
    return tuple(values)  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
