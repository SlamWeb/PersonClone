"""Command line entrypoint for PersonaForge."""

from __future__ import annotations

import argparse
from collections import Counter
import getpass
import os
from pathlib import Path
from typing import Iterable

from personaforge import __version__
from personaforge.crawler.exceptions import CrawlError
from personaforge.crawler.markdown import write_markdown_corpus, write_profile
from personaforge.crawler.models import ContentItem, ContentKind, CreatorProfile
from personaforge.crawler.zhihu import ZhihuPublicCrawler, fallback_profile, parse_user_token
from personaforge.crawler.zhihu_browser import ZhihuBrowserCrawler, save_zhihu_session
from personaforge.env import load_env_file
from personaforge.ingest.embeddings import BgeM3Encoder
from personaforge.ingest.build import build_corpus
from personaforge.ingest.index import index_corpus
from personaforge.ingest.query_understanding import build_grounded_query_plan, plan_to_trace
from personaforge.ingest.retrieve import retrieve_parents, retrieve_parents_for_queries
from personaforge.llm import DeepSeekJsonClient
from personaforge.persona.narrative import load_narrative_schema, load_narrative_schema_for_index
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
        "--writer-context-top-k",
        type=int,
        default=20,
        help="Number of retrieved parent documents sent to the writer. Use 5 for the MRPrompt RAG5 comparison.",
    )
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
    ask_parser.add_argument(
        "--narrative-schema-path",
        help="Optional Narrative Schema path. Required for --writer-prompt mrprompt unless found beside the author index.",
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
        "--writer-context-top-k",
        type=int,
        default=20,
        help="Number of retrieved parent documents sent to the exported writer prompt.",
    )
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
    prompt_pack_parser.add_argument(
        "--narrative-schema-path",
        help="Optional Narrative Schema path. Required for --writer-prompt mrprompt unless found beside the author index.",
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
    eval_prepare_parser.add_argument(
        "--test-only",
        action="store_true",
        help="Build a sparse-author dataset from all timestamped answers as Test; no Dev split is created.",
    )

    eval_run_parser = eval_subparsers.add_parser("run", help="Generate answers for one prepared eval split.")
    eval_run_parser.add_argument("author", help="Creator token.")
    eval_run_parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl from pf eval prepare.")
    eval_run_parser.add_argument("--index-dir", help="Directory containing parents.jsonl and nodes.jsonl.")
    eval_run_parser.add_argument("--qdrant-path", help="Local Qdrant storage path.")
    eval_run_parser.add_argument(
        "--persona-pack-path",
        help="Optional Persona Pack path when the eval index is outside the author directory.",
    )
    eval_run_parser.add_argument(
        "--narrative-schema-path",
        help="Optional Narrative Schema path when the eval index is outside the author directory.",
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
    eval_run_parser.add_argument(
        "--writer-context-top-k",
        type=int,
        default=20,
        help="Number of retrieved parent documents sent to the writer during evaluation.",
    )
    eval_run_parser.add_argument(
        "--content-context-top-k",
        type=int,
        default=5,
        help="Strong-style mode: number of parents reserved for current-answer content.",
    )
    eval_run_parser.add_argument(
        "--style-context-top-k",
        type=int,
        default=3,
        help="Strong-style mode: number of parents selected as expression exemplars.",
    )
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
    eval_run_parser.add_argument("--method-id", help="Stable method identifier shown in evaluation reports.")
    eval_run_parser.add_argument("--display-name", help="Human-readable method name shown in evaluation reports.")
    eval_run_parser.add_argument("--description", default="", help="Short change summary shown in evaluation reports.")
    eval_run_parser.add_argument("--parent-method", help="Optional parent method identifier for lineage display.")
    eval_run_parser.add_argument("--prompt-version", help="Explicit prompt version label for this run.")
    eval_run_parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 when loading BGE-M3.")

    eval_pool_parser = eval_subparsers.add_parser(
        "retrieval-pool",
        help="Freeze a seven-route retrieval candidate pool for relevance evaluation.",
    )
    eval_pool_parser.add_argument("author", help="Creator token.")
    eval_pool_parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl from pf eval prepare.")
    eval_pool_parser.add_argument("--dataset-id", help="Stable experiment dataset ID, for example temporal_dev10_v0.")
    eval_pool_parser.add_argument("--index-dir", help="Directory containing parents.jsonl and nodes.jsonl.")
    eval_pool_parser.add_argument("--qdrant-path", help="Local Qdrant storage path.")
    eval_pool_parser.add_argument("--out-dir", help="Output directory. Defaults beside the dataset.")
    eval_pool_parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    eval_pool_parser.add_argument(
        "--query-plan-file",
        help="Frozen JSONL query plans. When complete, no query-transform LLM or Web search is called.",
    )
    eval_pool_parser.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model name.")
    eval_pool_parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for query embedding.",
    )
    eval_pool_parser.add_argument("--child-top-k", type=int, default=100)
    eval_pool_parser.add_argument("--route-parent-k", type=int, default=30)
    eval_pool_parser.add_argument("--per-query-parent-k", type=int, default=30)
    eval_pool_parser.add_argument("--max-search-results", type=int, default=5)
    eval_pool_parser.add_argument("--bm25-k1", type=float, default=1.2)
    eval_pool_parser.add_argument("--bm25-b", type=float, default=0.75)
    eval_pool_parser.add_argument("--force", action="store_true", help="Replace an existing frozen pool.")
    eval_pool_parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 when loading BGE-M3.")

    eval_core_parser = eval_subparsers.add_parser(
        "retrieval-core",
        help="Derive a smaller human-labeling pool from an existing frozen retrieval pool.",
    )
    eval_core_parser.add_argument("--source-manifest", required=True, help="Manifest of the complete retrieval pool.")
    eval_core_parser.add_argument("--route-depth", type=int, default=3, help="Keep candidates in any route's Top N.")
    eval_core_parser.add_argument("--out-dir", help="Output directory beside the complete pool by default.")
    eval_core_parser.add_argument("--force", action="store_true", help="Replace an existing derived core pool.")

    eval_full_pool_parser = eval_subparsers.add_parser(
        "retrieval-full-pool",
        help="Freeze every temporally eligible parent for corpus-wide retrieval qrels.",
    )
    eval_full_pool_parser.add_argument("--source-manifest", required=True, help="Frozen six-route pool manifest.")
    eval_full_pool_parser.add_argument("--dataset", required=True, help="Matching temporal dataset.jsonl.")
    eval_full_pool_parser.add_argument("--index-dir", required=True, help="Directory containing parents.jsonl.")
    eval_full_pool_parser.add_argument("--out-dir", help="Output directory beside the source pool by default.")
    eval_full_pool_parser.add_argument("--force", action="store_true", help="Replace an existing exhaustive pool.")

    eval_rankings_parser = eval_subparsers.add_parser(
        "retrieval-rankings",
        help="Freeze independent seven-route Parent rankings for RAG metrics.",
    )
    eval_rankings_parser.add_argument("--pool-manifest", required=True, help="Frozen Qrels pool manifest.")
    eval_rankings_parser.add_argument("--index-dir", required=True, help="Directory containing parents.jsonl and nodes.jsonl.")
    eval_rankings_parser.add_argument("--qdrant-path", help="Local Qdrant storage path; defaults to index-dir/qdrant.")
    eval_rankings_parser.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model name or local path.")
    eval_rankings_parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for query embedding.",
    )
    eval_rankings_parser.add_argument("--depth", type=int, default=100, help="Parent ranking depth per route.")
    eval_rankings_parser.add_argument("--child-top-k", type=int, default=2000, help="Initial child retrieval depth.")
    eval_rankings_parser.add_argument("--max-child-top-k", type=int, default=10000, help="Maximum child depth after retries.")
    eval_rankings_parser.add_argument("--per-query-parent-k", type=int, default=100)
    eval_rankings_parser.add_argument("--rrf-k", type=int, default=60)
    eval_rankings_parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    eval_rankings_parser.add_argument("--ranking-id", default="seven_route_parent_top100_v1")
    eval_rankings_parser.add_argument("--out-dir", help="Optional output directory for the ranking snapshot.")
    eval_rankings_parser.add_argument("--force", action="store_true", help="Replace an existing incomplete or completed snapshot.")
    eval_rankings_parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 when loading BGE-M3.")

    eval_rerank_parser = eval_subparsers.add_parser(
        "retrieval-rerank",
        help="Append BGE cross-encoder reranked routes to a frozen Parent ranking snapshot.",
    )
    eval_rerank_parser.add_argument("--base-ranking-manifest", required=True)
    eval_rerank_parser.add_argument("--index-dir", required=True, help="Directory containing nodes.jsonl.")
    eval_rerank_parser.add_argument(
        "--model-name",
        default="BAAI/bge-reranker-v2-m3",
        help="Reranker model name or local path.",
    )
    eval_rerank_parser.add_argument("--cache-dir", help="Optional Hugging Face model cache directory.")
    eval_rerank_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    eval_rerank_parser.add_argument("--candidate-depth", type=int, default=100)
    eval_rerank_parser.add_argument("--batch-size", type=int, default=2)
    eval_rerank_parser.add_argument("--max-length", type=int, default=1024)
    eval_rerank_parser.add_argument(
        "--routes",
        nargs="+",
        default=[
            "raw_hybrid_rrf",
            "transformed_rrf",
            "transformed_dense_bm25_rrf",
        ],
    )
    eval_rerank_parser.add_argument(
        "--ranking-id",
        default="seven_route_parent_top100_bge_reranker_v2_m3_v1",
    )
    eval_rerank_parser.add_argument("--out-dir")
    eval_rerank_parser.add_argument("--force", action="store_true")
    eval_rerank_parser.add_argument("--no-fp16", action="store_true")

    eval_gold_units_parser = eval_subparsers.add_parser(
        "retrieval-gold-units",
        help="Freeze Gold answer units used only by the offline retrieval Judge.",
    )
    eval_gold_units_parser.add_argument("--dataset", required=True, help="Temporal dataset.jsonl containing Gold answers.")
    eval_gold_units_parser.add_argument("--out-file", help="Output JSONL; defaults beside the dataset.")
    eval_gold_units_parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    eval_gold_units_parser.add_argument("--max-tokens", type=int, default=1800)
    eval_gold_units_parser.add_argument("--max-attempts", type=int, default=3)

    eval_gold_label_parser = eval_subparsers.add_parser(
        "retrieval-gold-label",
        help="Create dual-axis, Gold-aware qrels for a frozen retrieval pool.",
    )
    eval_gold_label_parser.add_argument("--pool-manifest", required=True)
    eval_gold_label_parser.add_argument("--dataset", required=True)
    eval_gold_label_parser.add_argument("--gold-units", required=True)
    eval_gold_label_parser.add_argument("--label-set", default="gold_aware_dual_axis_v2")
    eval_gold_label_parser.add_argument("--seed-label-manifest", help="Reuse completed V2 labels shared with a larger pool.")
    eval_gold_label_parser.add_argument("--batch-size", type=int, default=10)
    eval_gold_label_parser.add_argument("--max-concurrency", type=int, default=4)
    eval_gold_label_parser.add_argument("--max-tokens", type=int, default=6500)
    eval_gold_label_parser.add_argument("--max-attempts", type=int, default=3)
    eval_gold_label_parser.add_argument("--stability-sample-rate", type=float, default=0.05)
    eval_gold_label_parser.add_argument("--budget-cny", type=float, help="Pause after this estimated CNY budget.")
    eval_gold_label_parser.add_argument("--candidate-warmup-count", type=int, default=2)
    eval_gold_label_parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    eval_gold_label_parser.add_argument("--limit", type=int, help="Optional prefix limit for smoke testing.")

    eval_gold_codex_export_parser = eval_subparsers.add_parser(
        "retrieval-gold-codex-export",
        help="Export a hash-bound dual-axis Codex handoff package.",
    )
    eval_gold_codex_export_parser.add_argument("--pool-manifest", required=True)
    eval_gold_codex_export_parser.add_argument("--dataset", required=True)
    eval_gold_codex_export_parser.add_argument("--gold-units", required=True)
    eval_gold_codex_export_parser.add_argument("--out-dir")
    eval_gold_codex_export_parser.add_argument("--label-set", default="codex_gold_aware_dual_axis_v1")
    eval_gold_codex_export_parser.add_argument("--split", choices=["dev", "test", "all"], default="all")

    eval_gold_codex_import_parser = eval_subparsers.add_parser(
        "retrieval-gold-codex-import",
        help="Validate and publish a completed dual-axis Codex handoff.",
    )
    eval_gold_codex_import_parser.add_argument("--pool-manifest", required=True)
    eval_gold_codex_import_parser.add_argument("--dataset", required=True)
    eval_gold_codex_import_parser.add_argument("--gold-units", required=True)
    eval_gold_codex_import_parser.add_argument("--review-file", required=True)
    eval_gold_codex_import_parser.add_argument("--label-set", default="codex_gold_aware_dual_axis_v1")
    eval_gold_codex_import_parser.add_argument("--split", choices=["dev", "test", "all"], default="all")

    eval_v1_v2_parser = eval_subparsers.add_parser(
        "retrieval-v1-v2-compare",
        help="Compare query-only V1 labels with Gold-aware V2 content support.",
    )
    eval_v1_v2_parser.add_argument("--pool-manifest", required=True)
    eval_v1_v2_parser.add_argument("--v1-label-manifest", required=True)
    eval_v1_v2_parser.add_argument("--v2-label-manifest", required=True)
    eval_v1_v2_parser.add_argument("--out-file")

    eval_llm_label_parser = eval_subparsers.add_parser(
        "retrieval-llm-label",
        help="Use an LLM to label every query-parent pair in a frozen retrieval pool.",
    )
    eval_llm_label_parser.add_argument("--pool-manifest", required=True, help="Manifest of the complete retrieval pool.")
    eval_llm_label_parser.add_argument("--label-set", default="llm_relevance_v1", help="Stable label-set name for resumable output.")
    eval_llm_label_parser.add_argument("--limit", type=int, help="Optional prefix limit for a smoke run.")
    eval_llm_label_parser.add_argument("--max-tokens", type=int, default=900)
    eval_llm_label_parser.add_argument("--max-attempts", type=int, default=3)

    eval_codex_label_parser = eval_subparsers.add_parser(
        "retrieval-codex-label",
        help="Materialize a completed offline Codex review without calling an API.",
    )
    eval_codex_label_parser.add_argument("--pool-manifest", required=True, help="Manifest of the complete retrieval pool.")
    eval_codex_label_parser.add_argument("--review-file", required=True, help="Completed Codex review JSON file.")
    eval_codex_label_parser.add_argument("--label-set", default="codex_relevance_v1", help="Stable output label-set name.")

    eval_judge_parser = eval_subparsers.add_parser(
        "judge",
        help="Run the frozen six-dimension Gold Judge for one discovered dev10 system.",
    )
    eval_judge_parser.add_argument("system_id", help="Immutable system ID shown by the Generate evaluation page.")
    eval_judge_parser.add_argument("--data-dir", default="data", help="Local data root containing eval runs.")

    eval_profile_pack_parser = eval_subparsers.add_parser(
        "generation-profile-pack",
        help="Convert a train-only Persona Pack into an evaluator-only evidence profile.",
    )
    eval_profile_pack_parser.add_argument("--persona-pack", required=True)
    eval_profile_pack_parser.add_argument("--out-file", required=True)
    eval_profile_pack_parser.add_argument("--author-id")

    eval_profile_corpus_parser = eval_subparsers.add_parser(
        "generation-profile-corpus",
        help="Build an LLM-free evidence profile from eligible historical excerpts.",
    )
    eval_profile_corpus_parser.add_argument("--parents", required=True, help="Author index parents.jsonl.")
    eval_profile_corpus_parser.add_argument("--author-id", required=True)
    eval_profile_corpus_parser.add_argument("--display-name", default="")
    eval_profile_corpus_parser.add_argument("--eval-dataset", help="Temporal dataset used to exclude evaluation items and set cutoff.")
    eval_profile_corpus_parser.add_argument("--max-evidence", type=int, default=24)
    eval_profile_corpus_parser.add_argument("--out-file", required=True)

    eval_pairwise_export_parser = eval_subparsers.add_parser(
        "generation-pairwise-export",
        help="Export a Gold-aware, evidence-profile pairwise handoff with A/B swap.",
    )
    eval_pairwise_export_parser.add_argument("--profile", required=True)
    eval_pairwise_export_parser.add_argument("--left-run", required=True, help="Completed Test run runs.jsonl.")
    eval_pairwise_export_parser.add_argument("--right-run", required=True, help="Completed Test run runs.jsonl.")
    eval_pairwise_export_parser.add_argument("--out-dir", required=True)

    eval_pairwise_import_parser = eval_subparsers.add_parser(
        "generation-pairwise-import",
        help="Validate and aggregate a completed offline pairwise handoff.",
    )
    eval_pairwise_import_parser.add_argument("--manifest", required=True)
    eval_pairwise_import_parser.add_argument("--responses", required=True)
    eval_pairwise_import_parser.add_argument("--out-file")

    web_parser = subparsers.add_parser("web", help="Start the local Web UI.")
    web_parser.add_argument("author", nargs="?", help="Creator token.")
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host. Use 0.0.0.0 inside Docker or on a server.",
    )
    web_parser.add_argument("--port", type=int, default=8000)
    web_parser.add_argument("--data-dir", default="data", help="Local data root.")
    web_parser.add_argument(
        "--model-name",
        default=os.environ.get("PERSONAFORGE_EMBEDDING_MODEL", "BAAI/bge-m3"),
        help="Embedding model ID or local directory (env: PERSONAFORGE_EMBEDDING_MODEL).",
    )
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
    web_parser.add_argument(
        "--no-deployment-guards",
        action="store_true",
        help="Disable local Chat/login protections for controlled development tests.",
    )

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
    forge_parser.add_argument(
        "--no-deployment-guards",
        action="store_true",
        help="Disable local Chat/login protections for controlled development tests.",
    )

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
    load_env_file()
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
    seen: set[tuple[str, str]] = set()
    errors: list[str] = []
    missing_kinds: list[str] = []

    def add_items(found: list[ContentItem]) -> None:
        for item in found:
            key = (item.kind, item.id)
            if key in seen:
                continue
            seen.add(key)
            items.append(item)

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
        for kind in _content_kinds(kinds):
            remaining = None if max_items is None else max_items - len(items)
            if remaining is not None and remaining <= 0:
                break
            try:
                found = public.crawl_user(token, kinds=(kind,), max_items=remaining)
            except CrawlError as exc:
                errors.append(f"public {kind}: {exc}")
                found = []
            add_items(found)
            if not found:
                missing_kinds.append(kind)
    else:
        missing_kinds.extend(_content_kinds(kinds))

    if missing_kinds and not args.no_browser and (max_items is None or len(items) < max_items):
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
        for kind in missing_kinds:
            remaining = None if max_items is None else max_items - len(items)
            if remaining is not None and remaining <= 0:
                break
            try:
                add_items(browser.crawl_user(token, kinds=(kind,), max_items=remaining))
            except (CrawlError, RuntimeError) as exc:
                errors.append(f"browser {kind}: {exc}")

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

    kind_counts = Counter(item.kind for item in items)
    count_detail = ", ".join(f"{kind}={kind_counts.get(kind, 0)}" for kind in kinds)
    print(f"Saved {len(paths)} item(s) ({count_detail}) to {out_dir}")
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


def _writer_context_parent_hits(args: argparse.Namespace, retrieve_result):
    limit = int(getattr(args, "writer_context_top_k", 20))
    if limit <= 0:
        raise ValueError("--writer-context-top-k must be positive.")
    return retrieve_result.parents[:limit]


def _run_ask(args: argparse.Namespace) -> int:
    retrieve_result, query_trace, objective_background = _retrieve_for_generation(args)
    index_dir = _generation_index_dir(args)
    persona_pack = load_persona_pack_for_index(
        index_dir,
        required=args.writer_prompt == "persona_pack",
    ) if args.writer_prompt == "persona_pack" else None
    narrative_schema = _load_narrative_schema_for_writer(args, index_dir)
    writer_parent_hits = _writer_context_parent_hits(args, retrieve_result)
    llm = DeepSeekJsonClient.from_env()
    answer = generate_answer(
        query=args.query,
        parent_hits=writer_parent_hits,
        llm=llm,
        objective_background=objective_background,
        writer_prompt=args.writer_prompt,
        persona_pack=persona_pack,
        narrative_schema=narrative_schema,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        deployment_guards_enabled=not args.no_deployment_guards,
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
    narrative_schema = _load_narrative_schema_for_writer(args, index_dir)
    writer_parent_hits = _writer_context_parent_hits(args, retrieve_result)
    prompt_pack = build_prompt_pack(
        query=args.query,
        parent_hits=writer_parent_hits,
        objective_background=objective_background,
        writer_prompt=args.writer_prompt,
        persona_pack=persona_pack,
        narrative_schema=narrative_schema,
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
            narrative_schema_id=narrative_schema.schema_id if narrative_schema else None,
            narrative_schema_sha256=narrative_schema.sha256 if narrative_schema else None,
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


def _load_narrative_schema_for_writer(
    args: argparse.Namespace,
    index_dir: Path,
):
    if args.writer_prompt != "mrprompt":
        return None
    explicit_path = getattr(args, "narrative_schema_path", None)
    if explicit_path:
        return load_narrative_schema(
            Path(explicit_path),
            parent_store_path=index_dir / "parents.jsonl",
            verify_evidence=True,
        )
    return load_narrative_schema_for_index(index_dir, required=True)


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
            test_only=args.test_only,
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
            writer_context_top_k=args.writer_context_top_k,
            content_context_top_k=args.content_context_top_k,
            style_context_top_k=args.style_context_top_k,
            max_search_results=args.max_search_results,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            limit=args.limit,
            persona_pack_path=Path(args.persona_pack_path) if args.persona_pack_path else None,
            narrative_schema_path=(
                Path(args.narrative_schema_path) if args.narrative_schema_path else None
            ),
            method_id=args.method_id,
            display_name=args.display_name,
            description=args.description,
            parent_method=args.parent_method,
            prompt_version=args.prompt_version,
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

    if args.eval_command == "retrieval-pool":
        from personaforge.eval.retrieval_pool import RetrievalPoolConfig, build_retrieval_pool

        dataset_path = Path(args.dataset)
        index_dir = Path(args.index_dir) if args.index_dir else Path("data/authors") / "zhihu" / args.author / "index"
        qdrant_path = Path(args.qdrant_path) if args.qdrant_path else index_dir / "qdrant"
        config = RetrievalPoolConfig(
            author=args.author,
            dataset_path=dataset_path,
            dataset_id=args.dataset_id,
            split=args.split,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            query_plan_path=Path(args.query_plan_file) if args.query_plan_file else None,
            child_top_k=args.child_top_k,
            route_parent_k=args.route_parent_k,
            per_query_parent_k=args.per_query_parent_k,
            max_search_results=args.max_search_results,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            force=args.force,
        )
        encoder = BgeM3Encoder(args.model_name, device=args.embedding_device, use_fp16=not args.no_fp16)
        query_plan_llm = None if args.query_plan_file else DeepSeekJsonClient.from_env()
        result = build_retrieval_pool(
            config,
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            llm=query_plan_llm,
        )
        print(f"Frozen retrieval pool {result.pool_id}:")
        print(f"- queries: {result.query_count}")
        print(f"- candidate pairs: {result.candidate_count}")
        print(f"- pool: {result.pool_path}")
        print(f"- manifest: {result.manifest_path}")
        return 0

    if args.eval_command == "retrieval-core":
        from personaforge.eval.retrieval_pool import derive_core_pool

        result = derive_core_pool(
            Path(args.source_manifest),
            route_depth=args.route_depth,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            force=args.force,
        )
        print(f"Derived core retrieval pool {result.pool_id}:")
        print(f"- queries: {result.query_count}")
        print(f"- candidate pairs: {result.candidate_count}")
        print(f"- pool: {result.pool_path}")
        print(f"- manifest: {result.manifest_path}")
        return 0

    if args.eval_command == "retrieval-full-pool":
        from personaforge.eval.retrieval_pool import build_exhaustive_retrieval_pool

        result = build_exhaustive_retrieval_pool(
            Path(args.source_manifest),
            dataset_path=Path(args.dataset),
            index_dir=Path(args.index_dir),
            out_dir=Path(args.out_dir) if args.out_dir else None,
            force=args.force,
        )
        print(f"Built exhaustive retrieval pool {result.pool_id}:")
        print(f"- queries: {result.query_count}")
        print(f"- candidate pairs: {result.candidate_count}")
        print(f"- pool: {result.pool_path}")
        print(f"- manifest: {result.manifest_path}")
        return 0

    if args.eval_command == "retrieval-rankings":
        from personaforge.eval.retrieval_rankings import (
            RetrievalRankingConfig,
            build_retrieval_ranking_snapshot,
        )

        index_dir = Path(args.index_dir)
        config = RetrievalRankingConfig(
            pool_manifest_path=Path(args.pool_manifest),
            index_dir=index_dir,
            qdrant_path=Path(args.qdrant_path) if args.qdrant_path else index_dir / "qdrant",
            depth=args.depth,
            child_top_k=args.child_top_k,
            max_child_top_k=args.max_child_top_k,
            per_query_parent_k=args.per_query_parent_k,
            rrf_k=args.rrf_k,
            split=args.split,
            ranking_id=args.ranking_id,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            force=args.force,
        )
        encoder = BgeM3Encoder(args.model_name, device=args.embedding_device, use_fp16=not args.no_fp16)
        result = build_retrieval_ranking_snapshot(config, encoder=encoder)
        print(f"Completed retrieval ranking snapshot {result.ranking_id}:")
        print(f"- queries: {result.query_count}")
        print(f"- requested depth: {result.requested_depth}")
        print(f"- actual route depths: {result.actual_depth_by_route}")
        print(f"- rankings: {result.rankings_path}")
        print(f"- manifest: {result.manifest_path}")
        return 0

    if args.eval_command == "retrieval-rerank":
        from personaforge.eval.retrieval_reranker import (
            BgeCrossEncoderReranker,
            RetrievalRerankerConfig,
            build_reranked_ranking_snapshot,
        )

        reranker = BgeCrossEncoderReranker(
            args.model_name,
            device=args.device,
            use_fp16=not args.no_fp16,
            batch_size=args.batch_size,
            max_length=args.max_length,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        )
        config = RetrievalRerankerConfig(
            base_ranking_manifest_path=Path(args.base_ranking_manifest),
            index_dir=Path(args.index_dir),
            ranking_id=args.ranking_id,
            routes=tuple(args.routes),
            candidate_depth=args.candidate_depth,
            max_length=args.max_length,
            batch_size=args.batch_size,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            force=args.force,
        )
        result = build_reranked_ranking_snapshot(config, reranker=reranker)
        print(f"Completed retrieval reranking snapshot {result.ranking_id}:")
        print(f"- queries: {result.query_count}")
        print(f"- reranked routes: {', '.join(result.reranked_routes)}")
        print(f"- pairs: {result.pair_count}")
        print(f"- inputs above max length: {result.truncated_pair_count}")
        print(f"- rankings: {result.rankings_path}")
        print(f"- manifest: {result.manifest_path}")
        return 0

    if args.eval_command == "retrieval-gold-units":
        from personaforge.eval.retrieval_gold_qrels import extract_gold_units

        client = DeepSeekJsonClient.from_env()

        def report_gold_progress(current: int, total: int) -> None:
            print(f"Gold units: {current}/{total}", flush=True)

        result = extract_gold_units(
            Path(args.dataset),
            client=client,
            out_path=Path(args.out_file) if args.out_file else None,
            splits=None if args.split == "all" else [args.split],
            max_tokens=args.max_tokens,
            max_attempts=args.max_attempts,
            progress=report_gold_progress,
        )
        print("Completed Gold unit extraction:")
        print(f"- units: {result['path']}")
        print(f"- manifest: {result['manifest_path']}")
        return 0

    if args.eval_command == "retrieval-gold-label":
        from personaforge.eval.retrieval_gold_qrels import label_gold_aware_pool

        client = DeepSeekJsonClient.from_env()

        def report_gold_label_progress(phase: str, current: int, total: int) -> None:
            if current == total or current == 1 or current % 10 == 0:
                print(f"Gold-aware labels {phase}: {current}/{total}", flush=True)

        result = label_gold_aware_pool(
            Path(args.pool_manifest),
            dataset_path=Path(args.dataset),
            gold_units_path=Path(args.gold_units),
            client=client,
            label_set=args.label_set,
            splits=None if args.split == "all" else [args.split],
            seed_label_manifest=Path(args.seed_label_manifest) if args.seed_label_manifest else None,
            batch_size=args.batch_size,
            max_concurrency=args.max_concurrency,
            max_tokens=args.max_tokens,
            max_attempts=args.max_attempts,
            stability_sample_rate=args.stability_sample_rate,
            budget_cny=args.budget_cny,
            candidate_warmup_count=args.candidate_warmup_count,
            limit=args.limit,
            progress=report_gold_label_progress,
        )
        manifest = result["manifest"]
        print("Completed Gold-aware retrieval labeling:")
        print(f"- label set: {manifest['label_set']}")
        print(f"- progress: {manifest['completed']} / {manifest['total']}")
        print(f"- stability: {manifest['stability']}")
        print(f"- estimated cost: CNY {manifest.get('estimated_cost_cny', 0):.4f}")
        print(f"- cache hit rate: {(manifest.get('usage') or {}).get('cache_hit_rate')}")
        print(f"- labels: {result['labels_path']}")
        print(f"- metrics: {result['manifest_path'].parent / 'metrics.json'}")
        return 0

    if args.eval_command == "retrieval-gold-codex-export":
        from personaforge.eval.retrieval_gold_qrels import export_codex_gold_handoff

        result = export_codex_gold_handoff(
            Path(args.pool_manifest),
            dataset_path=Path(args.dataset),
            gold_units_path=Path(args.gold_units),
            out_dir=Path(args.out_dir) if args.out_dir else None,
            label_set=args.label_set,
        )
        print("Created Codex Gold-aware handoff:")
        print(f"- handoff: {result['handoff_id']}")
        print(f"- package: {result['zip_path']}")
        print(f"- template: {result['template_path']}")
        return 0

    if args.eval_command == "retrieval-gold-codex-import":
        from personaforge.eval.retrieval_gold_qrels import materialize_codex_gold_labels

        result = materialize_codex_gold_labels(
            Path(args.pool_manifest),
            Path(args.review_file),
            dataset_path=Path(args.dataset),
            gold_units_path=Path(args.gold_units),
            label_set=args.label_set,
            splits=None if args.split == "all" else [args.split],
        )
        manifest = result["manifest"]
        print("Completed dual-axis Codex retrieval labeling:")
        print(f"- label set: {manifest['label_set']}")
        print(f"- progress: {manifest['completed']} / {manifest['total']}")
        print(f"- labels: {result['labels_path']}")
        print(f"- metrics: {result['manifest_path'].parent / 'metrics.json'}")
        return 0

    if args.eval_command == "retrieval-v1-v2-compare":
        from personaforge.eval.retrieval_gold_qrels import compare_v1_v2

        result = compare_v1_v2(
            Path(args.pool_manifest),
            v1_label_manifest=Path(args.v1_label_manifest),
            v2_label_manifest=Path(args.v2_label_manifest),
            out_path=Path(args.out_file) if args.out_file else None,
        )
        report = result["report"]
        print("Completed V1/V2 retrieval-label comparison:")
        print(f"- compared pairs: {report['all']['total']}")
        print(f"- V1 0 -> V2 1/2: {report['all']['v1_zero_to_v2_positive']}")
        print(f"- changed labels: {report['changed_count']}")
        print(f"- report: {result['path']}")
        return 0

    if args.eval_command == "retrieval-llm-label":
        from personaforge.eval.retrieval_judge import label_pool

        client = DeepSeekJsonClient.from_env()
        last_printed = {"value": -1}

        def report_progress(current: int, total: int) -> None:
            if current == total or current - last_printed["value"] >= 10:
                print(f"LLM retrieval labels: {current}/{total}", flush=True)
                last_printed["value"] = current

        result = label_pool(
            Path(args.pool_manifest),
            client=client,
            label_set=args.label_set,
            max_tokens=args.max_tokens,
            max_attempts=args.max_attempts,
            limit=args.limit,
            progress=report_progress,
        )
        manifest = result["manifest"]
        print("Completed retrieval LLM labeling:")
        print(f"- label set: {manifest['label_set']}")
        print(f"- progress: {manifest['completed']} / {manifest['total']}")
        print(f"- labels: {result['labels_path']}")
        print(f"- metrics: {result['manifest_path'].parent / 'metrics.json'}")
        return 0

    if args.eval_command == "retrieval-codex-label":
        from personaforge.eval.retrieval_judge import materialize_codex_labels

        result = materialize_codex_labels(
            Path(args.pool_manifest),
            Path(args.review_file),
            label_set=args.label_set,
        )
        manifest = result["manifest"]
        print("Completed offline Codex retrieval labeling:")
        print(f"- label set: {manifest['label_set']}")
        print(f"- progress: {manifest['completed']} / {manifest['total']}")
        print(f"- labels: {result['labels_path']}")
        print(f"- metrics: {result['manifest_path'].parent / 'metrics.json'}")
        return 0

    if args.eval_command == "generation-profile-pack":
        from personaforge.eval.generation_pairwise import profile_from_persona_pack

        profile = profile_from_persona_pack(
            Path(args.persona_pack),
            author_id=args.author_id,
            out_path=Path(args.out_file),
        )
        print("Completed evidence profile from Persona Pack:")
        print(f"- profile: {args.out_file}")
        print(f"- author: {profile['author_id']}")
        print(f"- evidence: {profile['stats']['evidence_count']}")
        return 0

    if args.eval_command == "generation-profile-corpus":
        from personaforge.eval.generation_pairwise import profile_from_parent_corpus

        profile = profile_from_parent_corpus(
            Path(args.parents),
            author_id=args.author_id,
            display_name=args.display_name,
            eval_dataset_path=Path(args.eval_dataset) if args.eval_dataset else None,
            max_evidence=args.max_evidence,
            out_path=Path(args.out_file),
        )
        print("Completed LLM-free evidence profile:")
        print(f"- profile: {args.out_file}")
        print(f"- author: {profile['author_id']}")
        print(f"- cutoff: {profile['source']['cutoff']}")
        print(f"- eligible parents: {profile['stats']['eligible_parent_count']}")
        print(f"- evidence: {profile['stats']['evidence_count']}")
        return 0

    if args.eval_command == "generation-pairwise-export":
        from personaforge.eval.generation_pairwise import build_handoff

        manifest = build_handoff(
            profile_path=Path(args.profile),
            left_run_path=Path(args.left_run),
            right_run_path=Path(args.right_run),
            out_dir=Path(args.out_dir),
        )
        print("Completed offline pairwise handoff:")
        print(f"- manifest: {Path(args.out_dir) / 'manifest.json'}")
        print(f"- requests: {manifest['request_count']} ({manifest['item_count']} items x forward/swapped)")
        print(f"- prompt hash: {manifest['prompt_hash']}")
        return 0

    if args.eval_command == "generation-pairwise-import":
        from personaforge.eval.generation_pairwise import import_handoff

        result = import_handoff(
            manifest_path=Path(args.manifest),
            response_path=Path(args.responses),
            out_path=Path(args.out_file) if args.out_file else None,
        )
        print("Completed offline pairwise import:")
        print(f"- result: {Path(args.out_file) if args.out_file else Path(args.manifest).parent / 'result.json'}")
        print(f"- items: {result['summary']['item_count']}")
        print(f"- position consistency: {result['summary']['position_consistency']}")
        print(f"- inconsistent items: {result['summary']['inconsistent_items']}")
        return 0

    if args.eval_command == "judge":
        from personaforge.web.generation_evaluation import (
            GenerationEvaluationStore,
            GenerationJudgeManager,
        )

        store = GenerationEvaluationStore(Path(args.data_dir))
        manager = GenerationJudgeManager(store)
        job = manager.create(args.system_id, repeats=3)
        if job["status"] == "completed":
            print(f"Gold Judge already completed: {job['id']}")
        elif job["status"] == "running":
            print(f"Gold Judge is already running in another process: {job['id']}")
            return 0
        else:
            manager.run_once()
            job = store.public_judge_job(job["id"])
            print(f"Gold Judge {job['status']}: {job['id']}")
        print(f"- system: {job['system_id']}")
        print(f"- model: {job['model']}")
        print(f"- progress: {job['completed_items']} / {job['total_items']}")
        if job.get("error_message"):
            print(f"- error: {job['error_message']}")
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
        "narrative_schema_id": answer.narrative_schema_id,
        "narrative_schema_sha256": answer.narrative_schema_sha256,
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
    narrative_schema_id: str | None,
    narrative_schema_sha256: str | None,
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
        "narrative_schema_id": narrative_schema_id,
        "narrative_schema_sha256": narrative_schema_sha256,
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
        deployment_guards_enabled=not getattr(args, "no_deployment_guards", False),
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
            no_deployment_guards=args.no_deployment_guards,
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
