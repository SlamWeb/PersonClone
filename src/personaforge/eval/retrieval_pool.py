"""Build and freeze retrieval-labeling candidate pools."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from personaforge.eval.bm25 import Bm25ChildIndex
from personaforge.eval.dataset import (
    load_dataset,
    load_dataset_manifest,
    sha256_json,
    utc_now,
    write_json,
    write_jsonl,
)
from personaforge.ingest.embeddings import TextEncoder
from personaforge.ingest.query_understanding import build_grounded_query_plan, plan_to_trace
from personaforge.ingest.retrieve import (
    ParentHit,
    fuse_parent_hits,
    fuse_parent_rankings,
    fuse_transformed_dense_parent_rankings,
    load_parents,
    retrieve_parents,
    retrieve_parents_for_queries,
)
from personaforge.ingest.query_understanding import (
    GroundedQueryPlan,
    QueryTransformResult,
    RetrievalQuery,
    SearchPlan,
)
from personaforge.llm import JsonChatClient


POOL_SCHEMA_VERSION = "personaforge.eval.retrieval_pool.v1"
LEGACY_POOL_SCHEMA_VERSION = "personaforge.eval.retrieval_pool.v0"
EXHAUSTIVE_POOL_SCHEMA_VERSION = "personaforge.eval.retrieval_pool.v2"
ROUTE_ORDER = (
    "raw_dense",
    "raw_sparse",
    "raw_hybrid_rrf",
    "transformed_dense_rrf",
    "transformed_rrf",
    "raw_bm25",
    "transformed_dense_bm25_rrf",
)


@dataclass(frozen=True, slots=True)
class RetrievalPoolConfig:
    author: str
    dataset_path: Path
    split: str = "dev"
    dataset_id: str | None = None
    out_dir: Path | None = None
    query_plan_path: Path | None = None
    child_top_k: int = 100
    route_parent_k: int = 30
    per_query_parent_k: int = 30
    rrf_k: int = 60
    max_search_results: int = 5
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    force: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalPoolResult:
    pool_id: str
    pool_path: Path
    manifest_path: Path
    query_count: int
    candidate_count: int


def build_exhaustive_retrieval_pool(
    source_manifest_path: Path,
    *,
    dataset_path: Path,
    index_dir: Path,
    out_dir: Path | None = None,
    force: bool = False,
) -> RetrievalPoolResult:
    """Freeze every parent visible at the temporal cutoff for every eval query.

    The source route pool supplies immutable query plans and route ranks. A
    parent absent from all routes is still included with empty
    ``route_ranks`` so corpus-wide recall has a real denominator.
    """

    source_manifest_path = source_manifest_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    index_dir = index_dir.expanduser().resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") not in {
        POOL_SCHEMA_VERSION,
        LEGACY_POOL_SCHEMA_VERSION,
    }:
        raise ValueError(f"Unsupported source retrieval pool: {source_manifest_path}")
    source_pool_path = source_manifest_path.parent / str(source_manifest.get("pool_file") or "pool.jsonl")
    source_records = {
        str(row.get("item_id") or ""): row
        for row in (
            json.loads(line)
            for line in source_pool_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    dataset_manifest = load_dataset_manifest(dataset_path)
    dataset = load_dataset(dataset_path)
    if str(source_manifest.get("dataset_sha256") or "") != str(dataset_manifest.get("dataset_sha256") or ""):
        raise ValueError("Source retrieval pool and dataset SHA-256 do not match")
    if {str(row.get("item_id") or "") for row in dataset} != set(source_records):
        raise ValueError("Source retrieval pool does not cover the same dataset items")

    parents_by_id = load_parents(index_dir / "parents.jsonl")
    excluded_parent_ids = {str(value) for value in dataset_manifest.get("excluded_parent_ids") or []}
    eligible_parent_ids = sorted(set(parents_by_id).difference(excluded_parent_ids))
    expected_train_count = int((dataset_manifest.get("counts") or {}).get("train_parents") or 0)
    if expected_train_count and len(eligible_parent_ids) != expected_train_count:
        raise ValueError(
            f"Eligible parent count changed: expected {expected_train_count}, got {len(eligible_parent_ids)}"
        )
    for item in dataset:
        target_parent_id = str(item.get("parent_id") or "")
        if target_parent_id in eligible_parent_ids:
            raise AssertionError(f"Gold target leaked into eligible corpus: {target_parent_id}")

    source_pool_id = str(source_manifest.get("pool_id") or "retrieval")
    pool_id = f"{source_pool_id}.exhaustive_qrels.v2"
    target_dir = out_dir or source_manifest_path.parent.parent / "all30_exhaustive_qrels_v2"
    pool_path = target_dir / "pool.jsonl"
    manifest_path = target_dir / "manifest.json"
    if (pool_path.exists() or manifest_path.exists()) and not force:
        raise FileExistsError(f"Exhaustive retrieval pool already exists: {target_dir}. Pass --force to replace it.")
    target_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for item in dataset:
        item_id = str(item.get("item_id") or "")
        source_record = source_records[item_id]
        retrieved_by_id = {
            str(candidate.get("parent_id") or ""): candidate
            for candidate in source_record.get("candidates") or []
        }
        candidates: list[dict[str, Any]] = []
        for parent_id in eligible_parent_ids:
            parent = parents_by_id[parent_id]
            source_candidate = retrieved_by_id.get(parent_id) or {}
            candidates.append(
                {
                    "parent_id": parent_id,
                    "title": str(parent.get("title") or ""),
                    "text": str(parent.get("text") or ""),
                    "url": str(parent.get("url") or ""),
                    "path": str(parent.get("path") or ""),
                    "kind": str(parent.get("kind") or ""),
                    "created_at": str(parent.get("created_at") or ""),
                    "route_ranks": source_candidate.get("route_ranks") or {},
                }
            )
        records.append(
            {
                "pool_id": pool_id,
                "item_id": item_id,
                "split": str(item.get("split") or ""),
                "query": str(item.get("query") or ""),
                "target_parent_id": str(item.get("parent_id") or ""),
                "query_trace": source_record.get("query_trace") or {},
                "candidates": candidates,
            }
        )

    write_jsonl(records, pool_path)
    candidate_count = len(records) * len(eligible_parent_ids)
    selection = dataset_manifest.get("selection") or {}
    sparse_retrospective = selection.get("protocol") == "sparse_author_test_only"
    manifest = {
        "schema_version": EXHAUSTIVE_POOL_SCHEMA_VERSION,
        "pool_id": pool_id,
        "dataset_id": str(source_manifest.get("dataset_id") or dataset_path.parent.name),
        "display_name": (
            "全量 Qrels · Test 题 × 回溯作者语料"
            if sparse_retrospective
            else "全量 Qrels · 30 题 × cutoff 前全部材料"
        ),
        "author": str(source_manifest.get("author") or dataset_manifest.get("author") or ""),
        "split": "all",
        "created_at": utc_now(),
        "status": "completed",
        "pool_kind": "exhaustive_eligible_parents",
        "recall_scope": (
            "eligible_author_corpus_excluding_eval_answers"
            if sparse_retrospective
            else "eligible_author_corpus_before_cutoff"
        ),
        "pool_file": pool_path.name,
        "pool_sha256": sha256_json(records),
        "pool_file_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_manifest.get("dataset_sha256"),
        "excluded_parent_ids_sha256": dataset_manifest.get("excluded_parent_ids_sha256"),
        "source_parents_sha256": dataset_manifest.get("source_parents_sha256"),
        "source_route_pool_id": source_pool_id,
        "source_route_pool_sha256": source_manifest.get("pool_sha256"),
        "eligible_parent_ids_sha256": hashlib.sha256(
            json.dumps(eligible_parent_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "counts": {
            "queries": len(records),
            "eligible_parents_per_query": len(eligible_parent_ids),
            "candidate_pairs": candidate_count,
            "unique_parents": len(eligible_parent_ids),
            "route_ranked_pairs": sum(
                bool(candidate.get("route_ranks"))
                for record in records
                for candidate in record["candidates"]
            ),
        },
        "routes": list(ROUTE_ORDER),
        "git": _git_revision(),
    }
    write_json(manifest, manifest_path)
    return RetrievalPoolResult(
        pool_id=pool_id,
        pool_path=pool_path,
        manifest_path=manifest_path,
        query_count=len(records),
        candidate_count=candidate_count,
    )


def derive_core_pool(
    source_manifest_path: Path,
    *,
    route_depth: int = 3,
    out_dir: Path | None = None,
    force: bool = False,
) -> RetrievalPoolResult:
    """Derive a human-sized pool from candidates ranked in any route's head."""
    if route_depth < 1:
        raise ValueError("route_depth must be at least 1")
    source_manifest_path = source_manifest_path.expanduser().resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") not in {POOL_SCHEMA_VERSION, LEGACY_POOL_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported retrieval pool manifest: {source_manifest_path}")
    source_pool_path = source_manifest_path.parent / str(source_manifest.get("pool_file") or "pool.jsonl")
    source_records = [
        json.loads(line)
        for line in source_pool_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_pool_id = str(source_manifest["pool_id"])
    pool_id = f"{source_pool_id}.core_top{route_depth}.v0"
    target_dir = out_dir or source_manifest_path.parent.parent / f"{source_manifest_path.parent.name}_core_top{route_depth}"
    pool_path = target_dir / "pool.jsonl"
    manifest_path = target_dir / "manifest.json"
    if (pool_path.exists() or manifest_path.exists()) and not force:
        raise FileExistsError(f"Core retrieval pool already exists: {target_dir}. Pass --force to replace it.")
    target_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for source_record in source_records:
        candidates = [
            candidate
            for candidate in source_record.get("candidates") or []
            if any(
                int(details.get("rank") or 0) <= route_depth
                for details in (candidate.get("route_ranks") or {}).values()
                if int(details.get("rank") or 0) > 0
            )
        ]
        records.append({**source_record, "pool_id": pool_id, "candidates": candidates})

    write_jsonl(records, pool_path)
    candidate_count = sum(len(row["candidates"]) for row in records)
    dataset_id = str(source_manifest.get("dataset_id") or "retrieval")
    manifest = {
        **source_manifest,
        "pool_id": pool_id,
        "dataset_id": f"{dataset_id}_core_top{route_depth}",
        "display_name": f"核心评估集 · 四路 Top{route_depth}",
        "created_at": utc_now(),
        "pool_file": pool_path.name,
        "pool_sha256": sha256_json(records),
        "pool_kind": "core",
        "source_pool_id": source_pool_id,
        "source_pool_sha256": source_manifest.get("pool_sha256"),
        "label_namespace_pool_id": str(source_manifest.get("label_namespace_pool_id") or source_pool_id),
        "selection": {"rule": "any_route_rank_lte", "route_depth": route_depth},
        "counts": {
            "queries": len(records),
            "candidate_pairs": candidate_count,
            "unique_parents": len({candidate["parent_id"] for row in records for candidate in row["candidates"]}),
        },
        "git": _git_revision(),
    }
    write_json(manifest, manifest_path)
    return RetrievalPoolResult(
        pool_id=pool_id,
        pool_path=pool_path,
        manifest_path=manifest_path,
        query_count=len(records),
        candidate_count=candidate_count,
    )


def build_retrieval_pool(
    config: RetrievalPoolConfig,
    *,
    index_dir: Path,
    qdrant_path: Path,
    encoder: TextEncoder,
    llm: JsonChatClient | None = None,
) -> RetrievalPoolResult:
    dataset_manifest = load_dataset_manifest(config.dataset_path)
    dataset = [
        row
        for row in load_dataset(config.dataset_path)
        if config.split == "all" or row.get("split") == config.split
    ]
    if not dataset:
        raise ValueError(f"No {config.split!r} items found in {config.dataset_path}.")

    dataset_id = config.dataset_id or str(dataset_manifest.get("dataset_id") or config.dataset_path.parent.name)
    pool_id = f"{dataset_id}.{config.split}.seven_route_retrieval_pool.v1"
    default_dir_name = f"{config.split}_seven_route_v1"
    out_dir = config.out_dir or config.dataset_path.parent / "retrieval_pool" / default_dir_name
    pool_path = out_dir / "pool.jsonl"
    manifest_path = out_dir / "manifest.json"
    if (pool_path.exists() or manifest_path.exists()) and not config.force:
        raise FileExistsError(f"Retrieval pool already exists: {out_dir}. Pass --force to replace it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded_parent_ids = {str(value) for value in dataset_manifest.get("excluded_parent_ids", [])}
    parents_by_id = load_parents(index_dir / "parents.jsonl")
    bm25 = Bm25ChildIndex.from_jsonl(
        index_dir / "nodes.jsonl",
        exclude_parent_ids=excluded_parent_ids,
        k1=config.bm25_k1,
        b=config.bm25_b,
    )
    frozen_plans = _load_frozen_query_plans(config.query_plan_path) if config.query_plan_path else {}

    records: list[dict[str, Any]] = []
    for item in dataset:
        query = str(item["query"])
        raw = retrieve_parents(
            query,
            author=config.author,
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            child_top_k=config.child_top_k,
            parent_top_k=config.route_parent_k,
            rrf_k=config.rrf_k,
            exclude_parent_ids=excluded_parent_ids,
        )
        raw_dense = fuse_parent_hits(
            {"raw_dense": raw.routes["dense"]},
            rrf_k=config.rrf_k,
            parent_top_k=config.route_parent_k,
        )
        raw_sparse = fuse_parent_hits(
            {"raw_sparse": raw.routes["sparse"]},
            rrf_k=config.rrf_k,
            parent_top_k=config.route_parent_k,
        )

        plan = frozen_plans.get(str(item["item_id"]))
        if plan is None:
            if llm is None:
                raise ValueError(
                    f"Missing frozen query plan for {item['item_id']}; provide --query-plan-file "
                    "or an LLM client."
                )
            plan = build_grounded_query_plan(
                query,
                llm=llm,
                max_results_per_query=config.max_search_results,
            )
        if plan.original_query != query:
            raise ValueError(f"Frozen query plan mismatch for {item['item_id']}: {plan.original_query!r}")

        transformed_result = retrieve_parents_for_queries(
            query,
            plan.transform.retrieval_queries,
            author=config.author,
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            child_top_k=config.child_top_k,
            per_query_parent_k=config.per_query_parent_k,
            parent_top_k=config.route_parent_k,
            rrf_k=config.rrf_k,
            exclude_parent_ids=excluded_parent_ids,
        )
        transformed_dense = fuse_transformed_dense_parent_rankings(
            plan.transform.retrieval_queries,
            transformed_result.routes,
            rrf_k=config.rrf_k,
            per_query_parent_k=config.per_query_parent_k,
            parent_top_k=config.route_parent_k,
        )

        dense_bm25_per_query: dict[str, list[ParentHit]] = {}
        for retrieval_query in plan.transform.retrieval_queries:
            dense_route = f"{retrieval_query.route}:dense"
            bm25_route = f"{retrieval_query.route}:bm25"
            dense_bm25_per_query[retrieval_query.route] = fuse_parent_hits(
                {
                    dense_route: transformed_result.routes.get(dense_route, []),
                    bm25_route: bm25.search(retrieval_query.query, child_top_k=config.child_top_k),
                },
                rrf_k=config.rrf_k,
                parent_top_k=config.per_query_parent_k,
            )
        transformed_dense_bm25 = fuse_parent_rankings(
            dense_bm25_per_query,
            rrf_k=config.rrf_k,
            parent_top_k=config.route_parent_k,
        )

        bm25_children = bm25.search(query, child_top_k=config.child_top_k)
        bm25_parents = fuse_parent_hits(
            {"raw_bm25": bm25_children},
            rrf_k=config.rrf_k,
            parent_top_k=config.route_parent_k,
        )
        routes = {
            "raw_dense": raw_dense,
            "raw_sparse": raw_sparse,
            "raw_hybrid_rrf": raw.parents,
            "transformed_dense_rrf": transformed_dense,
            "transformed_rrf": transformed_result.parents,
            "raw_bm25": bm25_parents,
            "transformed_dense_bm25_rrf": transformed_dense_bm25,
        }
        record = make_pool_record(
            item=item,
            pool_id=pool_id,
            routes=routes,
            parents_by_id=parents_by_id,
            query_trace=plan_to_trace(plan),
            excluded_parent_ids=excluded_parent_ids,
        )
        records.append(record)

    write_jsonl(records, pool_path)
    candidate_count = sum(len(row["candidates"]) for row in records)
    manifest = {
        "schema_version": POOL_SCHEMA_VERSION,
        "pool_id": pool_id,
        "dataset_id": dataset_id,
        "author": config.author,
        "split": config.split,
        "created_at": utc_now(),
        "status": "completed",
        "pool_file": pool_path.name,
        "pool_sha256": sha256_json(records),
        "dataset_path": str(config.dataset_path),
        "dataset_sha256": dataset_manifest.get("dataset_sha256"),
        "excluded_parent_ids_sha256": dataset_manifest.get("excluded_parent_ids_sha256"),
        "source_parents_sha256": dataset_manifest.get("source_parents_sha256"),
        "counts": {
            "queries": len(records),
            "candidate_pairs": candidate_count,
            "unique_parents": len({candidate["parent_id"] for row in records for candidate in row["candidates"]}),
        },
        "config": _jsonable_config(config),
        "routes": list(ROUTE_ORDER),
        "git": _git_revision(),
    }
    write_json(manifest, manifest_path)
    return RetrievalPoolResult(
        pool_id=pool_id,
        pool_path=pool_path,
        manifest_path=manifest_path,
        query_count=len(records),
        candidate_count=candidate_count,
    )


def make_pool_record(
    *,
    item: dict[str, Any],
    pool_id: str,
    routes: dict[str, list[ParentHit]],
    parents_by_id: dict[str, dict[str, Any]],
    query_trace: dict[str, Any],
    excluded_parent_ids: set[str],
) -> dict[str, Any]:
    route_ranks: dict[str, dict[str, dict[str, Any]]] = {}
    for route_name in ROUTE_ORDER:
        for hit in routes.get(route_name, []):
            if hit.parent_id in excluded_parent_ids:
                raise AssertionError(f"Retrieval leak in {route_name}: {hit.parent_id}")
            route_ranks.setdefault(hit.parent_id, {})[route_name] = {
                "rank": hit.rank,
                "score": hit.score,
            }

    candidates: list[dict[str, Any]] = []
    for parent_id in sorted(route_ranks):
        parent = parents_by_id.get(parent_id)
        if parent is None:
            raise KeyError(f"Missing parent document for candidate: {parent_id}")
        candidates.append(
            {
                "parent_id": parent_id,
                "title": str(parent.get("title") or ""),
                "text": str(parent.get("text") or ""),
                "url": str(parent.get("url") or ""),
                "path": str(parent.get("path") or ""),
                "kind": str(parent.get("kind") or ""),
                "created_at": str(parent.get("created_at") or ""),
                "route_ranks": route_ranks[parent_id],
            }
        )
    return {
        "pool_id": pool_id,
        "item_id": str(item["item_id"]),
        "split": str(item["split"]),
        "query": str(item["query"]),
        "target_parent_id": str(item["parent_id"]),
        "query_trace": query_trace,
        "candidates": candidates,
    }


def _jsonable_config(config: RetrievalPoolConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("dataset_path", "out_dir", "query_plan_path"):
        value = payload.get(key)
        payload[key] = str(value) if value is not None else None
    return payload


def _load_frozen_query_plans(path: Path) -> dict[str, GroundedQueryPlan]:
    path = path.expanduser().resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    plans: dict[str, GroundedQueryPlan] = {}
    for row in rows:
        item_id = str(row.get("item_id") or "").strip()
        original_query = str(row.get("query") or row.get("original_query") or "").strip()
        if not item_id or not original_query:
            raise ValueError(f"Frozen query-plan row requires item_id and query: {row!r}")
        if item_id in plans:
            raise ValueError(f"Duplicate frozen query plan: {item_id}")
        raw_queries = row.get("retrieval_queries") or []
        retrieval_queries = [
            RetrievalQuery(route=str(item["route"]), query=str(item["query"]).strip())
            for item in raw_queries
            if isinstance(item, dict) and str(item.get("route") or "").strip() and str(item.get("query") or "").strip()
        ]
        routes = {item.route for item in retrieval_queries}
        expected_routes = {"literal_question", "event_background", "mechanism_scene", "colloquial_surface"}
        if routes != expected_routes or len(retrieval_queries) != 4:
            raise ValueError(f"Frozen query plan {item_id} must contain exactly the four retrieval routes.")
        plans[item_id] = GroundedQueryPlan(
            original_query=original_query,
            search_plan=SearchPlan(needs_web=False, search_queries=[]),
            search_results=[],
            transform=QueryTransformResult(
                objective_background=str(row.get("objective_background") or ""),
                retrieval_queries=retrieval_queries,
            ),
        )
    return plans


def _git_revision() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(run("status", "--porcelain")),
    }
