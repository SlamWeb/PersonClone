"""Build independent retrieval ranking snapshots for frozen Qrels.

The frozen Qrels pool answers "which parents are judged".  This module
answers a different question: "how far did each retrieval route actually
rank".  Keeping those two artifacts separate makes Recall@50/100 meaningful
without pretending that the original Top30 candidate pool was a Top100 run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from personaforge.eval.bm25 import Bm25ChildIndex
from personaforge.eval.dataset import load_dataset_manifest, utc_now, write_json, write_jsonl
from personaforge.eval.retrieval_pool import ROUTE_ORDER
from personaforge.ingest.embeddings import TextEncoder
from personaforge.ingest.query_understanding import RetrievalQuery
from personaforge.ingest.retrieve import (
    ChildHit,
    ParentHit,
    fuse_parent_hits,
    fuse_parent_rankings,
    fuse_transformed_dense_parent_rankings,
    load_parents,
    retrieve_parents,
    retrieve_parents_for_queries,
)


LEGACY_RANKING_SCHEMA_VERSION = "personaforge.eval.retrieval_rankings.v1"
RANKING_SCHEMA_VERSION = "personaforge.eval.retrieval_rankings.v2"
DEFAULT_RANKING_ID = "seven_route_parent_top100_v1"


@dataclass(frozen=True, slots=True)
class RetrievalRankingConfig:
    pool_manifest_path: Path
    index_dir: Path
    qdrant_path: Path
    depth: int = 100
    child_top_k: int = 2000
    max_child_top_k: int = 10000
    per_query_parent_k: int = 100
    rrf_k: int = 60
    split: str = "all"
    ranking_id: str = DEFAULT_RANKING_ID
    out_dir: Path | None = None
    force: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalRankingResult:
    ranking_id: str
    ranking_dir: Path
    manifest_path: Path
    rankings_path: Path
    query_count: int
    requested_depth: int
    actual_depth_by_route: dict[str, int]


def build_retrieval_ranking_snapshot(
    config: RetrievalRankingConfig,
    *,
    encoder: TextEncoder,
) -> RetrievalRankingResult:
    """Run seven routes and freeze ordered Parent rankings.

    The input pool must already contain the frozen query traces and qrels
    namespace.  Query understanding is therefore replayed from disk and no
    query-transform or web-search model call is made here.
    """

    if config.depth < 1:
        raise ValueError("depth must be at least 1")
    if config.child_top_k < 1 or config.max_child_top_k < config.child_top_k:
        raise ValueError("child_top_k and max_child_top_k are invalid")
    if config.per_query_parent_k < config.depth:
        raise ValueError("per_query_parent_k must cover the requested parent depth")

    pool_manifest_path = config.pool_manifest_path.expanduser().resolve()
    pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    pool_path = pool_manifest_path.parent / str(pool_manifest.get("pool_file") or "pool.jsonl")
    records = _load_jsonl(pool_path)
    if config.split != "all":
        records = [row for row in records if str(row.get("split") or "") == config.split]
    if not records:
        raise ValueError(f"No records found for split {config.split!r} in {pool_path}")

    index_dir = config.index_dir.expanduser().resolve()
    qdrant_path = config.qdrant_path.expanduser().resolve()
    parents_by_id = load_parents(index_dir / "parents.jsonl")
    excluded_parent_ids = _excluded_parent_ids(pool_manifest)
    eligible_parent_count = len(set(parents_by_id).difference(excluded_parent_ids))
    expected_depth = min(config.depth, eligible_parent_count) if eligible_parent_count else config.depth

    ranking_dir = config.out_dir.expanduser().resolve() if config.out_dir else pool_manifest_path.parent / "rankings" / config.ranking_id
    rankings_path = ranking_dir / "rankings.jsonl"
    manifest_path = ranking_dir / "manifest.json"
    if (rankings_path.exists() or manifest_path.exists()) and not config.force:
        existing = _read_manifest_if_complete(manifest_path)
        if existing is not None:
            return RetrievalRankingResult(
                ranking_id=str(existing.get("ranking_id") or config.ranking_id),
                ranking_dir=ranking_dir,
                manifest_path=manifest_path,
                rankings_path=ranking_dir / str(existing.get("rankings_file") or rankings_path.name),
                query_count=int((existing.get("counts") or {}).get("queries") or 0),
                requested_depth=int(existing.get("requested_depth") or config.depth),
                actual_depth_by_route={
                    str(key): int(value)
                    for key, value in (existing.get("actual_depth_by_route") or {}).items()
                },
            )
        raise FileExistsError(f"Incomplete ranking snapshot exists: {ranking_dir}. Pass --force to replace it.")

    ranking_dir.mkdir(parents=True, exist_ok=True)
    partial_path = ranking_dir / "rankings.partial.jsonl"
    if config.force:
        for path in (rankings_path, manifest_path, partial_path):
            path.unlink(missing_ok=True)

    bm25 = Bm25ChildIndex.from_jsonl(
        index_dir / "nodes.jsonl",
        exclude_parent_ids=excluded_parent_ids,
    )
    actual_depth_by_route = {route: 0 for route in ROUTE_ORDER}
    output_rows: list[dict[str, Any]] = []
    for record in records:
        row = _rank_one_query(
            record,
            author=str(pool_manifest.get("author") or ""),
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            bm25=bm25,
            parents_by_id=parents_by_id,
            excluded_parent_ids=excluded_parent_ids,
            depth=config.depth,
            child_top_k=config.child_top_k,
            max_child_top_k=config.max_child_top_k,
            per_query_parent_k=config.per_query_parent_k,
            rrf_k=config.rrf_k,
        )
        output_rows.append(row)
        for route, entries in row["routes"].items():
            actual_depth_by_route[route] = max(actual_depth_by_route[route], len(entries))
        # The partial file is intentionally overwritten after each query.  A
        # crash cannot invalidate already completed query snapshots.
        write_jsonl(output_rows, partial_path)

    write_jsonl(output_rows, rankings_path)
    partial_path.unlink(missing_ok=True)
    manifest = {
        "schema_version": RANKING_SCHEMA_VERSION,
        "ranking_id": config.ranking_id,
        "status": "completed",
        "created_at": utc_now(),
        "pool_id": str(pool_manifest.get("pool_id") or ""),
        "qrels_pool_id": str(pool_manifest.get("pool_id") or ""),
        "pool_manifest_path": str(pool_manifest_path),
        "pool_sha256": pool_manifest.get("pool_sha256"),
        "dataset_sha256": pool_manifest.get("dataset_sha256"),
        "author": str(pool_manifest.get("author") or ""),
        "split": config.split,
        "routes": list(ROUTE_ORDER),
        "requested_depth": config.depth,
        "expected_depth": expected_depth,
        "eligible_parent_count": eligible_parent_count,
        "actual_depth_by_route": actual_depth_by_route,
        "rankings_file": rankings_path.name,
        "rankings_sha256": hashlib.sha256(rankings_path.read_bytes()).hexdigest(),
        "config": _jsonable_config(config),
        "counts": {"queries": len(output_rows)},
        "git": _git_revision(),
    }
    write_json(manifest, manifest_path)
    return RetrievalRankingResult(
        ranking_id=config.ranking_id,
        ranking_dir=ranking_dir,
        manifest_path=manifest_path,
        rankings_path=rankings_path,
        query_count=len(output_rows),
        requested_depth=config.depth,
        actual_depth_by_route=actual_depth_by_route,
    )


def load_ranking_snapshot(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {
        LEGACY_RANKING_SCHEMA_VERSION,
        RANKING_SCHEMA_VERSION,
    }:
        raise ValueError(f"Unsupported ranking snapshot: {manifest_path}")
    rankings_path = manifest_path.parent / str(manifest.get("rankings_file") or "rankings.jsonl")
    return manifest, _load_jsonl(rankings_path)


def _rank_one_query(
    record: dict[str, Any],
    *,
    author: str,
    index_dir: Path,
    qdrant_path: Path,
    encoder: TextEncoder,
    bm25: Bm25ChildIndex,
    parents_by_id: dict[str, dict[str, Any]],
    excluded_parent_ids: set[str],
    depth: int,
    child_top_k: int,
    max_child_top_k: int,
    per_query_parent_k: int,
    rrf_k: int,
) -> dict[str, Any]:
    retrieval_queries = _retrieval_queries_from_record(record)
    fetch = max(child_top_k, depth * 20)
    fetch = min(fetch, max_child_top_k)
    target = min(depth, max(len(parents_by_id) - len(excluded_parent_ids), 1))
    last_routes: dict[str, list[ParentHit]] = {}
    while True:
        raw = retrieve_parents(
            str(record.get("query") or ""),
            author=author,
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            child_top_k=fetch,
            parent_top_k=depth,
            rrf_k=rrf_k,
            exclude_parent_ids=excluded_parent_ids,
        )
        raw_dense = fuse_parent_hits(
            {"raw_dense": raw.routes.get("dense", [])},
            rrf_k=rrf_k,
            parent_top_k=depth,
        )
        raw_sparse = fuse_parent_hits(
            {"raw_sparse": raw.routes.get("sparse", [])},
            rrf_k=rrf_k,
            parent_top_k=depth,
        )
        transformed = retrieve_parents_for_queries(
            str(record.get("query") or ""),
            retrieval_queries,
            author=author,
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            child_top_k=fetch,
            per_query_parent_k=per_query_parent_k,
            parent_top_k=depth,
            rrf_k=rrf_k,
            exclude_parent_ids=excluded_parent_ids,
        )
        transformed_dense = fuse_transformed_dense_parent_rankings(
            retrieval_queries,
            transformed.routes,
            rrf_k=rrf_k,
            per_query_parent_k=per_query_parent_k,
            parent_top_k=depth,
        )
        raw_bm25_children = bm25.search(str(record.get("query") or ""), child_top_k=fetch)
        raw_bm25 = fuse_parent_hits(
            {"raw_bm25": raw_bm25_children},
            rrf_k=rrf_k,
            parent_top_k=depth,
        )
        transformed_dense_bm25_per_query: dict[str, list[ParentHit]] = {}
        transformed_dense_bm25_children: dict[str, list[ChildHit]] = {}
        for retrieval_query in retrieval_queries:
            dense_route = f"{retrieval_query.route}:dense"
            bm25_route = f"{retrieval_query.route}:bm25"
            bm25_hits = bm25.search(retrieval_query.query, child_top_k=fetch)
            transformed_dense_bm25_children[dense_route] = transformed.routes.get(dense_route, [])
            transformed_dense_bm25_children[bm25_route] = bm25_hits
            transformed_dense_bm25_per_query[retrieval_query.route] = fuse_parent_hits(
                {
                    dense_route: transformed.routes.get(dense_route, []),
                    bm25_route: bm25_hits,
                },
                rrf_k=rrf_k,
                parent_top_k=per_query_parent_k,
            )
        transformed_dense_bm25 = fuse_parent_rankings(
            transformed_dense_bm25_per_query,
            rrf_k=rrf_k,
            parent_top_k=depth,
        )
        last_routes = {
            "raw_dense": raw_dense,
            "raw_sparse": raw_sparse,
            "raw_hybrid_rrf": raw.parents,
            "transformed_dense_rrf": transformed_dense,
            "transformed_rrf": transformed.parents,
            "raw_bm25": raw_bm25,
            "transformed_dense_bm25_rrf": transformed_dense_bm25,
        }
        evidence_routes = {
            "raw_dense": {"raw_dense": raw.routes.get("dense", [])},
            "raw_sparse": {"raw_sparse": raw.routes.get("sparse", [])},
            "raw_hybrid_rrf": raw.routes,
            "transformed_dense_rrf": {
                route: hits
                for route, hits in transformed.routes.items()
                if route.endswith(":dense")
            },
            "transformed_rrf": transformed.routes,
            "raw_bm25": {"raw_bm25": raw_bm25_children},
            "transformed_dense_bm25_rrf": transformed_dense_bm25_children,
        }
        for route, parent_hits in last_routes.items():
            _attach_best_evidence(parent_hits, evidence_routes.get(route, {}))
        if all(len(hits) >= target for hits in last_routes.values()) or fetch >= max_child_top_k:
            break
        next_fetch = min(max_child_top_k, max(fetch * 2, fetch + 500))
        if next_fetch == fetch:
            break
        fetch = next_fetch

    return {
        "item_id": str(record.get("item_id") or ""),
        "split": str(record.get("split") or ""),
        "query": str(record.get("query") or ""),
        "target_parent_id": str(record.get("target_parent_id") or ""),
        "routes": {
            route: [_serialise_parent_hit(hit, parents_by_id) for hit in hits]
            for route, hits in last_routes.items()
        },
        "child_fetch_k": fetch,
    }


def _serialise_parent_hit(hit: ParentHit, parents_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    parent = parents_by_id.get(hit.parent_id) or {}
    result = {
        "rank": int(hit.rank),
        "parent_id": str(hit.parent_id),
        "score": float(hit.score),
        "title": str(hit.title or parent.get("title") or ""),
        "path": str(hit.path or parent.get("path") or ""),
    }
    if hit.evidence_hit is not None:
        result["evidence"] = {
            "node_id": hit.evidence_hit.node_id,
            "node_type": hit.evidence_hit.node_type,
            "route": hit.evidence_hit.route,
            "rank": int(hit.evidence_hit.rank),
        }
    return result


def _attach_best_evidence(
    parent_hits: list[ParentHit],
    child_routes: dict[str, list[ChildHit]],
) -> None:
    """Attach one route-local evidence node without changing RRF scores."""

    best_by_parent: dict[str, tuple[tuple[int, int, str, str], ChildHit]] = {}
    type_priority = {"passage": 0, "lead": 1, "title": 2}
    for route, child_hits in child_routes.items():
        for hit in child_hits:
            if not hit.parent_id:
                continue
            key = (
                type_priority.get(hit.node_type, 3),
                int(hit.rank),
                route,
                hit.node_id,
            )
            current = best_by_parent.get(hit.parent_id)
            if current is None or key < current[0]:
                best_by_parent[hit.parent_id] = (key, hit)
    for parent_hit in parent_hits:
        selected = best_by_parent.get(parent_hit.parent_id)
        parent_hit.evidence_hit = selected[1] if selected else None


def _retrieval_queries_from_record(record: dict[str, Any]) -> list[RetrievalQuery]:
    trace = record.get("query_trace") or {}
    raw_queries = trace.get("retrieval_queries") if isinstance(trace, dict) else None
    queries = [
        RetrievalQuery(route=str(row.get("route") or ""), query=str(row.get("query") or "").strip())
        for row in raw_queries or []
        if isinstance(row, dict) and str(row.get("route") or "").strip() and str(row.get("query") or "").strip()
    ]
    expected = {"literal_question", "event_background", "mechanism_scene", "colloquial_surface"}
    if {row.route for row in queries} != expected or len(queries) != 4:
        raise ValueError(f"Query trace for {record.get('item_id')} must contain the four frozen transform routes")
    return queries


def _excluded_parent_ids(pool_manifest: dict[str, Any]) -> set[str]:
    dataset_path = Path(str(pool_manifest.get("dataset_path") or ""))
    if dataset_path.is_file():
        try:
            manifest = load_dataset_manifest(dataset_path)
            return {str(value) for value in manifest.get("excluded_parent_ids") or []}
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
    return set()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_manifest_if_complete(path: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("schema_version") != RANKING_SCHEMA_VERSION or manifest.get("status") != "completed":
        return None
    rankings_path = path.parent / str(manifest.get("rankings_file") or "rankings.jsonl")
    return manifest if rankings_path.is_file() else None


def _jsonable_config(config: RetrievalRankingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("pool_manifest_path", "index_dir", "qdrant_path", "out_dir"):
        value = payload.get(key)
        payload[key] = str(value) if value is not None else None
    return payload


def _git_revision() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}
