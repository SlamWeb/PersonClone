"""Retrieve parent documents from a Qdrant child-node index."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from personaforge.ingest.embeddings import BgeM3Encoder, TextEncoder
from personaforge.ingest.query_understanding import RetrievalQuery
from personaforge.ingest.qdrant_index import collection_name_for_author, create_local_client


@dataclass(slots=True)
class ChildHit:
    rank: int
    score: float
    node_id: str
    parent_id: str
    node_type: str
    title: str
    path: str
    route: str


@dataclass(slots=True)
class ParentHit:
    rank: int
    parent_id: str
    score: float
    title: str
    path: str
    first_hits: list[ChildHit] = field(default_factory=list)
    evidence_hit: ChildHit | None = None
    parent: dict[str, Any] | None = None


@dataclass(slots=True)
class RetrieveResult:
    query: str
    collection_name: str
    child_top_k: int
    parent_top_k: int
    routes: dict[str, list[ChildHit]]
    parents: list[ParentHit]
    retrieval_queries: list[RetrievalQuery] = field(default_factory=list)
    timing: dict[str, int] = field(default_factory=dict)


def retrieve_parents(
    query: str,
    *,
    author: str,
    index_dir: Path,
    qdrant_path: Path | None = None,
    encoder: TextEncoder | None = None,
    source: str = "zhihu",
    child_top_k: int = 100,
    parent_top_k: int = 20,
    rrf_k: int = 60,
    exclude_parent_ids: set[str] | None = None,
) -> RetrieveResult:
    collection_name = collection_name_for_author(source, author)
    qdrant_path = qdrant_path or index_dir / "qdrant"
    client = create_local_client(qdrant_path)
    encoder = encoder or BgeM3Encoder()
    timing: dict[str, int] = {}
    started_at = perf_counter()
    embedding = encoder.encode_texts([query], batch_size=1)[0]
    timing["embedding"] = _elapsed_ms(started_at)

    started_at = perf_counter()
    dense_hits = query_child_nodes(
        client,
        collection_name,
        query_vector=embedding.dense,
        route="dense",
        child_top_k=child_top_k,
        exclude_parent_ids=exclude_parent_ids,
    )
    timing["dense"] = _elapsed_ms(started_at)
    started_at = perf_counter()
    sparse_hits = query_child_nodes(
        client,
        collection_name,
        query_vector={
            "indices": embedding.sparse.indices,
            "values": embedding.sparse.values,
        },
        route="sparse",
        child_top_k=child_top_k,
        exclude_parent_ids=exclude_parent_ids,
    )
    timing["sparse"] = _elapsed_ms(started_at)

    routes = {"dense": dense_hits, "sparse": sparse_hits}
    started_at = perf_counter()
    parent_hits = fuse_parent_hits(routes, rrf_k=rrf_k, parent_top_k=parent_top_k)
    timing["parent_aggregation"] = _elapsed_ms(started_at)
    started_at = perf_counter()
    parents_by_id = load_parents(index_dir / "parents.jsonl")
    for hit in parent_hits:
        hit.parent = parents_by_id.get(hit.parent_id)
    timing["parent_load"] = _elapsed_ms(started_at)

    client.close()
    return RetrieveResult(
        query=query,
        collection_name=collection_name,
        child_top_k=child_top_k,
        parent_top_k=parent_top_k,
        routes=routes,
        parents=parent_hits,
        retrieval_queries=[RetrievalQuery(route="original_semantics", query=query)],
        timing=timing,
    )


def retrieve_parents_for_queries(
    query: str,
    retrieval_queries: list[RetrievalQuery],
    *,
    author: str,
    index_dir: Path,
    qdrant_path: Path | None = None,
    encoder: TextEncoder | None = None,
    source: str = "zhihu",
    child_top_k: int = 100,
    per_query_parent_k: int = 30,
    parent_top_k: int = 20,
    rrf_k: int = 60,
    exclude_parent_ids: set[str] | None = None,
) -> RetrieveResult:
    collection_name = collection_name_for_author(source, author)
    qdrant_path = qdrant_path or index_dir / "qdrant"
    client = create_local_client(qdrant_path)
    encoder = encoder or BgeM3Encoder()

    timing: dict[str, int] = {}
    child_routes: dict[str, list[ChildHit]] = {}
    parent_routes: dict[str, list[ParentHit]] = {}
    for retrieval_query in retrieval_queries:
        started_at = perf_counter()
        embedding = encoder.encode_texts([retrieval_query.query], batch_size=1)[0]
        timing[f"{retrieval_query.route}:embedding"] = _elapsed_ms(started_at)
        dense_route = f"{retrieval_query.route}:dense"
        sparse_route = f"{retrieval_query.route}:sparse"
        started_at = perf_counter()
        dense_hits = query_child_nodes(
            client,
            collection_name,
            query_vector=embedding.dense,
            route=dense_route,
            vector_name="dense",
            child_top_k=child_top_k,
            exclude_parent_ids=exclude_parent_ids,
        )
        timing[dense_route] = _elapsed_ms(started_at)
        started_at = perf_counter()
        sparse_hits = query_child_nodes(
            client,
            collection_name,
            query_vector={
                "indices": embedding.sparse.indices,
                "values": embedding.sparse.values,
            },
            route=sparse_route,
            vector_name="sparse",
            child_top_k=child_top_k,
            exclude_parent_ids=exclude_parent_ids,
        )
        timing[sparse_route] = _elapsed_ms(started_at)
        child_routes[dense_route] = dense_hits
        child_routes[sparse_route] = sparse_hits
        started_at = perf_counter()
        parent_routes[retrieval_query.route] = fuse_parent_hits(
            {dense_route: dense_hits, sparse_route: sparse_hits},
            rrf_k=rrf_k,
            parent_top_k=per_query_parent_k,
        )
        timing[f"{retrieval_query.route}:parent_rrf"] = _elapsed_ms(started_at)

    started_at = perf_counter()
    parent_hits = fuse_parent_rankings(parent_routes, rrf_k=rrf_k, parent_top_k=parent_top_k)
    timing["parent_aggregation"] = _elapsed_ms(started_at)
    started_at = perf_counter()
    parents_by_id = load_parents(index_dir / "parents.jsonl")
    for hit in parent_hits:
        hit.parent = parents_by_id.get(hit.parent_id)
    timing["parent_load"] = _elapsed_ms(started_at)

    client.close()
    return RetrieveResult(
        query=query,
        collection_name=collection_name,
        child_top_k=child_top_k,
        parent_top_k=parent_top_k,
        routes=child_routes,
        parents=parent_hits,
        retrieval_queries=retrieval_queries,
        timing=timing,
    )


def query_child_nodes(
    client: Any,
    collection_name: str,
    *,
    query_vector: Any,
    route: str,
    vector_name: str | None = None,
    child_top_k: int,
    exclude_parent_ids: set[str] | None = None,
) -> list[ChildHit]:
    using = vector_name or route
    if using == "sparse" and isinstance(query_vector, dict):
        try:
            from qdrant_client import models
        except ImportError:
            # Unit tests can use fake clients without installing the optional
            # qdrant-client dependency.
            pass
        else:
            query_vector = models.SparseVector(
                indices=query_vector["indices"],
                values=query_vector["values"],
            )

    query_options: dict[str, Any] = {
        "collection_name": collection_name,
        "query": query_vector,
        "using": using,
        "limit": child_top_k,
        "with_payload": True,
        "with_vectors": False,
    }
    if exclude_parent_ids:
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError("Excluded-parent retrieval requires qdrant-client.") from exc
        query_options["query_filter"] = models.Filter(
            must_not=[
                models.FieldCondition(
                    key="parent_id",
                    match=models.MatchAny(any=sorted(exclude_parent_ids)),
                )
            ]
        )

    response = client.query_points(
        **query_options,
    )
    points = getattr(response, "points", response)
    hits: list[ChildHit] = []
    for index, point in enumerate(points, start=1):
        payload = point.payload or {}
        hits.append(
            ChildHit(
                rank=index,
                score=float(point.score),
                node_id=str(payload.get("node_id", "")),
                parent_id=str(payload.get("parent_id", "")),
                node_type=str(payload.get("node_type", "")),
                title=str(payload.get("title", "")),
                path=str(payload.get("path", "")),
                route=route,
            )
        )
    return hits


def fuse_parent_hits(
    routes: dict[str, list[ChildHit]],
    *,
    rrf_k: int = 60,
    parent_top_k: int = 20,
) -> list[ParentHit]:
    scores: dict[str, float] = {}
    first_hits: dict[str, list[ChildHit]] = {}
    display: dict[str, ChildHit] = {}

    for hits in routes.values():
        seen_in_route: set[str] = set()
        for hit in hits:
            if not hit.parent_id or hit.parent_id in seen_in_route:
                continue
            seen_in_route.add(hit.parent_id)
            scores[hit.parent_id] = scores.get(hit.parent_id, 0.0) + 1.0 / (rrf_k + hit.rank)
            first_hits.setdefault(hit.parent_id, []).append(hit)
            display.setdefault(hit.parent_id, hit)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:parent_top_k]
    return [
        ParentHit(
            rank=index,
            parent_id=parent_id,
            score=score,
            title=display[parent_id].title,
            path=display[parent_id].path,
            first_hits=first_hits[parent_id],
        )
        for index, (parent_id, score) in enumerate(ranked, start=1)
    ]


def fuse_parent_rankings(
    routes: dict[str, list[ParentHit]],
    *,
    rrf_k: int = 60,
    parent_top_k: int = 20,
) -> list[ParentHit]:
    scores: dict[str, float] = {}
    first_hits: dict[str, list[ChildHit]] = {}
    display: dict[str, ParentHit] = {}

    for parent_hits in routes.values():
        seen_in_route: set[str] = set()
        for hit in parent_hits:
            if not hit.parent_id or hit.parent_id in seen_in_route:
                continue
            seen_in_route.add(hit.parent_id)
            scores[hit.parent_id] = scores.get(hit.parent_id, 0.0) + 1.0 / (rrf_k + hit.rank)
            first_hits.setdefault(hit.parent_id, []).extend(hit.first_hits)
            display.setdefault(hit.parent_id, hit)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:parent_top_k]
    return [
        ParentHit(
            rank=index,
            parent_id=parent_id,
            score=score,
            title=display[parent_id].title,
            path=display[parent_id].path,
            first_hits=first_hits[parent_id],
        )
        for index, (parent_id, score) in enumerate(ranked, start=1)
    ]


def fuse_transformed_dense_parent_rankings(
    retrieval_queries: list[RetrievalQuery],
    child_routes: dict[str, list[ChildHit]],
    *,
    rrf_k: int = 60,
    per_query_parent_k: int = 30,
    parent_top_k: int = 20,
) -> list[ParentHit]:
    """Fuse only the Dense side of each frozen transformed query.

    ``retrieve_parents_for_queries`` computes both Dense and Sparse hits for
    every transformed query.  This helper projects those same hits onto the
    Dense-only ablation before doing the existing two-level Parent RRF:
    first within each transformed query, then across transformed queries.
    It therefore isolates the effect of query transformation from the effect
    of adding Sparse retrieval.
    """

    per_query_rankings: dict[str, list[ParentHit]] = {}
    for retrieval_query in retrieval_queries:
        dense_route = f"{retrieval_query.route}:dense"
        per_query_rankings[retrieval_query.route] = fuse_parent_hits(
            {dense_route: child_routes.get(dense_route, [])},
            rrf_k=rrf_k,
            parent_top_k=per_query_parent_k,
        )
    return fuse_parent_rankings(
        per_query_rankings,
        rrf_k=rrf_k,
        parent_top_k=parent_top_k,
    )


def load_parents(path: Path) -> dict[str, dict[str, Any]]:
    parents: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        parents[str(row["doc_id"])] = row
    return parents


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)
