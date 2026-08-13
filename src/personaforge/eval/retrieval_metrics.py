"""Metrics for frozen multi-route retrieval pools.

Labels are keyed by ``(item_id, parent_id)``.  A parent retrieved by several
routes is judged once, while every route keeps its own rank for evaluation.
"""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Mapping, Sequence


PREFERRED_ROUTE_ORDER = (
    "raw_dense",
    "raw_sparse",
    "raw_hybrid_rrf",
    "transformed_rrf",
    "raw_bm25",
    "transformed_dense_bm25_rrf",
)
DEFAULT_CUTOFFS = (1, 3, 5, 10, 20, 30)


def compute_retrieval_metrics(
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[tuple[str, str], int],
    *,
    cutoff: int = 3,
    cutoffs: Sequence[int] | None = None,
    relevance_threshold: int = 1,
    recall_scope: str = "six_route_candidate_union",
) -> dict[str, Any]:
    """Compute route-wise metrics without inventing labels.

    ``Recall@K`` is recall inside the frozen union of all route candidates, not
    corpus-wide recall.  It is emitted only when every candidate for that query
    has a label.  nDCG uses the same judged union as its ideal ranking.
    """

    requested_cutoffs = _normalise_cutoffs(cutoff=cutoff, cutoffs=cutoffs)
    if relevance_threshold < 0:
        raise ValueError("relevance_threshold must be non-negative")

    routes = discover_routes(records)
    route_stats: dict[str, dict[str, Any]] = {}
    for route in routes:
        by_cutoff = {
            str(k): _compute_route_at_k(
                records,
                labels,
                route=route,
                cutoff=k,
                relevance_threshold=relevance_threshold,
            )
            for k in requested_cutoffs
        }
        selected = dict(by_cutoff[str(cutoff)])
        selected["by_cutoff"] = by_cutoff
        route_stats[route] = selected

    total_candidates = sum(len(record.get("candidates") or []) for record in records)
    judged_candidates = sum(
        (str(record.get("item_id") or ""), _parent_id(candidate)) in labels
        for record in records
        for candidate in record.get("candidates") or []
    )
    relevant_candidates = sum(
        int(labels.get((str(record.get("item_id") or ""), _parent_id(candidate)), -1)) >= relevance_threshold
        for record in records
        for candidate in record.get("candidates") or []
    )
    return {
        "schema_version": "personaforge.eval.retrieval_metrics.v2",
        "cutoff": cutoff,
        "cutoffs": requested_cutoffs,
        "recall_scope": recall_scope,
        "relevance_threshold": relevance_threshold,
        "query_count": len(records),
        "candidate_count": total_candidates,
        "judged_candidate_count": judged_candidates,
        "relevant_candidate_count": relevant_candidates,
        "coverage": round(judged_candidates / total_candidates, 6) if total_candidates else 0.0,
        "routes": route_stats,
    }


def compute_split_metrics(
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[tuple[str, str], int],
    *,
    cutoff: int = 3,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    relevance_threshold: int = 1,
    recall_scope: str = "six_route_candidate_union",
) -> dict[str, Any]:
    """Return an overall report plus independently auditable split reports."""

    report = compute_retrieval_metrics(
        records,
        labels,
        cutoff=cutoff,
        cutoffs=cutoffs,
        relevance_threshold=relevance_threshold,
        recall_scope=recall_scope,
    )
    split_names = sorted({str(record.get("split") or "unknown") for record in records})
    report["splits"] = {
        split: compute_retrieval_metrics(
            [record for record in records if str(record.get("split") or "unknown") == split],
            labels,
            cutoff=cutoff,
            cutoffs=cutoffs,
            relevance_threshold=relevance_threshold,
            recall_scope=recall_scope,
        )
        for split in split_names
    }
    return report


def discover_routes(records: Sequence[Mapping[str, Any]]) -> list[str]:
    seen = {
        str(route)
        for record in records
        for candidate in record.get("candidates") or []
        for route in (candidate.get("route_ranks") or {})
    }
    preferred = [route for route in PREFERRED_ROUTE_ORDER if route in seen]
    return preferred + sorted(seen.difference(preferred))


def sort_candidates_by_relevance(
    candidates: Sequence[Mapping[str, Any]],
    labels: Mapping[tuple[str, str], Mapping[str, Any] | int],
    *,
    item_id: str,
) -> list[dict[str, Any]]:
    """Order an admin report by judged relevance without changing retrieval."""

    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        parent_id = _parent_id(candidate)
        raw_label = labels.get((item_id, parent_id))
        label = raw_label if isinstance(raw_label, Mapping) else {"score": raw_label}
        route_ranks = candidate.get("route_ranks") or {}
        best_rank = min(
            (
                int(details.get("rank"))
                for details in route_ranks.values()
                if isinstance(details, Mapping) and details.get("rank") is not None
            ),
            default=10**9,
        )
        row = dict(candidate)
        row["label"] = dict(label) if isinstance(label, Mapping) else {"score": None}
        row["best_route_rank"] = best_rank
        row["route_count"] = len(route_ranks)
        enriched.append(row)
    enriched.sort(
        key=lambda row: (
            -_label_score(row.get("label")),
            int(row.get("best_route_rank") or 10**9),
            -int(row.get("route_count") or 0),
            str(row.get("parent_id") or ""),
        )
    )
    for index, row in enumerate(enriched, start=1):
        row["relevance_order"] = index
    return enriched


def _compute_route_at_k(
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[tuple[str, str], int],
    *,
    route: str,
    cutoff: int,
    relevance_threshold: int,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "route": route,
        "cutoff": cutoff,
        "query_count": len(records),
        "candidate_count": 0,
        "judged_candidate_count": 0,
        "judged_query_count": 0,
        "fully_judged_query_count": 0,
        "coverage": 0.0,
        "hit_at_k": None,
        "mrr_at_k": None,
        "ndcg_at_k": None,
        "precision_at_k": None,
        "recall_at_k": None,
        "map_at_k": None,
        "relevant_query_count": 0,
        "no_relevant_query_count": 0,
    }
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    average_precisions: list[float] = []

    for record in records:
        item_id = str(record.get("item_id") or "")
        all_candidates = list(record.get("candidates") or [])
        ranked = _route_candidates(all_candidates, route, cutoff)
        stats["candidate_count"] += len(ranked)
        ranked_labels = [labels.get((item_id, _parent_id(candidate))) for candidate in ranked]
        stats["judged_candidate_count"] += sum(value is not None for value in ranked_labels)
        if not ranked or any(value is None for value in ranked_labels):
            continue

        scores = [int(value) for value in ranked_labels if value is not None]
        relevant = [score >= relevance_threshold for score in scores]
        stats["judged_query_count"] += 1
        hits.append(1.0 if any(relevant) else 0.0)
        reciprocal_ranks.append(
            next((1.0 / index for index, value in enumerate(relevant, start=1) if value), 0.0)
        )
        precisions.append(sum(relevant) / len(relevant))

        all_labels = [labels.get((item_id, _parent_id(candidate))) for candidate in all_candidates]
        if any(value is None for value in all_labels):
            continue
        stats["fully_judged_query_count"] += 1
        all_scores = [int(value) for value in all_labels if value is not None]
        ideal_gains = sorted((_gain(score) for score in all_scores), reverse=True)[: len(scores)]
        ideal_dcg = _dcg(ideal_gains)
        total_relevant = sum(score >= relevance_threshold for score in all_scores)
        if total_relevant:
            stats["relevant_query_count"] += 1
            ndcgs.append(_dcg([_gain(score) for score in scores]) / ideal_dcg if ideal_dcg else 0.0)
            recalls.append(sum(relevant) / total_relevant)
            precision_sum = sum(
                sum(relevant[:index]) / index
                for index, is_relevant in enumerate(relevant, start=1)
                if is_relevant
            )
            average_precisions.append(precision_sum / min(total_relevant, cutoff))
        else:
            stats["no_relevant_query_count"] += 1
            average_precisions.append(0.0)

    query_count = int(stats["query_count"])
    stats["coverage"] = round(
        stats["judged_candidate_count"] / stats["candidate_count"] if stats["candidate_count"] else 0.0,
        6,
    )
    stats["hit_at_k"] = _mean_or_none(hits)
    stats["mrr_at_k"] = _mean_or_none(reciprocal_ranks)
    stats["ndcg_at_k"] = _mean_or_none(ndcgs)
    stats["precision_at_k"] = _mean_or_none(precisions)
    stats["recall_at_k"] = _mean_or_none(recalls)
    stats["map_at_k"] = _mean_or_none(average_precisions)
    stats["unjudged_query_count"] = max(query_count - len(hits), 0)
    return stats


def _normalise_cutoffs(*, cutoff: int, cutoffs: Sequence[int] | None) -> list[int]:
    values = list(cutoffs) if cutoffs is not None else [cutoff]
    values.append(cutoff)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("cutoffs must contain positive integers")
    return sorted(set(values))


def _route_candidates(
    candidates: Sequence[Mapping[str, Any]], route: str, cutoff: int
) -> list[Mapping[str, Any]]:
    rows = []
    for candidate in candidates:
        details = (candidate.get("route_ranks") or {}).get(route)
        if isinstance(details, Mapping) and details.get("rank") is not None:
            rows.append((int(details["rank"]), candidate))
    rows.sort(key=lambda item: (item[0], _parent_id(item[1])))
    return [candidate for _, candidate in rows[:cutoff]]


def _parent_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("parent_id") or "")


def _label_score(label: Any) -> int:
    if isinstance(label, Mapping):
        label = label.get("score")
    return int(label) if isinstance(label, (int, float)) else -1


def _gain(score: int) -> float:
    return float((2**score) - 1)


def _dcg(values: Sequence[float]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(values))


def _mean_or_none(values: Sequence[float]) -> float | None:
    return round(fmean(values), 6) if values else None
