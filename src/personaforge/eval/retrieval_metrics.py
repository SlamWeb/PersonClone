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
DEFAULT_CUTOFFS = (1, 3, 5, 10, 20, 30, 50, 100, 200)
RANKING_NDCG_CUTOFFS = (1, 3, 5, 10, 20, 30)
RANKING_PRECISION_CUTOFFS = (1, 3, 5, 10, 20, 30)
RANKING_RECALL_CUTOFFS = (10, 20, 30, 50, 100)


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

    route_depths = _route_depths(records)
    max_supported_cutoff = min(route_depths.values(), default=0)
    requested_cutoffs = _normalise_cutoffs(
        cutoff=cutoff,
        cutoffs=cutoffs,
        max_supported_cutoff=max_supported_cutoff,
    )
    effective_cutoff = cutoff if cutoff in requested_cutoffs else requested_cutoffs[-1]
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
        selected = dict(by_cutoff[str(effective_cutoff)])
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
        "schema_version": "personaforge.eval.retrieval_metrics.v3",
        "cutoff": effective_cutoff,
        "cutoffs": requested_cutoffs,
        "max_supported_cutoff": max_supported_cutoff,
        "route_depths": route_depths,
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


def compute_split_metrics_from_rankings(
    records: Sequence[Mapping[str, Any]],
    ranking_records: Sequence[Mapping[str, Any]],
    labels: Mapping[tuple[str, str], int],
    *,
    ranking_id: str = "",
    requested_depth: int = 100,
    ndcg_cutoffs: Sequence[int] = RANKING_NDCG_CUTOFFS,
    precision_cutoffs: Sequence[int] = RANKING_PRECISION_CUTOFFS,
    recall_cutoffs: Sequence[int] = RANKING_RECALL_CUTOFFS,
    recall_scope: str = "frozen_qrels_candidate_pool",
    _include_splits: bool = True,
) -> dict[str, Any]:
    """Compute metrics from independent ordered Parent rankings.

    ``records`` defines the judged Qrels universe, while ``ranking_records``
    defines the actual output of each route.  A ranked parent outside Qrels is
    counted as zero relevance and exposed through ``pool_outside_*`` fields.
    This is intentionally separate from :func:`compute_split_metrics`, which
    evaluates the old route-rank fields embedded in a candidate pool.
    """

    if requested_depth < 1:
        raise ValueError("requested_depth must be positive")
    ndcg = _normalise_metric_cutoffs(ndcg_cutoffs, "nDCG")
    precision = _normalise_metric_cutoffs(precision_cutoffs, "precision")
    recall = _normalise_metric_cutoffs(recall_cutoffs, "recall")
    ranking_by_item = {
        str(row.get("item_id") or ""): row
        for row in ranking_records
        if str(row.get("item_id") or "")
    }
    route_names = [
        route
        for route in PREFERRED_ROUTE_ORDER
        if any(route in (row.get("routes") or {}) for row in ranking_records)
    ]
    route_names.extend(
        sorted(
            {
                str(route)
                for row in ranking_records
                for route in (row.get("routes") or {})
            }.difference(route_names)
        )
    )
    all_cutoffs = sorted(set(ndcg) | set(precision) | set(recall))
    route_stats: dict[str, dict[str, Any]] = {}
    for route in route_names:
        by_cutoff = {
            str(k): _compute_ranked_route_at_k(
                records,
                ranking_by_item,
                labels,
                route=route,
                cutoff=k,
                ndcg_enabled=k in ndcg,
                precision_enabled=k in precision,
                recall_enabled=k in recall,
            )
            for k in all_cutoffs
        }
        selected_cutoff = max(all_cutoffs, default=1)
        selected = dict(by_cutoff.get(str(selected_cutoff), {}))
        selected["by_cutoff"] = by_cutoff
        selected["available_cutoffs"] = {
            "ndcg": [k for k in ndcg if k <= _route_max_depth(ranking_by_item, route)],
            "precision": [k for k in precision if k <= _route_max_depth(ranking_by_item, route)],
            "recall": [k for k in recall if k <= _route_max_depth(ranking_by_item, route)],
        }
        selected["route_depth"] = _route_max_depth(ranking_by_item, route)
        route_stats[route] = selected

    qrels_pairs = {
        (str(record.get("item_id") or ""), str(candidate.get("parent_id") or ""))
        for record in records
        for candidate in record.get("candidates") or []
        if str(record.get("item_id") or "") and str(candidate.get("parent_id") or "")
    }
    labelled_pairs = {
        (str(item_id), str(parent_id)): int(score)
        for (item_id, parent_id), score in labels.items()
        if (str(item_id), str(parent_id)) in qrels_pairs
    }
    candidate_count = sum(len(record.get("candidates") or []) for record in records)
    judged_candidate_count = len(labelled_pairs)
    relevant_candidate_count = sum(score >= 1 for score in labelled_pairs.values())
    report = {
        "schema_version": "personaforge.eval.retrieval_metrics.v4",
        "ranking_id": ranking_id,
        "requested_depth": requested_depth,
        "cutoff": max(all_cutoffs, default=1),
        "cutoffs": all_cutoffs,
        "cutoff_groups": {"ndcg": ndcg, "precision": precision, "recall": recall},
        "max_supported_cutoff": max(
            (_route_max_depth(ranking_by_item, route) for route in route_names),
            default=0,
        ),
        "route_depths": {
            route: _route_max_depth(ranking_by_item, route)
            for route in route_names
        },
        "recall_scope": recall_scope,
        "relevance_threshold": 1,
        "query_count": len(records),
        "ranking_query_count": len(ranking_by_item),
        "candidate_count": candidate_count,
        "judged_candidate_count": judged_candidate_count,
        "relevant_candidate_count": relevant_candidate_count,
        "coverage": judged_candidate_count / candidate_count if candidate_count else 0.0,
        "routes": route_stats,
    }
    if _include_splits:
        split_names = sorted({str(record.get("split") or "unknown") for record in records})
        report["splits"] = {
            split: compute_split_metrics_from_rankings(
                [record for record in records if str(record.get("split") or "unknown") == split],
                [row for row in ranking_records if str(row.get("split") or "unknown") == split],
                labels,
                ranking_id=ranking_id,
                requested_depth=requested_depth,
                ndcg_cutoffs=ndcg,
                precision_cutoffs=precision,
                recall_cutoffs=recall,
                recall_scope=recall_scope,
                _include_splits=False,
            )
            for split in split_names
        }
    return report


def _compute_ranked_route_at_k(
    records: Sequence[Mapping[str, Any]],
    ranking_by_item: Mapping[str, Mapping[str, Any]],
    labels: Mapping[tuple[str, str], int],
    *,
    route: str,
    cutoff: int,
    ndcg_enabled: bool,
    precision_enabled: bool,
    recall_enabled: bool,
) -> dict[str, Any]:
    ndcg_values: list[float] = []
    precision_values: list[float] = []
    useful_precision_values: list[float] = []
    strong_precision_values: list[float] = []
    recall_values: list[float] = []
    useful_recall_values: list[float] = []
    strong_recall_values: list[float] = []
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    average_precisions: list[float] = []
    outside_counts: list[float] = []
    ranked_count = 0
    qrels_label_count = 0
    qrels_relevant_count = 0
    qrels_useful_count = 0
    qrels_strong_count = 0
    valid_recall_queries = 0

    for record in records:
        item_id = str(record.get("item_id") or "")
        ranking = ranking_by_item.get(item_id) or {}
        entries = list((ranking.get("routes") or {}).get(route) or [])
        entries.sort(key=lambda row: (int(row.get("rank") or 10**9), str(row.get("parent_id") or "")))
        top = entries[:cutoff]
        if not top:
            continue
        ranked_count += len(top)
        qrels_ids = {
            str(candidate.get("parent_id") or "")
            for candidate in record.get("candidates") or []
        }
        item_labels = {
            parent_id: int(score)
            for (label_item_id, parent_id), score in labels.items()
            if label_item_id == item_id and parent_id in qrels_ids
        }
        qrels_label_count += len(item_labels)
        qrels_relevant_count += sum(score >= 1 for score in item_labels.values())
        qrels_useful_count += sum(score >= 1 for score in item_labels.values())
        qrels_strong_count += sum(score == 2 for score in item_labels.values())
        outside = sum(str(entry.get("parent_id") or "") not in qrels_ids for entry in top)
        outside_counts.append(float(outside))
        scores = [item_labels.get(str(entry.get("parent_id") or ""), 0) for entry in top]
        useful = [score >= 1 for score in scores]
        strong = [score == 2 for score in scores]
        if precision_enabled:
            actual_k = len(top)
            precision_values.append(sum(useful) / actual_k)
            useful_precision_values.append(sum(useful) / actual_k)
            strong_precision_values.append(sum(strong) / actual_k)
        if ndcg_enabled:
            ideal_scores = sorted(item_labels.values(), reverse=True)[: len(top)]
            ideal = _dcg([_gain(score) for score in ideal_scores])
            ndcg_values.append(_dcg([_gain(score) for score in scores]) / ideal if ideal else 0.0)
        if recall_enabled:
            useful_total = sum(score >= 1 for score in item_labels.values())
            strong_total = sum(score == 2 for score in item_labels.values())
            if useful_total:
                useful_recall_values.append(sum(useful) / useful_total)
                valid_recall_queries += 1
            if strong_total:
                strong_recall_values.append(sum(strong) / strong_total)
            # Useful relevance is the main recall axis for backwards-compatible
            # ``recall_at_k``; strong recall is reported separately.
            if useful_total:
                recall_values.append(sum(useful) / useful_total)
        relevant = useful
        hits.append(1.0 if any(relevant) else 0.0)
        reciprocal_ranks.append(
            next((1.0 / index for index, value in enumerate(relevant, start=1) if value), 0.0)
        )
        relevant_total = sum(score >= 1 for score in item_labels.values())
        if relevant_total:
            average_precisions.append(
                sum(
                    sum(relevant[:index]) / index
                    for index, is_relevant in enumerate(relevant, start=1)
                    if is_relevant
                )
                / min(relevant_total, cutoff)
            )
        else:
            average_precisions.append(0.0)

    query_count = len(records)
    return {
        "route": route,
        "cutoff": cutoff,
        "query_count": query_count,
        "ranked_query_count": sum(1 for row in records if (ranking_by_item.get(str(row.get("item_id") or "")) or {}).get("routes", {}).get(route)),
        "candidate_count": ranked_count,
        "valid_recall_query_count": valid_recall_queries,
        "qrels_label_count": qrels_label_count,
        "qrels_relevant_count": qrels_relevant_count,
        "qrels_useful_count": qrels_useful_count,
        "qrels_strong_count": qrels_strong_count,
        "pool_outside_count": int(sum(outside_counts)),
        "pool_outside_rate": round(sum(outside_counts) / ranked_count, 6) if ranked_count else 0.0,
        "hit_at_k": _mean_or_none(hits),
        "mrr_at_k": _mean_or_none(reciprocal_ranks),
        "ndcg_at_k": _mean_or_none(ndcg_values),
        "precision_at_k": _mean_or_none(precision_values),
        "recall_at_k": _mean_or_none(recall_values),
        "map_at_k": _mean_or_none(average_precisions),
        "useful_precision_at_k": _mean_or_none(useful_precision_values),
        "strong_precision_at_k": _mean_or_none(strong_precision_values),
        "useful_recall_at_k": _mean_or_none(useful_recall_values),
        "strong_recall_at_k": _mean_or_none(strong_recall_values),
    }


def _route_max_depth(ranking_by_item: Mapping[str, Mapping[str, Any]], route: str) -> int:
    return max(
        (
            len((row.get("routes") or {}).get(route) or [])
            for row in ranking_by_item.values()
        ),
        default=0,
    )


def _normalise_metric_cutoffs(values: Sequence[int], name: str) -> list[int]:
    normalised = sorted(set(values))
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in normalised):
        raise ValueError(f"{name} cutoffs must contain positive integers")
    return normalised


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
        "useful_precision_at_k": None,
        "strong_precision_at_k": None,
        "useful_recall_at_k": None,
        "strong_recall_at_k": None,
        "relevant_query_count": 0,
        "no_relevant_query_count": 0,
    }
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    average_precisions: list[float] = []
    useful_precisions: list[float] = []
    strong_precisions: list[float] = []
    useful_recalls: list[float] = []
    strong_recalls: list[float] = []

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
        useful = [score >= 1 for score in scores]
        strong = [score == 2 for score in scores]
        stats["judged_query_count"] += 1
        hits.append(1.0 if any(relevant) else 0.0)
        reciprocal_ranks.append(
            next((1.0 / index for index, value in enumerate(relevant, start=1) if value), 0.0)
        )
        precisions.append(sum(relevant) / cutoff)
        useful_precisions.append(sum(useful) / cutoff)
        strong_precisions.append(sum(strong) / cutoff)

        all_labels = [labels.get((item_id, _parent_id(candidate))) for candidate in all_candidates]
        if any(value is None for value in all_labels):
            continue
        stats["fully_judged_query_count"] += 1
        all_scores = [int(value) for value in all_labels if value is not None]
        ideal_gains = sorted((_gain(score) for score in all_scores), reverse=True)[: len(scores)]
        ideal_dcg = _dcg(ideal_gains)
        total_relevant = sum(score >= relevance_threshold for score in all_scores)
        total_useful = sum(score >= 1 for score in all_scores)
        total_strong = sum(score == 2 for score in all_scores)
        if total_useful:
            useful_recalls.append(sum(useful) / total_useful)
        if total_strong:
            strong_recalls.append(sum(strong) / total_strong)
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
    stats["useful_precision_at_k"] = _mean_or_none(useful_precisions)
    stats["strong_precision_at_k"] = _mean_or_none(strong_precisions)
    stats["useful_recall_at_k"] = _mean_or_none(useful_recalls)
    stats["strong_recall_at_k"] = _mean_or_none(strong_recalls)
    stats["unjudged_query_count"] = max(query_count - len(hits), 0)
    return stats


def _normalise_cutoffs(
    *,
    cutoff: int,
    cutoffs: Sequence[int] | None,
    max_supported_cutoff: int,
) -> list[int]:
    values = list(cutoffs) if cutoffs is not None else [cutoff]
    values.append(cutoff)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("cutoffs must contain positive integers")
    if max_supported_cutoff <= 0:
        return [cutoff]
    values.append(max_supported_cutoff)
    return sorted({value for value in values if value <= max_supported_cutoff})


def _route_depths(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    depths: dict[str, int] = {}
    for record in records:
        for candidate in record.get("candidates") or []:
            for route, details in (candidate.get("route_ranks") or {}).items():
                if not isinstance(details, Mapping) or details.get("rank") is None:
                    continue
                key = str(route)
                depths[key] = max(depths.get(key, 0), int(details["rank"]))
    return depths


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
