"""Persistent human labels for frozen retrieval candidate pools."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from personaforge.eval.retrieval_judge import load_label_set
from personaforge.eval.retrieval_metrics import (
    compute_query_diagnostics,
    compute_split_metrics,
    compute_split_metrics_from_rankings,
)
from personaforge.eval.retrieval_rankings import load_ranking_snapshot


@dataclass(frozen=True, slots=True)
class PoolFiles:
    manifest_path: Path
    pool_path: Path
    manifest: dict[str, Any]


class RetrievalEvaluationStore:
    """Read frozen pools and keep one independent label set per user."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.database_path = self.data_dir / "system" / "personaforge.sqlite3"
        self.eval_root = self.data_dir / "eval"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._pool_cache: dict[str, tuple[int, list[dict[str, Any]]]] = {}
        self._label_set_cache: dict[
            str,
            tuple[tuple[tuple[int, int], tuple[int, int]], dict[str, Any], dict[tuple[str, str], dict[str, Any]]],
        ] = {}
        self._metrics_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._global_report_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._initialize()

    def list_pools(self, user_id: str, author: str | None = None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for files in self._discover().values():
            manifest_author = str(files.manifest.get("author") or "").strip()
            if author is not None and manifest_author != author:
                continue
            records = self._load_records(files)
            total = sum(len(row.get("candidates") or []) for row in records)
            labeled = self._labeled_count_for_records(files, records, user_id)
            llm_label_sets = self._llm_label_set_status(files)
            results.append(
                {
                    "pool_id": str(files.manifest["pool_id"]),
                    "dataset_id": str(files.manifest.get("dataset_id") or ""),
                    "display_name": str(files.manifest.get("display_name") or ""),
                    "author": manifest_author,
                    "author_status": "assigned" if manifest_author else "unassigned",
                    "split": str(files.manifest.get("split") or ""),
                    "query_count": len(records),
                    "candidate_count": total,
                    "labeled_count": labeled,
                    "completed": total > 0 and labeled >= total,
                    "llm_label_sets": llm_label_sets,
                    "ranking_snapshots": self._ranking_status(files),
                    "created_at": str(files.manifest.get("created_at") or ""),
                }
            )
        return sorted(results, key=lambda row: (row["created_at"], row["pool_id"]), reverse=True)

    def _llm_label_set_status(self, files: PoolFiles) -> list[dict[str, Any]]:
        """Return lightweight report availability for the pool selector.

        Candidate pools are useful before any machine labels exist, so the
        report UI must distinguish pool discovery from report availability.
        Keep this metadata small and avoid loading candidate records here.
        """
        label_root = files.manifest_path.parent / "llm_labels"
        if not label_root.exists():
            return []
        results: list[dict[str, Any]] = []
        for manifest_path in label_root.glob("*/manifest.json"):
            try:
                manifest, _labels = self._load_label_set_cached(manifest_path)
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
            if str(manifest.get("pool_id") or "") != str(files.manifest.get("pool_id") or ""):
                continue
            results.append(
                {
                    "label_set": str(manifest.get("label_set") or manifest_path.parent.name),
                    "status": str(manifest.get("status") or "unknown"),
                    "completed": int(manifest.get("completed") or 0),
                    "total": int(manifest.get("total") or 0),
                    "label_source": str(manifest.get("label_source") or manifest.get("labeler") or ""),
                    "provisional": bool(manifest.get("provisional") or manifest.get("not_gold")),
                }
            )
        return sorted(results, key=lambda row: row["label_set"])

    def _ranking_status(self, files: PoolFiles) -> list[dict[str, Any]]:
        ranking_root = files.manifest_path.parent / "rankings"
        if not ranking_root.exists():
            return []
        results: list[dict[str, Any]] = []
        for manifest_path in ranking_root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(manifest.get("pool_id") or "") != str(files.manifest.get("pool_id") or ""):
                continue
            results.append(
                {
                    "ranking_id": str(manifest.get("ranking_id") or manifest_path.parent.name),
                    "status": str(manifest.get("status") or "unknown"),
                    "requested_depth": int(manifest.get("requested_depth") or 0),
                    "expected_depth": int(manifest.get("expected_depth") or 0),
                    "actual_depth_by_route": {
                        str(key): int(value)
                        for key, value in (manifest.get("actual_depth_by_route") or {}).items()
                    },
                    "query_count": int((manifest.get("counts") or {}).get("queries") or 0),
                    "created_at": str(manifest.get("created_at") or ""),
                    "reranker": _public_reranker(manifest.get("reranker")),
                }
            )
        return sorted(results, key=lambda row: (row["created_at"], row["ranking_id"]), reverse=True)

    def workspace(self, pool_id: str, user_id: str) -> dict[str, Any]:
        files = self._pool(pool_id)
        records = self._load_records(files)
        label_pool_id = self._label_namespace(files)
        queries = []
        for index, row in enumerate(records, start=1):
            item_id = str(row["item_id"])
            candidate_count = len(row.get("candidates") or [])
            candidate_ids = {str(candidate.get("parent_id") or "") for candidate in row.get("candidates") or []}
            labels = self._labels_for_query(label_pool_id, item_id, user_id)
            labeled_count = len(candidate_ids.intersection(labels))
            queries.append(
                {
                    "item_id": item_id,
                    "ordinal": index,
                    "query": str(row.get("query") or ""),
                    "candidate_count": candidate_count,
                    "labeled_count": labeled_count,
                    "completed": candidate_count > 0 and labeled_count >= candidate_count,
                }
            )
        total = sum(row["candidate_count"] for row in queries)
        labeled = sum(row["labeled_count"] for row in queries)
        return {
            "pool": self._public_manifest(files.manifest),
            "progress": {"labeled": labeled, "total": total, "completed": total > 0 and labeled >= total},
            "queries": queries,
        }

    def query(self, pool_id: str, item_id: str, user_id: str) -> dict[str, Any]:
        files = self._pool(pool_id)
        record = next(
            (row for row in self._load_records(files) if str(row.get("item_id")) == item_id),
            None,
        )
        if record is None:
            raise KeyError(f"Unknown evaluation query: {item_id}")
        labels = self._labels_for_query(self._label_namespace(files), item_id, user_id)
        candidates = sorted(
            record.get("candidates") or [],
            key=lambda row: self._shuffle_key(pool_id, item_id, str(row.get("parent_id") or ""), user_id),
        )
        public_candidates = []
        for index, candidate in enumerate(candidates, start=1):
            parent_id = str(candidate.get("parent_id") or "")
            score = labels.get(parent_id)
            public_candidates.append(
                {
                    "parent_id": parent_id,
                    "ordinal": index,
                    "title": str(candidate.get("title") or ""),
                    "text": str(candidate.get("text") or ""),
                    "url": str(candidate.get("url") or ""),
                    "kind": str(candidate.get("kind") or ""),
                    "score": score,
                    "retrieval_details": candidate.get("route_ranks") if score is not None else None,
                }
            )
        return {
            "pool_id": pool_id,
            "item_id": item_id,
            "query": str(record.get("query") or ""),
            "candidate_count": len(public_candidates),
            "labeled_count": sum(candidate["score"] is not None for candidate in public_candidates),
            "candidates": public_candidates,
        }

    def list_llm_label_sets(self, pool_id: str) -> list[dict[str, Any]]:
        """List shared machine-label runs attached to one frozen pool."""

        files = self._pool(pool_id)
        label_root = files.manifest_path.parent / "llm_labels"
        results: list[dict[str, Any]] = []
        if not label_root.exists():
            return results
        records = self._load_records(files)
        for manifest_path in label_root.glob("*/manifest.json"):
            try:
                manifest, labels = self._load_label_set_cached(manifest_path)
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
            if str(manifest.get("pool_id") or "") != pool_id:
                continue
            label_records = _records_for_label_manifest(records, manifest)
            axes, default_axis = _label_axes(manifest)
            score_labels = _axis_labels(labels, default_axis)
            raw_metrics = self._load_metrics_cached(
                manifest_path.parent,
                files.pool_path,
                label_records,
                score_labels,
                metric_axis=default_axis,
                recall_scope=str(files.manifest.get("recall_scope") or "six_route_candidate_union"),
            )
            metrics = _metrics_for_axis(raw_metrics, default_axis)
            results.append(
                {
                    "label_set": str(manifest.get("label_set") or manifest_path.parent.name),
                    "status": str(manifest.get("status") or "unknown"),
                    "model": str(manifest.get("model") or ""),
                    "label_source": str(manifest.get("label_source") or manifest.get("labeler") or ""),
                    "provisional": bool(manifest.get("provisional") or manifest.get("not_gold")),
                    "prompt_version": str(manifest.get("prompt_version") or ""),
                    "completed": int(manifest.get("completed") or len(score_labels)),
                    "total": int(
                        manifest.get("total")
                        or sum(len(row.get("candidates") or []) for row in label_records)
                    ),
                    "progress": dict(manifest.get("progress") or {}),
                    "updated_at": str(manifest.get("updated_at") or ""),
                    "axes": axes,
                    "default_axis": default_axis,
                    "selected_splits": list(manifest.get("selected_splits") or []),
                    "metrics": metrics,
                }
            )
        return sorted(results, key=lambda row: (row["updated_at"], row["label_set"]), reverse=True)

    def llm_workspace(
        self,
        pool_id: str,
        label_set: str,
        *,
        axis: str | None = None,
        ranking_id: str | None = None,
    ) -> dict[str, Any]:
        files = self._pool(pool_id)
        manifest, labels = self._load_llm_label_set(files, label_set)
        records = _records_for_label_manifest(self._load_records(files), manifest)
        axes, default_axis = _label_axes(manifest)
        active_axis = _validate_axis(axis, axes, default_axis)
        score_labels = _axis_labels(labels, active_axis)
        query_rows = []
        for index, record in enumerate(records, start=1):
            item_id = str(record.get("item_id") or "")
            candidate_ids = [str(candidate.get("parent_id") or "") for candidate in record.get("candidates") or []]
            labeled_count = sum(
                labels.get((item_id, parent_id), {}).get("status") == "completed"
                for parent_id in candidate_ids
            )
            query_rows.append(
                {
                    "item_id": item_id,
                    "ordinal": index,
                    "query": str(record.get("query") or ""),
                    "split": str(record.get("split") or files.manifest.get("split") or "unknown"),
                    "candidate_count": len(candidate_ids),
                    "labeled_count": labeled_count,
                    "completed": labeled_count >= len(candidate_ids) and bool(candidate_ids),
                    "qrels": compute_query_diagnostics(record, score_labels)["qrels"],
                }
            )
        ranking = None
        query_ranking = None
        if ranking_id:
            ranking_manifest_path = self._ranking_manifest_path(files, ranking_id)
            ranking, ranking_records = load_ranking_snapshot(ranking_manifest_path)
            raw_metrics = compute_split_metrics_from_rankings(
                records,
                ranking_records,
                score_labels,
                ranking_id=str(ranking.get("ranking_id") or ranking_id),
                requested_depth=int(ranking.get("requested_depth") or 100),
                recall_scope=str(files.manifest.get("recall_scope") or "frozen_qrels_candidate_pool"),
            )
        else:
            ranking_records = []
            raw_metrics = self._load_metrics_cached(
                files.manifest_path.parent / "llm_labels" / label_set,
                files.pool_path,
                records,
                score_labels,
                metric_axis=active_axis,
                recall_scope=str(files.manifest.get("recall_scope") or "six_route_candidate_union"),
            )
        ranking_by_item = {
            str(row.get("item_id") or ""): row
            for row in ranking_records
        }
        for row, record in zip(query_rows, records):
            diagnostics = compute_query_diagnostics(
                record,
                score_labels,
                ranking=ranking_by_item.get(str(record.get("item_id") or "")),
            )
            row["qrels"] = diagnostics["qrels"]
            row["route_metrics"] = diagnostics["routes"]
        return {
            "pool": self._public_manifest(files.manifest),
            "label_set": {
                "label_set": str(manifest.get("label_set") or label_set),
                "status": str(manifest.get("status") or "unknown"),
                "model": str(manifest.get("model") or ""),
                "prompt_version": str(manifest.get("prompt_version") or ""),
                "completed": int(manifest.get("completed") or len(score_labels)),
                "total": int(manifest.get("total") or sum(len(row.get("candidates") or []) for row in records)),
                "progress": dict(manifest.get("progress") or {}),
                "axes": axes,
                "default_axis": default_axis,
                "selected_splits": list(manifest.get("selected_splits") or []),
            },
            "active_axis": active_axis,
            "metrics": _metrics_for_axis(raw_metrics, active_axis),
            "comparison": _load_v1_v2_comparison(
                files.manifest_path.parent / "llm_labels" / label_set
            ),
            "ranking": _public_ranking(ranking) if ranking else None,
            "ranking_snapshots": self._ranking_status(files),
            "queries": query_rows,
        }

    def global_llm_report(
        self,
        *,
        axis: str | None = None,
        split: str = "all",
    ) -> dict[str, Any]:
        """Aggregate compatible LLM retrieval reports across authors.

        The aggregation is deliberately macro-averaged: one author contributes
        one report, regardless of how many candidate pairs their corpus has.
        When an author only has separate Dev/Test label runs, those runs are
        merged within that author before the cross-author average is computed.
        No retrieval or LLM call is performed here; this reads the frozen
        reports already materialized under ``data/eval``.
        """

        if split not in {"all", "dev", "test"}:
            raise ValueError("split must be all, dev, or test")

        discovered_pools = self._discover()
        cache_key = (
            str(axis or ""),
            split,
            _global_report_signature(discovered_pools),
        )
        with self._lock:
            cached_report = self._global_report_cache.get(cache_key)
            if cached_report is not None:
                return cached_report

        discovered_authors = sorted(
            {
                str(files.manifest.get("author") or "").strip()
                for files in discovered_pools.values()
                if str(files.manifest.get("author") or "").strip()
            }
        )
        candidates_by_author: dict[str, list[dict[str, Any]]] = {
            author: [] for author in discovered_authors
        }
        available_axes: dict[str, dict[str, Any]] = {}
        available_splits: set[str] = set()
        resolved_axis: str | None = None

        for files in discovered_pools.values():
            author = str(files.manifest.get("author") or "").strip()
            if not author:
                continue
            label_root = files.manifest_path.parent / "llm_labels"
            if not label_root.exists():
                continue
            records = self._load_records(files)
            for manifest_path in label_root.glob("*/manifest.json"):
                try:
                    label_manifest, labels = self._load_label_set_cached(manifest_path)
                except (OSError, json.JSONDecodeError, KeyError, TypeError):
                    continue
                if str(label_manifest.get("pool_id") or "") != str(files.manifest.get("pool_id") or ""):
                    continue
                if str(label_manifest.get("status") or "") != "completed":
                    continue
                label_records = _records_for_label_manifest(records, label_manifest)
                axes, default_axis = _label_axes(label_manifest)
                active_axis = str(axis or default_axis)
                if active_axis not in axes:
                    continue
                if resolved_axis is None:
                    resolved_axis = active_axis
                available_axes.update({str(key): dict(value) for key, value in axes.items()})
                raw_metrics = self._load_metrics_cached(
                    manifest_path.parent,
                    files.pool_path,
                    label_records,
                    _axis_labels(labels, active_axis),
                    metric_axis=active_axis,
                    recall_scope=str(files.manifest.get("recall_scope") or "six_route_candidate_union"),
                )
                axis_metrics = _metrics_for_axis(raw_metrics, active_axis)
                selected_splits = {
                    str(value).strip()
                    for value in (label_manifest.get("selected_splits") or [])
                    if str(value).strip() in {"dev", "test"}
                }
                # An empty selection is the historical convention for a
                # report covering the complete available pool.
                coverage = selected_splits or {
                    str(row.get("split") or "")
                    for row in label_records
                    if str(row.get("split") or "") in {"dev", "test"}
                }
                if not coverage:
                    coverage = {"all"}
                available_splits.update(coverage)
                candidates_by_author[author].append(
                    {
                        "files": files,
                        "label_manifest": label_manifest,
                        "label_set": str(label_manifest.get("label_set") or manifest_path.parent.name),
                        "metrics": axis_metrics,
                        "coverage": coverage,
                        "updated_at": str(label_manifest.get("updated_at") or ""),
                    }
                )

        included: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for author in discovered_authors:
            selected = _select_global_author_reports(candidates_by_author.get(author, []), split)
            if not selected:
                reason = "没有完成且包含当前评估维度的 LLM 标注"
                if split != "all" and candidates_by_author.get(author):
                    reason = f"没有覆盖 {split.upper()} 的完整 LLM 标注"
                skipped.append({"author": author, "reason": reason})
                continue
            reports = [
                _metrics_for_requested_split(candidate["metrics"], split, candidate["coverage"])
                for candidate in selected
            ]
            reports = [report for report in reports if isinstance(report, dict) and report.get("routes")]
            if not reports:
                skipped.append({"author": author, "reason": "标注存在，但没有可用的指标报告"})
                continue
            author_metrics = _combine_metric_reports(reports, weights=[_metric_query_count(report) for report in reports])
            included.append(
                {
                    "author": author,
                    "pool_ids": sorted({str(candidate["files"].manifest.get("pool_id") or "") for candidate in selected}),
                    "label_sets": [str(candidate["label_set"]) for candidate in selected],
                    "effective_split": split if split != "all" else (
                        "all" if any("all" in candidate["coverage"] for candidate in selected) else " + ".join(sorted({value for candidate in selected for value in candidate["coverage"]}))
                    ),
                    "query_count": int(author_metrics.get("query_count") or 0),
                    "routes": sorted((author_metrics.get("routes") or {}).keys()),
                    "metrics": author_metrics,
                }
            )

        global_metrics = _combine_metric_reports(
            [row["metrics"] for row in included],
            weights=[1.0 for _row in included],
        )
        for route, route_metrics in (global_metrics.get("routes") or {}).items():
            route_metrics["author_count"] = sum(
                route in (row.get("routes") or []) for row in included
            )
        # The cross-author selector must not expose a partial K as if it were
        # a five-author average. Keep the union for audit, but publish only
        # cutoffs supported by every included author on every route.
        author_cutoff_sets: list[set[int]] = []
        for row in included:
            route_cutoff_sets = [
                {
                    int(cutoff)
                    for cutoff in (route_metrics.get("by_cutoff") or {})
                    if str(cutoff).isdigit()
                }
                for route_metrics in (row.get("metrics", {}).get("routes") or {}).values()
            ]
            route_cutoff_sets = [values for values in route_cutoff_sets if values]
            if route_cutoff_sets:
                author_cutoff_sets.append(set.intersection(*route_cutoff_sets))
        if author_cutoff_sets:
            common_cutoffs = sorted(set.intersection(*author_cutoff_sets))
            all_cutoffs = sorted(set.union(*author_cutoff_sets))
            global_metrics["cutoffs"] = common_cutoffs
            global_metrics["available_cutoffs"] = all_cutoffs
            global_metrics["partial_cutoffs"] = [
                cutoff for cutoff in all_cutoffs if cutoff not in common_cutoffs
            ]
        if not available_splits:
            available_splits = {"all"}
        if "all" not in available_splits and included:
            available_splits.add("all")
        ordered_splits = [value for value in ("all", "dev", "test") if value in available_splits]

        result = {
            "schema_version": "personaforge.eval.retrieval_global_report.v1",
            "active_axis": resolved_axis or str(axis or ""),
            "axes": available_axes,
            "split": split,
            "available_splits": ordered_splits,
            "total_authors": len(discovered_authors),
            "included_authors": len(included),
            "skipped_authors": skipped,
            "authors": [
                {key: value for key, value in row.items() if key != "metrics"}
                for row in included
            ],
            "metrics": global_metrics,
        }
        with self._lock:
            self._global_report_cache[cache_key] = result
        return result

    def llm_query(
        self,
        pool_id: str,
        label_set: str,
        item_id: str,
        *,
        axis: str | None = None,
        ranking_id: str | None = None,
    ) -> dict[str, Any]:
        files = self._pool(pool_id)
        manifest, labels = self._load_llm_label_set(files, label_set)
        axes, default_axis = _label_axes(manifest)
        active_axis = _validate_axis(axis, axes, default_axis)
        records = _records_for_label_manifest(self._load_records(files), manifest)
        record = next((row for row in records if str(row.get("item_id") or "") == item_id), None)
        if record is None:
            raise KeyError(f"Unknown evaluation query: {item_id}")
        candidates = _sort_report_candidates(
            record.get("candidates") or [],
            labels,
            item_id=item_id,
            axis=active_axis,
        )
        public_candidates = []
        for candidate in candidates:
            label = candidate.get("label") or {}
            axis_scores = {
                key: label.get(key)
                for key in axes
                if label.get(key) in {0, 1, 2}
            }
            public_candidates.append(
                {
                    "parent_id": str(candidate.get("parent_id") or ""),
                    "relevance_order": int(candidate.get("relevance_order") or 0),
                    "best_route_rank": int(candidate.get("best_route_rank") or 0),
                    "route_count": int(candidate.get("route_count") or 0),
                    "title": str(candidate.get("title") or ""),
                    "text": str(candidate.get("text") or ""),
                    "url": str(candidate.get("url") or ""),
                    "kind": str(candidate.get("kind") or ""),
                    "score": label.get(active_axis),
                    "axis_scores": axis_scores,
                    "confidence": label.get("confidence"),
                    "evidence": _axis_evidence(label, active_axis),
                    "reason": label.get("reason") or "",
                    "content_candidate_evidence": label.get("content_candidate_evidence") or "",
                    "content_gold_unit_ids": label.get("content_gold_unit_ids") or [],
                    "persona_candidate_evidence": label.get("persona_candidate_evidence") or "",
                    "persona_gold_unit_ids": label.get("persona_gold_unit_ids") or [],
                    "repeat_count": int(label.get("repeat_count") or 0),
                    "exact_agreement": label.get("exact_agreement"),
                    "status": label.get("status") or "unjudged",
                    "route_ranks": candidate.get("route_ranks") or {},
                }
            )
        gold_context = _load_gold_context(manifest, item_id)
        ranking = None
        query_ranking = None
        if ranking_id:
            ranking_manifest_path = self._ranking_manifest_path(files, ranking_id)
            ranking, ranking_records = load_ranking_snapshot(ranking_manifest_path)
            ranking_row = next(
                (row for row in ranking_records if str(row.get("item_id") or "") == item_id),
                None,
            )
            if ranking_row:
                query_ranking = ranking_row
                ranking_routes = ranking_row.get("routes") or {}
                for candidate in public_candidates:
                    parent_id = candidate["parent_id"]
                    candidate["ranking_routes"] = {
                        route: entry
                        for route, entries in ranking_routes.items()
                        for entry in entries
                        if str(entry.get("parent_id") or "") == parent_id
                    }
        diagnostics = compute_query_diagnostics(
            record,
            _axis_labels(labels, active_axis),
            ranking=query_ranking,
        )
        return {
            "pool_id": pool_id,
            "label_set": str(manifest.get("label_set") or label_set),
            "axes": axes,
            "active_axis": active_axis,
            "item_id": item_id,
            "query": str(record.get("query") or ""),
            **gold_context,
            "ranking": _public_ranking(ranking) if ranking else None,
            "candidate_count": len(public_candidates),
            "labeled_count": sum(row["status"] == "completed" for row in public_candidates),
            "qrels": diagnostics["qrels"],
            "route_metrics": diagnostics["routes"],
            "candidates": public_candidates,
        }

    def set_label(self, pool_id: str, item_id: str, parent_id: str, user_id: str, score: int) -> dict[str, Any]:
        if score not in {0, 1, 2}:
            raise ValueError("score must be 0, 1, or 2")
        files = self._pool(pool_id)
        record = next(
            (row for row in self._load_records(files) if str(row.get("item_id")) == item_id),
            None,
        )
        candidate = next(
            (
                row
                for row in (record.get("candidates") if record else []) or []
                if str(row.get("parent_id") or "") == parent_id
            ),
            None,
        )
        if record is None or candidate is None:
            raise KeyError("Unknown evaluation candidate")
        now = _utc_now()
        label_pool_id = self._label_namespace(files)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO retrieval_labels (
                    pool_id, item_id, parent_id, user_id, score, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pool_id, item_id, parent_id, user_id)
                DO UPDATE SET score = excluded.score, updated_at = excluded.updated_at
                """,
                (label_pool_id, item_id, parent_id, user_id, score, now, now),
            )
        return {
            "pool_id": pool_id,
            "item_id": item_id,
            "parent_id": parent_id,
            "score": score,
            "retrieval_details": candidate.get("route_ranks") or {},
            "saved_at": now,
        }

    def export(self, pool_id: str, user_id: str, *, format: str) -> tuple[str, str, str]:
        files = self._pool(pool_id)
        records = self._load_records(files)
        label_pool_id = self._label_namespace(files)
        rows: list[dict[str, Any]] = []
        with self._connect() as connection:
            labels = {
                (str(row["item_id"]), str(row["parent_id"])): int(row["score"])
                for row in connection.execute(
                    "SELECT item_id, parent_id, score FROM retrieval_labels WHERE pool_id = ? AND user_id = ?",
                    (label_pool_id, user_id),
                ).fetchall()
            }
        for record in records:
            for candidate in record.get("candidates") or []:
                key = (str(record["item_id"]), str(candidate["parent_id"]))
                if key not in labels:
                    continue
                rows.append(
                    {
                        "pool_id": pool_id,
                        "item_id": key[0],
                        "query": str(record.get("query") or ""),
                        "parent_id": key[1],
                        "title": str(candidate.get("title") or ""),
                        "score": labels[key],
                        "url": str(candidate.get("url") or ""),
                        "route_ranks": candidate.get("route_ranks") or {},
                    }
                )
        if format == "jsonl":
            content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
            return content, "application/x-ndjson; charset=utf-8", f"{pool_id}.labels.jsonl"
        if format == "csv":
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output,
                fieldnames=["pool_id", "item_id", "query", "parent_id", "title", "score", "url", "route_ranks"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "route_ranks": json.dumps(row["route_ranks"], ensure_ascii=False, sort_keys=True)})
            return "\ufeff" + output.getvalue(), "text/csv; charset=utf-8", f"{pool_id}.labels.csv"
        raise ValueError("format must be jsonl or csv")

    def _discover(self) -> dict[str, PoolFiles]:
        pools: dict[str, PoolFiles] = {}
        if not self.eval_root.exists():
            return pools
        for manifest_path in self.eval_root.rglob("manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("schema_version") not in {
                "personaforge.eval.retrieval_pool.v0",
                "personaforge.eval.retrieval_pool.v1",
                "personaforge.eval.retrieval_pool.v2",
            }:
                continue
            pool_id = str(manifest.get("pool_id") or "")
            pool_path = manifest_path.parent / str(manifest.get("pool_file") or "pool.jsonl")
            if pool_id and pool_path.exists():
                pools[pool_id] = PoolFiles(manifest_path=manifest_path, pool_path=pool_path, manifest=manifest)
        return pools

    def _pool(self, pool_id: str) -> PoolFiles:
        try:
            return self._discover()[pool_id]
        except KeyError as exc:
            raise KeyError(f"Unknown retrieval pool: {pool_id}") from exc

    def _load_llm_label_set(self, files: PoolFiles, label_set: str) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
        safe_label_set = str(label_set).strip()
        if not safe_label_set or Path(safe_label_set).name != safe_label_set:
            raise KeyError("Unknown LLM label set")
        manifest_path = files.manifest_path.parent / "llm_labels" / safe_label_set / "manifest.json"
        if not manifest_path.exists():
            raise KeyError(f"Unknown LLM label set: {label_set}")
        manifest, labels = self._load_label_set_cached(manifest_path)
        if str(manifest.get("pool_id") or "") != str(files.manifest.get("pool_id") or ""):
            raise KeyError("LLM label set does not belong to this pool")
        return manifest, labels

    def _load_label_set_cached(
        self,
        manifest_path: Path,
    ) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
        """Load a machine-label run once until its manifest or labels change."""

        manifest_path = manifest_path.expanduser().resolve()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise
        labels_path = manifest_path.parent / str(manifest.get("labels_file") or "labels.jsonl")
        signature = (_path_signature(manifest_path), _path_signature(labels_path))
        cache_key = str(manifest_path)
        with self._lock:
            cached = self._label_set_cache.get(cache_key)
            if cached and cached[0] == signature:
                return cached[1], cached[2]
        loaded_manifest, labels = load_label_set(manifest_path)
        with self._lock:
            self._label_set_cache[cache_key] = (signature, loaded_manifest, labels)
        return loaded_manifest, labels

    def _load_metrics_cached(
        self,
        label_dir: Path,
        pool_path: Path,
        records: list[dict[str, Any]],
        labels: dict[tuple[str, str], int],
        *,
        metric_axis: str,
        recall_scope: str,
    ) -> dict[str, Any]:
        """Reuse computed metrics while the frozen pool or labels are unchanged."""

        label_dir = label_dir.expanduser().resolve()
        label_manifest_path = label_dir / "manifest.json"
        try:
            label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            label_manifest = {}
        labels_path = label_dir / str(label_manifest.get("labels_file") or "labels.jsonl")
        record_signature = (
            len(records),
            str(records[0].get("item_id") or "") if records else "",
            str(records[-1].get("item_id") or "") if records else "",
            tuple(sorted({str(row.get("split") or "") for row in records})),
        )
        cache_key = (
            str(label_dir),
            _path_signature(pool_path),
            _path_signature(label_manifest_path),
            _path_signature(labels_path),
            record_signature,
            metric_axis,
            recall_scope,
        )
        with self._lock:
            cached = self._metrics_cache.get(cache_key)
            if cached is not None:
                return cached
        result = _load_metrics(label_dir, records, labels, recall_scope=recall_scope)
        with self._lock:
            self._metrics_cache[cache_key] = result
        return result

    @staticmethod
    def _ranking_manifest_path(files: PoolFiles, ranking_id: str) -> Path:
        safe_ranking_id = str(ranking_id).strip()
        if not safe_ranking_id or Path(safe_ranking_id).name != safe_ranking_id:
            raise KeyError("Unknown retrieval ranking")
        path = files.manifest_path.parent / "rankings" / safe_ranking_id / "manifest.json"
        if not path.is_file():
            raise KeyError(f"Unknown retrieval ranking: {ranking_id}")
        return path

    def _load_records(self, files: PoolFiles) -> list[dict[str, Any]]:
        modified = files.pool_path.stat().st_mtime_ns
        with self._lock:
            cached = self._pool_cache.get(str(files.pool_path))
            if cached and cached[0] == modified:
                return cached[1]
            records = [
                json.loads(line)
                for line in files.pool_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self._pool_cache[str(files.pool_path)] = (modified, records)
            return records

    def _labeled_count_for_records(
        self,
        files: PoolFiles,
        records: list[dict[str, Any]],
        user_id: str,
    ) -> int:
        label_pool_id = self._label_namespace(files)
        total = 0
        for record in records:
            item_id = str(record.get("item_id") or "")
            candidate_ids = {str(candidate.get("parent_id") or "") for candidate in record.get("candidates") or []}
            total += len(candidate_ids.intersection(self._labels_for_query(label_pool_id, item_id, user_id)))
        return total

    @staticmethod
    def _label_namespace(files: PoolFiles) -> str:
        return str(files.manifest.get("label_namespace_pool_id") or files.manifest.get("pool_id") or "")

    def _label_count(self, pool_id: str, user_id: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM retrieval_labels WHERE pool_id = ? AND user_id = ?",
                    (pool_id, user_id),
                ).fetchone()[0]
            )

    def _label_counts_by_query(self, pool_id: str, user_id: str) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT item_id, COUNT(*) AS count
                FROM retrieval_labels
                WHERE pool_id = ? AND user_id = ?
                GROUP BY item_id
                """,
                (pool_id, user_id),
            ).fetchall()
        return {str(row["item_id"]): int(row["count"]) for row in rows}

    def _labels_for_query(self, pool_id: str, item_id: str, user_id: str) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT parent_id, score FROM retrieval_labels
                WHERE pool_id = ? AND item_id = ? AND user_id = ?
                """,
                (pool_id, item_id, user_id),
            ).fetchall()
        return {str(row["parent_id"]): int(row["score"]) for row in rows}

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_labels (
                    pool_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    score INTEGER NOT NULL CHECK(score BETWEEN 0 AND 2),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(pool_id, item_id, parent_id, user_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_retrieval_labels_user_pool
                ON retrieval_labels(user_id, pool_id, item_id)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @staticmethod
    def _shuffle_key(pool_id: str, item_id: str, parent_id: str, user_id: str) -> str:
        return hashlib.sha256(f"{pool_id}|{item_id}|{parent_id}|{user_id}".encode("utf-8")).hexdigest()

    @staticmethod
    def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
        return {
            "pool_id": str(manifest.get("pool_id") or ""),
            "dataset_id": str(manifest.get("dataset_id") or ""),
            "display_name": str(manifest.get("display_name") or ""),
            "author": str(manifest.get("author") or ""),
            "author_status": "assigned" if str(manifest.get("author") or "").strip() else "unassigned",
            "split": str(manifest.get("split") or ""),
            "created_at": str(manifest.get("created_at") or ""),
            "recall_scope": str(manifest.get("recall_scope") or "six_route_candidate_union"),
            "counts": dict(manifest.get("counts") or {}),
        }


def _path_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (0, 0)
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _global_report_signature(pools: dict[str, PoolFiles]) -> tuple[Any, ...]:
    """Return a cheap invalidation key without reading large JSONL files."""

    signature: list[tuple[Any, ...]] = []
    for pool_id, files in sorted(pools.items()):
        signature.append(
            (
                pool_id,
                _path_signature(files.manifest_path),
                _path_signature(files.pool_path),
            )
        )
        label_root = files.manifest_path.parent / "llm_labels"
        if not label_root.exists():
            continue
        for manifest_path in sorted(label_root.glob("*/manifest.json")):
            signature.append((str(manifest_path), _path_signature(manifest_path)))
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            labels_path = manifest_path.parent / str(manifest.get("labels_file") or "labels.jsonl")
            signature.append((str(labels_path), _path_signature(labels_path)))
            metrics_path = manifest_path.parent / "metrics.json"
            signature.append((str(metrics_path), _path_signature(metrics_path)))
    return tuple(signature)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _records_for_label_manifest(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Restrict a shared all-split pool to the split evaluated by one label run."""

    selected = {
        str(value).strip()
        for value in (manifest.get("selected_splits") or [])
        if str(value).strip()
    }
    if not selected:
        return records
    return [row for row in records if str(row.get("split") or "").strip() in selected]


def _load_metrics(
    label_dir: Path,
    records: list[dict[str, Any]],
    labels: dict[tuple[str, str], int],
    *,
    recall_scope: str = "six_route_candidate_union",
) -> dict[str, Any]:
    metrics_path = label_dir / "metrics.json"
    if metrics_path.exists():
        try:
            cached = json.loads(metrics_path.read_text(encoding="utf-8"))
            if _has_current_retrieval_metrics(cached):
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    return compute_split_metrics(records, labels, cutoff=3, recall_scope=recall_scope)


def _has_current_retrieval_metrics(metrics: dict[str, Any]) -> bool:
    axes = metrics.get("axes")
    reports = axes.values() if isinstance(axes, dict) else [metrics]
    return all(
        isinstance(report, dict)
        and report.get("schema_version") == "personaforge.eval.retrieval_metrics.v4"
        for report in reports
    )


def _label_axes(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    raw_axes = manifest.get("axes")
    if isinstance(raw_axes, dict) and raw_axes:
        axes = {
            str(key): dict(value) if isinstance(value, dict) else {"label": str(key)}
            for key, value in raw_axes.items()
        }
        default_axis = str(manifest.get("default_axis") or next(iter(axes)))
        if default_axis not in axes:
            default_axis = next(iter(axes))
        return axes, default_axis
    return {"score": {"label": "问题相关性", "values": [0, 1, 2]}}, "score"


def _validate_axis(
    requested: str | None,
    axes: dict[str, dict[str, Any]],
    default_axis: str,
) -> str:
    active = str(requested or default_axis)
    if active not in axes:
        raise KeyError(f"Unknown retrieval label axis: {active}")
    return active


def _axis_labels(
    labels: dict[tuple[str, str], dict[str, Any]],
    axis: str,
) -> dict[tuple[str, str], int]:
    return {
        key: int(row[axis])
        for key, row in labels.items()
        if row.get("status") == "completed" and row.get(axis) in {0, 1, 2}
    }


def _metrics_for_axis(metrics: dict[str, Any], axis: str) -> dict[str, Any]:
    axes = metrics.get("axes")
    if isinstance(axes, dict):
        selected = axes.get(axis)
        if not isinstance(selected, dict):
            raise KeyError(f"Metrics do not contain retrieval label axis: {axis}")
        return selected
    return metrics


_METRIC_SUM_KEYS = {
    "query_count",
    "candidate_count",
    "judged_candidate_count",
    "judged_query_count",
    "fully_judged_query_count",
    "unjudged_query_count",
    "relevant_candidate_count",
    "relevant_query_count",
    "no_relevant_query_count",
    "strict_metric_query_count",
    "qrels_label_count",
    "qrels_unlabelled_count",
    "qrels_zero_count",
    "qrels_relevant_count",
    "qrels_useful_count",
    "qrels_strong_count",
    "unjudged_top_count",
    "pool_outside_count",
    "useful_recall_micro_numerator",
    "useful_recall_micro_denominator",
    "strong_recall_micro_numerator",
    "strong_recall_micro_denominator",
    "useful_recall_zero_query_count",
    "useful_recall_one_query_count",
    "strong_recall_zero_query_count",
    "strong_recall_one_query_count",
}

_METRIC_DERIVED_KEYS = {
    "useful_recall_micro_at_k",
    "strong_recall_micro_at_k",
}


def _metric_query_count(metrics: dict[str, Any]) -> float:
    try:
        return max(1.0, float(metrics.get("query_count") or 0))
    except (TypeError, ValueError):
        return 1.0


def _numeric_average(values: list[tuple[float, float]]) -> float | None:
    if not values:
        return None
    total_weight = sum(weight for _value, weight in values)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in values) / total_weight


def _combine_metric_maps(
    reports: list[dict[str, Any]],
    weights: list[float],
) -> dict[str, Any]:
    """Merge metric dictionaries while preserving by-cutoff values.

    Rate fields are weighted means. Count fields are summed so the returned
    report remains auditable. The caller controls the weights: query counts
    when merging split runs within one author, and equal weights when taking
    the macro average across authors.
    """

    result: dict[str, Any] = {}
    keys = sorted({key for report in reports for key in report})
    for key in keys:
        values = [(report.get(key), weights[index]) for index, report in enumerate(reports) if key in report]
        if key == "by_cutoff":
            cutoff_keys = sorted({str(cutoff) for value, _weight in values if isinstance(value, dict) for cutoff in value})
            result[key] = {
                cutoff: _combine_metric_maps(
                    [value[cutoff] for value, _weight in values if isinstance(value, dict) and cutoff in value],
                    [weight for value, weight in values if isinstance(value, dict) and cutoff in value],
                )
                for cutoff in cutoff_keys
            }
            continue
        if key in {"cutoffs", "available_cutoffs"}:
            flattened = {
                int(item)
                for value, _weight in values
                if isinstance(value, (list, tuple))
                for item in value
                if str(item).isdigit()
            }
            result[key] = sorted(flattened)
            continue
        if key in {"route", "schema_version", "ranking_id", "recall_scope"}:
            result[key] = next((value for value, _weight in values if value not in {None, ""}), None)
            continue
        if key in _METRIC_DERIVED_KEYS:
            continue
        numeric_values: list[tuple[float, float]] = []
        for value, weight in values:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_values.append((float(value), weight))
        if numeric_values:
            if key in _METRIC_SUM_KEYS:
                result[key] = sum(value for value, _weight in numeric_values)
            else:
                result[key] = _numeric_average(numeric_values)
        else:
            result[key] = next((value for value, _weight in values if value is not None), None)
    _refresh_micro_recall(result)
    return result


def _refresh_micro_recall(metrics: dict[str, Any]) -> None:
    """Recompute pooled recall after count aggregation.

    Macro recall is averaged per query. Micro recall is derived from pooled
    useful/strong numerators and denominators, so it must never be averaged as
    an ordinary rate when combining authors or splits.
    """

    for prefix in ("useful", "strong"):
        numerator = metrics.get(f"{prefix}_recall_micro_numerator")
        denominator = metrics.get(f"{prefix}_recall_micro_denominator")
        if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)) and denominator:
            metrics[f"{prefix}_recall_micro_at_k"] = numerator / denominator
        else:
            metrics[f"{prefix}_recall_micro_at_k"] = None
    by_cutoff = metrics.get("by_cutoff")
    if isinstance(by_cutoff, dict):
        for value in by_cutoff.values():
            if isinstance(value, dict):
                _refresh_micro_recall(value)


def _combine_metric_reports(
    reports: list[dict[str, Any]],
    *,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    if not reports:
        return {"routes": {}, "query_count": 0}
    effective_weights = weights or [1.0 for _report in reports]
    if len(effective_weights) != len(reports):
        effective_weights = [1.0 for _report in reports]
    result = _combine_metric_maps(reports, effective_weights)
    result["routes"] = {}
    routes = sorted({route for report in reports for route in (report.get("routes") or {})})
    for route in routes:
        route_reports = [report["routes"][route] for report in reports if route in (report.get("routes") or {})]
        route_weights = [
            effective_weights[index]
            for index, report in enumerate(reports)
            if route in (report.get("routes") or {})
        ]
        result["routes"][route] = _combine_metric_maps(route_reports, route_weights)
    # Some historical metric files only materialized ``by_cutoff`` and
    # omitted the convenience list consumed by the web selector. Rebuild it
    # from the aggregate so K switching never falls back to one value.
    cutoff_keys = sorted(
        {
            int(cutoff)
            for cutoff in (result.get("by_cutoff") or {})
            if str(cutoff).isdigit()
        }
    )
    if cutoff_keys:
        result["cutoffs"] = cutoff_keys
        result["available_cutoffs"] = cutoff_keys
    # ``splits`` are intentionally removed: the caller has already selected
    # one split, and carrying nested split averages would be misleading.
    result.pop("splits", None)
    return result


def _metrics_for_requested_split(
    metrics: dict[str, Any],
    split: str,
    coverage: set[str],
) -> dict[str, Any]:
    if split == "all" and "all" in coverage:
        return metrics
    splits = metrics.get("splits") or {}
    if split in {"dev", "test"}:
        selected = splits.get(split)
        if isinstance(selected, dict):
            return selected
    for name in ("dev", "test"):
        if name in coverage and isinstance(splits.get(name), dict):
            return splits[name]
    return metrics


def _select_global_author_reports(
    candidates: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda row: (str(row.get("updated_at") or ""), str(row.get("label_set") or "")),
        reverse=True,
    )
    if split in {"dev", "test"}:
        matching = [row for row in ordered if split in row.get("coverage", set()) or "all" in row.get("coverage", set())]
        return matching[:1]
    complete = [row for row in ordered if "all" in row.get("coverage", set())]
    if complete:
        return complete[:1]
    # Some authors were labeled in two resumable tasks, one for Dev and one
    # for Test. Keep the newest run for each split and merge them within the
    # author before the macro average.
    selected: list[dict[str, Any]] = []
    for name in ("dev", "test"):
        match = next((row for row in ordered if name in row.get("coverage", set())), None)
        if match is not None:
            selected.append(match)
    return selected


def _axis_evidence(label: dict[str, Any], axis: str) -> str:
    if axis == "content_support":
        return str(label.get("content_candidate_evidence") or "")
    if axis == "persona_expression_support":
        return str(label.get("persona_candidate_evidence") or "")
    return str(label.get("evidence") or "")


def _sort_report_candidates(
    candidates: list[dict[str, Any]],
    labels: dict[tuple[str, str], dict[str, Any]],
    *,
    item_id: str,
    axis: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        label = labels.get((item_id, str(candidate.get("parent_id") or "")), {})
        route_ranks = candidate.get("route_ranks") or {}
        row["label"] = dict(label)
        row["best_route_rank"] = min(
            (
                int(details["rank"])
                for details in route_ranks.values()
                if isinstance(details, dict) and details.get("rank") is not None
            ),
            default=10**9,
        )
        row["route_count"] = len(route_ranks)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -int(row["label"].get(axis) if row["label"].get(axis) in {0, 1, 2} else -1),
            int(row.get("best_route_rank") or 10**9),
            -int(row.get("route_count") or 0),
            str(row.get("parent_id") or ""),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["relevance_order"] = index
    return rows


def _load_gold_context(manifest: dict[str, Any], item_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {"gold_answer": "", "gold_units": []}
    dataset_path = Path(str(manifest.get("dataset_path") or ""))
    if dataset_path.is_file():
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("item_id") or "") == item_id:
                result["gold_answer"] = str(row.get("gold_answer") or row.get("answer") or "")
                break
    units_path = Path(str(manifest.get("gold_units_path") or ""))
    if units_path.is_file():
        for line in units_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("item_id") or "") == item_id:
                result["gold_units"] = row.get("units") or []
                break
    return result


def _public_ranking(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if not manifest:
        return None
    return {
        "ranking_id": str(manifest.get("ranking_id") or ""),
        "status": str(manifest.get("status") or "unknown"),
        "requested_depth": int(manifest.get("requested_depth") or 0),
        "expected_depth": int(manifest.get("expected_depth") or 0),
        "eligible_parent_count": int(manifest.get("eligible_parent_count") or 0),
        "actual_depth_by_route": {
            str(key): int(value)
            for key, value in (manifest.get("actual_depth_by_route") or {}).items()
        },
        "query_count": int((manifest.get("counts") or {}).get("queries") or 0),
        "created_at": str(manifest.get("created_at") or ""),
        "reranker": _public_reranker(manifest.get("reranker")),
    }


def _public_reranker(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    timing = value.get("timing_ms") if isinstance(value.get("timing_ms"), dict) else {}
    return {
        "model": str(value.get("model") or ""),
        "representation": str(value.get("representation") or ""),
        "base_routes": [str(route) for route in value.get("base_routes") or []],
        "reranked_routes": [str(route) for route in value.get("reranked_routes") or []],
        "candidate_depth": int(value.get("candidate_depth") or 0),
        "batch_size": int(value.get("batch_size") or 0),
        "max_length": int(value.get("max_length") or 0),
        "pair_count": int(value.get("pair_count") or 0),
        "truncated_pair_count": int(value.get("truncated_pair_count") or 0),
        "truncated_pair_rate": value.get("truncated_pair_rate"),
        "timing_ms": timing,
    }


def _load_v1_v2_comparison(label_dir: Path) -> dict[str, Any] | None:
    path = label_dir / "comparison_v1_vs_v2.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    overall = raw.get("all") if isinstance(raw.get("all"), dict) else {}
    return {
        "v1_label_set": str(raw.get("v1_label_set") or ""),
        "v2_label_set": str(raw.get("v2_label_set") or ""),
        "comparison_axis": str(raw.get("comparison_axis") or "content_support"),
        "total": int(overall.get("total") or 0),
        "changed_count": int(raw.get("changed_count") or 0),
        "v1_zero_to_v2_positive": int(overall.get("v1_zero_to_v2_positive") or 0),
        "transition_counts": overall.get("counts") or {},
    }
