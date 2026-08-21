"""Cross-encoder reranking over frozen Parent ranking snapshots."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Protocol, Sequence

from personaforge.eval.dataset import utc_now, write_json, write_jsonl
from personaforge.eval.retrieval_rankings import RANKING_SCHEMA_VERSION, load_ranking_snapshot


DEFAULT_RERANK_ROUTES = (
    "raw_hybrid_rrf",
    "transformed_rrf",
    "transformed_dense_bm25_rrf",
)
RERANK_ROUTE_SUFFIX = "_reranked"


class PairReranker(Protocol):
    model_name: str

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class BgeCrossEncoderReranker:
    """BGE cross-encoder adapter using the official Transformers inference path."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        use_fp16: bool = True,
        batch_size: int = 2,
        max_length: int = 1024,
        cache_dir: Path | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError(
                "BGE reranking requires torch and transformers. "
                "Install with: pip install -e \".[index]\""
            ) from exc
        cache_dir_value = None
        if cache_dir is not None:
            cache_dir_value = str(cache_dir.expanduser().resolve())
        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA reranking was requested, but torch cannot access a CUDA device.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = resolved_device
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir_value)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir_value,
        )
        if use_fp16 and resolved_device == "cuda":
            self.model.half()
        self.model.to(resolved_device)
        self.model.eval()

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                encoded = self.tokenizer(
                    [[query, document] for query, document in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded, return_dict=True).logits.view(-1).float().cpu()
                scores.extend(float(score) for score in logits.tolist())
        return scores

    def input_lengths(self, pairs: Sequence[tuple[str, str]]) -> list[int]:
        if not pairs:
            return []
        encoded = self.tokenizer(
            [query for query, _document in pairs],
            [document for _query, document in pairs],
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        return [len(row) for row in encoded.get("input_ids") or []]


@dataclass(frozen=True, slots=True)
class RetrievalRerankerConfig:
    base_ranking_manifest_path: Path
    index_dir: Path
    ranking_id: str
    routes: tuple[str, ...] = DEFAULT_RERANK_ROUTES
    candidate_depth: int = 100
    max_length: int = 1024
    batch_size: int = 2
    out_dir: Path | None = None
    force: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalRerankerResult:
    ranking_id: str
    ranking_dir: Path
    manifest_path: Path
    rankings_path: Path
    query_count: int
    reranked_routes: tuple[str, ...]
    truncated_pair_count: int
    pair_count: int


def build_reranked_ranking_snapshot(
    config: RetrievalRerankerConfig,
    *,
    reranker: PairReranker,
) -> RetrievalRerankerResult:
    """Append reranked route variants to one immutable Parent ranking snapshot."""

    if config.candidate_depth < 1:
        raise ValueError("candidate_depth must be at least 1")
    if config.max_length < 32:
        raise ValueError("max_length must be at least 32")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not config.routes:
        raise ValueError("At least one base route is required")

    base_manifest_path = config.base_ranking_manifest_path.expanduser().resolve()
    base_manifest, base_rows = load_ranking_snapshot(base_manifest_path)
    index_dir = config.index_dir.expanduser().resolve()
    node_store = _NodeStore.load(index_dir / "nodes.jsonl")
    ranking_dir = (
        config.out_dir.expanduser().resolve()
        if config.out_dir
        else base_manifest_path.parent.parent / config.ranking_id
    )
    rankings_path = ranking_dir / "rankings.jsonl"
    manifest_path = ranking_dir / "manifest.json"
    partial_path = ranking_dir / "rankings.partial.jsonl"
    if (rankings_path.exists() or manifest_path.exists()) and not config.force:
        existing = _read_complete_manifest(manifest_path)
        if existing is not None:
            return _result_from_manifest(existing, ranking_dir)
        raise FileExistsError(
            f"Incomplete reranking snapshot exists: {ranking_dir}. Pass --force to replace it."
        )
    ranking_dir.mkdir(parents=True, exist_ok=True)
    if config.force:
        for path in (rankings_path, manifest_path, partial_path):
            path.unlink(missing_ok=True)

    output_rows: list[dict[str, Any]] = []
    actual_depth_by_route = {
        str(route): int(depth)
        for route, depth in (base_manifest.get("actual_depth_by_route") or {}).items()
    }
    route_timings: dict[str, list[int]] = {}
    all_input_lengths: list[int] = []
    evidence_fallback_count = 0
    pair_count = 0
    for base_row in base_rows:
        row = json.loads(json.dumps(base_row, ensure_ascii=False))
        routes = row.setdefault("routes", {})
        query = str(row.get("query") or "").strip()
        for base_route in config.routes:
            entries = list(routes.get(base_route) or [])
            if not entries:
                raise ValueError(
                    f"Base route {base_route!r} is absent for query {row.get('item_id')}"
                )
            selected = entries[: config.candidate_depth]
            prepared: list[tuple[dict[str, Any], dict[str, Any], str]] = []
            for entry in selected:
                evidence, used_fallback = node_store.evidence_for_entry(entry)
                evidence_fallback_count += int(used_fallback)
                document = _reranker_document(entry, evidence)
                prepared.append((entry, evidence, document))
            pairs = [(query, document) for _entry, _evidence, document in prepared]
            lengths = _input_lengths(reranker, pairs)
            all_input_lengths.extend(lengths)
            started_at = perf_counter()
            scores = reranker.score_pairs(pairs)
            elapsed_ms = round((perf_counter() - started_at) * 1000)
            if len(scores) != len(prepared):
                raise RuntimeError(
                    f"Reranker returned {len(scores)} scores for {len(prepared)} pairs"
                )
            pair_count += len(prepared)
            reranked_head = []
            for index, ((entry, evidence, _document), score) in enumerate(
                zip(prepared, scores, strict=True)
            ):
                base_rank = int(entry.get("rank") or index + 1)
                reranked_head.append(
                    {
                        **entry,
                        "base_rank": base_rank,
                        "base_score": float(entry.get("score") or 0.0),
                        "score": float(score),
                        "rerank_score": float(score),
                        "reranked": True,
                        "evidence": _public_evidence(evidence),
                        "input_tokens": lengths[index] if index < len(lengths) else None,
                        "input_truncated": (
                            lengths[index] > config.max_length if index < len(lengths) else None
                        ),
                    }
                )
            reranked_head.sort(
                key=lambda entry: (
                    -float(entry.get("rerank_score") or 0.0),
                    int(entry.get("base_rank") or 0),
                )
            )
            tail = [
                {
                    **entry,
                    "base_rank": int(entry.get("rank") or index + 1),
                    "base_score": float(entry.get("score") or 0.0),
                    "rerank_score": None,
                    "reranked": False,
                }
                for index, entry in enumerate(entries[config.candidate_depth :], start=config.candidate_depth)
            ]
            reranked_route = f"{base_route}{RERANK_ROUTE_SUFFIX}"
            routes[reranked_route] = [
                {**entry, "rank": rank}
                for rank, entry in enumerate(reranked_head + tail, start=1)
            ]
            actual_depth_by_route[reranked_route] = len(routes[reranked_route])
            route_timings.setdefault(reranked_route, []).append(elapsed_ms)
        output_rows.append(row)
        write_jsonl(output_rows, partial_path)

    write_jsonl(output_rows, rankings_path)
    partial_path.unlink(missing_ok=True)
    reranked_routes = tuple(f"{route}{RERANK_ROUTE_SUFFIX}" for route in config.routes)
    truncated_pair_count = sum(length > config.max_length for length in all_input_lengths)
    manifest = {
        "schema_version": RANKING_SCHEMA_VERSION,
        "ranking_id": config.ranking_id,
        "status": "completed",
        "created_at": utc_now(),
        "pool_id": str(base_manifest.get("pool_id") or ""),
        "qrels_pool_id": str(base_manifest.get("qrels_pool_id") or base_manifest.get("pool_id") or ""),
        "pool_manifest_path": str(base_manifest.get("pool_manifest_path") or ""),
        "pool_sha256": base_manifest.get("pool_sha256"),
        "dataset_sha256": base_manifest.get("dataset_sha256"),
        "author": str(base_manifest.get("author") or ""),
        "split": str(base_manifest.get("split") or "all"),
        "routes": list((base_manifest.get("routes") or [])) + list(reranked_routes),
        "requested_depth": int(base_manifest.get("requested_depth") or config.candidate_depth),
        "expected_depth": int(base_manifest.get("expected_depth") or config.candidate_depth),
        "eligible_parent_count": int(base_manifest.get("eligible_parent_count") or 0),
        "actual_depth_by_route": actual_depth_by_route,
        "rankings_file": rankings_path.name,
        "rankings_sha256": hashlib.sha256(rankings_path.read_bytes()).hexdigest(),
        "base_ranking": {
            "ranking_id": str(base_manifest.get("ranking_id") or ""),
            "manifest_path": str(base_manifest_path),
            "rankings_sha256": base_manifest.get("rankings_sha256"),
        },
        "reranker": {
            "model": reranker.model_name,
            "representation": "title_plus_best_passage",
            "base_routes": list(config.routes),
            "reranked_routes": list(reranked_routes),
            "candidate_depth": config.candidate_depth,
            "batch_size": config.batch_size,
            "max_length": config.max_length,
            "pair_count": pair_count,
            "input_token_count": len(all_input_lengths),
            "truncated_pair_count": truncated_pair_count,
            "truncated_pair_rate": (
                truncated_pair_count / len(all_input_lengths) if all_input_lengths else None
            ),
            "input_tokens_mean": fmean(all_input_lengths) if all_input_lengths else None,
            "input_tokens_p95": _percentile(all_input_lengths, 0.95),
            "evidence_fallback_count": evidence_fallback_count,
            "timing_ms": {
                route: {
                    "mean": fmean(values) if values else None,
                    "p95": _percentile(values, 0.95),
                    "total": sum(values),
                }
                for route, values in route_timings.items()
            },
        },
        "config": _jsonable_config(config),
        "counts": {"queries": len(output_rows), "reranker_pairs": pair_count},
        "git": _git_revision(),
    }
    write_json(manifest, manifest_path)
    return RetrievalRerankerResult(
        ranking_id=config.ranking_id,
        ranking_dir=ranking_dir,
        manifest_path=manifest_path,
        rankings_path=rankings_path,
        query_count=len(output_rows),
        reranked_routes=reranked_routes,
        truncated_pair_count=truncated_pair_count,
        pair_count=pair_count,
    )


@dataclass(frozen=True, slots=True)
class _NodeStore:
    by_id: dict[str, dict[str, Any]]
    fallback_by_parent: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> "_NodeStore":
        by_id: dict[str, dict[str, Any]] = {}
        by_parent: dict[str, list[dict[str, Any]]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            node_id = str(row.get("node_id") or "")
            parent_id = str(row.get("parent_id") or "")
            if node_id:
                by_id[node_id] = row
            if parent_id:
                by_parent.setdefault(parent_id, []).append(row)
        type_priority = {"passage": 0, "lead": 1, "title": 2}
        fallback_by_parent = {
            parent_id: min(
                rows,
                key=lambda row: (
                    type_priority.get(str(row.get("node_type") or ""), 3),
                    int(row.get("index") or 0),
                    str(row.get("node_id") or ""),
                ),
            )
            for parent_id, rows in by_parent.items()
            if rows
        }
        return cls(by_id=by_id, fallback_by_parent=fallback_by_parent)

    def evidence_for_entry(self, entry: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        evidence = entry.get("evidence") or {}
        node_id = str(evidence.get("node_id") or "")
        selected = self.by_id.get(node_id)
        if selected is not None and str(selected.get("node_type") or "") != "title":
            return selected, False
        fallback = self.fallback_by_parent.get(str(entry.get("parent_id") or ""))
        if fallback is None:
            return {
                "node_id": node_id,
                "parent_id": str(entry.get("parent_id") or ""),
                "node_type": "title",
                "title": str(entry.get("title") or ""),
                "text": str(entry.get("title") or ""),
                "index": 0,
            }, True
        return fallback, True


def _reranker_document(entry: dict[str, Any], evidence: dict[str, Any]) -> str:
    title = str(entry.get("title") or evidence.get("title") or "").strip()
    text = str(evidence.get("text") or "").strip()
    if not title:
        return text
    if not text or text == title:
        return title
    return f"{title}\n\n{text}"


def _public_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    text = str(evidence.get("text") or "")
    return {
        "node_id": str(evidence.get("node_id") or ""),
        "node_type": str(evidence.get("node_type") or ""),
        "index": int(evidence.get("index") or 0),
        "characters": len(text),
        "text": text,
    }


def _input_lengths(
    reranker: PairReranker,
    pairs: Sequence[tuple[str, str]],
) -> list[int]:
    measure = getattr(reranker, "input_lengths", None)
    if not callable(measure):
        return []
    return [int(value) for value in measure(pairs)]


def _percentile(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    return ordered[int((len(ordered) - 1) * percentile)]


def _jsonable_config(config: RetrievalRerankerConfig) -> dict[str, Any]:
    result = asdict(config)
    for key in ("base_ranking_manifest_path", "index_dir", "out_dir"):
        value = result.get(key)
        if value is not None:
            result[key] = str(value)
    result["routes"] = list(config.routes)
    return result


def _read_complete_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        return None
    rankings_path = path.parent / str(manifest.get("rankings_file") or "rankings.jsonl")
    if not rankings_path.is_file():
        return None
    return manifest


def _result_from_manifest(manifest: dict[str, Any], ranking_dir: Path) -> RetrievalRerankerResult:
    reranker = manifest.get("reranker") or {}
    return RetrievalRerankerResult(
        ranking_id=str(manifest.get("ranking_id") or ranking_dir.name),
        ranking_dir=ranking_dir,
        manifest_path=ranking_dir / "manifest.json",
        rankings_path=ranking_dir / str(manifest.get("rankings_file") or "rankings.jsonl"),
        query_count=int((manifest.get("counts") or {}).get("queries") or 0),
        reranked_routes=tuple(str(route) for route in reranker.get("reranked_routes") or []),
        truncated_pair_count=int(reranker.get("truncated_pair_count") or 0),
        pair_count=int(reranker.get("pair_count") or 0),
    )


def _git_revision() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
