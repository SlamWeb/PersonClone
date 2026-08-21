from __future__ import annotations

import json
from pathlib import Path

from personaforge.eval.retrieval_reranker import (
    RetrievalRerankerConfig,
    build_reranked_ranking_snapshot,
)
from personaforge.eval.retrieval_rankings import RANKING_SCHEMA_VERSION, load_ranking_snapshot


class _MarkerReranker:
    model_name = "fake-marker-reranker"

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [10.0 if "preferred evidence" in document else 1.0 for _query, document in pairs]

    def input_lengths(self, pairs: list[tuple[str, str]]) -> list[int]:
        return [len(query) + len(document) for query, document in pairs]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    base_dir = tmp_path / "pool" / "rankings" / "base"
    base_rows = [{
        "item_id": "q-1",
        "split": "test",
        "query": "current question",
        "routes": {
            "raw_hybrid_rrf": [
                {
                    "rank": 1,
                    "parent_id": "parent-a",
                    "score": 0.9,
                    "title": "Title A",
                    "path": "a.md",
                    "evidence": {"node_id": "a-title", "node_type": "title"},
                },
                {
                    "rank": 2,
                    "parent_id": "parent-b",
                    "score": 0.8,
                    "title": "Title B",
                    "path": "b.md",
                    "evidence": {"node_id": "b-passage", "node_type": "passage"},
                },
                {
                    "rank": 3,
                    "parent_id": "parent-c",
                    "score": 0.7,
                    "title": "Title C",
                    "path": "c.md",
                    "evidence": {"node_id": "c-passage", "node_type": "passage"},
                },
            ]
        },
    }]
    _write_jsonl(base_dir / "rankings.jsonl", base_rows)
    _write_json(base_dir / "manifest.json", {
        "schema_version": RANKING_SCHEMA_VERSION,
        "ranking_id": "base",
        "status": "completed",
        "pool_id": "pool-1",
        "author": "author-1",
        "split": "test",
        "routes": ["raw_hybrid_rrf"],
        "requested_depth": 3,
        "expected_depth": 3,
        "eligible_parent_count": 3,
        "actual_depth_by_route": {"raw_hybrid_rrf": 3},
        "rankings_file": "rankings.jsonl",
        "counts": {"queries": 1},
    })
    index_dir = tmp_path / "index"
    _write_jsonl(index_dir / "nodes.jsonl", [
        {
            "node_id": "a-title",
            "parent_id": "parent-a",
            "node_type": "title",
            "title": "Title A",
            "text": "Title A",
            "index": 0,
        },
        {
            "node_id": "a-passage",
            "parent_id": "parent-a",
            "node_type": "passage",
            "title": "Title A",
            "text": "ordinary fallback passage",
            "index": 1,
        },
        {
            "node_id": "b-passage",
            "parent_id": "parent-b",
            "node_type": "passage",
            "title": "Title B",
            "text": "preferred evidence",
            "index": 2,
        },
        {
            "node_id": "c-passage",
            "parent_id": "parent-c",
            "node_type": "passage",
            "title": "Title C",
            "text": "ordinary tail evidence",
            "index": 3,
        },
    ])
    return base_dir / "manifest.json", index_dir


def test_reranker_appends_route_and_keeps_parent_candidates(tmp_path: Path) -> None:
    manifest_path, index_dir = _fixture(tmp_path)
    result = build_reranked_ranking_snapshot(
        RetrievalRerankerConfig(
            base_ranking_manifest_path=manifest_path,
            index_dir=index_dir,
            ranking_id="reranked",
            routes=("raw_hybrid_rrf",),
            candidate_depth=2,
            max_length=64,
            batch_size=2,
        ),
        reranker=_MarkerReranker(),
    )

    manifest, rows = load_ranking_snapshot(result.manifest_path)
    base = rows[0]["routes"]["raw_hybrid_rrf"]
    reranked = rows[0]["routes"]["raw_hybrid_rrf_reranked"]

    assert [entry["parent_id"] for entry in base] == ["parent-a", "parent-b", "parent-c"]
    assert [entry["parent_id"] for entry in reranked] == ["parent-b", "parent-a", "parent-c"]
    assert reranked[0]["base_rank"] == 2
    assert reranked[0]["rank"] == 1
    assert reranked[0]["evidence"]["node_id"] == "b-passage"
    assert reranked[1]["evidence"]["node_id"] == "a-passage"
    assert reranked[2]["reranked"] is False
    assert manifest["reranker"]["evidence_fallback_count"] == 1
    assert manifest["reranker"]["pair_count"] == 2


def test_reranker_snapshot_is_idempotent_when_completed(tmp_path: Path) -> None:
    manifest_path, index_dir = _fixture(tmp_path)
    config = RetrievalRerankerConfig(
        base_ranking_manifest_path=manifest_path,
        index_dir=index_dir,
        ranking_id="reranked",
        routes=("raw_hybrid_rrf",),
        candidate_depth=3,
    )
    first = build_reranked_ranking_snapshot(config, reranker=_MarkerReranker())
    second = build_reranked_ranking_snapshot(config, reranker=_MarkerReranker())

    assert first.manifest_path == second.manifest_path
    assert second.pair_count == 3
