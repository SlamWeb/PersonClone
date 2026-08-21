from __future__ import annotations

import json
from pathlib import Path

import pytest

from personaforge.eval.bm25 import Bm25ChildIndex
from personaforge.eval.dataset import sha256_json
from personaforge.eval.retrieval_pool import (
    build_exhaustive_retrieval_pool,
    derive_core_pool,
    make_pool_record,
)
from personaforge.ingest.query_understanding import RetrievalQuery
from personaforge.ingest.retrieve import (
    ChildHit,
    ParentHit,
    fuse_transformed_dense_parent_rankings,
)


def test_bm25_uses_existing_nodes_and_excludes_future_parents(tmp_path: Path) -> None:
    pytest.importorskip("jieba")
    pytest.importorskip("rank_bm25")
    nodes = [
        {
            "node_id": "n1",
            "parent_id": "p1",
            "node_type": "passage",
            "title": "婚恋材料",
            "path": "a.md",
            "text": "女生在婚恋关系中讨论配得感",
        },
        {
            "node_id": "n2",
            "parent_id": "future",
            "node_type": "passage",
            "title": "泄漏材料",
            "path": "b.md",
            "text": "配得感 配得感 配得感",
        },
    ]
    path = tmp_path / "nodes.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in nodes), encoding="utf-8")

    index = Bm25ChildIndex.from_jsonl(path, exclude_parent_ids={"future"})
    hits = index.search("女生配得感", child_top_k=30)

    assert [hit.parent_id for hit in hits] == ["p1"]
    assert hits[0].route == "raw_bm25"


def test_pool_record_unions_routes_and_rejects_holdout_leaks() -> None:
    child = ChildHit(1, 0.9, "n1", "p1", "title", "材料", "a.md", "dense")
    parent = ParentHit(1, "p1", 1.0, "材料", "a.md", [child])
    parents = {
        "p1": {
            "doc_id": "p1",
            "title": "材料",
            "text": "完整正文",
            "url": "https://example.com",
            "path": "a.md",
            "kind": "answer",
            "created_at": "2026-01-01",
        }
    }
    routes = {"raw_dense": [parent], "raw_sparse": [], "transformed_rrf": [parent], "raw_bm25": []}
    record = make_pool_record(
        item={"item_id": "dev-01", "split": "dev", "query": "问题", "parent_id": "future"},
        pool_id="pool",
        routes=routes,
        parents_by_id=parents,
        query_trace={},
        excluded_parent_ids={"future"},
    )
    assert len(record["candidates"]) == 1
    assert set(record["candidates"][0]["route_ranks"]) == {"raw_dense", "transformed_rrf"}

    with pytest.raises(AssertionError):
        make_pool_record(
            item={"item_id": "dev-01", "split": "dev", "query": "问题", "parent_id": "future"},
            pool_id="pool",
            routes={**routes, "raw_bm25": [ParentHit(1, "future", 1.0, "泄漏", "future.md")]},
            parents_by_id=parents,
            query_trace={},
            excluded_parent_ids={"future"},
        )


def test_transformed_dense_route_reuses_only_dense_child_hits() -> None:
    queries = [
        RetrievalQuery(route="literal_question", query="问题一"),
        RetrievalQuery(route="mechanism_scene", query="问题二"),
    ]
    child_routes = {
        "literal_question:dense": [
            ChildHit(1, 0.9, "n1", "p1", "passage", "一", "a.md", "literal_question:dense"),
            ChildHit(2, 0.8, "n2", "p2", "passage", "二", "b.md", "literal_question:dense"),
        ],
        "literal_question:sparse": [
            ChildHit(1, 0.99, "n3", "p3", "passage", "三", "c.md", "literal_question:sparse"),
        ],
        "mechanism_scene:dense": [
            ChildHit(1, 0.7, "n4", "p2", "passage", "二", "b.md", "mechanism_scene:dense"),
        ],
    }

    ranked = fuse_transformed_dense_parent_rankings(
        queries,
        child_routes,
        per_query_parent_k=10,
        parent_top_k=10,
    )

    assert [hit.parent_id for hit in ranked] == ["p2", "p1"]
    assert "p3" not in {hit.parent_id for hit in ranked}


def test_derive_core_pool_keeps_any_route_head_and_shares_label_namespace(tmp_path: Path) -> None:
    source_dir = tmp_path / "retrieval_pool" / "dev"
    source_dir.mkdir(parents=True)
    source_pool_id = "dataset.dev.retrieval_pool.v0"
    records = [
        {
            "pool_id": source_pool_id,
            "item_id": "dev-01",
            "split": "dev",
            "query": "问题",
            "candidates": [
                {"parent_id": "dense-head", "route_ranks": {"raw_dense": {"rank": 3, "score": 1.0}}},
                {"parent_id": "bm25-head", "route_ranks": {"raw_bm25": {"rank": 1, "score": 2.0}}},
                {"parent_id": "tail", "route_ranks": {"raw_dense": {"rank": 4, "score": 0.5}}},
            ],
        }
    ]
    (source_dir / "pool.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "personaforge.eval.retrieval_pool.v0",
        "pool_id": source_pool_id,
        "dataset_id": "dataset",
        "author": "author-a",
        "split": "dev",
        "pool_file": "pool.jsonl",
        "pool_sha256": "source-sha",
    }
    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = derive_core_pool(manifest_path, route_depth=3)

    core_record = json.loads(result.pool_path.read_text(encoding="utf-8"))
    assert {candidate["parent_id"] for candidate in core_record["candidates"]} == {"dense-head", "bm25-head"}
    core_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert core_manifest["label_namespace_pool_id"] == source_pool_id
    assert core_manifest["display_name"] == "核心评估集 · 四路 Top3"
    assert result.candidate_count == 2


def test_exhaustive_pool_keeps_unretrieved_eligible_parents_without_gold_leak(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    parents = [
        {"doc_id": "p1", "title": "召回材料", "text": "A", "kind": "answer", "created_at": "2025-01-01"},
        {"doc_id": "p2", "title": "未召回材料", "text": "B", "kind": "pin", "created_at": "2025-01-02"},
        {"doc_id": "gold", "title": "目标答案", "text": "Gold", "kind": "answer", "created_at": "2026-01-01"},
    ]
    (index_dir / "parents.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in parents),
        encoding="utf-8",
    )
    dataset = [
        {
            "item_id": "dev-01",
            "split": "dev",
            "query": "问题",
            "parent_id": "gold",
            "gold_answer": "真实答案",
        }
    ]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps(dataset[0], ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_sha256": sha256_json(dataset),
                "excluded_parent_ids": ["gold"],
                "excluded_parent_ids_sha256": "excluded",
                "source_parents_sha256": sha256_json(parents),
                "counts": {"train_parents": 2},
            }
        ),
        encoding="utf-8",
    )
    source_dir = tmp_path / "retrieval_pool" / "six-route"
    source_dir.mkdir(parents=True)
    source_record = {
        "item_id": "dev-01",
        "split": "dev",
        "query": "问题",
        "target_parent_id": "gold",
        "query_trace": {},
        "candidates": [
            {"parent_id": "p1", "route_ranks": {"raw_dense": {"rank": 1, "score": 1.0}}}
        ],
    }
    (source_dir / "pool.jsonl").write_text(json.dumps(source_record, ensure_ascii=False) + "\n", encoding="utf-8")
    source_manifest_path = source_dir / "manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "personaforge.eval.retrieval_pool.v1",
                "pool_id": "source-pool",
                "dataset_id": "demo",
                "dataset_sha256": sha256_json(dataset),
                "pool_file": "pool.jsonl",
                "pool_sha256": "source-sha",
                "author": "author-a",
            }
        ),
        encoding="utf-8",
    )

    result = build_exhaustive_retrieval_pool(
        source_manifest_path,
        dataset_path=dataset_path,
        index_dir=index_dir,
    )

    record = json.loads(result.pool_path.read_text(encoding="utf-8"))
    assert {row["parent_id"] for row in record["candidates"]} == {"p1", "p2"}
    assert next(row for row in record["candidates"] if row["parent_id"] == "p2")["route_ranks"] == {}
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["recall_scope"] == "eligible_author_corpus_before_cutoff"
    assert manifest["counts"]["candidate_pairs"] == 2
