from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from personaforge.web.app import create_app
from personaforge.web.retrieval_evaluation import (
    RetrievalEvaluationStore,
    _combine_metric_reports,
)
from personaforge.web.service import WebConfig


def write_pool(data_dir: Path, *, core: bool = False, author: str = "author-a") -> str:
    pool_id = "temporal_dev10_v0.dev.retrieval_pool.v0"
    directory = data_dir / "eval" / "demo" / "retrieval_pool" / "dev"
    directory.mkdir(parents=True)
    manifest = {
        "schema_version": "personaforge.eval.retrieval_pool.v0",
        "pool_id": pool_id,
        "dataset_id": "temporal_dev10_v0",
        "author": author,
        "split": "dev",
        "created_at": "2026-08-03T00:00:00+00:00",
        "pool_file": "pool.jsonl",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    record = {
        "pool_id": pool_id,
        "item_id": "dev-01",
        "split": "dev",
        "query": "测试问题",
        "target_parent_id": "future-parent",
        "candidates": [
            {
                "parent_id": "parent-a",
                "title": "材料 A",
                "text": "第一篇完整材料。",
                "url": "https://example.com/a",
                "kind": "answer",
                "route_ranks": {"raw_dense": {"rank": 1, "score": 0.9}},
            },
            {
                "parent_id": "parent-b",
                "title": "材料 B",
                "text": "第二篇完整材料。",
                "url": "https://example.com/b",
                "kind": "article",
                "route_ranks": {"raw_bm25": {"rank": 2, "score": 3.2}},
            },
        ],
    }
    (directory / "pool.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    if core:
        core_id = f"{pool_id}.core_top3.v0"
        core_directory = data_dir / "eval" / "demo" / "retrieval_pool" / "dev_core_top3"
        core_directory.mkdir(parents=True)
        core_manifest = {
            **manifest,
            "pool_id": core_id,
            "dataset_id": "temporal_dev10_v0_core_top3",
            "display_name": "核心评估集 · 四路 Top3",
            "label_namespace_pool_id": pool_id,
        }
        (core_directory / "manifest.json").write_text(json.dumps(core_manifest), encoding="utf-8")
        core_record = {**record, "pool_id": core_id, "candidates": record["candidates"][:1]}
        (core_directory / "pool.jsonl").write_text(
            json.dumps(core_record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return pool_id


def test_labels_are_user_scoped_persistent_and_hide_routes_before_scoring(tmp_path: Path) -> None:
    pool_id = write_pool(tmp_path)
    store = RetrievalEvaluationStore(tmp_path)

    first = store.query(pool_id, "dev-01", "user-a")
    second = store.query(pool_id, "dev-01", "user-a")
    assert [row["parent_id"] for row in first["candidates"]] == [row["parent_id"] for row in second["candidates"]]
    assert all(row["retrieval_details"] is None for row in first["candidates"])

    parent_id = first["candidates"][0]["parent_id"]
    store.set_label(pool_id, "dev-01", parent_id, "user-a", 2)
    store.set_label(pool_id, "dev-01", parent_id, "user-a", 1)

    user_a = store.query(pool_id, "dev-01", "user-a")
    scored = next(row for row in user_a["candidates"] if row["parent_id"] == parent_id)
    assert scored["score"] == 1
    assert scored["retrieval_details"]
    assert store.query(pool_id, "dev-01", "user-b")["labeled_count"] == 0
    assert store.workspace(pool_id, "user-a")["progress"] == {
        "labeled": 1,
        "total": 2,
        "completed": False,
    }

    content, media_type, filename = store.export(pool_id, "user-a", format="jsonl")
    assert json.loads(content)["score"] == 1
    assert media_type.startswith("application/x-ndjson")
    assert filename.endswith(".labels.jsonl")


def test_pool_listing_reports_each_users_progress_independently(tmp_path: Path) -> None:
    pool_id = write_pool(tmp_path)
    store = RetrievalEvaluationStore(tmp_path)
    store.set_label(pool_id, "dev-01", "parent-a", "user-a", 0)

    assert store.list_pools("user-a")[0]["labeled_count"] == 1
    assert store.list_pools("user-b")[0]["labeled_count"] == 0


def test_pool_listing_can_be_scoped_to_one_author(tmp_path: Path) -> None:
    first = write_pool(tmp_path, author="author-a")
    second_dir = tmp_path / "eval" / "demo-b" / "retrieval_pool" / "dev"
    second_dir.mkdir(parents=True)
    second_manifest = {
        "schema_version": "personaforge.eval.retrieval_pool.v0",
        "pool_id": "pool-b",
        "dataset_id": "temporal_dev10_v0",
        "author": "author-b",
        "split": "dev",
        "pool_file": "pool.jsonl",
    }
    (second_dir / "manifest.json").write_text(json.dumps(second_manifest), encoding="utf-8")
    (second_dir / "pool.jsonl").write_text(
        json.dumps({"pool_id": "pool-b", "item_id": "dev-01", "query": "问题", "candidates": []}),
        encoding="utf-8",
    )
    store = RetrievalEvaluationStore(tmp_path)
    assert [row["pool_id"] for row in store.list_pools("u1", author="author-a")] == [first]
    assert [row["pool_id"] for row in store.list_pools("u1", author="author-b")] == ["pool-b"]


def test_pool_listing_accepts_v1_six_route_manifests(tmp_path: Path) -> None:
    pool_id = write_pool(tmp_path)
    manifest_path = tmp_path / "eval" / "demo" / "retrieval_pool" / "dev" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "personaforge.eval.retrieval_pool.v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = RetrievalEvaluationStore(tmp_path)
    assert store.list_pools("user-a")[0]["pool_id"] == pool_id


def test_core_pool_reuses_source_labels_but_counts_only_core_candidates(tmp_path: Path) -> None:
    pool_id = write_pool(tmp_path, core=True)
    core_id = f"{pool_id}.core_top3.v0"
    store = RetrievalEvaluationStore(tmp_path)
    store.set_label(pool_id, "dev-01", "parent-a", "user-a", 2)
    store.set_label(pool_id, "dev-01", "parent-b", "user-a", 0)

    core = store.workspace(core_id, "user-a")
    assert core["progress"] == {"labeled": 1, "total": 1, "completed": True}
    assert store.query(core_id, "dev-01", "user-a")["labeled_count"] == 1

    store.set_label(core_id, "dev-01", "parent-a", "user-a", 1)
    source_label = next(
        candidate for candidate in store.query(pool_id, "dev-01", "user-a")["candidates"]
        if candidate["parent_id"] == "parent-a"
    )
    assert source_label["score"] == 1


def test_retrieval_evaluation_api_saves_and_exports_labels(tmp_path: Path) -> None:
    pool_id = write_pool(tmp_path)
    app = create_app(WebConfig(data_dir=tmp_path, auth_required=False))

    with TestClient(app) as client:
        pools = client.get("/api/evaluations/retrieval/pools")
        assert pools.status_code == 200
        assert pools.json()["pools"][0]["pool_id"] == pool_id

        query = client.get(f"/api/evaluations/retrieval/pools/{pool_id}/queries/dev-01")
        parent_id = query.json()["candidates"][0]["parent_id"]
        saved = client.put(
            f"/api/evaluations/retrieval/pools/{pool_id}/queries/dev-01/candidates/{parent_id}",
            json={"score": 2},
        )
        assert saved.status_code == 200
        assert saved.json()["retrieval_details"]

        workspace = client.get(f"/api/evaluations/retrieval/pools/{pool_id}")
        assert workspace.json()["progress"]["labeled"] == 1
        exported = client.get(f"/api/evaluations/retrieval/pools/{pool_id}/export?format=csv")
        assert exported.status_code == 200
        assert "dev-01" in exported.text


def test_retrieval_pool_api_accepts_author_scope(tmp_path: Path) -> None:
    first = write_pool(tmp_path, author="author-a")
    second_dir = tmp_path / "eval" / "demo-b" / "retrieval_pool" / "dev"
    second_dir.mkdir(parents=True)
    (second_dir / "manifest.json").write_text(json.dumps({
        "schema_version": "personaforge.eval.retrieval_pool.v0",
        "pool_id": "pool-b",
        "dataset_id": "temporal_dev10_v0",
        "author": "author-b",
        "split": "dev",
        "pool_file": "pool.jsonl",
    }), encoding="utf-8")
    (second_dir / "pool.jsonl").write_text(
        json.dumps({"pool_id": "pool-b", "item_id": "dev-01", "query": "问题", "candidates": []}),
        encoding="utf-8",
    )
    app = create_app(WebConfig(data_dir=tmp_path, auth_required=False))
    with TestClient(app) as client:
        response = client.get("/api/evaluations/retrieval/pools?author=author-b")
    assert response.status_code == 200
    assert [row["pool_id"] for row in response.json()["pools"]] == ["pool-b"]


def test_gold_aware_v2_report_supports_axes_gold_and_exhaustive_recall(tmp_path: Path) -> None:
    pool_id = write_pool(tmp_path)
    pool_dir = tmp_path / "eval" / "demo" / "retrieval_pool" / "dev"
    pool_path = pool_dir / "pool.jsonl"
    dev_record = json.loads(pool_path.read_text(encoding="utf-8").strip())
    test_record = {
        **dev_record,
        "item_id": "test-01",
        "split": "test",
        "query": "不属于本次 dev 标注的问题",
    }
    pool_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in (dev_record, test_record)
        ),
        encoding="utf-8",
    )
    pool_manifest_path = pool_dir / "manifest.json"
    pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    pool_manifest.update(
        {
            "schema_version": "personaforge.eval.retrieval_pool.v2",
            "recall_scope": "eligible_author_corpus_before_cutoff",
        }
    )
    pool_manifest_path.write_text(json.dumps(pool_manifest), encoding="utf-8")

    dataset_path = tmp_path / "eval" / "demo" / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {"item_id": "dev-01", "gold_answer": "作者真实回答。"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    gold_units_path = tmp_path / "eval" / "demo" / "gold_units.jsonl"
    gold_units_path.write_text(
        json.dumps(
            {
                "item_id": "dev-01",
                "units": {
                    "stance": [{"id": "stance-1", "text": "作者的核心判断。"}],
                    "expression": [{"id": "expression-1", "text": "作者的表达动作。"}],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    label_dir = pool_dir / "llm_labels" / "gold-v2"
    label_dir.mkdir(parents=True)
    label_manifest = {
        "schema_version": "personaforge.eval.retrieval_gold_labels.v2",
        "status": "completed",
        "label_set": "gold-v2",
        "pool_id": pool_id,
        "model": "judge",
        "prompt_version": "v2",
        "selected_splits": ["dev"],
        "completed": 2,
        "total": 2,
        "labels_file": "labels.jsonl",
        "dataset_path": str(dataset_path),
        "gold_units_path": str(gold_units_path),
        "axes": {
            "content_support": {"label": "内容支撑", "values": [0, 1, 2]},
            "persona_expression_support": {"label": "作者表达支撑", "values": [0, 1, 2]},
        },
        "default_axis": "content_support",
    }
    (label_dir / "manifest.json").write_text(json.dumps(label_manifest), encoding="utf-8")
    labels = [
        {
            "item_id": "dev-01",
            "parent_id": "parent-a",
            "status": "completed",
            "content_support": 2,
            "persona_expression_support": 0,
            "content_candidate_evidence": "内容证据 A",
            "content_gold_unit_ids": ["stance-1"],
            "persona_candidate_evidence": "",
            "persona_gold_unit_ids": [],
            "reason": "A 提供内容支撑。",
            "confidence": "high",
            "repeat_count": 2,
            "exact_agreement": True,
        },
        {
            "item_id": "dev-01",
            "parent_id": "parent-b",
            "status": "completed",
            "content_support": 0,
            "persona_expression_support": 2,
            "content_candidate_evidence": "",
            "content_gold_unit_ids": [],
            "persona_candidate_evidence": "表达证据 B",
            "persona_gold_unit_ids": ["expression-1"],
            "reason": "B 提供表达支撑。",
            "confidence": "medium",
            "repeat_count": 3,
            "exact_agreement": False,
        },
    ]
    (label_dir / "labels.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels),
        encoding="utf-8",
    )

    app = create_app(WebConfig(data_dir=tmp_path, auth_required=False))
    with TestClient(app) as client:
        pools = client.get("/api/evaluations/retrieval/pools").json()["pools"]
        assert pools[0]["pool_id"] == pool_id

        label_sets = client.get(
            f"/api/evaluations/retrieval/pools/{pool_id}/llm-labels"
        ).json()["label_sets"]
        assert label_sets[0]["default_axis"] == "content_support"
        assert set(label_sets[0]["axes"]) == {"content_support", "persona_expression_support"}

        workspace = client.get(
            f"/api/evaluations/retrieval/pools/{pool_id}/llm-labels/gold-v2"
            "?axis=persona_expression_support"
        ).json()
        assert workspace["active_axis"] == "persona_expression_support"
        assert workspace["metrics"]["recall_scope"] == "eligible_author_corpus_before_cutoff"
        assert [row["item_id"] for row in workspace["queries"]] == ["dev-01"]

        report = client.get(
            f"/api/evaluations/retrieval/pools/{pool_id}/llm-labels/gold-v2/queries/dev-01"
            "?axis=persona_expression_support"
        ).json()
        assert report["gold_answer"] == "作者真实回答。"
        assert report["gold_units"]["stance"][0]["id"] == "stance-1"
        assert report["candidates"][0]["parent_id"] == "parent-b"
        assert report["candidates"][0]["score"] == 2
        assert report["candidates"][0]["persona_gold_unit_ids"] == ["expression-1"]

        global_report = client.get(
            "/api/evaluations/retrieval/global"
            "?axis=persona_expression_support&split=dev"
        ).json()
        assert global_report["active_axis"] == "persona_expression_support"
        assert global_report["included_authors"] == 1
        assert global_report["total_authors"] == 1
        assert global_report["split"] == "dev"


def test_global_metric_combination_is_equal_author_macro_average() -> None:
    """Cross-author summaries must not let a larger query set dominate."""

    author_a = {
        "query_count": 2,
        "routes": {
            "raw_dense": {"ndcg_at_k": 0.2, "hit_at_k": 0.4, "relevant_query_count": 1},
        },
    }
    author_b = {
        "query_count": 20,
        "routes": {
            "raw_dense": {"ndcg_at_k": 0.8, "hit_at_k": 1.0, "relevant_query_count": 18},
        },
    }

    macro = _combine_metric_reports([author_a, author_b], weights=[1.0, 1.0])
    assert macro["routes"]["raw_dense"]["ndcg_at_k"] == 0.5
    assert macro["routes"]["raw_dense"]["hit_at_k"] == 0.7
    assert macro["routes"]["raw_dense"]["relevant_query_count"] == 19.0

    within_author = _combine_metric_reports([author_a, author_b], weights=[2.0, 20.0])
    assert round(within_author["routes"]["raw_dense"]["ndcg_at_k"], 6) == round(0.745454545, 6)
