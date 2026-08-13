from __future__ import annotations

import json
from pathlib import Path

from personaforge.web.retrieval_eval_jobs import RetrievalEvalJobConfig, RetrievalEvalJobManager


def prepare_author(data_dir: Path, author: str = "demo-author") -> Path:
    index_dir = data_dir / "authors" / "zhihu" / author / "index"
    index_dir.mkdir(parents=True)
    (index_dir / "qdrant").mkdir()
    (index_dir / "parents.jsonl").write_text(
        json.dumps({"parent_id": "answer-1", "title": "题目", "text": "答案"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (index_dir / "nodes.jsonl").write_text(
        json.dumps({"node_id": "answer-1::title", "parent_id": "answer-1", "text": "题目"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return index_dir


def test_retrieval_eval_job_is_persistent_deduplicated_and_split_scoped(tmp_path: Path) -> None:
    prepare_author(tmp_path)
    manager = RetrievalEvalJobManager(RetrievalEvalJobConfig(data_dir=tmp_path, working_dir=tmp_path))
    first = manager.create(
        author="demo-author",
        labeler="codex_handoff",
        split="dev",
        budget_cny=5,
        owner_id="user-1",
    )
    duplicate = manager.create(
        author="demo-author",
        labeler="codex_handoff",
        split="dev",
        budget_cny=5,
        owner_id="user-1",
    )
    test_job = manager.create(
        author="demo-author",
        labeler="codex_handoff",
        split="test",
        budget_cny=5,
        owner_id="user-1",
    )

    assert duplicate["id"] == first["id"]
    assert first["label_set"].endswith("_dev")
    assert test_job["id"] != first["id"]
    assert manager._paths(manager._raw(first["id"]))["gold_units"].name == "gold_units_dev_v2.jsonl"
    assert manager._paths(manager._raw(test_job["id"]))["gold_units"].name == "gold_units_test_v2.jsonl"
    assert test_job["label_set"].endswith("_test")
    assert first["label_set"] != test_job["label_set"]
    reopened = RetrievalEvalJobManager(RetrievalEvalJobConfig(data_dir=tmp_path, working_dir=tmp_path))
    assert {job["id"] for job in reopened.list()} == {first["id"], test_job["id"]}


def test_gold_units_cache_must_cover_the_current_split(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"item_id": f"test-{index:02d}", "split": "test"}, ensure_ascii=False)
            for index in range(1, 4)
        )
        + "\n",
        encoding="utf-8",
    )
    gold = tmp_path / "gold.jsonl"
    category = {"id": "anchor", "text": "anchor"}
    units = {
        "stance": [category],
        "reasoning": [category],
        "example": [category],
        "expression": [category],
    }
    gold.write_text(
        json.dumps({"item_id": "test-01", "units": units}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert not RetrievalEvalJobManager._gold_units_complete(gold, dataset, "test")
    with gold.open("a", encoding="utf-8") as handle:
        for index in (2, 3):
            handle.write(
                json.dumps({"item_id": f"test-{index:02d}", "units": units}, ensure_ascii=False) + "\n"
            )
    assert RetrievalEvalJobManager._gold_units_complete(gold, dataset, "test")

def test_budget_paused_job_requires_a_larger_budget_to_resume(tmp_path: Path) -> None:
    prepare_author(tmp_path)
    manager = RetrievalEvalJobManager(RetrievalEvalJobConfig(data_dir=tmp_path, working_dir=tmp_path))
    job = manager.create(
        author="demo-author",
        labeler="deepseek_api",
        split="dev",
        budget_cny=2,
        owner_id="admin-1",
    )
    manager._update(job["id"], status="paused_budget", stage="paused_budget")

    try:
        manager.resume(job["id"], budget_cny=2)
    except ValueError as exc:
        assert "greater" in str(exc)
    else:
        raise AssertionError("Expected an unchanged budget to be rejected")

    resumed = manager.resume(job["id"], budget_cny=5)
    assert resumed["status"] == "queued"
    assert resumed["budget_cny"] == 5
