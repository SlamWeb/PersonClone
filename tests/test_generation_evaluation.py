from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from personaforge.eval.gold_judge import GROUP_DIMENSIONS
from personaforge.web.app import create_app
from personaforge.web.generation_evaluation import (
    GenerationEvaluationStore,
    GenerationJudgeManager,
)
from personaforge.web.service import WebConfig


DATASET_SHA = "a" * 64


def write_system(
    data_dir: Path,
    run_name: str,
    *,
    suffix: str,
    author: str = "alice",
    split: str = "dev",
    count: int = 10,
) -> str:
    dataset_dir = data_dir / "eval" / "demo"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "personaforge.eval.dataset.v0",
                "dataset_id": "temporal_test20_v0" if split == "test" else "temporal_dev10_v0",
                "author": author,
                "dataset_sha256": DATASET_SHA,
                "counts": {"dev": 10, "test": 20},
            }
        ),
        encoding="utf-8",
    )
    run_dir = dataset_dir / "runs" / run_name
    run_dir.mkdir(parents=True)
    rows = [
        {
            "item_id": f"{split}-{index:02d}",
            "split": split,
            "status": "completed",
            "query": f"问题 {index}",
            "gold_answer": f"作者回答 {index}",
            "answer": f"系统{suffix}回答 {index}",
        }
        for index in range(1, count + 1)
    ]
    (run_dir / "runs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    system_id = (suffix.lower() * 64)[:64]
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "personaforge.eval.run.v0",
                "status": "completed",
                "item_count": count,
                "dataset_sha256": DATASET_SHA,
                "run_sha256": system_id,
                "finished_at": "2026-08-03T00:00:00+00:00",
                "writer_model": "fake-model",
                "config": {
                    "author": author,
                    "run_name": run_name,
                    "split": split,
                    "writer_prompt": "strong_identity",
                },
            }
        ),
        encoding="utf-8",
    )
    return system_id


def test_generation_systems_rubric_and_pairwise_are_persistent(tmp_path: Path) -> None:
    left = write_system(tmp_path, "baseline-v1", suffix="b")
    right = write_system(tmp_path, "persona-pack-v1", suffix="c")
    store = GenerationEvaluationStore(tmp_path)

    assert {row["system_id"] for row in store.list_systems("u1")} == {left, right}

    partial = store.set_rubric(left, "dev-01", "u1", {"d1_stance_value": 4})
    assert partial["completed"] is False
    scores = {
        "d1_stance_value": 4,
        "d2_argumentation": 3,
        "d3_lexicon_register": 3,
        "d4_tone_posture": 4,
        "d5_syntax_rhythm": 3,
        "d6_naturalness_artifacts": 4,
    }
    complete = store.set_rubric(left, "dev-01", "u1", scores, "可接受")
    assert complete["completed"] is True
    assert store.workspace(left, "u1")["progress"] == {"completed": 1, "total": 10}

    first = store.comparison_item(left, right, "dev-01", "u1")
    again = store.comparison_item(left, right, "dev-01", "u1")
    assert first == again
    assert first["revealed"] is None
    voted = store.set_pair_vote(left, right, "dev-01", "u1", "A")
    assert voted["revealed"]["A"]["system_id"] == voted["winner"]["system_id"]
    assert store.comparison(left, right, "u1")["progress"] == {"completed": 1, "total": 10}


def test_generation_systems_are_author_scoped_and_cross_author_pairs_are_rejected(tmp_path: Path) -> None:
    alice = write_system(tmp_path, "alice-v1", suffix="b", author="alice")
    bob = write_system(tmp_path, "bob-v1", suffix="c", author="bob")
    store = GenerationEvaluationStore(tmp_path)

    assert {row["system_id"] for row in store.list_systems("u1", author="alice")} == {alice}
    assert {row["system_id"] for row in store.list_systems("u1", author="bob")} == {bob}
    try:
        store.comparison(alice, bob, "u1")
    except ValueError as exc:
        assert str(exc) == "Systems belong to different authors"
    else:
        raise AssertionError("cross-author comparison must be rejected")


def test_generation_store_discovers_test20_runs(tmp_path: Path) -> None:
    system_id = write_system(
        tmp_path,
        "test20-v1",
        suffix="t",
        split="test",
        count=20,
    )
    store = GenerationEvaluationStore(tmp_path)

    systems = store.list_systems("u1")
    assert systems[0]["system_id"] == system_id
    assert systems[0]["split"] == "test"
    assert systems[0]["item_count"] == 20
    assert store.workspace(system_id, "u1")["progress"] == {
        "completed": 0,
        "total": 20,
    }


class FakeJudgeClient:
    def complete_json(self, messages, *, temperature=0.0, max_tokens=2500):
        system = messages[0]["content"]
        group = next(group for group, keys in GROUP_DIMENSIONS.items() if all(key in system for key in keys))
        return {
            "dimensions": {
                key: {
                    "score": 4,
                    "status": "scored",
                    "gold_evidence": ["作者证据"],
                    "candidate_evidence": ["候选证据"],
                    "reason": "整体一致。",
                }
                for key in GROUP_DIMENSIONS[group]
            }
        }


class FailOnceJudgeClient(FakeJudgeClient):
    def __init__(self) -> None:
        self.failures_remaining = 9

    def complete_json(self, messages, *, temperature=0.0, max_tokens=2500):
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise ValueError("transient judge response error")
        return super().complete_json(messages, temperature=temperature, max_tokens=max_tokens)


def test_generation_judge_job_runs_without_blocking_the_api_contract(tmp_path: Path) -> None:
    system_id = write_system(tmp_path, "baseline-v1", suffix="b")
    store = GenerationEvaluationStore(tmp_path)
    manager = GenerationJudgeManager(store, client_factory=FakeJudgeClient)

    job = manager.create(system_id)
    assert job["status"] == "queued"
    assert manager.run_once() is True

    completed = store.public_judge_job(job["id"])
    assert completed["status"] == "completed"
    assert completed["completed_items"] == 10
    assert completed["prompt_version"] == "gold-judge-v1.0"
    assert completed["result"]["dimensions"]["d1_stance_value"]["mean"] == 4.0


def test_failed_judge_job_reuses_partial_output_on_retry(tmp_path: Path) -> None:
    system_id = write_system(tmp_path, "baseline-v1", suffix="b")
    store = GenerationEvaluationStore(tmp_path)
    flaky_client = FailOnceJudgeClient()
    manager = GenerationJudgeManager(store, client_factory=lambda: flaky_client)

    job = manager.create(system_id)
    assert manager.run_once() is True
    failed = store.public_judge_job(job["id"])
    assert failed["status"] == "failed"
    assert failed["completed_items"] == 0

    retry = manager.create(system_id)
    assert retry["id"] == job["id"]
    assert retry["status"] == "queued"
    assert manager.run_once() is True

    completed = store.public_judge_job(job["id"])
    assert completed["status"] == "completed"
    assert completed["completed_items"] == 10


def test_generation_evaluation_api_lists_complete_runs(tmp_path: Path) -> None:
    system_id = write_system(tmp_path, "baseline-v1", suffix="b")
    app = create_app(WebConfig(data_dir=tmp_path, auth_required=False))

    with TestClient(app) as client:
        response = client.get("/api/evaluations/generation/systems")
        detail = client.get(f"/api/evaluations/generation/systems/{system_id}")

    assert response.status_code == 200
    assert response.json()["systems"][0]["system_id"] == system_id
    assert detail.status_code == 200
    assert detail.json()["progress"] == {"completed": 0, "total": 10}


def test_generation_system_api_accepts_author_scope(tmp_path: Path) -> None:
    alice = write_system(tmp_path, "alice-v1", suffix="b", author="alice")
    write_system(tmp_path, "bob-v1", suffix="c", author="bob")
    app = create_app(WebConfig(data_dir=tmp_path, auth_required=False))

    with TestClient(app) as client:
        response = client.get("/api/evaluations/generation/systems?author=alice")

    assert response.status_code == 200
    assert [row["system_id"] for row in response.json()["systems"]] == [alice]
