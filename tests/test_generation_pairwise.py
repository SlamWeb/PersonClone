from __future__ import annotations

import json
from pathlib import Path

import pytest

from personaforge.eval.generation_pairwise import (
    PAIRWISE_SCHEMA_VERSION,
    build_handoff,
    build_messages,
    import_handoff,
    profile_from_parent_corpus,
    profile_from_persona_pack,
)


def _write_pack(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "pack_id": "pack-v1",
                "author_id": "alice",
                "display_name": "Alice",
                "sections": {
                    "voice": [
                        {
                            "claim_id": "V1",
                            "claim": "先指出问题预设，再给判断。",
                            "confidence": 0.9,
                            "evidence": [{"doc_id": "answer-1", "excerpt": "先说结论"}],
                        }
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_run(path: Path, run_sha: str, suffix: str) -> None:
    path.mkdir(parents=True)
    rows = [
        {
            "item_id": "test-01",
            "split": "test",
            "status": "completed",
            "query": "问题",
            "gold_answer": "作者回答",
            "answer": f"{suffix}回答",
        }
    ]
    (path / "runs.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "personaforge.eval.run.v0",
                "status": "completed",
                "run_sha256": run_sha,
                "config": {
                    "split": "test",
                    "author": "alice",
                    "run_name": f"run-{suffix}",
                    "method_id": f"method-{suffix}",
                    "display_name": f"方法 {suffix}",
                },
            }
        ),
        encoding="utf-8",
    )


def test_profile_conversion_keeps_evidence_and_source_hash(tmp_path: Path) -> None:
    pack = tmp_path / "pack.json"
    _write_pack(pack)
    profile = profile_from_persona_pack(pack)
    assert profile["schema_version"] == "personaforge.eval.author_evidence_profile.v1"
    assert profile["stats"] == {"claim_count": 1, "evidence_count": 1}
    assert profile["sections"]["voice"][0]["evidence"][0]["doc_id"] == "answer-1"
    assert profile["source"]["sha256"]


def test_pairwise_prompt_contains_question_gold_profile_and_candidates(tmp_path: Path) -> None:
    messages = build_messages(
        question="问题",
        gold_answer="作者回答",
        profile={"profile_id": "p1", "sections": {"voice": []}},
        candidate_a={"label": "A", "answer": "回答 A"},
        candidate_b={"label": "B", "answer": "回答 B"},
    )
    joined = "\n".join(message["content"] for message in messages)
    assert all(value in joined for value in ("问题", "作者回答", "回答 A", "回答 B", "p1"))


def test_corpus_profile_is_evidence_only_and_excludes_eval_rows(tmp_path: Path) -> None:
    parents = tmp_path / "parents.jsonl"
    rows = [
        {"doc_id": "train-1", "kind": "answer", "title": "旧文", "created_at": "2020-01-01", "text": "历史材料。" * 30},
        {"doc_id": "gold-1", "kind": "answer", "title": "测试题", "created_at": "2021-01-01", "text": "测试材料。" * 30},
    ]
    parents.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"item_id": "test-1", "parent_id": "gold-1", "created_at": "2021-01-01"}) + "\n", encoding="utf-8")
    profile = profile_from_parent_corpus(parents, author_id="alice", eval_dataset_path=dataset)
    evidence = profile["sections"]["historical_evidence"][0]["evidence"]
    assert [row["doc_id"] for row in evidence] == ["train-1"]
    assert profile["judge_policy"]["profile_is_llm_free"] is True


def test_handoff_swaps_positions_and_imports_consistent_result(tmp_path: Path) -> None:
    pack = tmp_path / "pack.json"
    _write_pack(pack)
    profile_path = tmp_path / "profile.json"
    profile_from_persona_pack(pack, out_path=profile_path)
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left, "a" * 64, "左")
    _write_run(right, "b" * 64, "右")
    handoff = build_handoff(
        profile_path=profile_path,
        left_run_path=left / "runs.jsonl",
        right_run_path=right / "runs.jsonl",
        out_dir=tmp_path / "handoff",
    )
    requests = [
        json.loads(line)
        for line in (tmp_path / "handoff" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["order"] for row in requests] == ["forward", "swapped"]
    assert handoff["left_run_name"] == "run-左"
    assert handoff["right_method_id"] == "method-右"
    assert handoff["right_display_name"] == "方法 右"
    prompt_text = json.dumps(requests[0]["messages"], ensure_ascii=False)
    assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in prompt_text
    responses = []
    for request in requests:
        responses.append(
            {
                "schema_version": PAIRWISE_SCHEMA_VERSION,
                "task_id": request["task_id"],
                "item_id": request["item_id"],
                "order": request["order"],
                "prompt_hash": handoff["prompt_hash"],
                "winner": "A" if request["order"] == "forward" else "B",
                "confidence": "high",
                "profile_evidence_ids": ["V1:answer-1"],
                "reason": "证据一致。",
            }
        )
    response_path = tmp_path / "responses.jsonl"
    response_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in responses) + "\n",
        encoding="utf-8",
    )
    result = import_handoff(
        manifest_path=tmp_path / "handoff" / "manifest.json",
        response_path=response_path,
    )
    assert result["summary"]["position_consistency"] == 1.0
    assert result["items"][0]["winner_system_id"] == "a" * 64


def test_handoff_rejects_non_test_runs(tmp_path: Path) -> None:
    pack = tmp_path / "pack.json"
    _write_pack(pack)
    profile_path = tmp_path / "profile.json"
    profile_from_persona_pack(pack, out_path=profile_path)
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left, "a" * 64, "左")
    _write_run(right, "b" * 64, "右")
    manifest_path = right / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["config"]["split"] = "dev"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split=test"):
        build_handoff(
            profile_path=profile_path,
            left_run_path=left / "runs.jsonl",
            right_run_path=right / "runs.jsonl",
            out_dir=tmp_path / "handoff",
        )
