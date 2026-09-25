"""Focused source-boundary checks for the new Wiki pilot."""

from __future__ import annotations

import hashlib
import json

from personaforge.persona.wiki_pilot import validate_and_render_pilot


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _fixture(tmp_path):
    index_dir = tmp_path / "index"
    pilot_dir = tmp_path / "pilot"
    index_dir.mkdir()
    pilot_dir.mkdir()
    source = {"doc_id": "source:1", "kind": "answer", "text": "我做事先考虑风险，再考虑回报。"}
    held_out = {"doc_id": "source:2", "kind": "answer", "text": "未来的保留回答。"}
    (index_dir / "parents.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (source, held_out)), encoding="utf-8"
    )
    eval_manifest = tmp_path / "dataset_manifest.json"
    _write_json(eval_manifest, {"excluded_parent_ids": ["source:2"]})
    _write_json(pilot_dir / "sample_manifest.json", [{
        "doc_id": "source:1", "kind": "answer",
        "text_sha256": hashlib.sha256(source["text"].encode("utf-8")).hexdigest(),
    }])
    (pilot_dir / "source_outcomes.jsonl").write_text(
        '{"doc_id":"source:1","status":"observed"}\n', encoding="utf-8"
    )
    (pilot_dir / "observations.jsonl").write_text(json.dumps({
        "observation_id": "o001", "doc_id": "source:1", "excerpt": "先考虑风险",
        "observation": "遇事先评估风险", "type": "stance", "attribution": "self",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_json(pilot_dir / "wiki_draft.json", {
        "subject_name": "示例人物",
        "overview": [{"text": "他做决定时常先评估风险。", "observation_ids": ["o001"]}],
        "topics": [{"topic_id": "decision", "title": "如何做决定", "scope": "风险与回报",
                    "claims": [{"text": "他先考虑风险，再考虑回报。", "condition": "需要做取舍时",
                                "observation_ids": ["o001"]}]}],
        "style": [],
    })
    return index_dir, pilot_dir, eval_manifest


def test_wiki_pilot_checks_sources_and_renders_without_ids(tmp_path):
    index_dir, pilot_dir, eval_manifest = _fixture(tmp_path)
    report = validate_and_render_pilot(index_dir=index_dir, pilot_dir=pilot_dir, eval_manifest=eval_manifest)
    assert report["errors"] == []
    readable = (pilot_dir / "persona_wiki_readable.md").read_text(encoding="utf-8")
    assert "## 人物概览" in readable
    assert "source:1" not in readable


def test_wiki_pilot_renders_optional_topic_hierarchy(tmp_path):
    index_dir, pilot_dir, eval_manifest = _fixture(tmp_path)
    draft_path = pilot_dir / "wiki_draft.json"
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    draft["topics"][0]["section_path"] = ["工作观", "决策"]
    _write_json(draft_path, draft)
    report = validate_and_render_pilot(index_dir=index_dir, pilot_dir=pilot_dir, eval_manifest=eval_manifest)
    assert report["errors"] == []
    readable = (pilot_dir / "persona_wiki_readable.md").read_text(encoding="utf-8")
    assert "## 工作观\n\n### 决策\n\n#### 如何做决定" in readable


def test_wiki_pilot_rejects_holdout_and_fabricated_excerpt(tmp_path):
    index_dir, pilot_dir, eval_manifest = _fixture(tmp_path)
    sample = json.loads((pilot_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    sample.append({"doc_id": "source:2", "kind": "answer", "text_sha256": "irrelevant"})
    _write_json(pilot_dir / "sample_manifest.json", sample)
    observations = json.loads((pilot_dir / "observations.jsonl").read_text(encoding="utf-8"))
    observations["excerpt"] = "原文里不存在的句子"
    _write_json(pilot_dir / "observations.jsonl", observations)
    report = validate_and_render_pilot(index_dir=index_dir, pilot_dir=pilot_dir, eval_manifest=eval_manifest)
    assert any("excluded" in error for error in report["errors"])
    assert any("verbatim" in error for error in report["errors"])
    assert not (pilot_dir / "persona_wiki_readable.md").exists()


def test_wiki_pilot_keeps_uncertain_observations_out_of_readable_wiki(tmp_path):
    index_dir, pilot_dir, eval_manifest = _fixture(tmp_path)
    observation = json.loads((pilot_dir / "observations.jsonl").read_text(encoding="utf-8"))
    observation["type"] = "uncertain"
    _write_json(pilot_dir / "observations.jsonl", observation)
    report = validate_and_render_pilot(index_dir=index_dir, pilot_dir=pilot_dir, eval_manifest=eval_manifest)
    assert any("promotes an uncertain observation" in error for error in report["errors"])
    assert not (pilot_dir / "persona_wiki_readable.md").exists()


def test_wiki_pilot_does_not_promote_uncertain_source_outcome(tmp_path):
    index_dir, pilot_dir, eval_manifest = _fixture(tmp_path)
    _write_json(pilot_dir / "source_outcomes.jsonl", {"doc_id": "source:1", "status": "uncertain"})
    report = validate_and_render_pilot(index_dir=index_dir, pilot_dir=pilot_dir, eval_manifest=eval_manifest)
    assert any("from an uncertain source" in error for error in report["errors"])
    assert not (pilot_dir / "persona_wiki_readable.md").exists()


def test_wiki_pilot_rejects_observation_ids_in_writer_prose(tmp_path):
    index_dir, pilot_dir, eval_manifest = _fixture(tmp_path)
    draft = json.loads((pilot_dir / "wiki_draft.json").read_text(encoding="utf-8"))
    draft["topics"][0]["claims"][0]["condition"] = "参考观察 o001"
    _write_json(pilot_dir / "wiki_draft.json", draft)
    report = validate_and_render_pilot(index_dir=index_dir, pilot_dir=pilot_dir, eval_manifest=eval_manifest)
    assert any("internal source or observation ID" in error for error in report["errors"])
    assert not (pilot_dir / "persona_wiki_readable.md").exists()


def test_wiki_pilot_full_train_requires_frozen_index_and_complete_author_scope(tmp_path):
    index_dir, pilot_dir, eval_manifest = _fixture(tmp_path)
    parents = [json.loads(line) for line in (index_dir / "parents.jsonl").read_text(encoding="utf-8").splitlines()]
    canonical_hash = hashlib.sha256(
        json.dumps(parents, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    excluded_hash = hashlib.sha256(
        json.dumps(["source:2"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _write_json(eval_manifest, {
        "author": "demo", "excluded_parent_ids": ["source:2"],
        "excluded_parent_ids_sha256": excluded_hash,
        "source_parents_sha256": canonical_hash, "counts": {"train_parents": 1},
    })
    sample = json.loads((pilot_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    _write_json(pilot_dir / "sample_manifest.json", {"author": "demo", "documents": sample})
    outcomes_path = pilot_dir / "source_outcomes.jsonl"
    reviewed_outcome = json.loads(outcomes_path.read_text(encoding="utf-8"))
    reviewed_outcome["reviewed"] = True
    _write_json(outcomes_path, reviewed_outcome)
    draft_path = pilot_dir / "wiki_draft.json"
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    observations = [json.loads(line) for line in (pilot_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    draft["source_observations_sha256"] = hashlib.sha256(
        json.dumps(observations, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _write_json(draft_path, draft)
    good = validate_and_render_pilot(
        index_dir=index_dir, pilot_dir=pilot_dir, eval_manifest=eval_manifest, require_full_train=True,
    )
    assert good["errors"] == []
    assert good["full_train_coverage"] is True
    assert good["writer_view_sha256"]

    stale_draft = dict(draft)
    stale_draft["source_observations_sha256"] = "0" * 64
    _write_json(draft_path, stale_draft)
    stale = validate_and_render_pilot(
        index_dir=index_dir, pilot_dir=pilot_dir, eval_manifest=eval_manifest, require_full_train=True,
    )
    assert any("not bound to current reviewed observations" in error for error in stale["errors"])
    _write_json(draft_path, draft)

    reviewed_outcome["reviewed"] = False
    _write_json(outcomes_path, reviewed_outcome)
    unreviewed = validate_and_render_pilot(
        index_dir=index_dir, pilot_dir=pilot_dir, eval_manifest=eval_manifest, require_full_train=True,
    )
    assert any("without completed semantic review" in error for error in unreviewed["errors"])
    reviewed_outcome["reviewed"] = True
    _write_json(outcomes_path, reviewed_outcome)

    _write_json(pilot_dir / "sample_manifest.json", {"author": "other", "documents": sample})
    wrong_author = validate_and_render_pilot(
        index_dir=index_dir, pilot_dir=pilot_dir, eval_manifest=eval_manifest, require_full_train=True,
    )
    assert any("author differs" in error for error in wrong_author["errors"])

    _write_json(pilot_dir / "sample_manifest.json", {"author": "demo", "documents": []})
    incomplete = validate_and_render_pilot(
        index_dir=index_dir, pilot_dir=pilot_dir, eval_manifest=eval_manifest, require_full_train=True,
    )
    assert any("cover every frozen training document" in error for error in incomplete["errors"])
