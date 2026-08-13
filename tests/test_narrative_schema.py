from __future__ import annotations

import json
from pathlib import Path

import pytest

from personaforge.ingest.retrieve import ParentHit
from personaforge.persona.narrative import (
    load_narrative_schema_for_index,
    render_narrative_schema_prompt,
)
from personaforge.persona.writer import WRITER_PROMPT_CHOICES, build_writer_messages


def _parent_hit() -> ParentHit:
    return ParentHit(
        rank=1,
        parent_id="example:answer:1",
        score=0.1,
        title="标题",
        path="answer/answer-1.md",
        parent={"doc_id": "example:answer:1", "title": "标题", "text": "正文。"},
    )


def _write_narrative_fixture(tmp_path: Path) -> Path:
    author_dir = tmp_path / "authors" / "example-author"
    index_dir = author_dir / "index"
    index_dir.mkdir(parents=True)
    parent_rows = [
        {
            "doc_id": "example:answer:1",
            "title": "示例一",
            "text": "遇到复杂问题时，先指出题目里的隐藏前提。",
        },
        {
            "doc_id": "example:answer:2",
            "title": "示例二",
            "text": "信息不足时，明确承认边界，不假装知道答案。",
        },
    ]
    (index_dir / "parents.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in parent_rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "schema_id": "example-author.narrative.v1",
        "author_id": "example-author",
        "display_name": "示例作者",
        "source": {"platform": "fixture"},
        "corpus_snapshot": {"parent_count": 2},
        "identity": {"public_identity": "一位公开写作者"},
        "global_summary": "重视问题前提，也明确知识边界。",
        "core_traits": ["先检查前提", "不越界编造"],
        "scene_facets": [
            {
                "facet_id": "premise",
                "title": "检查问题前提",
                "cue_keys": ["判断", "原因"],
                "situation": "问题包含未经验证的前提时",
                "thinking_pattern": "先判断前提是否成立",
                "behavior_pattern": "指出前提，再展开自己的判断",
                "expression_signals": ["直接切入关键前提"],
                "boundary_anchors": ["不把单个例子推广为普遍规律"],
                "source_evidence": [
                    {
                        "claim_id": "premise-1",
                        "doc_id": "example:answer:1",
                        "excerpt": "先指出题目里的隐藏前提",
                    }
                ],
            },
            {
                "facet_id": "boundary",
                "title": "承认信息边界",
                "cue_keys": ["未知", "预测"],
                "situation": "公开材料不足以支持结论时",
                "thinking_pattern": "区分已知事实和推测",
                "behavior_pattern": "说明不知道的部分",
                "expression_signals": ["简洁说明证据边界"],
                "boundary_anchors": ["不虚构经历或事实"],
                "source_evidence": [
                    {
                        "claim_id": "boundary-1",
                        "doc_id": "example:answer:2",
                        "excerpt": "明确承认边界",
                    }
                ],
            },
        ],
        "generation_policy": {"select_facets": "only_relevant"},
    }
    (author_dir / "narrative_schema.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index_dir


def test_narrative_schema_fixture_is_evidence_backed(tmp_path: Path) -> None:
    schema = load_narrative_schema_for_index(
        _write_narrative_fixture(tmp_path), required=True
    )

    assert schema is not None
    assert schema.schema_id == "example-author.narrative.v1"
    assert schema.facet_count == 2
    assert schema.evidence_count == 2


def test_narrative_renderer_keeps_audit_evidence_out_of_writer_context(
    tmp_path: Path,
) -> None:
    schema = load_narrative_schema_for_index(
        _write_narrative_fixture(tmp_path), required=True
    )
    assert schema is not None

    prompt = render_narrative_schema_prompt(schema)

    assert "Anchoring" in prompt
    assert "Selecting" in prompt
    assert "Bounding" in prompt
    assert "Enacting" in prompt
    assert "example:answer:" not in prompt
    assert "先指出题目里的隐藏前提" not in prompt


def test_mrprompt_isolated_from_existing_writer_variants(tmp_path: Path) -> None:
    schema = load_narrative_schema_for_index(
        _write_narrative_fixture(tmp_path), required=True
    )
    assert schema is not None

    messages = build_writer_messages(
        query="问题",
        parent_hits=[_parent_hit()],
        writer_prompt="mrprompt",
        narrative_schema=schema,
    )
    system_prompt = messages[0]["content"]

    assert "Narrative Schema" in system_prompt
    assert "Magic-If 执行协议" in system_prompt
    assert "长期叙事记忆" in messages[0]["content"]
    assert WRITER_PROMPT_CHOICES[:3] == ("current", "strong_identity", "persona_pack")
    assert "mrprompt" in WRITER_PROMPT_CHOICES


def test_mrprompt_requires_schema() -> None:
    with pytest.raises(ValueError, match="Narrative Schema"):
        build_writer_messages(query="问题", parent_hits=[], writer_prompt="mrprompt")
