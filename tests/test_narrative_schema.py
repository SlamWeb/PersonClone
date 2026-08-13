from __future__ import annotations

from pathlib import Path

import pytest

from personaforge.ingest.retrieve import ParentHit
from personaforge.persona.narrative import (
    load_narrative_schema_for_index,
    render_narrative_schema_prompt,
)
from personaforge.persona.writer import WRITER_PROMPT_CHOICES, build_writer_messages


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_INDEX = ROOT / "data" / "authors" / "zhihu" / "wu-ren-jun-28" / "index"


def _parent_hit() -> ParentHit:
    return ParentHit(
        rank=1,
        parent_id="zhihu:answer:1",
        score=0.1,
        title="标题",
        path="answer/answer-1.md",
        parent={"doc_id": "zhihu:answer:1", "title": "标题", "text": "正文。"},
    )


def test_wu_ren_jun_narrative_schema_is_evidence_backed() -> None:
    schema = load_narrative_schema_for_index(AUTHOR_INDEX, required=True)

    assert schema is not None
    assert schema.schema_id == "wu-ren-jun-28.narrative.v1"
    assert schema.facet_count == 6
    assert schema.evidence_count >= 12


def test_narrative_renderer_keeps_audit_evidence_out_of_writer_context() -> None:
    schema = load_narrative_schema_for_index(AUTHOR_INDEX, required=True)
    assert schema is not None

    prompt = render_narrative_schema_prompt(schema)

    assert "Anchoring" in prompt
    assert "Selecting" in prompt
    assert "Bounding" in prompt
    assert "Enacting" in prompt
    assert "zhihu:answer:" not in prompt
    assert "本质上来说，你的老婆不是你的伴侣" not in prompt


def test_mrprompt_isolated_from_existing_writer_variants() -> None:
    schema = load_narrative_schema_for_index(AUTHOR_INDEX, required=True)
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
