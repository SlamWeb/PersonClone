from __future__ import annotations

import json
from pathlib import Path

import pytest

from personaforge.eval.runner import assert_persona_pack_no_leak
from personaforge.ingest.retrieve import ParentHit
from personaforge.persona.pack import load_persona_pack, render_persona_pack_prompt
from personaforge.persona.writer import build_writer_messages


def _payload(excerpt: str = "作者原话") -> dict:
    def claim(claim_id: str) -> dict:
        return {
            "claim_id": claim_id,
            "claim": f"{claim_id} 的稳定倾向",
            "confidence": 0.9,
            "scopes": ["测试"],
            "activation_condition": "当前问题相关时",
            "avoid_overapplication": "不相关时不要使用",
            "evidence": [{"doc_id": "zhihu:answer:1", "excerpt": excerpt}],
        }

    return {
        "schema_version": 1,
        "pack_id": "author.temporal-train.v1",
        "author_id": "author",
        "display_name": "测试作者",
        "source": {"training_document_count": 1, "holdout_used": False},
        "corpus_stats": {"document_count": 1},
        "sections": {
            "response_strategy": [claim("S1")],
            "worldview": [claim("W1")],
            "reasoning": [claim("R1")],
            "voice": [claim("V1")],
        },
        "generation_policy": {
            "selection_rule": "只激活当前问题相关的少量倾向。",
            "forbidden_overfit": ["不要堆口癖。"],
        },
        "research_basis": [],
    }


def _write_fixture(tmp_path: Path, *, excerpt: str = "作者原话") -> tuple[Path, Path]:
    pack_path = tmp_path / "persona_pack.json"
    parent_path = tmp_path / "parents.jsonl"
    pack_path.write_text(
        json.dumps(_payload(excerpt), ensure_ascii=False),
        encoding="utf-8",
    )
    parent_path.write_text(
        json.dumps(
            {
                "doc_id": "zhihu:answer:1",
                "title": "标题",
                "text": "前文。作者原话。后文。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return pack_path, parent_path


def _parent_hit() -> ParentHit:
    return ParentHit(
        rank=1,
        parent_id="zhihu:answer:1",
        score=0.1,
        title="标题",
        path="answer/answer-1.md",
        parent={"doc_id": "zhihu:answer:1", "title": "标题", "text": "作者原话。"},
    )


def test_persona_pack_verifies_exact_evidence_and_renders_boundaries(tmp_path: Path) -> None:
    pack_path, parent_path = _write_fixture(tmp_path)

    pack = load_persona_pack(pack_path, parent_store_path=parent_path)
    rendered = render_persona_pack_prompt(pack)

    assert pack.claim_count == 4
    assert pack.evidence_count == 4
    assert "概率性倾向" in rendered
    assert "回应边界" in rendered
    assert "只服从当前 RAG 中最相似的作者原文" in rendered
    assert "S1 的稳定倾向" in rendered
    assert "作者原文证据：作者原话" in rendered
    assert "不相关时不要使用" in rendered
    assert "不要堆口癖" in rendered
    assert "直接作答、否定题面" not in rendered


def test_persona_pack_rejects_non_verbatim_evidence(tmp_path: Path) -> None:
    pack_path, parent_path = _write_fixture(tmp_path, excerpt="被改写的证据")

    with pytest.raises(ValueError, match="non-verbatim excerpt"):
        load_persona_pack(pack_path, parent_store_path=parent_path)


def test_persona_pack_keeps_backward_compatibility_without_response_strategy(
    tmp_path: Path,
) -> None:
    payload = _payload()
    del payload["sections"]["response_strategy"]
    pack_path, parent_path = _write_fixture(tmp_path)
    pack_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    pack = load_persona_pack(pack_path, parent_store_path=parent_path)

    assert pack.response_strategy == ()
    assert "回应边界" not in render_persona_pack_prompt(pack)


def test_persona_pack_writer_variant_requires_pack() -> None:
    with pytest.raises(ValueError, match="requires a validated Persona Pack"):
        build_writer_messages(
            query="问题",
            parent_hits=[_parent_hit()],
            writer_prompt="persona_pack",
        )


def test_persona_pack_writer_variant_adds_pack_without_changing_rag_context(
    tmp_path: Path,
) -> None:
    pack_path, parent_path = _write_fixture(tmp_path)
    pack = load_persona_pack(pack_path, parent_store_path=parent_path)

    messages = build_writer_messages(
        query="问题",
        parent_hits=[_parent_hit()],
        writer_prompt="persona_pack",
        persona_pack=pack,
    )
    combined = "\n".join(message["content"] for message in messages)

    assert "证据化 Persona Pack" in combined
    assert "W1 的稳定倾向" in combined
    assert "创作者过往公开表达" in combined
    assert "作者原话。" in combined


def test_eval_rejects_persona_pack_evidence_from_holdout(tmp_path: Path) -> None:
    pack_path, parent_path = _write_fixture(tmp_path)
    pack = load_persona_pack(pack_path, parent_store_path=parent_path)

    with pytest.raises(RuntimeError, match="Persona Pack cites excluded parent"):
        assert_persona_pack_no_leak(pack, {"zhihu:answer:1"})
