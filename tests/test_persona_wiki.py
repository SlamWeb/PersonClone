from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from personaforge.cli import main
from personaforge.ingest.retrieve import ParentHit
from personaforge.ingest.retrieve import RetrieveResult
from personaforge.persona.wiki import (
    build_persona_wiki,
    load_persona_wiki,
)
from personaforge.persona.writer import build_writer_messages
from personaforge.web.service import PersonaChatService, WebConfig
from personaforge.web.multiturn import raw_turn_plan
import personaforge.web.service as web_service


def _write_assets(tmp_path: Path) -> Path:
    author_dir = tmp_path / "authors" / "zhihu" / "writer"
    index_dir = author_dir / "index"
    index_dir.mkdir(parents=True)
    (index_dir / "parents.jsonl").write_text(
        json.dumps(
            {"doc_id": "example:answer:1", "text": "作者原话。先确认问题，再说明边界。"},
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    claim = lambda name: {
        "claim_id": name,
        "claim": f"{name} 的观点",
        "confidence": 0.8,
        "scopes": ["问题"],
        "activation_condition": "遇到相关问题",
        "avoid_overapplication": "不扩展到无关主题",
        "evidence": [{"doc_id": "example:answer:1", "excerpt": "先确认问题"}],
    }
    pack = {
        "schema_version": 1,
        "pack_id": "writer.pack.v1",
        "author_id": "writer",
        "display_name": "示例作者",
        "source": {},
        "corpus_stats": {},
        "sections": {
            "worldview": [claim("W1")],
            "reasoning": [claim("R1")],
            "voice": [claim("V1")],
        },
        "generation_policy": {},
    }
    (author_dir / "persona_pack.json").write_text(
        json.dumps(pack, ensure_ascii=False), encoding="utf-8"
    )
    schema = {
        "schema_version": 1,
        "schema_id": "writer.narrative.v1",
        "author_id": "writer",
        "display_name": "示例作者",
        "source": {},
        "corpus_snapshot": {},
        "identity": {"public_identity": "公开写作者"},
        "global_summary": "先确认问题。",
        "core_traits": ["注意问题边界"],
        "scene_facets": [
            {
                "facet_id": "boundary",
                "title": "说明边界",
                "cue_keys": ["边界"],
                "situation": "信息不足时",
                "thinking_pattern": "确认已知信息",
                "behavior_pattern": "说明边界",
                "expression_signals": ["简洁"],
                "boundary_anchors": ["不编造"],
                "source_evidence": [
                    {
                        "claim_id": "B1",
                        "doc_id": "example:answer:1",
                        "excerpt": "再说明边界",
                    }
                ],
            }
        ],
        "generation_policy": {},
    }
    (author_dir / "narrative_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False), encoding="utf-8"
    )
    return index_dir


def test_wiki_preserves_both_assets_and_verbatim_provenance(tmp_path: Path) -> None:
    index_dir = _write_assets(tmp_path)

    path = build_persona_wiki(index_dir)
    first_bytes = path.read_bytes()
    wiki = load_persona_wiki(index_dir)

    assert path == index_dir.parent / "persona_wiki.json"
    assert wiki["core"]["global_summary"] == "先确认问题。"
    assert set(wiki["source_hashes"]) == {"persona_pack", "narrative_schema"}
    assert {card["node_id"] for card in wiki["cards"]} == {
        "pack:W1", "pack:R1", "pack:V1", "narrative:boundary"
    }
    assert wiki["cards"][0]["evidence"] == [
        {"doc_id": "example:answer:1", "excerpt": "先确认问题"}
    ]
    assert "narrative:boundary" in wiki["cards"][0]["shared_source_links"]
    assert build_persona_wiki(index_dir).read_bytes() == first_bytes


def test_wiki_rejects_changed_raw_source_and_preserves_published_file(tmp_path: Path) -> None:
    index_dir = _write_assets(tmp_path)
    path = build_persona_wiki(index_dir)
    published = path.read_bytes()
    (index_dir / "parents.jsonl").write_text(
        json.dumps({"doc_id": "example:answer:1", "text": "被替换的内容"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-verbatim|altered raw text"):
        build_persona_wiki(index_dir)
    assert path.read_bytes() == published
    with pytest.raises(ValueError, match="altered raw text"):
        load_persona_wiki(index_dir)


def test_wiki_rejects_cross_author_assets(tmp_path: Path) -> None:
    index_dir = _write_assets(tmp_path)
    schema_path = index_dir.parent / "narrative_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["author_id"] = "another-writer"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="different authors"):
        build_persona_wiki(index_dir)


def test_wiki_requires_rebuild_when_source_asset_changes(tmp_path: Path) -> None:
    index_dir = _write_assets(tmp_path)
    build_persona_wiki(index_dir)
    pack_path = index_dir.parent / "persona_pack.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pack["sections"]["worldview"][0]["claim"] = "更新的有来源观点"
    pack_path.write_text(json.dumps(pack, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="source changed"):
        load_persona_wiki(index_dir)
    build_persona_wiki(index_dir)
    assert load_persona_wiki(index_dir)["cards"][0]["title"] == "更新的有来源观点"


def test_persona_wiki_cli_builds_the_author_asset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index_dir = _write_assets(tmp_path)

    assert main(["persona-wiki", "writer", "--data-dir", str(tmp_path)]) == 0

    assert (index_dir.parent / "persona_wiki.json").exists()
    assert "cards: 4" in capsys.readouterr().out


def test_wiki_writer_keeps_full_wiki_and_user_memory_before_history(tmp_path: Path) -> None:
    index_dir = _write_assets(tmp_path)
    build_persona_wiki(index_dir)
    wiki = load_persona_wiki(index_dir)
    hit = ParentHit(
        rank=1, parent_id="example:answer:1", score=1.0, title="作者文章", path="answer/1.md",
        parent={"doc_id": "example:answer:1", "title": "作者文章", "text": "作者原话。先确认问题，再说明边界。"},
    )
    trace: dict = {}

    messages = build_writer_messages(
        query="信息边界是什么？", parent_hits=[hit], writer_prompt="mrprompt",
        persona_wiki=wiki, user_memories=["用户偏好简短回答"],
        conversation_summary={"summary": "更早的会话"},
        conversation_messages=[{"role": "user", "content": "之前的问题"},
                               {"role": "assistant", "content": "之前的回答"}],
        response_depth="normal", budget_trace=trace,
    )

    assert [item["role"] for item in messages] == [
        "system", "system", "system", "user", "assistant", "user", "user"
    ]
    assert "Persona Wiki：作者总纲" in messages[0]["content"]
    assert "Narrative Schema" not in messages[0]["content"]
    assert "W1 的观点" in messages[0]["content"]
    assert "说明边界" in messages[0]["content"]
    assert "用户偏好简短回答" in messages[1]["content"]
    assert "更早的会话" in messages[2]["content"]
    assert "作者原话" in messages[-2]["content"]
    assert "example:answer:1" not in "\n".join(item["content"] for item in messages)
    assert len(trace["final_wiki_card_ids"]) == 4


def test_web_service_builds_and_refreshes_fixed_wiki_once_per_asset_version(tmp_path: Path) -> None:
    index_dir = _write_assets(tmp_path)
    service = PersonaChatService(WebConfig(data_dir=tmp_path))

    first = service._load_persona_wiki(index_dir)
    assert (index_dir.parent / "persona_wiki.json").exists()
    assert service._load_persona_wiki(index_dir) is first

    pack_path = index_dir.parent / "persona_pack.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pack["sections"]["worldview"][0]["claim"] = "更新后的观点"
    pack_path.write_text(json.dumps(pack, ensure_ascii=False), encoding="utf-8")

    refreshed = service._load_persona_wiki(index_dir)
    assert refreshed["cards"][0]["title"] == "更新后的观点"
    assert refreshed is not first


def test_web_chat_uses_full_wiki_when_mrprompt_is_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_assets(tmp_path)
    hit = ParentHit(
        rank=1, parent_id="example:answer:1", score=1.0, title="作者文章", path="answer/1.md",
        parent={"doc_id": "example:answer:1", "title": "作者文章", "text": "作者原话。先确认问题，再说明边界。"},
    )
    result = RetrieveResult(
        query="如何说明边界？", collection_name="example__writer", child_top_k=100,
        parent_top_k=20, routes={}, parents=[hit], retrieval_queries=[],
    )
    monkeypatch.setattr(web_service, "retrieve_parents", lambda *args, **kwargs: result)
    service = PersonaChatService(
        WebConfig(data_dir=tmp_path), encoder=object(), llm=object()
    )

    prepared = service.prepare_chat(
        author="writer", session_id=None, query="如何说明边界？",
        query_mode="raw", writer_prompt="mrprompt",
    )

    assert "Persona Wiki：作者总纲" in prepared.messages[0]["content"]
    assert "W1 的观点" in prepared.messages[0]["content"]
    assert "作者原话" in prepared.messages[-2]["content"]
    assert prepared.messages[-1] == {"role": "user", "content": "如何说明边界？"}
    writer_stage = next(stage for stage in prepared.stages if stage["id"] == "writer_pack")
    assert writer_stage["details"]["persona_wiki_mode"] == "fixed_prefix"


def test_grounded_wiki_chat_uses_all_active_user_memories_without_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_assets(tmp_path)
    hit = ParentHit(
        rank=1, parent_id="example:answer:1", score=1.0, title="作者文章", path="answer/1.md",
        parent={"doc_id": "example:answer:1", "title": "作者文章", "text": "作者原话。先确认问题，再说明边界。"},
    )
    result = RetrieveResult(
        query="如何说明边界？", collection_name="example__writer", child_top_k=100,
        parent_top_k=20, routes={}, parents=[hit], retrieval_queries=[],
    )
    monkeypatch.setattr(web_service, "retrieve_parents_for_queries", lambda *args, **kwargs: result)
    monkeypatch.setattr(web_service, "plan_conversation_turn", lambda query, **kwargs: raw_turn_plan(query))
    monkeypatch.setattr(
        web_service, "build_background_and_retrieval_queries",
        lambda *args, **kwargs: SimpleNamespace(objective_background="", retrieval_queries=[]),
    )
    monkeypatch.setattr(
        web_service, "recall_user_memories",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dynamic memory recall ran")),
    )
    service = PersonaChatService(WebConfig(data_dir=tmp_path), encoder=object(), llm=object())
    monkeypatch.setattr(service.user_memories, "settings", lambda owner_id: {"enabled": True})
    monkeypatch.setattr(service.user_memories, "list_active", lambda owner_id: [
        SimpleNamespace(id="m1", content="用户喜欢短回答"),
        SimpleNamespace(id="m2", content="用户正在写论文"),
    ])
    monkeypatch.setattr(service.user_memories, "mark_accessed", lambda owner_id, ids: None)

    prepared = service.prepare_chat(
        author="writer", session_id=None, query="如何说明边界？",
        query_mode="grounded", writer_prompt="mrprompt",
    )

    assert "用户喜欢短回答" in prepared.messages[1]["content"]
    assert "用户正在写论文" in prepared.messages[1]["content"]
    assert prepared.selected_memory_ids == ["m1", "m2"]
