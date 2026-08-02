from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from personaforge.ingest.embeddings import SparseEmbedding, TextEmbedding
from personaforge.web.conversations import ConversationTurn
from personaforge.web.user_memory import (
    UserMemoryStore,
    recall_user_memories,
    redact_sensitive_text,
    update_user_memories,
)


class FakeEncoder:
    def encode_texts(self, texts: list[str], *, batch_size: int = 12) -> list[TextEmbedding]:
        return [
            TextEmbedding(
                dense=[float("哥" in text), float("写作" in text), 1.0],
                sparse=SparseEmbedding(
                    indices=[1] if "哥" in text else [2],
                    values=[1.0],
                ),
            )
            for text in texts
        ]


class FakeLlm:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls = []

    def complete_json(self, messages, *, temperature, max_tokens):
        self.calls.append(messages)
        return self.payloads.pop(0)


def make_turn(
    message_id: str,
    query: str,
    sequence: int = 1,
    *,
    assistant_text: str = "irrelevant generated answer",
) -> ConversationTurn:
    return ConversationTurn(
        id=f"turn-{sequence}",
        conversation_id="conversation-1",
        author="author-1",
        user_message_id=message_id,
        assistant_message_id=f"assistant-{sequence}",
        query=query,
        assistant_text=assistant_text,
        assistant_status="completed",
        trace_id=None,
        parent_ids=[],
        sequence=sequence,
    )


def test_store_revision_and_forget(tmp_path: Path) -> None:
    store = UserMemoryStore(tmp_path)
    first = store.save_revision(
        "owner-1",
        kind="semantic",
        memory_key="user.preference.explanation",
        content="用户喜欢先讲结论。",
        sensitivity="normal",
        importance=3,
        confidence=0.95,
        event_status="stable",
        source_author="author-1",
        source_conversation_id="conversation-1",
        source_message_ids=["message-1"],
        evidence_quotes=["先讲结论"],
    )
    corrected = store.correct("owner-1", first.id, "用户喜欢先说明背景再讲结论。")

    assert corrected.supersedes_id == first.id
    assert store.get("owner-1", first.id).status == "superseded"
    store.set_pinned("owner-1", corrected.id, True)
    assert store.get("owner-1", corrected.id).pinned is True
    store.forget("owner-1", corrected.id)
    assert store.list_active("owner-1") == []


def test_automatic_revision_preserves_non_conflicting_evidence(tmp_path: Path) -> None:
    store = UserMemoryStore(tmp_path)
    first = store.save_revision(
        "owner-1", kind="episodic", memory_key="family.brother.finance",
        content="用户担心哥哥再次借贷。", sensitivity="restricted", importance=5,
        confidence=0.9, event_status="ongoing", source_author="a",
        source_conversation_id="c", source_message_ids=["m1"], evidence_quotes=[],
    )
    revised = store.save_revision(
        "owner-1", kind="episodic", memory_key="family.brother.finance",
        content="哥哥已注销自己的账户，但用户担心哥哥再次借贷。", sensitivity="restricted",
        importance=5, confidence=0.95, event_status="ongoing", source_author="a",
        source_conversation_id="c", source_message_ids=["m2"], evidence_quotes=[],
        supersedes_id=first.id,
    )

    assert revised.content == "哥哥已注销自己的账户，但用户担心哥哥再次借贷。"
    assert revised.source_message_ids == ["m1", "m2"]


def test_hybrid_recall_keeps_user_boundaries(tmp_path: Path) -> None:
    store = UserMemoryStore(tmp_path)
    brother = store.save_revision(
        "owner-1",
        kind="episodic",
        memory_key="family.brother.trading",
        content="用户担心哥哥再次借贷进行高风险交易。",
        sensitivity="restricted",
        importance=5,
        confidence=0.96,
        event_status="ongoing",
        source_author="author-1",
        source_conversation_id="conversation-1",
        source_message_ids=["message-1"],
        evidence_quotes=[],
    )
    store.save_revision(
        "owner-1",
        kind="procedural",
        memory_key="user.preference.writing",
        content="用户希望回答简短。",
        sensitivity="normal",
        importance=2,
        confidence=0.95,
        event_status="stable",
        source_author="author-1",
        source_conversation_id="conversation-2",
        source_message_ids=["message-2"],
        evidence_quotes=["回答简短"],
    )

    hits = recall_user_memories(
        store,
        "owner-1",
        "我哥又想充值怎么办",
        encoder=FakeEncoder(),
        model="fake",
    )

    assert hits[0].memory.id == brother.id
    assert hits[0].dense_rank == 1
    assert hits[0].sparse_rank == 1


def test_two_stage_write_rejects_question_promoted_to_belief(tmp_path: Path) -> None:
    store = UserMemoryStore(tmp_path)
    turn = make_turn("message-1", "是不是只要玩短线，频繁操作，最终一定亏光呢")
    candidate = {
        "decision": "approve",
        "reason": "",
        "kind": "semantic",
        "memory_key": "user.belief.trading",
        "content": "用户认为短线交易最终一定会亏光。",
        "sensitivity": "private",
        "importance": 3,
        "confidence": 0.96,
        "event_status": "stable",
        "source_message_ids": ["message-1"],
        "evidence_quotes": ["是不是只要玩短线，频繁操作，最终一定亏光呢"],
        "supersedes_id": None,
        "pinned": False,
    }
    llm = FakeLlm([{"candidates": [candidate]}, {"memories": [candidate]}])

    result = update_user_memories(
        store,
        "owner-1",
        author="author-1",
        conversation_id="conversation-1",
        user_turns=[turn],
        llm=llm,
        force_flush=True,
    )

    assert store.list_active("owner-1") == []
    assert result["rejections"] == [{"reason": "question_promoted_to_belief"}]


def test_two_stage_write_generalizes_third_party_finance(tmp_path: Path) -> None:
    store = UserMemoryStore(tmp_path)
    query = "我哥十倍杠杆贷款，两天亏掉20w，我担心他又贷款杠杆。"
    turn = make_turn("message-1", query)
    approved = {
        "decision": "approve",
        "reason": "",
        "kind": "episodic",
        "memory_key": "family.brother.high_risk_trading",
        "content": "用户担心哥哥在高风险交易亏损后再次借贷加杠杆。",
        "sensitivity": "restricted",
        "importance": 5,
        "confidence": 0.97,
        "event_status": "ongoing",
        "source_message_ids": ["message-1"],
        "evidence_quotes": [],
        "supersedes_id": None,
        "pinned": False,
    }
    llm = FakeLlm([{"candidates": [approved]}, {"memories": [approved]}])

    result = update_user_memories(
        store,
        "owner-1",
        author="author-1",
        conversation_id="conversation-1",
        user_turns=[turn],
        llm=llm,
        force_flush=True,
    )

    memories = store.list_active("owner-1")
    assert len(memories) == 1
    assert memories[0].sensitivity == "restricted"
    assert memories[0].evidence_quotes == []
    assert "20" not in memories[0].content
    assert result["operations"][0]["operation"] == "create"
    assert "20w" not in redact_sensitive_text(query)


def test_memory_window_defers_until_three_turns_and_advances_checkpoint(tmp_path: Path) -> None:
    store = UserMemoryStore(tmp_path)
    turns = [make_turn(f"message-{index}", f"普通问题{index}", index) for index in range(1, 4)]
    llm = FakeLlm([{"candidates": []}])

    first = update_user_memories(
        store, "owner-1", author="author-1", conversation_id="conversation-1",
        user_turns=turns[:1], llm=llm,
    )
    second = update_user_memories(
        store, "owner-1", author="author-1", conversation_id="conversation-1",
        user_turns=turns[:2], llm=llm,
    )
    third = update_user_memories(
        store, "owner-1", author="author-1", conversation_id="conversation-1",
        user_turns=turns, llm=llm,
    )

    assert first["status"] == second["status"] == "deferred"
    assert len(llm.calls) == 1
    assert third["status"] == "completed"
    assert third["critic_skipped"] is True
    assert store.window_checkpoint("owner-1", "conversation-1") == 3
    repeated = update_user_memories(
        store, "owner-1", author="author-1", conversation_id="conversation-1",
        user_turns=turns, llm=llm,
    )
    assert repeated["reason"] == "no_pending_turns"
    assert len(llm.calls) == 1


def test_assistant_messages_are_context_but_never_eligible_evidence(tmp_path: Path) -> None:
    store = UserMemoryStore(tmp_path)
    llm = FakeLlm([{"candidates": []}])
    turn = make_turn(
        "message-1",
        "对，他又开始了。",
        assistant_text="你是说你哥哥又开始高风险交易了吗？",
    )

    update_user_memories(
        store, "owner-1", author="author-1", conversation_id="conversation-1",
        user_turns=[turn], llm=llm, force_flush=True,
    )

    prompt = llm.calls[0][1]["content"]
    assert "你哥哥又开始高风险交易" in prompt
    assert '"eligible_evidence_message_ids": [\n    "message-1"' in prompt
    assert "assistant-1" not in prompt.split('"eligible_evidence_message_ids"', 1)[1].split("]", 1)[0]


def test_newer_conflicting_memory_replaces_older_content(tmp_path: Path) -> None:
    store = UserMemoryStore(tmp_path)
    old = store.save_revision(
        "owner-1", kind="semantic", memory_key="user.preference.answer_length",
        content="用户喜欢详细回答。", sensitivity="normal", importance=3,
        confidence=0.95, event_status="stable", source_author="author-1",
        source_conversation_id="conversation-0", source_message_ids=["old-message"],
        evidence_quotes=["详细回答"],
    )
    replacement = {
        "decision": "approve", "reason": "用户更新了偏好", "kind": "semantic",
        "memory_key": "user.preference.answer_length", "content": "用户现在喜欢简短回答。",
        "sensitivity": "normal", "importance": 3, "confidence": 0.98,
        "event_status": "stable", "source_message_ids": ["message-1"],
        "evidence_quotes": ["现在喜欢简短回答"], "supersedes_id": old.id,
        "revision_mode": "replace", "pinned": False,
    }
    llm = FakeLlm([{"candidates": [replacement]}, {"memories": [replacement]}])

    result = update_user_memories(
        store, "owner-1", author="author-1", conversation_id="conversation-1",
        user_turns=[make_turn("message-1", "我现在喜欢简短回答。")],
        llm=llm, force_flush=True, related_memories=[old],
    )

    active = store.list_active("owner-1")
    assert len(active) == 1
    assert active[0].content == "用户现在喜欢简短回答。"
    assert active[0].supersedes_id == old.id
    assert result["operations"][0]["revision_mode"] == "replace"


def test_explicit_forget_soft_deletes_selected_memory_without_llm(tmp_path: Path) -> None:
    store = UserMemoryStore(tmp_path)
    memory = store.save_revision(
        "owner-1",
        kind="episodic",
        memory_key="family.brother.trading",
        content="用户担心哥哥再次进行高风险交易。",
        sensitivity="restricted",
        importance=5,
        confidence=0.97,
        event_status="ongoing",
        source_author="author-1",
        source_conversation_id="conversation-1",
        source_message_ids=["message-1"],
        evidence_quotes=[],
    )

    result = update_user_memories(
        store,
        "owner-1",
        author="author-1",
        conversation_id="conversation-2",
        user_turns=[make_turn("message-2", "忘掉这件事")],
        llm=FakeLlm([]),
        related_memories=[memory],
    )

    assert result["operations"][0]["operation"] == "forget"
    assert store.list_active("owner-1") == []
