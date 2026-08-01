import json

import pytest

from personaforge.web.conversations import ConversationBusyError, ConversationStore


def test_store_persists_author_scoped_conversation_and_turn(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    turn = store.create_turn(
        author="alice",
        conversation_id=None,
        query="第一问",
        query_mode="grounded",
        writer_prompt="strong_identity",
        parent_top_k=20,
        trace_capture="summary",
    )

    assert store.claim_turn(turn.id) is True
    store.set_turn_plan(
        turn.id,
        {
            "turn_type": "new_topic",
            "retrieval_policy": "new",
            "response_depth": "normal",
        },
    )
    store.set_turn_evidence(turn.id, ["zhihu:answer:1"], "trace-1")
    completed = store.complete_turn(
        turn.id,
        answer="第一答",
        sources=[{"parent_id": "zhihu:answer:1", "title": "来源"}],
        trace_id="trace-1",
    )

    reloaded = ConversationStore(tmp_path)
    session = reloaded.get_conversation("alice", turn.conversation_id)
    turns = reloaded.get_completed_turns(turn.conversation_id)

    assert completed.status == "completed"
    assert [item["role"] for item in session["messages"]] == ["user", "assistant"]
    assert session["messages"][1]["trace_id"] == "trace-1"
    assert turns[0].query == "第一问"
    assert turns[0].assistant_text == "第一答"
    assert turns[0].parent_ids == ["zhihu:answer:1"]


def test_store_rejects_parallel_turns_in_same_conversation(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    first = store.create_turn(
        author="alice",
        conversation_id=None,
        query="第一问",
        query_mode="grounded",
        writer_prompt="strong_identity",
        parent_top_k=20,
        trace_capture="summary",
    )

    with pytest.raises(ConversationBusyError):
        store.create_turn(
            author="alice",
            conversation_id=first.conversation_id,
            query="第二问",
            query_mode="grounded",
            writer_prompt="strong_identity",
            parent_top_k=20,
            trace_capture="summary",
        )

    second = store.create_turn(
        author="bob",
        conversation_id=None,
        query="另一会话",
        query_mode="grounded",
        writer_prompt="strong_identity",
        parent_top_k=20,
        trace_capture="summary",
    )
    assert second.conversation_id != first.conversation_id


def test_store_marks_running_turn_interrupted_on_restart(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    turn = store.create_turn(
        author="alice",
        conversation_id=None,
        query="问题",
        query_mode="grounded",
        writer_prompt="strong_identity",
        parent_top_k=20,
        trace_capture="summary",
    )
    assert store.claim_turn(turn.id) is True

    reloaded = ConversationStore(tmp_path)

    assert reloaded.get_turn(turn.id).status == "interrupted"
    assert reloaded.get_conversation("alice", turn.conversation_id)["messages"][1]["status"] == "interrupted"


def test_store_migrates_legacy_json_idempotently(tmp_path) -> None:
    path = tmp_path / "authors" / "zhihu" / "alice" / "sessions" / "legacy.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "id": "legacy",
                "author": "alice",
                "title": "旧对话",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:01:00+00:00",
                "messages": [
                    {"role": "user", "text": "旧问题"},
                    {
                        "role": "assistant",
                        "text": "旧回答",
                        "sources": [{"parent_id": "zhihu:answer:1"}],
                        "trace_id": "trace-old",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    first = ConversationStore(tmp_path)
    second = ConversationStore(tmp_path)

    assert first.get_conversation("alice", "legacy")["title"] == "旧对话"
    assert second.list_conversations("alice")[0]["message_count"] == 2
    assert second.get_completed_turns("legacy")[0].parent_ids == ["zhihu:answer:1"]
    assert path.exists()


def test_store_round_trips_turn_embedding_and_cascades_delete(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    turn = store.save_completed_turn(
        conversation_id="s1",
        author="alice",
        query="问题",
        answer="回答",
        sources=[],
        trace_id="trace-1",
    )
    store.save_turn_embedding(turn.id, [0.25, -0.5, 0.75], model="bge", version="v1")

    assert store.get_turn_embedding(turn.id, model="bge", version="v1") == pytest.approx(
        [0.25, -0.5, 0.75]
    )
    assert store.get_turn_embedding(turn.id, model="bge", version="v2") is None

    assert store.delete_conversation("alice", "s1") == ["trace-1"]
    assert store.list_conversations("alice") == []
