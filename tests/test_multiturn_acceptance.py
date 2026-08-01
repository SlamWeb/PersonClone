from __future__ import annotations

import pytest

from personaforge.web.conversations import ConversationStore
from personaforge.web.multiturn import parse_turn_plan
from personaforge.web.service import response_token_limit


@pytest.mark.parametrize(
    ("payload", "available", "expected"),
    [
        pytest.param(
            {
                "turn_type": "follow_up",
                "resolved_question": "为什么女明星嫁入豪门后仍会觉得上当？",
                "retrieval_policy": "new",
                "response_depth": "normal",
            },
            {"turn-latest"},
            ("follow_up", "new", False, "normal"),
            id="pronoun-resolution",
        ),
        pytest.param(
            {
                "turn_type": "explain_previous",
                "resolved_question": "继续展开豪门婚姻中的权责冲突",
                "retrieval_policy": "reuse",
                "evidence_source_turn_id": "turn-latest",
                "response_depth": "deep",
            },
            {"turn-latest"},
            ("explain_previous", "reuse", False, "deep"),
            id="continue-expansion",
        ),
        pytest.param(
            {
                "turn_type": "explain_previous",
                "resolved_question": "解释上一回答中选择权的含义",
                "retrieval_policy": "reuse",
                "evidence_source_turn_id": "turn-latest",
                "response_depth": "brief",
            },
            {"turn-latest"},
            ("explain_previous", "reuse", False, "brief"),
            id="explain-previous",
        ),
        pytest.param(
            {
                "turn_type": "follow_up",
                "resolved_question": "婚姻中的经济分工是否公平？",
                "retrieval_policy": "new",
                "response_depth": "normal",
            },
            {"turn-latest"},
            ("follow_up", "new", False, "normal"),
            id="substantive-new-question",
        ),
        pytest.param(
            {
                "turn_type": "new_topic",
                "resolved_question": "如何看待年轻人不买房？",
                "retrieval_policy": "new",
                "response_depth": "normal",
            },
            {"turn-latest"},
            ("new_topic", "new", False, "normal"),
            id="explicit-topic-switch",
        ),
        pytest.param(
            {
                "turn_type": "explain_previous",
                "resolved_question": "回到之前谈过的配得感问题继续展开",
                "retrieval_policy": "reuse",
                "evidence_source_turn_id": "turn-old",
                "response_depth": "normal",
            },
            {"turn-old", "turn-latest"},
            ("explain_previous", "reuse", False, "normal"),
            id="recall-old-topic",
        ),
        pytest.param(
            {
                "turn_type": "follow_up",
                "resolved_question": "结合今天的新事件继续评价",
                "retrieval_policy": "reuse",
                "evidence_source_turn_id": "turn-latest",
                "needs_web": True,
                "search_queries": ["事件名 最新进展"],
                "response_depth": "normal",
            },
            {"turn-latest"},
            ("follow_up", "new", True, "normal"),
            id="web-grounded-follow-up",
        ),
        pytest.param(
            {
                "turn_type": "unclear",
                "resolved_question": "这个问题缺少明确指代",
                "retrieval_policy": "new",
                "needs_web": True,
                "search_queries": ["错误搜索词"],
                "response_depth": "normal",
                "clarification_focus": "需要确认你说的是哪件事",
            },
            {"turn-latest"},
            ("unclear", "none", False, "normal"),
            id="unclear-clarification",
        ),
        pytest.param(
            {
                "turn_type": "casual",
                "resolved_question": "你好",
                "retrieval_policy": "new",
                "response_depth": "deep",
            },
            {"turn-latest"},
            ("casual", "none", False, "brief"),
            id="casual-chat",
        ),
        pytest.param(
            {
                "turn_type": "explain_previous",
                "resolved_question": "把刚才的建议压缩成一句话",
                "retrieval_policy": "none",
                "response_depth": "brief",
            },
            {"turn-latest"},
            ("explain_previous", "none", False, "brief"),
            id="transform-previous-answer",
        ),
        pytest.param(
            {
                "turn_type": "follow_up",
                "resolved_question": "正确做短线的思维是什么？",
                "retrieval_policy": "reuse",
                "evidence_source_turn_id": "turn-latest",
                "response_depth": "normal",
            },
            {"turn-latest"},
            ("follow_up", "new", False, "normal"),
            id="same-topic-new-information-need",
        ),
    ],
)
def test_scripted_turn_routes(payload, available, expected) -> None:
    plan = parse_turn_plan(payload, query="当前消息", available_turn_ids=available)

    assert (
        plan.turn_type,
        plan.retrieval_policy,
        plan.needs_web,
        plan.response_depth,
    ) == expected


def test_scripted_length_control() -> None:
    brief = response_token_limit("brief", [], configured_max=1600)
    normal = response_token_limit("normal", [], configured_max=1600)
    deep = response_token_limit("deep", [], configured_max=1600)

    assert 192 <= brief < normal < deep <= 1600


def test_scripted_refresh_marks_running_turn_interrupted(tmp_path) -> None:
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

    assert store.mark_running_interrupted() == 1
    restored = store.get_turn(turn.id)
    session = store.get_conversation("alice", turn.conversation_id)

    assert restored.status == "interrupted"
    assert session["messages"][1]["status"] == "interrupted"


def test_scripted_retry_is_idempotent_for_user_message(tmp_path) -> None:
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
    store.claim_turn(turn.id)
    store.fail_turn(turn.id, {"message": "失败"})

    retried = store.retry_turn(turn.id)
    messages = store.get_conversation("alice", turn.conversation_id)["messages"]

    assert retried.status == "queued"
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["text"] == "问题"
    assert messages[1]["text"] == ""
