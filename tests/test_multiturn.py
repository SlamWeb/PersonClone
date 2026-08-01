from personaforge.ingest.embeddings import SparseEmbedding, TextEmbedding
from personaforge.web.conversations import ConversationStore
from personaforge.web.multiturn import (
    normalize_summary,
    parse_turn_plan,
    plan_conversation_turn,
    select_conversation_context,
    should_update_summary,
    update_conversation_summary,
)
from personaforge.web.user_memory import UserMemoryStore


class FakeJsonClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def complete_json(self, _messages, **_kwargs):
        return self.payloads.pop(0)


class FakeEncoder:
    def __init__(self, vectors):
        self.vectors = list(vectors)
        self.calls = []

    def encode_texts(self, texts, *, batch_size=12):
        self.calls.append((list(texts), batch_size))
        dense = self.vectors.pop(0)
        return [
            TextEmbedding(dense=list(vector), sparse=SparseEmbedding(indices=[], values=[]))
            for vector in dense
        ]


def complete_turn(store, conversation_id, query, answer):
    return store.save_completed_turn(
        conversation_id=conversation_id,
        author="alice",
        query=query,
        answer=answer,
        sources=[],
        trace_id=None,
    )


def test_turn_planner_resolves_follow_up_and_reuses_valid_turn(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    previous = complete_turn(store, "s1", "为什么女明星嫁豪门会后悔？", "因为预期和现实不同。")
    llm = FakeJsonClient(
        [
            {
                "turn_type": "explain_previous",
                "resolved_question": "解释女明星嫁豪门后预期与现实不同的含义",
                "retrieval_policy": "reuse",
                "evidence_source_turn_id": previous.id,
                "needs_web": False,
                "search_queries": [],
                "response_depth": "brief",
                "clarification_focus": "",
            }
        ]
    )

    plan = plan_conversation_turn(
        "你刚才那句是什么意思？",
        summary={},
        recent_turns=store.get_completed_turns("s1"),
        llm=llm,
    )

    assert plan.retrieval_policy == "reuse"
    assert plan.evidence_source_turn_id == previous.id
    assert "预期与现实" in plan.resolved_question


def test_turn_planner_can_select_only_available_memory_ids(tmp_path) -> None:
    memory = UserMemoryStore(tmp_path).save_revision(
        "owner-1",
        kind="episodic",
        memory_key="family.brother.trading",
        content="用户担心哥哥再次进行高风险交易。",
        sensitivity="restricted",
        importance=5,
        confidence=0.96,
        event_status="ongoing",
        source_author="alice",
        source_conversation_id="s1",
        source_message_ids=["m1"],
        evidence_quotes=[],
    )
    llm = FakeJsonClient([{
        "turn_type": "new_topic",
        "resolved_question": "哥哥再次想充值进行高风险交易时怎么办",
        "retrieval_policy": "new",
        "needs_web": False,
        "memory_ids": [memory.id, "missing"],
    }])

    plan = plan_conversation_turn(
        "我哥又想充值怎么办",
        summary={},
        recent_turns=[],
        memory_candidates=[memory],
        llm=llm,
    )

    assert plan.memory_ids == [memory.id]


def test_turn_plan_enforces_web_and_invalid_reuse_invariants() -> None:
    web = parse_turn_plan(
        {
            "turn_type": "follow_up",
            "resolved_question": "今天发生了什么",
            "retrieval_policy": "reuse",
            "evidence_source_turn_id": "turn-1",
            "needs_web": True,
            "search_queries": ["事件 今天"],
            "response_depth": "normal",
        },
        query="今天呢",
        available_turn_ids={"turn-1"},
    )
    invalid_reuse = parse_turn_plan(
        {
            "turn_type": "explain_previous",
            "resolved_question": "解释上一点",
            "retrieval_policy": "reuse",
            "evidence_source_turn_id": "missing",
            "needs_web": False,
        },
        query="解释一下",
        available_turn_ids={"turn-1"},
    )

    assert web.retrieval_policy == "new"
    assert web.evidence_source_turn_id is None
    assert web.search_queries == ["事件 今天"]
    assert invalid_reuse.retrieval_policy == "new"


def test_follow_up_with_new_information_need_cannot_reuse_previous_evidence() -> None:
    plan = parse_turn_plan(
        {
            "turn_type": "follow_up",
            "resolved_question": "当代年轻人是否有必要结婚？",
            "retrieval_policy": "reuse",
            "evidence_source_turn_id": "turn-1",
            "needs_web": False,
            "response_depth": "normal",
        },
        query="那当代年轻人有必要结婚吗",
        available_turn_ids={"turn-1"},
    )

    assert plan.retrieval_policy == "new"
    assert plan.evidence_source_turn_id is None


def test_previous_answer_transform_skips_retrieval() -> None:
    plan = parse_turn_plan(
        {
            "turn_type": "explain_previous",
            "resolved_question": "把刚才的建议压缩成一句话",
            "retrieval_policy": "none",
            "evidence_source_turn_id": "turn-1",
            "needs_web": False,
            "response_depth": "brief",
        },
        query="压缩成一句话",
        available_turn_ids={"turn-1"},
    )

    assert plan.retrieval_policy == "none"
    assert plan.evidence_source_turn_id is None


def test_short_history_returns_every_completed_turn(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    for index in range(4):
        complete_turn(store, "s1", f"问题{index}", f"回答{index}")

    context = select_conversation_context(
        store,
        "s1",
        resolved_question="继续",
        turn_type="follow_up",
        encoder=None,
        embedding_model="bge",
    )

    assert context.used_full_short_history is True
    assert len(context.selected_turns) == 4
    assert [turn.query for turn in context.selected_turns] == [f"问题{index}" for index in range(4)]


def test_long_history_uses_recent_three_and_semantic_top_two(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    turns = [complete_turn(store, "s1", f"问题{index}", f"回答{index}") for index in range(8)]
    # First call encodes the query. Second call encodes the five older turns.
    encoder = FakeEncoder(
        [
            [[1.0, 0.0]],
            [
                [0.1, 0.9],
                [0.9, 0.1],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.0, 1.0],
            ],
        ]
    )

    context = select_conversation_context(
        store,
        "s1",
        resolved_question="找相关旧话题",
        turn_type="follow_up",
        encoder=encoder,
        embedding_model="bge",
    )

    assert [turn.id for turn in context.recent_turns] == [turn.id for turn in turns[-3:]]
    assert [turn.id for turn in context.relevant_turns] == [turns[1].id, turns[3].id]
    assert len(context.selected_turns) == 5

    cached_encoder = FakeEncoder([[[1.0, 0.0]]])
    select_conversation_context(
        store,
        "s1",
        resolved_question="再次查找",
        turn_type="follow_up",
        encoder=cached_encoder,
        embedding_model="bge",
    )
    assert len(cached_encoder.calls) == 1


def test_new_topic_does_not_feed_old_dialogue_to_writer(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    complete_turn(store, "s1", "旧话题", "旧回答")

    context = select_conversation_context(
        store,
        "s1",
        resolved_question="新话题",
        turn_type="new_topic",
        encoder=None,
        embedding_model="bge",
    )

    assert context.selected_turns == []
    assert context.summary == {}


def test_summary_updates_after_more_than_six_turns(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    for index in range(7):
        complete_turn(store, "s1", f"问题{index}", f"回答{index}")
    turns = store.get_completed_turns("s1")
    state = store.get_summary("s1")
    llm = FakeJsonClient(
        [
            {
                "topics": ["婚恋"],
                "entities": [],
                "user_requests": ["用户要求展开"],
                "assistant_previous_claims": ["此前 assistant 曾表示关系存在预期差异"],
                "unresolved_references": [],
            }
        ]
    )

    assert should_update_summary(turns, state) is True
    updated = update_conversation_summary(store, "s1", llm=llm)

    assert updated is not None
    assert updated["version"] == 1
    assert updated["summary"]["topics"] == ["婚恋"]
    assert should_update_summary(turns, updated) is False


def test_normalize_summary_drops_unknown_fields_and_non_lists() -> None:
    summary = normalize_summary(
        {
            "topics": ["A"],
            "entities": "invalid",
            "user_requests": [],
            "assistant_previous_claims": [],
            "unresolved_references": [],
            "extra": ["ignored"],
        }
    )

    assert summary["topics"] == ["A"]
    assert summary["entities"] == []
    assert "extra" not in summary
