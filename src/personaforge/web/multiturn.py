"""Conversation-aware planning, history selection, and summary updates."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from personaforge.ingest.embeddings import TextEncoder
from personaforge.llm import JsonChatClient
from personaforge.web.conversations import ConversationStore, ConversationTurn
from personaforge.web.trace import estimated_usage_for_text
from personaforge.web.user_memory import UserMemory

TurnType = Literal["new_topic", "follow_up", "explain_previous", "casual", "unclear"]
RetrievalPolicy = Literal["new", "reuse", "none"]
ResponseDepth = Literal["brief", "normal", "deep"]
MemoryWritePolicy = Literal["defer", "flush"]

TURN_TYPES = {"new_topic", "follow_up", "explain_previous", "casual", "unclear"}
RETRIEVAL_POLICIES = {"new", "reuse", "none"}
RESPONSE_DEPTHS = {"brief", "normal", "deep"}
MEMORY_WRITE_POLICIES = {"defer", "flush"}
SUMMARY_FIELDS = (
    "topics",
    "entities",
    "user_requests",
    "assistant_previous_claims",
    "unresolved_references",
)


@dataclass(frozen=True, slots=True)
class TurnPlan:
    turn_type: TurnType
    resolved_question: str
    retrieval_policy: RetrievalPolicy
    evidence_source_turn_id: str | None
    needs_web: bool
    search_queries: list[str] = field(default_factory=list)
    response_depth: ResponseDepth = "normal"
    clarification_focus: str = ""
    memory_ids: list[str] = field(default_factory=list)
    memory_write_policy: MemoryWritePolicy = "defer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HistoryMatch:
    turn_id: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"turn_id": self.turn_id, "score": round(self.score, 6)}


@dataclass(frozen=True, slots=True)
class SelectedConversationContext:
    summary: dict[str, Any]
    summary_version: int
    summary_through_sequence: int
    recent_turns: list[ConversationTurn]
    relevant_turns: list[ConversationTurn]
    selected_turns: list[ConversationTurn]
    history_matches: list[HistoryMatch]
    used_full_short_history: bool

    def trace_payload(self) -> dict[str, Any]:
        return {
            "summary_version": self.summary_version,
            "summary_through_sequence": self.summary_through_sequence,
            "recent_turn_ids": [turn.id for turn in self.recent_turns],
            "relevant_turn_ids": [turn.id for turn in self.relevant_turns],
            "selected_turn_ids": [turn.id for turn in self.selected_turns],
            "history_matches": [item.to_dict() for item in self.history_matches],
            "used_full_short_history": self.used_full_short_history,
        }


def plan_conversation_turn(
    query: str,
    *,
    summary: dict[str, Any],
    recent_turns: list[ConversationTurn],
    available_turns: list[ConversationTurn] | None = None,
    memory_candidates: list[UserMemory] | None = None,
    llm: JsonChatClient,
) -> TurnPlan:
    available_turns = available_turns if available_turns is not None else recent_turns
    available_turn_ids = {turn.id for turn in available_turns}
    memory_candidates = memory_candidates or []
    available_memory_ids = {memory.id for memory in memory_candidates}
    payload = llm.complete_json(
        [
            {"role": "system", "content": TURN_PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _turn_planner_user_prompt(
                    query=query,
                    summary=summary,
                    recent_turns=recent_turns,
                    available_turns=available_turns,
                    memory_candidates=memory_candidates,
                ),
            },
        ],
        temperature=0.0,
        max_tokens=900,
    )
    return parse_turn_plan(
        payload,
        query=query,
        available_turn_ids=available_turn_ids,
        available_memory_ids=available_memory_ids,
    )


def raw_turn_plan(query: str) -> TurnPlan:
    """Preserve raw developer mode while still allowing history-aware Writer input."""

    return TurnPlan(
        turn_type="new_topic",
        resolved_question=query.strip(),
        retrieval_policy="new",
        evidence_source_turn_id=None,
        needs_web=False,
        search_queries=[],
        response_depth="normal",
        clarification_focus="",
        memory_ids=[],
        memory_write_policy=_explicit_memory_write_policy(query),
    )


def parse_turn_plan(
    payload: dict[str, object],
    *,
    query: str,
    available_turn_ids: set[str],
    available_memory_ids: set[str] | None = None,
) -> TurnPlan:
    available_memory_ids = available_memory_ids or set()
    turn_type = str(payload.get("turn_type") or "new_topic").strip()
    if turn_type not in TURN_TYPES:
        turn_type = "new_topic"
    retrieval_policy = str(payload.get("retrieval_policy") or "new").strip()
    if retrieval_policy not in RETRIEVAL_POLICIES:
        retrieval_policy = "new"
    response_depth = str(payload.get("response_depth") or "normal").strip()
    if response_depth not in RESPONSE_DEPTHS:
        response_depth = "normal"
    resolved_question = str(payload.get("resolved_question") or query).strip() or query.strip()
    evidence_source_turn_id = str(payload.get("evidence_source_turn_id") or "").strip() or None
    needs_web = bool(payload.get("needs_web"))
    search_queries = _string_list(payload.get("search_queries"))[:3]
    clarification_focus = str(payload.get("clarification_focus") or "").strip()
    memory_write_policy = str(payload.get("memory_write_policy") or "defer").strip()
    if memory_write_policy not in MEMORY_WRITE_POLICIES:
        memory_write_policy = "defer"
    memory_ids = [
        memory_id
        for memory_id in _string_list(payload.get("memory_ids"))
        if memory_id in available_memory_ids
    ][:4]

    if turn_type == "unclear":
        retrieval_policy = "none"
        evidence_source_turn_id = None
        needs_web = False
        search_queries = []
    elif turn_type == "casual":
        retrieval_policy = "none"
        evidence_source_turn_id = None
        needs_web = False
        search_queries = []
        response_depth = "brief"
    elif needs_web:
        retrieval_policy = "new"
        evidence_source_turn_id = None
    elif turn_type in {"new_topic", "follow_up"}:
        # A follow-up is a new content question inside the current topic. It
        # needs evidence for the resolved question instead of inheriting the
        # previous turn's entire retrieval set.
        retrieval_policy = "new"
        evidence_source_turn_id = None
    elif retrieval_policy == "reuse" and evidence_source_turn_id not in available_turn_ids:
        retrieval_policy = "new"
        evidence_source_turn_id = None
    if retrieval_policy != "reuse":
        evidence_source_turn_id = None
    if retrieval_policy != "new":
        needs_web = False
        search_queries = []
    if not needs_web:
        search_queries = []

    return TurnPlan(
        turn_type=turn_type,  # type: ignore[arg-type]
        resolved_question=resolved_question,
        retrieval_policy=retrieval_policy,  # type: ignore[arg-type]
        evidence_source_turn_id=evidence_source_turn_id,
        needs_web=needs_web,
        search_queries=search_queries,
        response_depth=response_depth,  # type: ignore[arg-type]
        clarification_focus=clarification_focus,
        memory_ids=memory_ids,
        memory_write_policy=memory_write_policy,  # type: ignore[arg-type]
    )


def select_conversation_context(
    store: ConversationStore,
    conversation_id: str,
    *,
    resolved_question: str,
    turn_type: TurnType,
    encoder: TextEncoder | None,
    embedding_model: str,
    embedding_version: str = "turn-memory-v1",
    recent_count: int = 3,
    relevant_count: int = 2,
    short_history_turns: int = 6,
) -> SelectedConversationContext:
    summary_state = store.get_summary(conversation_id)
    turns = store.get_completed_turns(conversation_id)
    recent_turns = turns[-recent_count:]

    if turn_type == "new_topic":
        return SelectedConversationContext(
            summary={},
            summary_version=int(summary_state["version"]),
            summary_through_sequence=int(summary_state["through_sequence"]),
            recent_turns=[],
            relevant_turns=[],
            selected_turns=[],
            history_matches=[],
            used_full_short_history=len(turns) <= short_history_turns,
        )

    if len(turns) <= short_history_turns:
        return SelectedConversationContext(
            summary={},
            summary_version=int(summary_state["version"]),
            summary_through_sequence=int(summary_state["through_sequence"]),
            recent_turns=recent_turns,
            relevant_turns=[],
            selected_turns=turns,
            history_matches=[],
            used_full_short_history=True,
        )

    older_turns = turns[:-recent_count]
    relevant_turns: list[ConversationTurn] = []
    matches: list[HistoryMatch] = []
    if encoder is not None and older_turns and resolved_question.strip():
        query_vector = encoder.encode_texts([resolved_question], batch_size=1)[0].dense
        turn_vectors = _load_or_create_turn_vectors(
            store,
            older_turns,
            encoder=encoder,
            model=embedding_model,
            version=embedding_version,
        )
        scored = sorted(
            (
                (_cosine_similarity(query_vector, turn_vectors[turn.id]), turn)
                for turn in older_turns
                if turn.id in turn_vectors
            ),
            key=lambda item: (-item[0], item[1].sequence),
        )[:relevant_count]
        relevant_turns = [turn for _, turn in scored]
        matches = [HistoryMatch(turn_id=turn.id, score=score) for score, turn in scored]

    selected_by_id = {turn.id: turn for turn in [*relevant_turns, *recent_turns]}
    selected_turns = sorted(selected_by_id.values(), key=lambda turn: turn.sequence)
    return SelectedConversationContext(
        summary=dict(summary_state["summary"]),
        summary_version=int(summary_state["version"]),
        summary_through_sequence=int(summary_state["through_sequence"]),
        recent_turns=recent_turns,
        relevant_turns=relevant_turns,
        selected_turns=selected_turns,
        history_matches=matches,
        used_full_short_history=False,
    )


def should_update_summary(
    turns: list[ConversationTurn],
    summary_state: dict[str, Any],
    *,
    short_history_turns: int = 3,
    token_threshold: int = 8000,
    update_every_turns: int = 3,
) -> bool:
    if not turns:
        return False
    through_sequence = int(summary_state.get("through_sequence") or 0)
    version = int(summary_state.get("version") or 0)
    eligible = turns[:-short_history_turns] if len(turns) > short_history_turns else []
    unsummarized = [turn for turn in eligible if turn.sequence > through_sequence]
    if not unsummarized:
        return False
    if version > 0:
        return len(unsummarized) >= update_every_turns
    estimated_tokens = int(
        estimated_usage_for_text(*(turn.memory_text for turn in unsummarized)).get(
            "estimated_tokens", 0
        )
    )
    return len(unsummarized) >= update_every_turns or estimated_tokens >= token_threshold


def update_conversation_summary(
    store: ConversationStore,
    conversation_id: str,
    *,
    llm: JsonChatClient,
) -> dict[str, Any] | None:
    turns = store.get_completed_turns(conversation_id)
    summary_state = store.get_summary(conversation_id)
    if not should_update_summary(turns, summary_state):
        return None
    through_sequence = int(summary_state.get("through_sequence") or 0)
    eligible = turns[:-3] if len(turns) > 3 else []
    unsummarized = [turn for turn in eligible if turn.sequence > through_sequence]
    if not unsummarized:
        return None
    payload = llm.complete_json(
        [
            {"role": "system", "content": CONVERSATION_SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _summary_user_prompt(
                    existing=dict(summary_state.get("summary") or {}),
                    turns=unsummarized,
                ),
            },
        ],
        temperature=0.0,
        max_tokens=1200,
    )
    summary = normalize_summary(payload)
    latest_sequence = max(turn.sequence for turn in unsummarized)
    return store.save_summary(
        conversation_id,
        summary,
        through_sequence=latest_sequence,
    )


def normalize_summary(payload: dict[str, object]) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {}
    for field_name in SUMMARY_FIELDS:
        summary[field_name] = _string_list(payload.get(field_name))[:20]
    return summary


def turns_to_chat_messages(turns: list[ConversationTurn]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        messages.append({"role": "user", "content": turn.query})
        messages.append({"role": "assistant", "content": turn.assistant_text})
    return messages


def _load_or_create_turn_vectors(
    store: ConversationStore,
    turns: list[ConversationTurn],
    *,
    encoder: TextEncoder,
    model: str,
    version: str,
) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    missing: list[ConversationTurn] = []
    for turn in turns:
        vector = store.get_turn_embedding(turn.id, model=model, version=version)
        if vector is None:
            missing.append(turn)
        else:
            vectors[turn.id] = vector
    if missing:
        encoded = encoder.encode_texts([turn.memory_text for turn in missing])
        for turn, embedding in zip(missing, encoded, strict=True):
            vectors[turn.id] = embedding.dense
            store.save_turn_embedding(turn.id, embedding.dense, model=model, version=version)
    return vectors


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _turn_planner_user_prompt(
    *,
    query: str,
    summary: dict[str, Any],
    recent_turns: list[ConversationTurn],
    available_turns: list[ConversationTurn],
    memory_candidates: list[UserMemory],
) -> str:
    summary_block = json.dumps(summary, ensure_ascii=False, indent=2) if summary else "无。"
    if recent_turns:
        history_blocks = []
        for turn in recent_turns:
            history_blocks.append(
                "\n".join(
                    [
                        f"[turn_id={turn.id}]",
                        f"用户：{turn.query}",
                        f"助手：{turn.assistant_text}",
                    ]
                )
            )
        history_block = "\n\n".join(history_blocks)
    else:
        history_block = "无。"
    recent_ids = {turn.id for turn in recent_turns}
    older_index = [
        f"- turn_id={turn.id}；用户问题：{turn.query}"
        for turn in available_turns
        if turn.id not in recent_ids
    ][-50:]
    older_block = "\n".join(older_index) or "无。"
    memory_block = (
        "\n".join(memory.prompt_line() for memory in memory_candidates)
        if memory_candidates
        else "无。"
    )
    return f"""会话摘要：
{summary_block}

最近对话：
{history_block}

更早 Turn 索引（仅用于选择需要复用的证据轮次）：
{older_block}

跨会话用户记忆候选（只选择当前问题真正需要的项）：
{memory_block}

当前用户消息：
{query}
"""


def _summary_user_prompt(
    *,
    existing: dict[str, Any],
    turns: list[ConversationTurn],
) -> str:
    existing_block = json.dumps(existing, ensure_ascii=False, indent=2) if existing else "无。"
    turn_blocks = [
        "\n".join(
            [
                f"[turn_id={turn.id}]",
                f"用户：{turn.query}",
                f"助手：{turn.assistant_text}",
            ]
        )
        for turn in turns
    ]
    return f"""已有摘要：
{existing_block}

新增对话：
{chr(10).join(turn_blocks)}
"""


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _explicit_memory_write_policy(query: str) -> MemoryWritePolicy:
    compact = "".join(query.split())
    markers = (
        "请记住", "帮我记住", "以后记得", "忘掉这件事", "忘记这件事",
        "不要记住这个", "删除这条记忆", "刚才记错了", "纠正一下",
    )
    return "flush" if any(marker in compact for marker in markers) else "defer"


TURN_PLANNER_SYSTEM_PROMPT = """你是 PersonaForge 的 Conversation-aware Turn Planner。

你只负责理解当前用户在延续什么话题、把残缺追问补成可独立检索的问题，并决定
是否需要联网和作者历史检索。你不能模仿作者，不能预测作者立场，不能生成答案。

只能输出 JSON object：
{
  "turn_type": "new_topic|follow_up|explain_previous|casual|unclear",
  "resolved_question": "...",
  "retrieval_policy": "new|reuse|none",
  "evidence_source_turn_id": null,
  "needs_web": false,
  "search_queries": [],
  "response_depth": "brief|normal|deep",
  "clarification_focus": "",
  "memory_ids": [],
  "memory_write_policy": "defer|flush"
}

分类和路由规则：
- resolved_question 只做必要的指代消解和省略补全，不加入用户没有表达的观点、原因或框架。
- new_topic：与当前对话无关的新主题，必须使用 new。
- follow_up：沿用当前主题、人物或情境，但提出了新的内容问题、观点判断、行动建议或
  因果问题，必须使用 new。是否还是同一主题，不是复用材料的理由。
- explain_previous：操作对象是已有 assistant 回答本身，而不是继续向作者询问新内容。
  “展开/解释/论证你刚才那一点”使用 reuse，并填写对应 evidence_source_turn_id；
  “压缩/改写/翻译/纠正字数或格式”使用 none。
- 闲聊、致谢、简单情绪回应使用 none 和 brief。
- 只有不同理解会明显改变检索与回答时才使用 unclear，并在 clarification_focus
  客观说明缺少什么信息；不要写最终澄清话术。
- 涉及近期事件、具体人物近期言论、新闻、热搜、新梗、反常用词或不确定的新义时
  needs_web=true，并生成最多 3 条只用于查清客观背景的 search_queries。
- needs_web=true 时 retrieval_policy 必须是 new。
- 用户明确要求“一句话、简单说”时 brief；明确要求“展开、详细分析”时 deep；
  否则根据当前话语动作选择 normal 或 brief。
- 历史 assistant 的说法只是对话上下文，不是作者真实观点或可靠外部事实。
- memory_ids 最多选择 4 个候选 id。它们只用于理解用户本人和延续跨会话上下文；
  不相关时必须返回空数组，不能把用户记忆当作作者观点或外部事实。
- memory_write_policy 默认 defer。只有用户明确要求记住、忘记或纠正长期记忆，或者
  当前消息明确披露了未来跨会话仍有帮助的稳定偏好、个人事实或持续事件时才使用 flush。
  普通知识问题、假设、一次性要求、闲聊和仅由助手提出的信息必须 defer。

典型边界：
- 前文讨论婚姻博弈，用户问“那当代年轻人有必要结婚吗”：
  follow_up + new。
- 前文解释配得感，用户问“那这种人应该怎么改变”：
  follow_up + new。
- 前文讨论短线亏损，用户问“那正确做短线的思维是什么”：
  follow_up + new。
- 用户说“展开解释你刚才第二段”：
  explain_previous + reuse。
- 用户说“把刚才的建议压缩成一句话”或“这不是八个字吗”：
  explain_previous + none + brief。
"""


CONVERSATION_SUMMARY_SYSTEM_PROMPT = """你是 PersonaForge 的会话摘要节点。

你要把已有摘要和新增对话合并成结构化 JSON，方便后续恢复指代和用户要求。
只能输出：
{
  "topics": [],
  "entities": [],
  "user_requests": [],
  "assistant_previous_claims": [],
  "unresolved_references": []
}

约束：
- 只保留对后续对话可能有用的内容，使用简短中文短句。
- 不推断长期人格、隐私属性或用户没有说出的偏好。
- assistant 的生成观点必须写成“此前 assistant 曾表示……”，不能写成作者事实。
- 不把 assistant 的回答当作外部事实。
- 合并重复项，删除已经解决的 unresolved reference。
"""
