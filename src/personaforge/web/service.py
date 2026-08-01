"""Service layer used by the FastAPI Web app."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Iterator
from uuid import uuid4

from personaforge.ingest.embeddings import BgeM3Encoder, TextEncoder
from personaforge.ingest.query_understanding import (
    GroundedQueryPlan,
    RetrievalQuery,
    SearchResult,
    TavilySearchClient,
    build_background_and_retrieval_queries,
    plan_to_trace,
    plan_web_search,
)
from personaforge.ingest.retrieve import ParentHit, RetrieveResult, retrieve_parents, retrieve_parents_for_queries
from personaforge.llm import DeepSeekJsonClient, JsonChatClient
from personaforge.persona.pack import load_persona_pack_for_index
from personaforge.persona.writer import build_writer_messages
from personaforge.web.conversations import ConversationStore
from personaforge.web.multiturn import (
    SelectedConversationContext,
    TurnPlan,
    plan_conversation_turn,
    raw_turn_plan,
    select_conversation_context,
    turns_to_chat_messages,
)
from personaforge.web.trace import (
    DEFAULT_TRACE_RETENTION,
    TRACE_SCHEMA_VERSION,
    estimated_usage_for_text,
    new_stage,
    new_trace_id,
    provider_usage,
    read_trace,
    write_trace,
)
from personaforge.web.user_memory import (
    MemoryRecallHit,
    UserMemory,
    UserMemoryStore,
    recall_user_memories,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WebConfig:
    author: str | None = None
    data_dir: Path = Path("data")
    host: str = "127.0.0.1"
    port: int = 8000
    model_name: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    use_fp16: bool = True
    child_top_k: int = 100
    per_query_parent_k: int = 30
    parent_top_k: int = 20
    max_search_results: int = 5
    temperature: float = 0.85
    max_tokens: int = 1600
    trace_retention: int = DEFAULT_TRACE_RETENTION
    index_batch_size: int = 12
    auth_required: bool = True
    secure_cookies: bool | None = None
    session_days: int = 30


@dataclass(slots=True)
class LocalPersona:
    author: str
    source: str
    display_name: str
    avatar_url: str | None
    headline: str
    content_count: int | None
    persona_pack_available: bool
    profile_url: str | None
    last_synced_at: str | None
    author_dir: Path
    index_dir: Path
    qdrant_path: Path


@dataclass(slots=True)
class PreparedChat:
    session_id: str
    author: str
    query: str
    query_mode: str
    writer_prompt: str
    objective_background: str
    query_trace: dict[str, Any] | None
    retrieve_result: RetrieveResult
    messages: list[dict[str, str]]
    trace_capture: str = "summary"
    trace_id: str = ""
    trace_created_at: str = ""
    trace_started_at: float = 0.0
    query_understanding_duration_ms: int = 0
    retrieval_duration_ms: int = 0
    writer_build_duration_ms: int = 0
    generation_started_at: float | None = None
    generation_duration_ms: int = 0
    generation_ttft_ms: int | None = None
    generation_usage: dict[str, Any] | None = None
    stages: list[dict[str, Any]] | None = None
    persona_pack_id: str | None = None
    persona_pack_sha256: str | None = None
    persona_pack_claim_count: int = 0
    turn_id: str | None = None
    turn_plan: dict[str, Any] | None = None
    conversation_context: dict[str, Any] | None = None
    response_max_tokens: int | None = None
    memory_update: dict[str, Any] | None = None
    owner_id: str = "local-user"
    recalled_memories: list[UserMemory] | None = None
    selected_memory_ids: list[str] | None = None


@dataclass(frozen=True, slots=True)
class ChatProgress:
    """A user-visible execution stage emitted before the work actually starts."""

    stage: str
    label: str


class _SynchronizedEncoder:
    """Serialize access to one shared local embedding model across chat workers."""

    def __init__(self, encoder: TextEncoder, lock: threading.RLock) -> None:
        self.encoder = encoder
        self.lock = lock

    def encode_texts(self, texts: list[str], *, batch_size: int = 12):
        with self.lock:
            return self.encoder.encode_texts(texts, batch_size=batch_size)


class PersonaChatService:
    def __init__(
        self,
        config: WebConfig,
        *,
        encoder: TextEncoder | None = None,
        llm: JsonChatClient | None = None,
        conversation_store: ConversationStore | None = None,
        user_memory_store: UserMemoryStore | None = None,
    ) -> None:
        self.config = config
        self._encoder = encoder
        self._encoder_proxy: TextEncoder | None = None
        self._encoder_lock = threading.RLock()
        self._encoder_warmup_thread: threading.Thread | None = None
        self._encoder_warmup_state = "ready" if encoder is not None else "idle"
        self._encoder_warmup_duration_ms: int | None = 0 if encoder is not None else None
        self._encoder_warmup_error: str | None = None
        self._llm = llm
        self._llm_local = threading.local()
        self.conversations = conversation_store or ConversationStore(config.data_dir)
        self.user_memories = user_memory_store or UserMemoryStore(config.data_dir)

    def start_encoder_warmup(self) -> bool:
        """Load the shared embedding model off the first user request."""

        with self._encoder_lock:
            if self._encoder_warmup_state == "ready":
                return False
            if self._encoder_warmup_thread is not None and self._encoder_warmup_thread.is_alive():
                return False
            self._encoder_warmup_state = "loading"
            self._encoder_warmup_duration_ms = None
            self._encoder_warmup_error = None
            thread = threading.Thread(
                target=self._warm_encoder,
                name="personaforge-embedding-warmup",
                daemon=True,
            )
            self._encoder_warmup_thread = thread
            thread.start()
            return True

    def encoder_status(self) -> dict[str, Any]:
        with self._encoder_lock:
            return {
                "state": self._encoder_warmup_state,
                "device": self.config.embedding_device,
                "duration_ms": self._encoder_warmup_duration_ms,
                "error": self._encoder_warmup_error,
            }

    def _warm_encoder(self) -> None:
        started_at = perf_counter()
        try:
            encoder = self._get_encoder()
            encoder.encode_texts(["PersonaForge 启动预热"], batch_size=1)
        except Exception as exc:  # pragma: no cover - depends on local CUDA/model state.
            duration_ms = round((perf_counter() - started_at) * 1000)
            with self._encoder_lock:
                self._encoder_warmup_state = "failed"
                self._encoder_warmup_duration_ms = duration_ms
                self._encoder_warmup_error = f"{type(exc).__name__}: {exc}"
            logger.exception("BGE-M3 background warmup failed")
            return

        duration_ms = round((perf_counter() - started_at) * 1000)
        with self._encoder_lock:
            self._encoder_warmup_state = "ready"
            self._encoder_warmup_duration_ms = duration_ms
            self._encoder_warmup_error = None
        logger.info("BGE-M3 background warmup completed in %.2f s", duration_ms / 1000)

    def list_personas(self) -> list[LocalPersona]:
        return list_local_personas(self.config.data_dir)

    def default_author(self) -> str | None:
        if self.config.author:
            return self.config.author
        personas = self.list_personas()
        return personas[0].author if personas else None

    def prepare_chat(
        self,
        *,
        author: str | None,
        session_id: str | None,
        query: str,
        query_mode: str,
        writer_prompt: str,
        parent_top_k: int | None = None,
        trace_capture: str = "summary",
    ) -> PreparedChat:
        prepared: PreparedChat | None = None
        for item in self.iter_prepare_chat(
            author=author,
            session_id=session_id,
            query=query,
            query_mode=query_mode,
            writer_prompt=writer_prompt,
            parent_top_k=parent_top_k,
            trace_capture=trace_capture,
        ):
            if isinstance(item, PreparedChat):
                prepared = item
        if prepared is None:  # pragma: no cover - defensive invariant.
            raise RuntimeError("Chat preparation finished without a prepared request.")
        return prepared

    def iter_prepare_chat(
        self,
        *,
        author: str | None,
        session_id: str | None,
        query: str,
        query_mode: str,
        writer_prompt: str,
        parent_top_k: int | None = None,
        trace_capture: str = "summary",
        turn_id: str | None = None,
    ) -> Iterator[ChatProgress | PreparedChat]:
        selected_author = (author or self.default_author() or "").strip()
        if not selected_author:
            raise ValueError("No local persona index found. Run `pf build` and `pf index` first.")

        if turn_id:
            run = self.conversations.get_turn(turn_id)
            if run.author != selected_author:
                raise ValueError("Turn belongs to a different author.")
            if session_id and run.conversation_id != session_id:
                raise ValueError("Turn belongs to a different conversation.")
            session_id = run.conversation_id
            query = run.query
            query_mode = run.query_mode
            writer_prompt = run.writer_prompt
            parent_top_k = run.parent_top_k
            trace_capture = run.trace_capture

        selected_session_id = session_id or new_session_id()
        trace_id = new_trace_id()
        trace_created_at = utc_now()
        trace_started_at = perf_counter()
        resolved_parent_top_k = parent_top_k or self.config.parent_top_k
        if trace_capture not in {"summary", "full"}:
            raise ValueError(f"Unknown trace_capture: {trace_capture}")

        try:
            index_dir = self.config.data_dir / "authors" / "zhihu" / selected_author / "index"
            qdrant_path = index_dir / "qdrant"
            persona_pack = (
                load_persona_pack_for_index(index_dir, required=True)
                if writer_prompt == "persona_pack"
                else None
            )
            stages: list[dict[str, Any]] = []
            llm = self._get_llm()
            completed_turns = self.conversations.get_completed_turns(selected_session_id)
            try:
                owner_id = self.conversations.get_conversation_owner(selected_session_id)
            except FileNotFoundError:
                owner_id = "local-user"
            try:
                summary_state = self.conversations.get_summary(selected_session_id)
            except FileNotFoundError:
                summary_state = {"summary": {}, "through_sequence": 0, "version": 0}

            yield ChatProgress(stage="conversation_context", label="正在理解对话")
            context_started_at = perf_counter()
            recent_turns = completed_turns[-3:]
            stages.append(
                self._stage(
                    "conversation_context",
                    "读取会话上下文",
                    context_started_at,
                    details={
                        "completed_turn_count": len(completed_turns),
                        "recent_turn_ids": [item.id for item in recent_turns],
                        "summary_version": int(summary_state.get("version") or 0),
                    },
                )
            )

            planner_started_at = perf_counter()
            memory_hits: list[MemoryRecallHit] = []
            memory_candidates: list[UserMemory] = []
            if query_mode == "grounded":
                memory_started_at = perf_counter()
                memory_hits = recall_user_memories(
                    self.user_memories,
                    owner_id,
                    query,
                    encoder=self._get_encoder(),
                    model=self.config.model_name,
                    limit=8,
                )
                memory_candidates = [hit.memory for hit in memory_hits]
                stages.append(
                    self._stage(
                        "user_memory_recall",
                        "召回跨会话用户记忆",
                        memory_started_at,
                        details={
                            "candidate_count": len(memory_hits),
                            "candidates": [hit.trace_payload() for hit in memory_hits],
                        },
                    )
                )
                plan = plan_conversation_turn(
                    query,
                    summary=dict(summary_state.get("summary") or {}),
                    recent_turns=recent_turns,
                    available_turns=completed_turns,
                    memory_candidates=memory_candidates,
                    llm=llm,
                )
                planner_usage = self._usage_or_estimate(
                    llm,
                    query,
                    json.dumps(summary_state.get("summary") or {}, ensure_ascii=False),
                    *(turn.memory_text for turn in recent_turns),
                )
            elif query_mode == "raw":
                plan = raw_turn_plan(query)
                planner_usage = None
            else:
                raise ValueError(f"Unknown query_mode: {query_mode}")
            stages.append(
                self._stage(
                    "turn_planner",
                    "判断当前对话动作",
                    planner_started_at,
                    details={
                        "turn_type": plan.turn_type,
                        "retrieval_policy": plan.retrieval_policy,
                        "needs_web": plan.needs_web,
                        "response_depth": plan.response_depth,
                        "evidence_source_turn_id": plan.evidence_source_turn_id,
                        "memory_ids": plan.memory_ids,
                    },
                    usage=planner_usage,
                )
            )
            planner_duration_ms = elapsed_ms(planner_started_at)
            selected_memories = [
                memory for memory in memory_candidates if memory.id in set(plan.memory_ids)
            ]
            if selected_memories:
                self.user_memories.mark_accessed(owner_id, [memory.id for memory in selected_memories])
            if turn_id:
                self.conversations.set_turn_plan(turn_id, plan.to_dict())

            history_started_at = perf_counter()
            if self._conversation_exists(selected_session_id):
                conversation_context = select_conversation_context(
                    self.conversations,
                    selected_session_id,
                    resolved_question=plan.resolved_question,
                    turn_type=plan.turn_type,
                    encoder=self._get_encoder() if len(completed_turns) > 6 and plan.turn_type != "new_topic" else None,
                    embedding_model=self.config.model_name,
                )
            else:
                conversation_context = _empty_conversation_context()
            stages.append(
                self._stage(
                    "history_recall",
                    "选择相关历史对话",
                    history_started_at,
                    details=conversation_context.trace_payload(),
                )
            )

            source_turn = None
            if plan.retrieval_policy == "reuse":
                source_turn = next(
                    (item for item in completed_turns if item.id == plan.evidence_source_turn_id),
                    None,
                )
                if source_turn is None or not source_turn.parent_ids:
                    plan = _fallback_to_new_retrieval(plan)
                    if turn_id:
                        self.conversations.set_turn_plan(turn_id, plan.to_dict())

            search_results: list[SearchResult] = []
            objective_background = ""
            retrieval_queries: list[RetrievalQuery] = []
            query_trace: dict[str, Any] | None = None
            understanding_started_at = perf_counter()

            if plan.retrieval_policy == "new" and query_mode == "grounded":
                grounding_error: dict[str, str] | None = None
                if plan.needs_web:
                    yield ChatProgress(stage="web_grounding", label="正在查询相关背景")
                    stage_started_at = perf_counter()
                    try:
                        search_results = TavilySearchClient.from_env().search_many(
                            plan.search_queries,
                            max_results=self.config.max_search_results,
                        )
                        stages.append(
                            self._stage(
                                "tavily_search",
                                "获取公开背景资料",
                                stage_started_at,
                                details={
                                    "search_query_count": len(plan.search_queries),
                                    "result_count": len(search_results),
                                },
                            )
                        )
                    except Exception as exc:
                        grounding_error = trace_error(exc)
                        stages.append(
                            self._stage(
                                "tavily_search",
                                "获取公开背景资料",
                                stage_started_at,
                                status="fallback",
                                details={
                                    "search_query_count": len(plan.search_queries),
                                    "error": grounding_error,
                                },
                            )
                        )
                        yield ChatProgress(
                            stage="web_fallback",
                            label="未获得额外背景，继续检索作者历史表达",
                        )
                yield ChatProgress(stage="query_transform", label="正在整理检索线索")
                stage_started_at = perf_counter()
                transform = build_background_and_retrieval_queries(
                    query,
                    resolved_query=plan.resolved_question,
                    search_results=search_results,
                    llm=llm,
                )
                objective_background = transform.objective_background
                retrieval_queries = transform.retrieval_queries
                stages.append(
                    self._stage(
                        "query_transform",
                        "生成多路检索表达",
                        stage_started_at,
                        details={
                            "retrieval_query_count": len(retrieval_queries),
                            "has_background": bool(objective_background),
                        },
                        usage=self._usage_or_estimate(
                            llm,
                            query,
                            plan.resolved_question,
                            *(item.content for item in search_results),
                        ),
                    )
                )
                query_trace = _multiturn_query_trace(
                    query=query,
                    plan=plan,
                    search_results=search_results,
                    objective_background=objective_background,
                    retrieval_queries=retrieval_queries,
                    grounding_error=grounding_error,
                )
            elif plan.retrieval_policy == "new":
                query_trace = _multiturn_query_trace(
                    query=query,
                    plan=plan,
                    search_results=[],
                    objective_background="",
                    retrieval_queries=[],
                )
            else:
                query_trace = _multiturn_query_trace(
                    query=query,
                    plan=plan,
                    search_results=[],
                    objective_background="",
                    retrieval_queries=[],
                )
            understanding_ms = elapsed_ms(understanding_started_at) + planner_duration_ms

            retrieval_started_at = perf_counter()
            if plan.retrieval_policy == "new":
                yield ChatProgress(stage="retrieval", label="正在检索历史表达")
                if query_mode == "raw":
                    retrieve_result = retrieve_parents(
                        plan.resolved_question,
                        author=selected_author,
                        index_dir=index_dir,
                        qdrant_path=qdrant_path,
                        encoder=self._get_encoder(),
                        child_top_k=self.config.child_top_k,
                        parent_top_k=resolved_parent_top_k,
                    )
                else:
                    retrieve_result = retrieve_parents_for_queries(
                        plan.resolved_question,
                        retrieval_queries,
                        author=selected_author,
                        index_dir=index_dir,
                        qdrant_path=qdrant_path,
                        encoder=self._get_encoder(),
                        child_top_k=self.config.child_top_k,
                        per_query_parent_k=self.config.per_query_parent_k,
                        parent_top_k=resolved_parent_top_k,
                    )
                stages.extend(self._retrieval_stages(retrieve_result, retrieval_started_at))
            elif plan.retrieval_policy == "reuse" and source_turn is not None:
                yield ChatProgress(stage="retrieval_reuse", label="正在延续上次回答")
                retrieve_result = _retrieve_result_from_parent_ids(
                    query=plan.resolved_question,
                    author=selected_author,
                    parent_ids=source_turn.parent_ids,
                    parents_path=index_dir / "parents.jsonl",
                    child_top_k=self.config.child_top_k,
                )
                stages.append(
                    self._stage(
                        "retrieval_reuse",
                        "复用既有作者证据",
                        retrieval_started_at,
                        details={
                            "source_turn_id": source_turn.id,
                            "parent_count": len(retrieve_result.parents),
                        },
                    )
                )
            else:
                retrieve_result = _empty_retrieve_result(
                    query=plan.resolved_question,
                    author=selected_author,
                    child_top_k=self.config.child_top_k,
                )
                stages.append(
                    self._stage(
                        "retrieval_skipped",
                        "跳过作者材料检索",
                        retrieval_started_at,
                        details={"reason": plan.turn_type},
                    )
                )
            retrieval_ms = elapsed_ms(retrieval_started_at)

            selected_turns = list(conversation_context.selected_turns)
            if source_turn is not None and source_turn.id not in {item.id for item in selected_turns}:
                selected_turns.append(source_turn)
                selected_turns.sort(key=lambda item: item.sequence)

            yield ChatProgress(stage="writer", label="正在准备回答")
            writer_started_at = perf_counter()
            messages = build_writer_messages(
                query=query,
                parent_hits=retrieve_result.parents,
                objective_background=objective_background,
                writer_prompt=writer_prompt,
                persona_pack=persona_pack,
                conversation_summary=conversation_context.summary,
                conversation_messages=turns_to_chat_messages(selected_turns),
                response_depth=plan.response_depth,
                clarification_focus=plan.clarification_focus if plan.turn_type == "unclear" else "",
                user_memories=[memory.content for memory in selected_memories],
            )
            response_max_tokens = response_token_limit(
                plan.response_depth,
                retrieve_result.parents,
                configured_max=self.config.max_tokens,
            )
            stages.append(
                self._stage(
                    "writer_pack",
                    "组织作者身份、证据与对话",
                    writer_started_at,
                    details={
                        "parent_count": len(retrieve_result.parents),
                        "history_turn_count": len(selected_turns),
                        "message_count": len(messages),
                        "context_characters": sum(
                            len(message.get("content", "")) for message in messages
                        ),
                        "response_depth": plan.response_depth,
                        "response_max_tokens": response_max_tokens,
                        "persona_pack_id": persona_pack.pack_id if persona_pack else None,
                        "persona_pack_sha256": persona_pack.sha256 if persona_pack else None,
                        "persona_pack_claim_count": persona_pack.claim_count if persona_pack else 0,
                        "selected_user_memory_count": len(selected_memories),
                    },
                    usage=estimated_usage_for_text(
                        *(message.get("content", "") for message in messages)
                    ),
                )
            )
        except Exception as exc:
            self._write_prepare_failure_trace(
                author=selected_author,
                session_id=selected_session_id,
                trace_id=trace_id,
                created_at=trace_created_at,
                query=query,
                query_mode=query_mode,
                writer_prompt=writer_prompt,
                parent_top_k=resolved_parent_top_k,
                error=exc,
            )
            raise

        prepared = PreparedChat(
            session_id=selected_session_id,
            author=selected_author,
            query=query,
            query_mode=query_mode,
            writer_prompt=writer_prompt,
            trace_capture=trace_capture,
            objective_background=objective_background,
            query_trace=query_trace,
            retrieve_result=retrieve_result,
            messages=messages,
            trace_id=trace_id,
            trace_created_at=trace_created_at,
            trace_started_at=trace_started_at,
            query_understanding_duration_ms=understanding_ms,
            retrieval_duration_ms=retrieval_ms,
            writer_build_duration_ms=elapsed_ms(writer_started_at),
            stages=stages,
            persona_pack_id=persona_pack.pack_id if persona_pack else None,
            persona_pack_sha256=persona_pack.sha256 if persona_pack else None,
            persona_pack_claim_count=persona_pack.claim_count if persona_pack else 0,
            turn_id=turn_id,
            turn_plan=plan.to_dict(),
            conversation_context=conversation_context.trace_payload(),
            response_max_tokens=response_max_tokens,
            owner_id=owner_id,
            recalled_memories=memory_candidates,
            selected_memory_ids=[memory.id for memory in selected_memories],
        )
        if turn_id:
            self.conversations.set_turn_evidence(
                turn_id,
                [hit.parent_id for hit in retrieve_result.parents],
                trace_id,
            )
        self.record_prepared_trace(prepared)
        yield prepared

    def _iter_prepare_chat_v1(
        self,
        *,
        author: str | None,
        session_id: str | None,
        query: str,
        query_mode: str,
        writer_prompt: str,
        parent_top_k: int | None = None,
        trace_capture: str = "summary",
    ) -> Iterator[ChatProgress | PreparedChat]:
        """Keep the pre-multiturn pipeline available for fixture compatibility."""
        selected_author = (author or self.default_author() or "").strip()
        if not selected_author:
            raise ValueError("No local persona index found. Run `pf build` and `pf index` first.")

        selected_session_id = session_id or new_session_id()
        trace_id = new_trace_id()
        trace_created_at = utc_now()
        trace_started_at = perf_counter()
        resolved_parent_top_k = parent_top_k or self.config.parent_top_k
        if trace_capture not in {"summary", "full"}:
            raise ValueError(f"Unknown trace_capture: {trace_capture}")
        try:
            index_dir = self.config.data_dir / "authors" / "zhihu" / selected_author / "index"
            qdrant_path = index_dir / "qdrant"
            persona_pack = load_persona_pack_for_index(
                index_dir,
                required=writer_prompt == "persona_pack",
            ) if writer_prompt == "persona_pack" else None
            query_trace: dict[str, Any] | None = None
            objective_background = ""
            understanding_ms = 0
            stages: list[dict[str, Any]] = []
            llm = self._get_llm()

            if query_mode == "grounded":
                yield ChatProgress(stage="understanding", label="正在理解问题")
                stage_started_at = perf_counter()
                search_plan = plan_web_search(query, llm=llm)
                stages.append(
                    self._stage(
                        "search_planner",
                        "判断题目是否需要外部背景",
                        stage_started_at,
                        details={"needs_web": search_plan.needs_web, "search_query_count": len(search_plan.search_queries)},
                        usage=self._usage_or_estimate(llm, query),
                    )
                )
                search_results = []
                grounding_error: dict[str, str] | None = None
                if search_plan.needs_web:
                    yield ChatProgress(stage="web_grounding", label="正在查询相关背景")
                    stage_started_at = perf_counter()
                    try:
                        search_results = TavilySearchClient.from_env().search_many(
                            search_plan.search_queries,
                            max_results=self.config.max_search_results,
                        )
                        stages.append(
                            self._stage(
                                "tavily_search",
                                "获取公开背景资料",
                                stage_started_at,
                                details={"search_query_count": len(search_plan.search_queries), "result_count": len(search_results)},
                            )
                        )
                    except Exception as exc:  # An auxiliary source must not block a local RAG answer.
                        grounding_error = trace_error(exc)
                        stages.append(
                            self._stage(
                                "tavily_search",
                                "获取公开背景资料",
                                stage_started_at,
                                status="fallback",
                                details={"search_query_count": len(search_plan.search_queries), "error": grounding_error},
                            )
                        )
                        yield ChatProgress(
                            stage="web_fallback",
                            label="未获得额外背景，继续检索作者历史表达",
                        )

                yield ChatProgress(stage="query_transform", label="正在整理检索线索")
                stage_started_at = perf_counter()
                transform = build_background_and_retrieval_queries(
                    query,
                    search_results=search_results,
                    llm=llm,
                )
                stages.append(
                    self._stage(
                        "query_transform",
                        "生成多路检索表达",
                        stage_started_at,
                        details={"retrieval_query_count": len(transform.retrieval_queries), "has_background": bool(transform.objective_background)},
                        usage=self._usage_or_estimate(llm, query, *[str(item) for item in search_results]),
                    )
                )
                plan = GroundedQueryPlan(
                    original_query=query,
                    search_plan=search_plan,
                    search_results=search_results,
                    transform=transform,
                )
                query_trace = plan_to_trace(plan)
                if grounding_error is not None:
                    query_trace["web_grounding_error"] = grounding_error
                objective_background = transform.objective_background
                understanding_ms = sum(stage.get("duration_ms", 0) for stage in stages)
                retrieval_queries = transform.retrieval_queries
            elif query_mode == "raw":
                retrieval_queries = None
            else:
                raise ValueError(f"Unknown query_mode: {query_mode}")

            yield ChatProgress(stage="retrieval", label="正在检索历史表达")
            retrieval_started_at = perf_counter()
            if retrieval_queries is None:
                retrieve_result = retrieve_parents(
                    query,
                    author=selected_author,
                    index_dir=index_dir,
                    qdrant_path=qdrant_path,
                    encoder=self._get_encoder(),
                    child_top_k=self.config.child_top_k,
                    parent_top_k=resolved_parent_top_k,
                )
            else:
                retrieve_result = retrieve_parents_for_queries(
                    query,
                    retrieval_queries,
                    author=selected_author,
                    index_dir=index_dir,
                    qdrant_path=qdrant_path,
                    encoder=self._get_encoder(),
                    child_top_k=self.config.child_top_k,
                    per_query_parent_k=self.config.per_query_parent_k,
                    parent_top_k=resolved_parent_top_k,
                )
            retrieval_ms = elapsed_ms(retrieval_started_at)
            stages.extend(self._retrieval_stages(retrieve_result, retrieval_started_at))

            yield ChatProgress(stage="writer", label="正在准备回答")
            writer_started_at = perf_counter()
            messages = build_writer_messages(
                query=query,
                parent_hits=retrieve_result.parents,
                objective_background=objective_background,
                writer_prompt=writer_prompt,
                persona_pack=persona_pack,
            )
            stages.append(
                self._stage(
                    "writer_pack",
                    "组织作者历史表达与写作指令",
                    writer_started_at,
                    details={
                        "parent_count": len(retrieve_result.parents),
                        "message_count": len(messages),
                        "context_characters": sum(len(message.get("content", "")) for message in messages),
                        "persona_pack_id": persona_pack.pack_id if persona_pack else None,
                        "persona_pack_sha256": persona_pack.sha256 if persona_pack else None,
                        "persona_pack_claim_count": persona_pack.claim_count if persona_pack else 0,
                    },
                    usage=estimated_usage_for_text(*(message.get("content", "") for message in messages)),
                )
            )
        except Exception as exc:
            self._write_prepare_failure_trace(
                author=selected_author,
                session_id=selected_session_id,
                trace_id=trace_id,
                created_at=trace_created_at,
                query=query,
                query_mode=query_mode,
                writer_prompt=writer_prompt,
                parent_top_k=resolved_parent_top_k,
                error=exc,
            )
            raise

        prepared = PreparedChat(
            session_id=selected_session_id,
            author=selected_author,
            query=query,
            query_mode=query_mode,
            writer_prompt=writer_prompt,
            trace_capture=trace_capture,
            objective_background=objective_background,
            query_trace=query_trace,
            retrieve_result=retrieve_result,
            messages=messages,
            trace_id=trace_id,
            trace_created_at=trace_created_at,
            trace_started_at=trace_started_at,
            query_understanding_duration_ms=understanding_ms,
            retrieval_duration_ms=retrieval_ms,
            writer_build_duration_ms=elapsed_ms(writer_started_at),
            stages=stages,
            persona_pack_id=persona_pack.pack_id if persona_pack else None,
            persona_pack_sha256=persona_pack.sha256 if persona_pack else None,
            persona_pack_claim_count=persona_pack.claim_count if persona_pack else 0,
        )
        self.record_prepared_trace(prepared)
        yield prepared

    def stream_answer(self, prepared: PreparedChat) -> Iterator[str]:
        prepared.generation_started_at = perf_counter()
        first_token_at: float | None = None
        usage_payload: dict[str, Any] | None = None

        def receive_usage(usage: Any) -> None:
            nonlocal usage_payload
            usage_payload = provider_usage(usage)

        try:
            llm = self._get_llm()
            stream_with_usage = getattr(llm, "stream_text_with_usage", None)
            if callable(stream_with_usage):
                stream = stream_with_usage(
                    prepared.messages,
                    temperature=self.config.temperature,
                    max_tokens=prepared.response_max_tokens or self.config.max_tokens,
                    on_usage=receive_usage,
                )
            else:
                stream = llm.stream_text(
                    prepared.messages,
                    temperature=self.config.temperature,
                    max_tokens=prepared.response_max_tokens or self.config.max_tokens,
                )
            for token in stream:
                if first_token_at is None:
                    first_token_at = perf_counter()
                    prepared.generation_ttft_ms = elapsed_ms(prepared.generation_started_at)
                yield token
        finally:
            if prepared.generation_started_at is not None:
                prepared.generation_duration_ms = elapsed_ms(prepared.generation_started_at)
            prepared.generation_usage = usage_payload or self._usage_or_estimate(
                self._get_llm(), *(message.get("content", "") for message in prepared.messages)
            )

    def record_prepared_trace(self, prepared: PreparedChat) -> Path:
        return self._write_trace(prepared, status="prepared")

    def complete_trace(self, prepared: PreparedChat, answer: str) -> Path:
        return self._write_trace(prepared, status="completed", answer=answer)

    def fail_trace(self, prepared: PreparedChat, error: Exception) -> Path:
        return self._write_trace(prepared, status="failed", error=error)

    def update_memory_trace(
        self,
        *,
        author: str,
        trace_id: str,
        memory_update: dict[str, Any],
        answer: str,
    ) -> Path:
        payload = read_trace(self.config.data_dir, author, trace_id)
        payload["memory_update"] = memory_update
        payload["updated_at"] = utc_now()
        stages = payload.get("stages") if isinstance(payload.get("stages"), list) else []
        stages = [stage for stage in stages if isinstance(stage, dict) and stage.get("id") != "memory_update"]
        stages.append(
            new_stage(
                stage_id="memory_update",
                label="更新会话记忆",
                started_at=0.0,
                duration_ms=int(memory_update.get("duration_ms") or 0),
                status="failed" if memory_update.get("status") == "failed" else "completed",
                details={
                    "outcome": str(memory_update.get("status") or "completed"),
                    **{
                        key: value
                        for key, value in memory_update.items()
                        if key not in {"duration_ms", "status"}
                    },
                },
            )
        )
        offset_ms = 0
        for index, stage in enumerate(stages, start=1):
            stage["order"] = index
            stage["started_offset_ms"] = offset_ms
            offset_ms += int(stage.get("duration_ms") or 0)
        payload["stages"] = stages
        generation = payload.get("generation")
        if isinstance(generation, dict):
            generation["answer_characters"] = len(answer)
        return write_trace(
            self.config.data_dir,
            author,
            trace_id,
            payload,
            retention=self.config.trace_retention,
        )

    def get_trace(self, author: str, trace_id: str) -> dict[str, Any]:
        payload = read_trace(self.config.data_dir, author, trace_id)
        enrich_trace_source_urls(
            payload,
            self.config.data_dir / "authors" / "zhihu" / author / "index" / "parents.jsonl",
        )
        return payload

    def list_sessions(self, author: str, *, owner_id: str | None = None) -> list[dict[str, Any]]:
        return self.conversations.list_conversations(author, owner_id=owner_id)

    def get_session(
        self,
        author: str,
        session_id: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        payload = self.conversations.get_conversation(author, session_id, owner_id=owner_id)
        enrich_session_source_urls(
            payload,
            self.config.data_dir / "authors" / "zhihu" / author / "index" / "parents.jsonl",
        )
        return payload

    def delete_session(
        self,
        author: str,
        session_id: str,
        *,
        owner_id: str | None = None,
    ) -> None:
        trace_ids = self.conversations.delete_conversation(author, session_id, owner_id=owner_id)
        for trace_id in trace_ids:
            path = self.config.data_dir / "authors" / "zhihu" / author / "traces" / f"{trace_id}.json"
            try:
                path.unlink()
            except FileNotFoundError:
                continue

    def list_suggestions(self, author: str) -> list[str]:
        path = suggestions_path(self.config.data_dir, author)
        if not path.exists():
            return []
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            return []
        suggestions = payload.get("suggestions")
        if not isinstance(suggestions, list):
            return []
        return [str(item) for item in suggestions if isinstance(item, str)]

    def save_turn(self, prepared: PreparedChat, answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
        if prepared.turn_id:
            self.conversations.complete_turn(
                prepared.turn_id,
                answer=answer,
                sources=sources,
                trace_id=prepared.trace_id or None,
            )
        else:
            self.conversations.save_completed_turn(
                conversation_id=prepared.session_id,
                author=prepared.author,
                query=prepared.query,
                answer=answer,
                sources=sources,
                trace_id=prepared.trace_id or None,
                query_mode=prepared.query_mode,
                writer_prompt=prepared.writer_prompt,
                parent_top_k=prepared.retrieve_result.parent_top_k or self.config.parent_top_k,
                trace_capture=prepared.trace_capture,
            )
        return self.get_session(prepared.author, prepared.session_id)

    def _conversation_exists(self, conversation_id: str) -> bool:
        try:
            self.conversations.get_summary(conversation_id)
        except FileNotFoundError:
            return False
        return True

    def _get_encoder(self) -> TextEncoder:
        with self._encoder_lock:
            if self._encoder is None:
                self._encoder = BgeM3Encoder(
                    self.config.model_name,
                    device=self.config.embedding_device,
                    use_fp16=self.config.use_fp16,
                )
            if self._encoder_proxy is None:
                self._encoder_proxy = _SynchronizedEncoder(self._encoder, self._encoder_lock)
            return self._encoder_proxy

    def _get_llm(self) -> JsonChatClient:
        if self._llm is not None:
            return self._llm
        client = getattr(self._llm_local, "client", None)
        if client is None:
            client = DeepSeekJsonClient.from_env()
            self._llm_local.client = client
        return client

    def llm_client(self) -> JsonChatClient:
        return self._get_llm()

    def _write_trace(
        self,
        prepared: PreparedChat,
        *,
        status: str,
        answer: str | None = None,
        error: Exception | None = None,
    ) -> Path:
        payload = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_id": prepared.trace_id,
            "status": status,
            "created_at": prepared.trace_created_at,
            "updated_at": utc_now(),
            "input": {
                "author": prepared.author,
                "session_id": prepared.session_id,
                "turn_id": prepared.turn_id,
                "query": prepared.query,
                "query_mode": prepared.query_mode,
                "writer_prompt": prepared.writer_prompt,
                "retrieval_parameters": {
                    "child_top_k": self.config.child_top_k,
                    "per_query_parent_k": self.config.per_query_parent_k,
                    "parent_top_k": prepared.retrieve_result.parent_top_k,
                },
            },
            "capture": {"mode": prepared.trace_capture, "retention": self.config.trace_retention},
            "stages": self._finalized_stages(prepared, status=status, answer=answer, error=error),
            "conversation_context": prepared.conversation_context,
            "user_memory_recall": {
                "candidate_ids": [memory.id for memory in (prepared.recalled_memories or [])],
                "selected_ids": prepared.selected_memory_ids or [],
            },
            "turn_planner": prepared.turn_plan,
            "query_understanding": {
                "duration_ms": prepared.query_understanding_duration_ms,
                "trace": prepared.query_trace,
                "objective_background": prepared.objective_background,
            },
            "retrieval": serialize_retrieve_result(prepared.retrieve_result, prepared.retrieval_duration_ms),
            "writer": {
                "variant": prepared.writer_prompt,
                "persona_pack_id": prepared.persona_pack_id,
                "persona_pack_sha256": prepared.persona_pack_sha256,
                "persona_pack_claim_count": prepared.persona_pack_claim_count,
                "duration_ms": prepared.writer_build_duration_ms,
                "context_parents": [
                    {"rank": hit.rank, "parent_id": hit.parent_id, "title": hit.title}
                    for hit in prepared.retrieve_result.parents
                ],
                "messages": [
                    {"role": message.get("role", ""), "characters": len(message.get("content", ""))}
                    for message in prepared.messages
                ],
                "total_characters": sum(len(message.get("content", "")) for message in prepared.messages),
            },
            "generation": {
                "provider": type(self._get_llm()).__name__ if self._llm is not None else "DeepSeekJsonClient",
                "model": str(getattr(self._llm, "model", "")) if self._llm is not None else "",
                "temperature": self.config.temperature,
                "max_tokens": prepared.response_max_tokens or self.config.max_tokens,
                "duration_ms": prepared.generation_duration_ms,
                "time_to_first_token_ms": prepared.generation_ttft_ms,
                "usage": prepared.generation_usage,
                "answer_characters": len(answer) if answer is not None else 0,
            },
            "memory_update": prepared.memory_update,
            "timing": {"total_duration_ms": elapsed_ms(prepared.trace_started_at)},
        }
        if error is not None:
            payload["error"] = trace_error(error)
        if prepared.trace_capture == "full":
            payload["writer"]["full_messages"] = prepared.messages
            payload["retrieval"]["full_parent_context"] = [
                {"rank": hit.rank, "parent_id": hit.parent_id, "parent": hit.parent}
                for hit in prepared.retrieve_result.parents
            ]
        return write_trace(
            self.config.data_dir,
            prepared.author,
            prepared.trace_id,
            payload,
            retention=self.config.trace_retention,
        )

    def _write_prepare_failure_trace(
        self,
        *,
        author: str,
        session_id: str,
        trace_id: str,
        created_at: str,
        query: str,
        query_mode: str,
        writer_prompt: str,
        parent_top_k: int,
        error: Exception,
    ) -> Path:
        payload = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_id": trace_id,
            "status": "failed",
            "created_at": created_at,
            "updated_at": utc_now(),
            "input": {
                "author": author,
                "session_id": session_id,
                "query": query,
                "query_mode": query_mode,
                "writer_prompt": writer_prompt,
                "retrieval_parameters": {"parent_top_k": parent_top_k},
            },
            "query_understanding": None,
            "conversation_context": None,
            "turn_planner": None,
            "capture": {"mode": "summary", "retention": self.config.trace_retention},
            "stages": [],
            "retrieval": None,
            "writer": None,
            "generation": None,
            "memory_update": None,
            "error": trace_error(error),
        }
        return write_trace(
            self.config.data_dir,
            author,
            trace_id,
            payload,
            retention=self.config.trace_retention,
        )

    def _stage(
        self,
        stage_id: str,
        label: str,
        started_at: float,
        *,
        status: str = "completed",
        details: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return new_stage(
            stage_id=stage_id,
            label=label,
            started_at=0.0,
            duration_ms=elapsed_ms(started_at),
            status=status,
            details=details,
            usage=usage,
        )

    def _usage_or_estimate(self, llm: Any, *texts: str) -> dict[str, Any]:
        return provider_usage(getattr(llm, "last_usage", None)) or estimated_usage_for_text(*texts)

    def _retrieval_stages(self, result: RetrieveResult, started_at: float) -> list[dict[str, Any]]:
        timing = result.timing
        stages: list[dict[str, Any]] = []
        for key, duration_ms in timing.items():
            if key.endswith(":embedding") or key == "embedding":
                label = "编码检索问题"
            elif key.endswith(":dense") or key == "dense":
                label = "Dense 向量召回"
            elif key.endswith(":sparse") or key == "sparse":
                label = "Sparse 关键词召回"
            elif key.endswith("parent_rrf") or key == "parent_aggregation":
                label = "Parent RRF 聚合"
            elif key == "parent_load":
                label = "加载最终 Parent 全文"
            else:
                label = key
            stages.append(
                new_stage(
                    stage_id=f"retrieval:{key}",
                    label=label,
                    started_at=0.0,
                    duration_ms=duration_ms,
                    details={"metric": key},
                )
            )
        return stages

    def _finalized_stages(
        self,
        prepared: PreparedChat,
        *,
        status: str,
        answer: str | None,
        error: Exception | None,
    ) -> list[dict[str, Any]]:
        stages = list(prepared.stages or [])
        if prepared.generation_started_at is not None:
            generation_status = "failed" if error is not None else "completed" if status == "completed" else "running"
            details: dict[str, Any] = {
                "time_to_first_token_ms": prepared.generation_ttft_ms,
                "answer_characters": len(answer or ""),
            }
            if prepared.generation_duration_ms > 0 and answer is not None:
                details["characters_per_second"] = round(len(answer) / (prepared.generation_duration_ms / 1000), 1)
            if error is not None:
                details["error"] = trace_error(error)
            stages.append(
                new_stage(
                    stage_id="generation",
                    label="流式生成回答",
                    started_at=0.0,
                    duration_ms=prepared.generation_duration_ms,
                    status=generation_status,
                    details=details,
                    usage=prepared.generation_usage,
                )
            )
        offset_ms = 0
        for index, stage in enumerate(stages, start=1):
            stage["order"] = index
            stage["started_offset_ms"] = offset_ms
            offset_ms += int(stage.get("duration_ms") or 0)
        return stages


def _empty_conversation_context() -> SelectedConversationContext:
    return SelectedConversationContext(
        summary={},
        summary_version=0,
        summary_through_sequence=0,
        recent_turns=[],
        relevant_turns=[],
        selected_turns=[],
        history_matches=[],
        used_full_short_history=True,
    )


def _fallback_to_new_retrieval(plan: TurnPlan) -> TurnPlan:
    return TurnPlan(
        turn_type=plan.turn_type,
        resolved_question=plan.resolved_question,
        retrieval_policy="new",
        evidence_source_turn_id=None,
        needs_web=plan.needs_web,
        search_queries=plan.search_queries,
        response_depth=plan.response_depth,
        clarification_focus=plan.clarification_focus,
        memory_ids=plan.memory_ids,
    )


def _empty_retrieve_result(*, query: str, author: str, child_top_k: int) -> RetrieveResult:
    return RetrieveResult(
        query=query,
        collection_name=f"zhihu__{author}",
        child_top_k=child_top_k,
        parent_top_k=0,
        routes={},
        parents=[],
        retrieval_queries=[],
        timing={},
    )


def _retrieve_result_from_parent_ids(
    *,
    query: str,
    author: str,
    parent_ids: list[str],
    parents_path: Path,
    child_top_k: int,
) -> RetrieveResult:
    records = load_parents_by_id(parents_path, set(parent_ids))
    parent_hits: list[ParentHit] = []
    for rank, parent_id in enumerate(parent_ids, start=1):
        parent = records.get(parent_id)
        if parent is None:
            continue
        parent_hits.append(
            ParentHit(
                rank=rank,
                parent_id=parent_id,
                score=1.0 / (60 + rank),
                title=str(parent.get("title") or parent_id),
                path=str(parent.get("path") or ""),
                first_hits=[],
                parent=parent,
            )
        )
    return RetrieveResult(
        query=query,
        collection_name=f"zhihu__{author}",
        child_top_k=child_top_k,
        parent_top_k=len(parent_hits),
        routes={},
        parents=parent_hits,
        retrieval_queries=[],
        timing={"parent_load": 0},
    )


def _multiturn_query_trace(
    *,
    query: str,
    plan: TurnPlan,
    search_results: list[SearchResult],
    objective_background: str,
    retrieval_queries: list[RetrievalQuery],
    grounding_error: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "original_query": query,
        "resolved_question": plan.resolved_question,
        "turn_plan": plan.to_dict(),
        "search_plan": {
            "needs_web": plan.needs_web,
            "search_queries": plan.search_queries,
        },
        "search_results": [
            {"query": item.query, "title": item.title, "url": item.url}
            for item in search_results
        ],
        "objective_background": objective_background,
        "retrieval_queries": [
            {"route": item.route, "query": item.query}
            for item in retrieval_queries
        ],
    }
    if grounding_error is not None:
        payload["web_grounding_error"] = grounding_error
    return payload


def response_token_limit(
    depth: str,
    parent_hits: list[ParentHit],
    *,
    configured_max: int,
) -> int:
    configured_max = max(128, configured_max)
    parent_token_estimates = [
        int(estimated_usage_for_text(str((hit.parent or {}).get("text") or "")).get("estimated_tokens", 0))
        for hit in parent_hits[:5]
        if str((hit.parent or {}).get("text") or "").strip()
    ]
    typical = int(median(parent_token_estimates)) if parent_token_estimates else min(900, configured_max)
    if depth == "brief":
        return min(configured_max, max(192, min(512, round(typical * 0.55))))
    if depth == "deep":
        return min(configured_max, max(900, round(typical * 1.35)))
    return min(configured_max, max(600, typical))


def list_local_personas(data_dir: Path = Path("data")) -> list[LocalPersona]:
    root = data_dir / "authors" / "zhihu"
    if not root.exists():
        return []
    personas: list[LocalPersona] = []
    for author_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        index_dir = author_dir / "index"
        qdrant_path = index_dir / "qdrant"
        if (index_dir / "parents.jsonl").exists() and qdrant_path.exists():
            profile = load_persona_profile(author_dir)
            personas.append(
                LocalPersona(
                    author=author_dir.name,
                    source="zhihu",
                    display_name=str(profile.get("display_name") or profile.get("nickname") or author_dir.name),
                    avatar_url=local_persona_avatar_url(author_dir) or profile.get("avatar_url"),
                    headline=str(profile.get("headline") or ""),
                    content_count=count_parents(index_dir),
                    persona_pack_available=(
                        (author_dir / "persona_pack.json").exists()
                        or (index_dir / "persona_pack.json").exists()
                    ),
                    profile_url=profile.get("profile_url"),
                    last_synced_at=indexed_at(index_dir),
                    author_dir=author_dir,
                    index_dir=index_dir,
                    qdrant_path=qdrant_path,
                )
            )
    return personas


def sources_from_parent_hits(parent_hits: list[ParentHit]) -> list[dict[str, Any]]:
    return [
        {
            "rank": hit.rank,
            "parent_id": hit.parent_id,
            "score": hit.score,
            "title": hit.title,
            "path": hit.path,
            "url": public_source_url(hit.parent),
            "first_hits": [
                {
                    "rank": child.rank,
                    "score": child.score,
                    "node_id": child.node_id,
                    "node_type": child.node_type,
                    "route": child.route,
                }
                for child in hit.first_hits
            ],
        }
        for hit in parent_hits
    ]


def serialize_retrieve_result(result: RetrieveResult, duration_ms: int) -> dict[str, Any]:
    return {
        "duration_ms": duration_ms,
        "timing": result.timing,
        "query": result.query,
        "collection_name": result.collection_name,
        "child_top_k": result.child_top_k,
        "parent_top_k": result.parent_top_k,
        "retrieval_queries": [
            {"route": item.route, "query": item.query}
            for item in result.retrieval_queries
        ],
        "routes": {
            route: [serialize_child_hit(hit) for hit in hits]
            for route, hits in result.routes.items()
        },
        "parents": [serialize_parent_hit(hit) for hit in result.parents],
    }


def serialize_child_hit(hit: Any) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "score": hit.score,
        "node_id": hit.node_id,
        "parent_id": hit.parent_id,
        "node_type": hit.node_type,
        "title": hit.title,
        "path": hit.path,
        "route": hit.route,
    }


def serialize_parent_hit(hit: ParentHit) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "score": hit.score,
        "parent_id": hit.parent_id,
        "title": hit.title,
        "path": hit.path,
        "url": public_source_url(hit.parent),
        "first_hits": [serialize_child_hit(child) for child in hit.first_hits],
    }


def public_source_url(parent: dict[str, Any] | None) -> str | None:
    """Return a human-facing source URL instead of an ingestion API URL."""

    if not parent:
        return None
    kind = str(parent.get("kind") or "")
    source_id = str(parent.get("source_id") or "")
    metadata = parent.get("metadata") if isinstance(parent.get("metadata"), dict) else {}
    if kind == "answer" and source_id:
        question_id = str(metadata.get("question_id") or "")
        if question_id:
            return f"https://www.zhihu.com/question/{question_id}/answer/{source_id}"
    if kind == "article" and source_id:
        return f"https://zhuanlan.zhihu.com/p/{source_id}"
    if kind == "pin" and source_id:
        return f"https://www.zhihu.com/pins/{source_id}"
    url = str(parent.get("url") or "").strip()
    if url.startswith("http://"):
        return "https://" + url.removeprefix("http://")
    return url or None


def enrich_session_source_urls(payload: dict[str, Any], parents_path: Path) -> None:
    """Add public URLs to old session sources without rewriting session files."""

    messages = payload.get("messages")
    if not isinstance(messages, list) or not parents_path.exists():
        return
    missing_ids = {
        str(source.get("parent_id"))
        for message in messages
        if isinstance(message, dict)
        for source in (message.get("sources") or [])
        if isinstance(source, dict) and not source.get("url") and source.get("parent_id")
    }
    if not missing_ids:
        return
    parents = load_parents_by_id(parents_path, missing_ids)
    for message in messages:
        if not isinstance(message, dict):
            continue
        for source in message.get("sources") or []:
            if not isinstance(source, dict) or source.get("url"):
                continue
            source["url"] = public_source_url(parents.get(str(source.get("parent_id") or "")))


def enrich_trace_source_urls(payload: dict[str, Any], parents_path: Path) -> None:
    retrieval = payload.get("retrieval")
    parents = retrieval.get("parents") if isinstance(retrieval, dict) else None
    if not isinstance(parents, list) or not parents_path.exists():
        return
    missing_ids = {
        str(parent.get("parent_id"))
        for parent in parents
        if isinstance(parent, dict) and not parent.get("url") and parent.get("parent_id")
    }
    parent_records = load_parents_by_id(parents_path, missing_ids)
    for parent in parents:
        if not isinstance(parent, dict) or parent.get("url"):
            continue
        parent["url"] = public_source_url(parent_records.get(str(parent.get("parent_id") or "")))


def load_parents_by_id(parents_path: Path, parent_ids: set[str]) -> dict[str, dict[str, Any]]:
    parents: dict[str, dict[str, Any]] = {}
    if not parent_ids:
        return parents
    with parents_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parent = json.loads(line)
            parent_id = str(parent.get("doc_id") or "")
            if parent_id in parent_ids:
                parents[parent_id] = parent
    return parents


def load_persona_profile(author_dir: Path) -> dict[str, Any]:
    for path in [author_dir / "profile.json", author_dir / "raw" / "profile.json"]:
        if path.exists():
            return read_json(path)
    return {}


def count_parents(index_dir: Path) -> int | None:
    path = index_dir / "parents.jsonl"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def local_persona_avatar_url(author_dir: Path) -> str | None:
    assets_dir = author_dir / "raw" / "assets"
    if not assets_dir.exists():
        return None
    if any(path.is_file() for path in assets_dir.glob("avatar.*")):
        return f"/api/personas/{author_dir.name}/avatar"
    return None


def indexed_at(index_dir: Path) -> str | None:
    for name, field in (("qdrant_manifest.json", "indexed_at"), ("build_manifest.json", "built_at")):
        path = index_dir / name
        if path.exists():
            value = read_json(path).get(field)
            if value:
                return str(value)
    return None


def session_dir(data_dir: Path, author: str) -> Path:
    return data_dir / "authors" / "zhihu" / author / "sessions"


def session_path(data_dir: Path, author: str, session_id: str) -> Path:
    safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"})
    if not safe_id:
        safe_id = new_session_id()
    return session_dir(data_dir, author) / f"{safe_id}.json"


def suggestions_path(data_dir: Path, author: str) -> Path:
    return data_dir / "authors" / "zhihu" / author / "profile_suggestions.json"


def new_session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid4().hex[:8]}"


def session_title(query: str) -> str:
    title = " ".join(query.strip().split())
    return title[:32] if title else "未命名对话"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def trace_error(error: Exception) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error)[:1000],
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
