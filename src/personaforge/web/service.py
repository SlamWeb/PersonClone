"""Service layer used by the FastAPI Web app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator
from uuid import uuid4

from personaforge.ingest.embeddings import BgeM3Encoder, TextEncoder
from personaforge.ingest.query_understanding import (
    GroundedQueryPlan,
    TavilySearchClient,
    build_background_and_retrieval_queries,
    plan_to_trace,
    plan_web_search,
)
from personaforge.ingest.retrieve import ParentHit, RetrieveResult, retrieve_parents, retrieve_parents_for_queries
from personaforge.llm import DeepSeekJsonClient, JsonChatClient
from personaforge.persona.pack import load_persona_pack_for_index
from personaforge.persona.writer import build_writer_messages
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


@dataclass(frozen=True, slots=True)
class ChatProgress:
    """A user-visible execution stage emitted before the work actually starts."""

    stage: str
    label: str


class PersonaChatService:
    def __init__(
        self,
        config: WebConfig,
        *,
        encoder: TextEncoder | None = None,
        llm: JsonChatClient | None = None,
    ) -> None:
        self.config = config
        self._encoder = encoder
        self._llm = llm

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
    ) -> Iterator[ChatProgress | PreparedChat]:
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
                    max_tokens=self.config.max_tokens,
                    on_usage=receive_usage,
                )
            else:
                stream = llm.stream_text(
                    prepared.messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
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

    def get_trace(self, author: str, trace_id: str) -> dict[str, Any]:
        payload = read_trace(self.config.data_dir, author, trace_id)
        enrich_trace_source_urls(
            payload,
            self.config.data_dir / "authors" / "zhihu" / author / "index" / "parents.jsonl",
        )
        return payload

    def list_sessions(self, author: str) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in sorted(session_dir(self.config.data_dir, author).glob("*.json"), reverse=True):
            try:
                payload = read_json(path)
            except json.JSONDecodeError:
                continue
            messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
            sessions.append(
                {
                    "id": str(payload.get("id") or path.stem),
                    "author": author,
                    "title": str(payload.get("title") or "未命名对话"),
                    "created_at": str(payload.get("created_at") or ""),
                    "updated_at": str(payload.get("updated_at") or ""),
                    "message_count": len(messages),
                }
            )
        return sessions

    def get_session(self, author: str, session_id: str) -> dict[str, Any]:
        path = session_path(self.config.data_dir, author, session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        payload = read_json(path)
        payload.setdefault("id", session_id)
        payload.setdefault("author", author)
        payload.setdefault("title", "未命名对话")
        payload.setdefault("messages", [])
        enrich_session_source_urls(
            payload,
            self.config.data_dir / "authors" / "zhihu" / author / "index" / "parents.jsonl",
        )
        return payload

    def delete_session(self, author: str, session_id: str) -> None:
        path = session_path(self.config.data_dir, author, session_id)
        if path.exists():
            path.unlink()

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
        now = utc_now()
        path = session_path(self.config.data_dir, prepared.author, prepared.session_id)
        if path.exists():
            payload = read_json(path)
        else:
            payload = {
                "id": prepared.session_id,
                "author": prepared.author,
                "title": session_title(prepared.query),
                "created_at": now,
                "messages": [],
            }
        payload["updated_at"] = now
        messages = payload.setdefault("messages", [])
        messages.append({"role": "user", "text": prepared.query})
        messages.append(
            {
                "role": "assistant",
                "text": answer,
                "sources": sources,
                "trace_id": prepared.trace_id or None,
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def _get_encoder(self) -> TextEncoder:
        if self._encoder is None:
            self._encoder = BgeM3Encoder(
                self.config.model_name,
                device=self.config.embedding_device,
                use_fp16=self.config.use_fp16,
            )
        return self._encoder

    def _get_llm(self) -> JsonChatClient:
        if self._llm is None:
            self._llm = DeepSeekJsonClient.from_env()
        return self._llm

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
                "max_tokens": self.config.max_tokens,
                "duration_ms": prepared.generation_duration_ms,
                "time_to_first_token_ms": prepared.generation_ttft_ms,
                "usage": prepared.generation_usage,
                "answer_characters": len(answer) if answer is not None else 0,
            },
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
            "capture": {"mode": "summary", "retention": self.config.trace_retention},
            "stages": [],
            "retrieval": None,
            "writer": None,
            "generation": None,
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
