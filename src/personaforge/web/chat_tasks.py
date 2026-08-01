"""Persistent background execution for chat turns."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

from personaforge.web.conversations import ConversationStore, TurnRun
from personaforge.web.multiturn import update_conversation_summary
from personaforge.web.user_memory import update_user_memories
from personaforge.web.service import (
    ChatProgress,
    PersonaChatService,
    PreparedChat,
    sources_from_parent_hits,
    trace_error,
)


class ChatTaskManager:
    """Run persisted chat turns independently from HTTP client connections."""

    def __init__(
        self,
        service: PersonaChatService,
        *,
        store: ConversationStore | None = None,
        worker_count: int = 2,
        token_flush_characters: int = 48,
        token_flush_seconds: float = 0.08,
    ) -> None:
        self.service = service
        self.store = store or service.conversations
        self.worker_count = max(1, worker_count)
        self.token_flush_characters = max(1, token_flush_characters)
        self.token_flush_seconds = max(0.01, token_flush_seconds)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._queued_ids: set[str] = set()
        self._queued_lock = threading.Lock()

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for index in range(self.worker_count):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"personaforge-chat-{index + 1}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        for turn in self.store.list_queued_turns():
            self.enqueue(turn.id)

    def stop(self) -> None:
        self._stop.set()
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads.clear()

    def create_turn(
        self,
        *,
        author: str,
        conversation_id: str | None,
        query: str,
        query_mode: str,
        writer_prompt: str,
        parent_top_k: int,
        trace_capture: str,
        owner_id: str | None = None,
    ) -> TurnRun:
        turn = self.store.create_turn(
            author=author,
            conversation_id=conversation_id,
            query=query,
            query_mode=query_mode,
            writer_prompt=writer_prompt,
            parent_top_k=parent_top_k,
            trace_capture=trace_capture,
            owner_id=owner_id,
        )
        self.enqueue(turn.id)
        return turn

    def retry(self, turn_id: str) -> TurnRun:
        turn = self.store.retry_turn(turn_id)
        self.enqueue(turn.id)
        return turn

    def enqueue(self, turn_id: str) -> None:
        with self._queued_lock:
            if turn_id in self._queued_ids:
                return
            self._queued_ids.add(turn_id)
        self._queue.put(turn_id)

    def run_once(self) -> bool:
        queued = self.store.list_queued_turns(limit=1)
        if not queued:
            return False
        self._run_turn(queued[0].id)
        return True

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                turn_id = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if turn_id is None:
                return
            try:
                self._run_turn(turn_id)
            finally:
                with self._queued_lock:
                    self._queued_ids.discard(turn_id)
                self._queue.task_done()

    def _run_turn(self, turn_id: str) -> None:
        if not self.store.claim_turn(turn_id):
            return
        turn = self.store.get_turn(turn_id)
        prepared: PreparedChat | None = None
        answer_parts: list[str] = []
        buffered_tokens: list[str] = []
        last_flush = time.monotonic()
        try:
            for item in self.service.iter_prepare_chat(
                author=turn.author,
                session_id=turn.conversation_id,
                query=turn.query,
                query_mode=turn.query_mode,
                writer_prompt=turn.writer_prompt,
                parent_top_k=turn.parent_top_k,
                trace_capture=turn.trace_capture,
                turn_id=turn.id,
            ):
                if isinstance(item, ChatProgress):
                    self.store.update_turn_stage(turn.id, item.stage, item.label)
                    continue
                prepared = item
            if prepared is None:
                raise RuntimeError("Chat preparation finished without a prepared request.")

            self.store.append_event(
                turn.id,
                "meta",
                {
                    "session_id": prepared.session_id,
                    "turn_id": turn.id,
                    "trace_id": prepared.trace_id,
                    "author": prepared.author,
                    "query_mode": prepared.query_mode,
                    "writer_prompt": prepared.writer_prompt,
                    "objective_background": prepared.objective_background,
                    "turn_plan": prepared.turn_plan,
                    "retrieval_queries": [
                        {"route": item.route, "query": item.query}
                        for item in prepared.retrieve_result.retrieval_queries
                    ],
                },
            )
            generation_label = "已完成检索，正在生成回答"
            self.store.update_turn_stage(turn.id, "generation", generation_label)

            for token in self.service.stream_answer(prepared):
                answer_parts.append(token)
                buffered_tokens.append(token)
                now = time.monotonic()
                if (
                    sum(len(item) for item in buffered_tokens) >= self.token_flush_characters
                    or now - last_flush >= self.token_flush_seconds
                ):
                    self._flush_tokens(turn.id, answer_parts, buffered_tokens)
                    last_flush = now
            self._flush_tokens(turn.id, answer_parts, buffered_tokens)

            answer = "".join(answer_parts)
            sources = sources_from_parent_hits(prepared.retrieve_result.parents)
            self.service.save_turn(prepared, answer, sources)
            self.service.complete_trace(prepared, answer)
            self.store.append_event(
                turn.id,
                "done",
                {
                    "session_id": prepared.session_id,
                    "turn_id": turn.id,
                    "trace_id": prepared.trace_id,
                    "answer": answer,
                    "sources": sources,
                },
            )
            self._update_memory_after_done(prepared, answer)
        except Exception as exc:
            error = trace_error(exc)
            if prepared is not None:
                self.service.fail_trace(prepared, exc)
            self.store.fail_turn(turn.id, error)
            self.store.append_event(turn.id, "error", {"error": error["message"], "detail": error})

    def _flush_tokens(
        self,
        turn_id: str,
        answer_parts: list[str],
        buffered_tokens: list[str],
    ) -> None:
        if not buffered_tokens:
            return
        chunk = "".join(buffered_tokens)
        buffered_tokens.clear()
        self.store.update_partial_answer(turn_id, "".join(answer_parts))
        self.store.append_event(turn_id, "token", {"text": chunk})

    def _update_memory_after_done(self, prepared: PreparedChat, answer: str) -> None:
        started_at = time.perf_counter()
        summary_payload: dict[str, Any]
        try:
            result = update_conversation_summary(
                self.store,
                prepared.session_id,
                llm=self.service.llm_client(),
            )
            summary_payload = {
                "status": "completed" if result is not None else "skipped",
                "summary_version": int(result["version"]) if result else None,
                "through_sequence": int(result["through_sequence"]) if result else None,
            }
        except Exception as exc:
            summary_payload = {
                "status": "failed",
                "error": trace_error(exc),
            }
        user_memory_started_at = time.perf_counter()
        if not hasattr(self.service, "user_memories"):
            user_memory_payload = {
                "status": "skipped",
                "duration_ms": 0,
                "operations": [],
                "rejections": [],
            }
        else:
            try:
                user_memory_payload = update_user_memories(
                    self.service.user_memories,
                    prepared.owner_id,
                    author=prepared.author,
                    conversation_id=prepared.session_id,
                user_turns=self.store.get_completed_turns(prepared.session_id),
                llm=self.service.llm_client(),
                related_memories=[
                    memory
                    for memory in (prepared.recalled_memories or [])
                    if memory.id in set(prepared.selected_memory_ids or [])
                ],
            )
            except Exception as exc:
                user_memory_payload = {
                    "status": "failed",
                    "duration_ms": round((time.perf_counter() - user_memory_started_at) * 1000),
                    "error": trace_error(exc),
                    "operations": [],
                    "rejections": [],
                }
        statuses = {summary_payload.get("status"), user_memory_payload.get("status")}
        overall_status = "skipped" if statuses == {"skipped"} else "completed"
        if "failed" in statuses and statuses <= {"failed", "skipped"}:
            overall_status = "failed"
        payload: dict[str, Any] = {
            "status": overall_status,
            "duration_ms": round((time.perf_counter() - started_at) * 1000),
            "conversation_summary": summary_payload,
            "user_memory": user_memory_payload,
        }
        try:
            self.service.update_memory_trace(
                author=prepared.author,
                trace_id=prepared.trace_id,
                memory_update=payload,
                answer=answer,
            )
        except Exception:
            # Memory and trace enrichment are derived data. A completed answer
            # must remain completed even when this post-response write fails.
            return
