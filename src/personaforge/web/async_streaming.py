"""Cancellation-safe async SSE orchestration for interactive chat requests."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from personaforge.web.conversations import ConversationStore, TurnRun
from personaforge.web.schemas import ChatStreamRequest
from personaforge.web.service import (
    ChatProgress,
    PersonaChatService,
    PreparedChat,
    sources_from_parent_hits,
    trace_error,
)
from personaforge.web.streaming import sse_event


class ChatStreamDisconnected(RuntimeError):
    """Trace-safe marker for an HTTP client cancellation."""


class ChatConcurrencyLimiter:
    """One process-wide capacity limit shared by async streams and worker threads."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("Chat concurrency capacity must be positive")
        self.capacity = capacity
        self._semaphore = threading.BoundedSemaphore(capacity)

    @contextmanager
    def sync_slot(self):
        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()

    @asynccontextmanager
    async def async_slot(self):
        await self._acquire_async()
        try:
            yield
        finally:
            self._semaphore.release()

    async def _acquire_async(self) -> None:
        loop = asyncio.get_running_loop()
        state = {"cancelled": False, "acquired": False, "claimed": False}
        lock = threading.Lock()

        def wait_for_slot() -> bool:
            while True:
                with lock:
                    if state["cancelled"]:
                        return False
                if not self._semaphore.acquire(timeout=0.05):
                    continue
                with lock:
                    if state["cancelled"]:
                        self._semaphore.release()
                        return False
                    state["acquired"] = True
                return True

        future = loop.run_in_executor(None, wait_for_slot)
        try:
            acquired = await future
            with lock:
                state["claimed"] = acquired
            if not acquired:
                raise asyncio.CancelledError
        except asyncio.CancelledError:
            with lock:
                state["cancelled"] = True
                release = state["acquired"] and not state["claimed"]
                state["claimed"] = True
            if release:
                self._semaphore.release()
            raise


async def async_chat_stream_events(
    service: PersonaChatService,
    store: ConversationStore,
    turn: TurnRun,
    request: ChatStreamRequest,
    *,
    concurrency_limit: Any,
    after_complete: Callable[[PreparedChat, str], None] | None = None,
    token_flush_characters: int = 48,
    token_flush_seconds: float = 0.08,
) -> AsyncIterator[str]:
    """Run one turn and stream request-owned events.

    The semaphore deliberately queues excess requests. Cancellation propagates
    through the native async LLM iterator, whose response context is then closed.
    """

    answer_parts: list[str] = []
    buffered_tokens: list[str] = []
    prepared: PreparedChat | None = None
    completed = False
    last_flush = time.monotonic()

    yield sse_event(
        "accepted",
        {
            "session_id": turn.conversation_id,
            "turn_id": turn.id,
            "status": turn.status,
            "stage": turn.stage,
            "label": turn.label,
        },
    )

    try:
        slot = (
            concurrency_limit.async_slot()
            if hasattr(concurrency_limit, "async_slot")
            else concurrency_limit
        )
        async with slot:
            claimed = await asyncio.to_thread(store.claim_turn, turn.id)
            if not claimed:
                raise RuntimeError("Chat turn is no longer queued")

            async for item in service.aiter_prepare_chat(
                author=request.author,
                session_id=turn.conversation_id,
                query=request.query,
                query_mode=request.query_mode,
                writer_prompt=request.writer_prompt,
                parent_top_k=request.parent_top_k,
                trace_capture=request.trace_capture,
                turn_id=turn.id,
            ):
                if isinstance(item, ChatProgress):
                    await asyncio.to_thread(
                        store.update_turn_stage,
                        turn.id,
                        item.stage,
                        item.label,
                    )
                    yield sse_event("status", {"stage": item.stage, "label": item.label})
                    continue
                prepared = item

            if prepared is None:  # pragma: no cover - defensive invariant.
                raise RuntimeError("Chat preparation finished without a prepared request")

            meta = {
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
            }
            await asyncio.to_thread(store.append_event, turn.id, "meta", meta)
            yield sse_event("meta", meta)

            generation_label = "已完成检索，正在生成回答"
            await asyncio.to_thread(
                store.update_turn_stage,
                turn.id,
                "generation",
                generation_label,
            )
            yield sse_event(
                "status",
                {"stage": "generation", "label": generation_label},
            )

            async for token in service.astream_answer(prepared):
                answer_parts.append(token)
                buffered_tokens.append(token)
                now = time.monotonic()
                if (
                    sum(len(item) for item in buffered_tokens) >= token_flush_characters
                    or now - last_flush >= token_flush_seconds
                ):
                    await _persist_tokens(store, turn.id, answer_parts, buffered_tokens)
                    last_flush = now
                yield sse_event("token", {"text": token})
            await _persist_tokens(store, turn.id, answer_parts, buffered_tokens)

            answer = "".join(answer_parts)
            sources = sources_from_parent_hits(prepared.retrieve_result.parents)
            await asyncio.to_thread(service.save_turn, prepared, answer, sources)
            await asyncio.to_thread(service.complete_trace, prepared, answer)
            completed = True
            done = {
                "session_id": prepared.session_id,
                "turn_id": turn.id,
                "trace_id": prepared.trace_id,
                "answer": answer,
                "sources": sources,
            }
            await asyncio.to_thread(store.append_event, turn.id, "done", done)
            if after_complete is not None:
                after_complete(prepared, answer)
            yield sse_event("done", done)
    except asyncio.CancelledError:
        if not completed:
            await _record_interruption(service, store, turn.id, prepared, answer_parts)
        raise
    except Exception as exc:
        if prepared is not None:
            await asyncio.to_thread(service.fail_trace, prepared, exc)
        error = trace_error(exc)
        await asyncio.to_thread(store.fail_turn, turn.id, error)
        payload = {"error": error["message"], "detail": error, "turn_id": turn.id}
        await asyncio.to_thread(store.append_event, turn.id, "error", payload)
        yield sse_event("error", payload)
    finally:
        if not completed and buffered_tokens:
            try:
                await _persist_tokens(store, turn.id, answer_parts, buffered_tokens)
            except Exception:
                pass


async def _persist_tokens(
    store: ConversationStore,
    turn_id: str,
    answer_parts: list[str],
    buffered_tokens: list[str],
) -> None:
    if not buffered_tokens:
        return
    chunk = "".join(buffered_tokens)
    buffered_tokens.clear()
    answer = "".join(answer_parts)
    await asyncio.to_thread(store.update_partial_answer, turn_id, answer)
    await asyncio.to_thread(store.append_event, turn_id, "token", {"text": chunk})


async def _record_interruption(
    service: PersonaChatService,
    store: ConversationStore,
    turn_id: str,
    prepared: PreparedChat | None,
    answer_parts: list[str],
) -> None:
    cancellation = ChatStreamDisconnected("Chat stream client disconnected")
    if prepared is not None:
        try:
            await asyncio.to_thread(service.fail_trace, prepared, cancellation)
        except Exception:
            pass
    try:
        await asyncio.to_thread(
            store.interrupt_turn,
            turn_id,
            reason="客户端已断开，生成已取消",
            partial_answer="".join(answer_parts),
        )
    except Exception:
        pass
