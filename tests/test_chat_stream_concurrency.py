from __future__ import annotations

import asyncio
import json
from time import perf_counter

from fastapi.testclient import TestClient

from personaforge.ingest.retrieve import RetrieveResult
from personaforge.web.app import create_app
from personaforge.web.async_streaming import ChatConcurrencyLimiter, async_chat_stream_events
from personaforge.web.chat_tasks import ChatTaskManager
from personaforge.web.conversations import ConversationStore
from personaforge.web.schemas import ChatStreamRequest
from personaforge.web.service import ChatProgress, PersonaChatService, PreparedChat, WebConfig


class TimedAsyncChatService:
    def __init__(
        self,
        store: ConversationStore,
        *,
        prepare_delay: float = 0.04,
        token_delay: float = 0.04,
        failing_author: str | None = None,
    ) -> None:
        self.conversations = store
        self.config = WebConfig(data_dir=store.data_dir)
        self.prepare_delay = prepare_delay
        self.token_delay = token_delay
        self.failing_author = failing_author
        self.closed: dict[str, asyncio.Event] = {}
        self.failed_traces: list[tuple[str, str]] = []

    async def aiter_prepare_chat(self, **kwargs):
        await asyncio.sleep(self.prepare_delay)
        yield ChatProgress(stage="retrieval", label="正在检索历史表达")
        author = kwargs["author"]
        yield PreparedChat(
            session_id=kwargs["session_id"],
            author=author,
            query=kwargs["query"],
            query_mode=kwargs["query_mode"],
            writer_prompt=kwargs["writer_prompt"],
            objective_background="",
            query_trace=None,
            retrieve_result=RetrieveResult(
                query=kwargs["query"],
                collection_name=f"personaforge__zhihu__{author}",
                child_top_k=100,
                parent_top_k=kwargs["parent_top_k"],
                routes={},
                parents=[],
                retrieval_queries=[],
            ),
            messages=[{"role": "user", "content": kwargs["query"]}],
            trace_id=f"trace-{author}",
            turn_id=kwargs["turn_id"],
        )

    async def astream_answer(self, prepared: PreparedChat):
        closed = self.closed.setdefault(prepared.author, asyncio.Event())
        try:
            for token in (f"{prepared.author}-A", f"{prepared.author}-B", f"{prepared.author}-C"):
                await asyncio.sleep(self.token_delay)
                if prepared.author == self.failing_author and token.endswith("B"):
                    raise RuntimeError(f"{prepared.author} failed")
                yield token
        finally:
            closed.set()

    def save_turn(self, prepared: PreparedChat, answer: str, sources):
        self.conversations.complete_turn(
            prepared.turn_id,
            answer=answer,
            sources=sources,
            trace_id=prepared.trace_id,
        )

    def complete_trace(self, _prepared: PreparedChat, _answer: str):
        return None

    def fail_trace(self, prepared: PreparedChat, error: Exception):
        self.failed_traces.append((prepared.trace_id, str(error)))


def _request(author: str) -> ChatStreamRequest:
    return ChatStreamRequest(
        author=author,
        query=f"question for {author}",
        query_mode="raw",
        writer_prompt="strong_identity",
        parent_top_k=20,
    )


def _turn(store: ConversationStore, request: ChatStreamRequest):
    return store.create_turn(
        author=request.author,
        conversation_id=None,
        query=request.query,
        query_mode=request.query_mode,
        writer_prompt=request.writer_prompt,
        parent_top_k=request.parent_top_k,
        trace_capture=request.trace_capture,
    )


def _decode_event(chunk: str) -> tuple[str, dict[str, object]]:
    lines = chunk.strip().splitlines()
    event = lines[0].split(":", 1)[1].strip()
    payload = json.loads(lines[1].split(":", 1)[1].strip())
    return event, payload


async def _measure(
    service: TimedAsyncChatService,
    store: ConversationStore,
    request: ChatStreamRequest,
    semaphore: asyncio.Semaphore,
) -> dict[str, object]:
    turn = _turn(store, request)
    started_at = perf_counter()
    first_token_at: float | None = None
    completed_at: float | None = None
    answer = ""
    trace_id = ""
    events: list[str] = []
    async for chunk in async_chat_stream_events(
        service,  # type: ignore[arg-type]
        store,
        turn,
        request,
        concurrency_limit=semaphore,
        token_flush_characters=1,
    ):
        event, payload = _decode_event(chunk)
        events.append(event)
        if event == "token":
            first_token_at = first_token_at or perf_counter()
            answer += str(payload["text"])
        elif event == "done":
            completed_at = perf_counter()
            trace_id = str(payload["trace_id"])
    assert first_token_at is not None
    assert completed_at is not None
    assert events[-1] == "done"
    return {
        "author": request.author,
        "started_at": started_at,
        "first_token_at": first_token_at,
        "completed_at": completed_at,
        "characters": len(answer),
        "answer": answer,
        "trace_id": trace_id,
        "events": events,
    }


def test_two_authors_are_faster_concurrently_and_do_not_cross_streams(tmp_path) -> None:
    async def scenario() -> tuple[list[dict[str, object]], list[dict[str, object]], float, float]:
        store = ConversationStore(tmp_path)
        service = TimedAsyncChatService(store)

        serial_started = perf_counter()
        serial = []
        for author in ("alice", "bob"):
            serial.append(await _measure(service, store, _request(author), asyncio.Semaphore(2)))
        serial_total = perf_counter() - serial_started

        concurrent_started = perf_counter()
        shared_limit = asyncio.Semaphore(2)
        concurrent = await asyncio.gather(
            _measure(service, store, _request("alice"), shared_limit),
            _measure(service, store, _request("bob"), shared_limit),
        )
        concurrent_total = perf_counter() - concurrent_started
        return serial, concurrent, serial_total, concurrent_total

    serial, concurrent, serial_total, concurrent_total = asyncio.run(scenario())
    saved_ratio = 1 - concurrent_total / serial_total

    assert concurrent_total < serial_total * 0.75
    assert saved_ratio > 0.25
    for row in [*serial, *concurrent]:
        author = str(row["author"])
        assert row["answer"] == f"{author}-A{author}-B{author}-C"
        assert row["trace_id"] == f"trace-{author}"
        assert row["characters"] == len(str(row["answer"]))
        assert row["events"][-1] == "done"

    print(
        json.dumps(
            {
                "serial": serial,
                "concurrent": concurrent,
                "serial_total_seconds": serial_total,
                "concurrent_total_seconds": concurrent_total,
                "saved_ratio": saved_ratio,
            },
            ensure_ascii=False,
            default=str,
        )
    )


def test_one_author_failure_does_not_stop_the_other(tmp_path) -> None:
    async def scenario():
        store = ConversationStore(tmp_path)
        service = TimedAsyncChatService(store, failing_author="alice")
        limit = asyncio.Semaphore(2)

        async def collect(author: str):
            request = _request(author)
            turn = _turn(store, request)
            events = []
            async for chunk in async_chat_stream_events(
                service,  # type: ignore[arg-type]
                store,
                turn,
                request,
                concurrency_limit=limit,
                token_flush_characters=1,
            ):
                events.append(_decode_event(chunk))
            return turn, events

        return service, await asyncio.gather(collect("alice"), collect("bob"))

    service, results = asyncio.run(scenario())
    by_author = {turn.author: (turn, events) for turn, events in results}
    assert by_author["alice"][1][-1][0] == "error"
    assert by_author["bob"][1][-1][0] == "done"
    assert "trace-alice" in {trace_id for trace_id, _ in service.failed_traces}


def test_client_cancellation_closes_upstream_and_interrupts_turn(tmp_path) -> None:
    async def scenario():
        store = ConversationStore(tmp_path)
        service = TimedAsyncChatService(store, token_delay=0.2)
        request = _request("alice")
        turn = _turn(store, request)
        first_token = asyncio.Event()

        async def consume() -> None:
            async for chunk in async_chat_stream_events(
                service,  # type: ignore[arg-type]
                store,
                turn,
                request,
                concurrency_limit=asyncio.Semaphore(2),
                token_flush_characters=1,
            ):
                event, _payload = _decode_event(chunk)
                if event == "token":
                    first_token.set()

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(first_token.wait(), timeout=2)
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
        await asyncio.wait_for(service.closed["alice"].wait(), timeout=2)
        return store.get_turn(turn.id), service

    interrupted, service = asyncio.run(scenario())
    assert interrupted.status == "interrupted"
    assert interrupted.partial_answer == "alice-A"
    assert service.closed["alice"].is_set()
    assert service.failed_traces == [("trace-alice", "Chat stream client disconnected")]


def test_chat_concurrency_limit_comes_from_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PERSONAFORGE_CHAT_MAX_CONCURRENCY", "5")
    assert WebConfig(data_dir=tmp_path).chat_max_concurrency == 5


def test_cancelled_request_does_not_leak_a_queued_concurrency_slot() -> None:
    async def scenario() -> bool:
        limiter = ChatConcurrencyLimiter(1)
        entered = asyncio.Event()

        with limiter.sync_slot():
            async def wait_for_slot() -> None:
                async with limiter.async_slot():
                    entered.set()

            queued = asyncio.create_task(wait_for_slot())
            await asyncio.sleep(0.05)
            assert not entered.is_set()
            queued.cancel()
            try:
                await queued
            except asyncio.CancelledError:
                pass

        async with limiter.async_slot():
            return True

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=2)) is True


def test_http_chat_stream_keeps_request_and_core_sse_contract(tmp_path) -> None:
    class NoopEncoder:
        def encode_texts(self, _texts, *, batch_size=12):
            return []

    config = WebConfig(data_dir=tmp_path, auth_required=False, chat_max_concurrency=2)
    service = PersonaChatService(config, encoder=NoopEncoder())

    async def prepare(**kwargs):
        yield PreparedChat(
            session_id=kwargs["session_id"],
            author=kwargs["author"],
            query=kwargs["query"],
            query_mode=kwargs["query_mode"],
            writer_prompt=kwargs["writer_prompt"],
            objective_background="",
            query_trace=None,
            retrieve_result=RetrieveResult(
                query=kwargs["query"],
                collection_name="personaforge__zhihu__alice",
                child_top_k=100,
                parent_top_k=kwargs["parent_top_k"],
                routes={},
                parents=[],
                retrieval_queries=[],
            ),
            messages=[],
            trace_id="trace-http-alice",
            turn_id=kwargs["turn_id"],
        )

    async def answer(_prepared):
        yield "HTTP"
        yield "回答"

    service.aiter_prepare_chat = prepare  # type: ignore[method-assign]
    service.astream_answer = answer  # type: ignore[method-assign]
    service.complete_trace = lambda *_args: None  # type: ignore[method-assign]
    service.fail_trace = lambda *_args: None  # type: ignore[method-assign]
    manager = ChatTaskManager(service, worker_count=1)
    app = create_app(
        config,
        service=service,
        chat_manager=manager,
        startup_report={"status": "ready", "checks": []},
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/stream",
            json={
                "author": "alice",
                "query": "问题",
                "query_mode": "raw",
                "writer_prompt": "strong_identity",
                "parent_top_k": 20,
            },
        )

    assert response.status_code == 200
    events = [part for part in response.text.split("\n\n") if part.strip()]
    decoded = [_decode_event(part) for part in events]
    core = [event for event, _ in decoded if event in {"meta", "token", "error", "done"}]
    assert core == ["meta", "token", "token", "done"]
    done = next(payload for event, payload in decoded if event == "done")
    assert done["answer"] == "HTTP回答"
    assert done["trace_id"] == "trace-http-alice"
