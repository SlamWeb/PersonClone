from __future__ import annotations

import asyncio
import json
import threading

import httpx

from personaforge.llm import DeepSeekJsonClient
from personaforge.web.service import (
    ChatPreparationCancelled,
    _AsyncLoopBridge,
    _LoopJsonClient,
)


class _FakeResponse:
    is_error = False
    status_code = 200

    content = b""

    def __init__(self, lines, payload=None):
        self.lines = lines
        self.payload = payload

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self):
        for line in self.lines:
            if callable(line):
                line = await line()
            yield line

    def json(self):
        return self.payload


class _FakeStreamContext:
    def __init__(self, response: _FakeResponse, closed: asyncio.Event):
        self.response = response
        self.closed = closed

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *_args):
        self.closed.set()


class _FakeAsyncClient:
    response: _FakeResponse
    response_closed: asyncio.Event

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def stream(self, *_args, **_kwargs):
        return _FakeStreamContext(self.response, self.response_closed)

    async def post(self, *_args, **_kwargs):
        return self.response


def test_native_async_deepseek_json_completion_returns_request_local_usage(monkeypatch) -> None:
    async def scenario():
        _FakeAsyncClient.response = _FakeResponse(
            [],
            payload={
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
        client = DeepSeekJsonClient(api_key="test-only")
        return await client.acomplete_json_with_usage([])

    payload, usage = asyncio.run(scenario())
    assert payload == {"ok": True}
    assert usage is not None and usage.total_tokens == 7


def test_native_async_deepseek_stream_parses_tokens_and_request_local_usage(monkeypatch) -> None:
    async def scenario():
        closed = asyncio.Event()
        usage = []
        _FakeAsyncClient.response_closed = closed
        _FakeAsyncClient.response = _FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"你"}}]}',
                'data: {"choices":[{"delta":{"content":"好"}}]}',
                "data: "
                + json.dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                        },
                    }
                ),
                "data: [DONE]",
            ]
        )
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
        client = DeepSeekJsonClient(api_key="test-only")
        tokens = []
        async for token in client.astream_text_with_usage(
            [{"role": "user", "content": "hello"}],
            on_usage=usage.append,
        ):
            tokens.append(token)
        return tokens, usage, closed

    tokens, usage, closed = asyncio.run(scenario())
    assert tokens == ["你", "好"]
    assert usage[0].total_tokens == 12
    assert closed.is_set()


def test_native_async_deepseek_stream_closes_response_when_cancelled(monkeypatch) -> None:
    async def scenario():
        closed = asyncio.Event()
        first_token = asyncio.Event()
        never = asyncio.Event()

        async def wait_forever():
            await never.wait()
            return "data: [DONE]"

        _FakeAsyncClient.response_closed = closed
        _FakeAsyncClient.response = _FakeResponse(
            ['data: {"choices":[{"delta":{"content":"first"}}]}', wait_forever]
        )
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
        client = DeepSeekJsonClient(api_key="test-only")

        async def consume():
            async for _token in client.astream_text([]):
                first_token.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(first_token.wait(), timeout=1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.wait_for(closed.wait(), timeout=1)
        return closed

    assert asyncio.run(scenario()).is_set()


def test_sync_planner_protocol_is_bridged_to_native_async_http_loop() -> None:
    class AsyncDelegate:
        async def acomplete_json_with_usage(self, _messages, **_kwargs):
            return {"needs_web": False}, None

        def complete_json(self, *_args, **_kwargs):
            raise AssertionError("sync provider path must not be used")

    async def scenario():
        cancelled = threading.Event()
        bridge = _AsyncLoopBridge(asyncio.get_running_loop(), cancelled)
        client = _LoopJsonClient(AsyncDelegate(), bridge)  # type: ignore[arg-type]
        return await asyncio.to_thread(client.complete_json, [])

    assert asyncio.run(scenario()) == {"needs_web": False}


def test_cancelling_preparation_cancels_native_async_planner_call() -> None:
    class WaitingDelegate:
        def __init__(self):
            self.started = asyncio.Event()
            self.closed = asyncio.Event()
            self.never = asyncio.Event()

        async def acomplete_json_with_usage(self, _messages, **_kwargs):
            self.started.set()
            try:
                await self.never.wait()
            finally:
                self.closed.set()

        def complete_json(self, *_args, **_kwargs):
            raise AssertionError("sync provider path must not be used")

    async def scenario():
        delegate = WaitingDelegate()
        cancelled = threading.Event()
        bridge = _AsyncLoopBridge(asyncio.get_running_loop(), cancelled)
        client = _LoopJsonClient(delegate, bridge)  # type: ignore[arg-type]
        worker = asyncio.create_task(asyncio.to_thread(client.complete_json, []))
        await asyncio.wait_for(delegate.started.wait(), timeout=1)
        cancelled.set()
        bridge.cancel_all()
        try:
            await worker
        except ChatPreparationCancelled:
            pass
        await asyncio.wait_for(delegate.closed.wait(), timeout=1)
        return delegate.closed

    assert asyncio.run(scenario()).is_set()
