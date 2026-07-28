"""FastAPI application for the local PersonaForge Web UI."""

from __future__ import annotations

import traceback
import mimetypes
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from personaforge.web.schemas import (
    ChatSession,
    ChatStreamRequest,
    PersonaInfo,
    PersonasResponse,
    SuggestionsResponse,
    SessionsResponse,
)
from personaforge.web.service import ChatProgress, PreparedChat, PersonaChatService, WebConfig, sources_from_parent_hits
from personaforge.web.streaming import sse_event


def create_app(config: WebConfig | None = None, *, service: PersonaChatService | None = None) -> FastAPI:
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    config = config or WebConfig()
    service = service or PersonaChatService(config)
    app = FastAPI(title="PersonaForge", version="0.1.0")
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/personas", response_model=PersonasResponse)
    def personas() -> PersonasResponse:
        items = [
            PersonaInfo(
                author=item.author,
                source=item.source,
                index_dir=str(item.index_dir),
                display_name=item.display_name,
                avatar_url=item.avatar_url,
                headline=item.headline,
                content_count=item.content_count,
                persona_pack_available=item.persona_pack_available,
            )
            for item in service.list_personas()
        ]
        return PersonasResponse(personas=items, default_author=service.default_author())

    @app.get("/api/personas/{author}/sessions", response_model=SessionsResponse)
    def sessions(author: str) -> SessionsResponse:
        return SessionsResponse(sessions=service.list_sessions(author))

    @app.get("/api/personas/{author}/suggestions", response_model=SuggestionsResponse)
    def suggestions(author: str) -> SuggestionsResponse:
        return SuggestionsResponse(suggestions=service.list_suggestions(author))

    @app.get("/api/personas/{author}/sessions/{session_id}", response_model=ChatSession)
    def session(author: str, session_id: str) -> ChatSession | JSONResponse:
        try:
            return ChatSession(**service.get_session(author, session_id))
        except FileNotFoundError:
            return JSONResponse({"error": "Session not found"}, status_code=404)

    @app.delete("/api/personas/{author}/sessions/{session_id}")
    def delete_session(author: str, session_id: str) -> dict[str, str]:
        service.delete_session(author, session_id)
        return {"status": "ok"}

    @app.get("/api/personas/{author}/traces/{trace_id}", response_model=None)
    def trace(author: str, trace_id: str) -> dict[str, Any] | JSONResponse:
        try:
            return service.get_trace(author, trace_id)
        except FileNotFoundError:
            return JSONResponse({"error": "Trace not found"}, status_code=404)
        except ValueError:
            return JSONResponse({"error": "Trace is invalid"}, status_code=400)

    @app.post("/api/chat/stream")
    def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
        return StreamingResponse(
            _chat_stream_events(service, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    static_dir = _frontend_dist_dir()
    if static_dir.exists():
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

        @app.get("/{path:path}", response_model=None)
        def spa_fallback(path: str):
            if path.startswith("api/"):
                return JSONResponse({"error": "Not found"}, status_code=404)
            return FileResponse(static_dir / "index.html")

    return app


def _chat_stream_events(service: PersonaChatService, request: ChatStreamRequest) -> Iterator[str]:
    answer_parts: list[str] = []
    prepared: PreparedChat | None = None
    try:
        for item in service.iter_prepare_chat(
            author=request.author,
            session_id=request.session_id,
            query=request.query,
            query_mode=request.query_mode,
            writer_prompt=request.writer_prompt,
            parent_top_k=request.parent_top_k,
            trace_capture=request.trace_capture,
        ):
            if isinstance(item, ChatProgress):
                yield sse_event("status", {"stage": item.stage, "label": item.label})
                continue
            prepared = item
        if prepared is None:  # pragma: no cover - defensive invariant.
            raise RuntimeError("Chat preparation finished without a prepared request.")
        yield sse_event(
            "meta",
            {
                "session_id": prepared.session_id,
                "trace_id": prepared.trace_id,
                "author": prepared.author,
                "query_mode": prepared.query_mode,
                "writer_prompt": prepared.writer_prompt,
                "objective_background": prepared.objective_background,
                "query_understanding": prepared.query_trace,
                "retrieval_queries": [
                    {"route": item.route, "query": item.query}
                    for item in prepared.retrieve_result.retrieval_queries
                ],
            },
        )
        yield sse_event("status", {"stage": "generation", "label": "已完成检索，正在生成回答"})
        for token in service.stream_answer(prepared):
            answer_parts.append(token)
            yield sse_event("token", {"text": token})
        answer = "".join(answer_parts)
        sources = sources_from_parent_hits(prepared.retrieve_result.parents)
        service.save_turn(prepared, answer, sources)
        service.complete_trace(prepared, answer)
        yield sse_event(
            "done",
            {
                "session_id": prepared.session_id,
                "trace_id": prepared.trace_id,
                "answer": answer,
                "sources": sources,
            },
        )
    except Exception as exc:  # pragma: no cover - API boundary safety net.
        if prepared is not None:
            service.fail_trace(prepared, exc)
        yield sse_event(
            "error",
            {
                "error": str(exc),
                "traceback": traceback.format_exc(limit=6),
            },
        )


def _frontend_dist_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "web" / "dist"


def run_web(config: WebConfig) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - missing optional dependency.
        raise RuntimeError('Web server requires optional dependencies: pip install -e ".[web]"') from exc

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)
