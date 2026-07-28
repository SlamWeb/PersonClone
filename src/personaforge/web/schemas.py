"""Pydantic schemas for the FastAPI Web API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PersonaInfo(BaseModel):
    author: str
    source: str
    index_dir: str
    display_name: str
    avatar_url: str | None = None
    headline: str = ""
    content_count: int | None = None
    persona_pack_available: bool = False


class PersonasResponse(BaseModel):
    personas: list[PersonaInfo]
    default_author: str | None = None


class SessionSummary(BaseModel):
    id: str
    author: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "error"]
    text: str
    sources: list[dict] | None = None
    trace_id: str | None = None


class ChatSession(BaseModel):
    id: str
    author: str
    title: str
    created_at: str
    updated_at: str
    messages: list[ChatMessage]


class SessionsResponse(BaseModel):
    sessions: list[SessionSummary]


class SuggestionsResponse(BaseModel):
    suggestions: list[str]


class ChatStreamRequest(BaseModel):
    author: str | None = None
    session_id: str | None = None
    query: str = Field(min_length=1)
    query_mode: Literal["raw", "grounded"] = "grounded"
    writer_prompt: Literal["current", "strong_identity", "persona_pack"] = "strong_identity"
    parent_top_k: int = Field(default=20, ge=1, le=40)
    trace_capture: Literal["summary", "full"] = "summary"
