"""Pydantic schemas for the FastAPI Web API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AuthUserInfo(BaseModel):
    id: str
    username: str
    display_name: str
    role: Literal["admin", "member"]


class AuthStateResponse(BaseModel):
    configured: bool
    authenticated: bool
    user: AuthUserInfo | None = None


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class BootstrapRequest(BaseModel):
    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=8, max_length=256)
    display_name: str | None = Field(default=None, max_length=80)


class AdminUserCreateRequest(BaseModel):
    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=8, max_length=256)
    display_name: str | None = Field(default=None, max_length=80)
    role: Literal["admin", "member"] = "member"


class AdminUserInfo(AuthUserInfo):
    created_at: str


class AdminUsersResponse(BaseModel):
    users: list[AdminUserInfo]


class PersonaInfo(BaseModel):
    author: str
    source: str
    index_dir: str
    display_name: str
    avatar_url: str | None = None
    headline: str = ""
    content_count: int | None = None
    persona_pack_available: bool = False
    narrative_schema_available: bool = False
    profile_url: str | None = None
    last_synced_at: str | None = None


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
    id: str | None = None
    role: Literal["user", "assistant", "error"]
    text: str
    status: str = "completed"
    sources: list[dict] | None = None
    trace_id: str | None = None
    turn_id: str | None = None


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


class AuthorPreviewRequest(BaseModel):
    value: str = Field(min_length=1, max_length=500)


class AuthorPreview(BaseModel):
    author: str
    display_name: str
    avatar_url: str | None = None
    headline: str = ""
    profile_url: str
    exists: bool = False
    ready: bool = False


class AuthorJobCreateRequest(BaseModel):
    author: str = Field(min_length=1, max_length=500)
    kinds: list[Literal["answer", "article", "pin"]] = Field(
        default_factory=lambda: ["answer", "article", "pin"]
    )
    max_items: int | None = Field(default=None, ge=1, le=10000)


class AuthorJobResponse(BaseModel):
    id: str
    source: str
    author_input: str
    author: str
    operation: Literal["create", "sync"]
    status: Literal["queued", "running", "ready", "failed", "cancelled", "interrupted"]
    stage: str
    label: str
    kinds: list[str] | tuple[str, ...]
    max_items: int | None = None
    display_name: str
    avatar_url: str | None = None
    headline: str = ""
    profile_url: str
    item_count: int | None = None
    parent_count: int | None = None
    node_count: int | None = None
    error_message: str | None = None
    cancel_requested: bool = False
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None


class AuthorJobsResponse(BaseModel):
    jobs: list[AuthorJobResponse]


class ChatStreamRequest(BaseModel):
    author: str | None = None
    session_id: str | None = None
    query: str = Field(min_length=1, max_length=4000)
    query_mode: Literal["raw", "grounded"] = "grounded"
    writer_prompt: Literal["current", "strong_identity", "persona_pack", "mrprompt"] = "mrprompt"
    parent_top_k: int = Field(default=20, ge=1, le=40)
    trace_capture: Literal["summary", "full"] = "summary"


class TurnRunResponse(BaseModel):
    id: str
    conversation_id: str
    author: str
    query: str
    status: Literal["queued", "running", "completed", "failed", "interrupted"]
    stage: str
    label: str
    partial_answer: str = ""
    error: dict | None = None
    planner: dict | None = None
    response_depth: str | None = None
    trace_id: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None


class UserMemoryInfo(BaseModel):
    id: str
    kind: str
    memory_key: str
    content: str
    status: str
    pinned: bool
    sensitivity: str
    importance: int
    confidence: float
    event_status: str
    source_author: str | None = None
    source_conversation_id: str | None = None
    source_message_ids: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    supersedes_id: str | None = None
    created_at: str
    updated_at: str
    last_accessed_at: str | None = None
    access_count: int = 0


class UserMemoriesResponse(BaseModel):
    memories: list[UserMemoryInfo]


class UserMemoryPatchRequest(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=1000)
    pinned: bool | None = None


class UserMemorySettingsResponse(BaseModel):
    enabled: bool
    auto_write: bool


class UserMemorySettingsPatchRequest(BaseModel):
    enabled: bool | None = None
    auto_write: bool | None = None


class RetrievalLabelRequest(BaseModel):
    score: int = Field(ge=0, le=2)


class RetrievalEvalJobCreateRequest(BaseModel):
    author: str = Field(min_length=1, max_length=200)
    labeler: Literal["deepseek_api", "codex_handoff", "manual_import"] = "codex_handoff"
    split: Literal["dev", "test"] = "dev"
    budget_cny: float = Field(default=5.0, gt=0, le=1000)


class RetrievalEvalJobResumeRequest(BaseModel):
    budget_cny: float | None = Field(default=None, gt=0, le=1000)


class GenerationRubricRequest(BaseModel):
    scores: dict[str, int | None] = Field(default_factory=dict)
    note: str = Field(default="", max_length=2000)


class GenerationPairwiseRequest(BaseModel):
    choice: Literal["A", "B"]


class GenerationJudgeJobRequest(BaseModel):
    system_id: str = Field(min_length=16, max_length=128)
    repeats: int = Field(default=3, ge=3, le=3)
