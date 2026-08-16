"""FastAPI application for the local PersonaForge Web UI."""

from __future__ import annotations

import os
import traceback
import mimetypes
import time
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from collections.abc import Iterator
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from personaforge.web.schemas import (
    AdminUserCreateRequest,
    AdminUserInfo,
    AdminUsersResponse,
    AuthStateResponse,
    AuthUserInfo,
    AuthorJobCreateRequest,
    AuthorJobResponse,
    AuthorJobsResponse,
    AuthorPreview,
    AuthorPreviewRequest,
    BootstrapRequest,
    ChatSession,
    ChatStreamRequest,
    LoginRequest,
    PersonaInfo,
    PersonasResponse,
    SuggestionsResponse,
    SessionsResponse,
    TurnRunResponse,
    UserMemoriesResponse,
    UserMemoryInfo,
    UserMemoryPatchRequest,
    UserMemorySettingsPatchRequest,
    UserMemorySettingsResponse,
    RetrievalLabelRequest,
    RetrievalEvalJobCreateRequest,
    RetrievalEvalJobResumeRequest,
    GenerationRubricRequest,
    GenerationPairwiseRequest,
    GenerationJudgeJobRequest,
)
from personaforge.web.author_jobs import (
    AuthorJobConfig,
    AuthorJobManager,
    local_avatar_path,
    resolve_author_preview,
    safe_author_token,
)
from personaforge.web.chat_tasks import ChatTaskManager
from personaforge.web.conversations import ConversationBusyError, TurnRun
from personaforge.web.auth import AuthStore, AuthUser
from personaforge.web.service import ChatProgress, PreparedChat, PersonaChatService, WebConfig, sources_from_parent_hits
from personaforge.web.streaming import sse_event
from personaforge.web.deployment_guard import (
    DeploymentGuard,
    DeploymentGuardConfig,
    DeploymentGuardError,
)
from personaforge.web.startup_checks import run_startup_checks
from personaforge.web.retrieval_evaluation import RetrievalEvaluationStore
from personaforge.web.retrieval_eval_jobs import RetrievalEvalJobConfig, RetrievalEvalJobManager
from personaforge.web.generation_evaluation import (
    GenerationEvaluationStore,
    GenerationJudgeManager,
)
from personaforge.studies.study1_service import (
    Study1Store,
    StudyCodeCreateRequest,
    StudyDemoChatRequest,
    StudyExposureRequest,
    StudyFeedbackRequest,
    StudyNavigateRequest,
    StudyPairwiseRequest,
    StudyPointwiseRequest,
    StudyProfileRequest,
    StudyTransitionRequest,
)


AUTH_COOKIE_NAME = "personaforge_session"


def create_app(
    config: WebConfig | None = None,
    *,
    service: PersonaChatService | None = None,
    job_manager: AuthorJobManager | None = None,
    chat_manager: ChatTaskManager | None = None,
    auth_store: AuthStore | None = None,
    generation_judge_manager: GenerationJudgeManager | None = None,
    retrieval_eval_job_manager: RetrievalEvalJobManager | None = None,
    startup_report: dict[str, object] | None = None,
) -> FastAPI:
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    config = config or WebConfig()
    service = service or PersonaChatService(config)
    job_manager = job_manager or AuthorJobManager(
        AuthorJobConfig(
            data_dir=config.data_dir,
            model_name=config.model_name,
            embedding_device=config.embedding_device,
            use_fp16=config.use_fp16,
            batch_size=config.index_batch_size,
        )
    )
    chat_manager = chat_manager or ChatTaskManager(service)
    auth_store = auth_store or AuthStore(config.data_dir)
    retrieval_evaluations = RetrievalEvaluationStore(config.data_dir)
    generation_evaluations = GenerationEvaluationStore(config.data_dir)
    study1 = Study1Store(config.data_dir)
    study_rate_events: dict[str, deque[float]] = defaultdict(deque)
    study_rate_lock = Lock()
    deployment_guard = DeploymentGuard(
        DeploymentGuardConfig(enabled=config.deployment_guards_enabled)
    )
    startup_report = startup_report or run_startup_checks(
        data_dir=config.data_dir,
        model_name=config.model_name,
    )
    generation_judge_manager = generation_judge_manager or GenerationJudgeManager(
        generation_evaluations
    )
    retrieval_eval_job_manager = retrieval_eval_job_manager or RetrievalEvalJobManager(
        RetrievalEvalJobConfig(
            data_dir=config.data_dir,
            model_name=config.model_name,
            embedding_device=config.embedding_device,
            use_fp16=config.use_fp16,
            working_dir=Path.cwd(),
        )
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        has_personas = bool(service.list_personas())
        if has_personas:
            service.prepare_encoder_runtime()
        job_manager.start()
        chat_manager.start()
        generation_judge_manager.start()
        retrieval_eval_job_manager.start()
        if has_personas:
            service.start_encoder_warmup()
        try:
            yield
        finally:
            retrieval_eval_job_manager.stop()
            generation_judge_manager.stop()
            chat_manager.stop()
            job_manager.stop()

    app = FastAPI(title="PersonaForge", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.state.author_jobs = job_manager
    app.state.chat_tasks = chat_manager
    app.state.auth = auth_store
    app.state.retrieval_evaluations = retrieval_evaluations
    app.state.generation_evaluations = generation_evaluations
    app.state.study1 = study1
    app.state.generation_judge_jobs = generation_judge_manager
    app.state.retrieval_eval_jobs = retrieval_eval_job_manager
    app.state.deployment_guard = deployment_guard
    app.state.startup_report = startup_report
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        return response

    local_user = AuthUser(
        id="local-user",
        username="local-user",
        display_name="本地用户",
        role="admin",
        created_at="",
    )

    def optional_user(request: Request) -> AuthUser | None:
        if not config.auth_required:
            return local_user
        return auth_store.resolve_session(request.cookies.get(AUTH_COOKIE_NAME))

    def current_user(request: Request) -> AuthUser:
        user = optional_user(request)
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Session"},
            )
        return user

    def current_admin(user: AuthUser = Depends(current_user)) -> AuthUser:
        if user.role != "admin":
            raise HTTPException(status_code=403, detail="Administrator access required")
        return user

    @app.get("/api/auth/state", response_model=AuthStateResponse)
    def auth_state(request: Request) -> AuthStateResponse:
        user = optional_user(request)
        return AuthStateResponse(
            configured=not config.auth_required or auth_store.has_users(),
            authenticated=user is not None,
            user=AuthUserInfo(**user.to_api()) if user else None,
        )

    @app.post("/api/auth/bootstrap", response_model=AuthStateResponse)
    def bootstrap_auth(request: Request, payload: BootstrapRequest) -> JSONResponse:
        if not config.auth_required:
            raise HTTPException(status_code=409, detail="Authentication is disabled")
        try:
            user = auth_store.bootstrap_admin(
                username=payload.username,
                password=payload.password,
                display_name=payload.display_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        token = auth_store.create_session(user.id, days=config.session_days)
        response = JSONResponse(
            AuthStateResponse(
                configured=True,
                authenticated=True,
                user=AuthUserInfo(**user.to_api()),
            ).model_dump()
        )
        _set_auth_cookie(response, request, token, config)
        return response

    @app.post("/api/auth/login", response_model=AuthStateResponse)
    def login_auth(request: Request, payload: LoginRequest) -> JSONResponse:
        client_key = _client_key(request)
        if not deployment_guard.allow_login_attempt(client_key):
            raise HTTPException(
                status_code=429,
                detail="登录尝试过多，请稍后再试。",
                headers={"Retry-After": "300"},
            )
        user = auth_store.authenticate(payload.username, payload.password)
        if user is None:
            deployment_guard.record_login_failure(client_key)
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        deployment_guard.clear_login_failures(client_key)
        token = auth_store.create_session(user.id, days=config.session_days)
        response = JSONResponse(
            AuthStateResponse(
                configured=True,
                authenticated=True,
                user=AuthUserInfo(**user.to_api()),
            ).model_dump()
        )
        _set_auth_cookie(response, request, token, config)
        return response

    @app.post("/api/auth/logout")
    def logout_auth(request: Request) -> JSONResponse:
        auth_store.revoke_session(request.cookies.get(AUTH_COOKIE_NAME))
        response = JSONResponse({"status": "ok"})
        response.delete_cookie(AUTH_COOKIE_NAME, path="/")
        return response

    @app.get("/api/admin/users", response_model=AdminUsersResponse)
    def list_admin_users(_admin: AuthUser = Depends(current_admin)) -> AdminUsersResponse:
        return AdminUsersResponse(
            users=[
                AdminUserInfo(**user.to_api(), created_at=user.created_at)
                for user in auth_store.list_users()
            ]
        )

    @app.post("/api/admin/users", response_model=AdminUserInfo, status_code=201)
    def create_admin_user(
        payload: AdminUserCreateRequest,
        _admin: AuthUser = Depends(current_admin),
    ) -> AdminUserInfo:
        try:
            user = auth_store.create_user(
                username=payload.username,
                password=payload.password,
                display_name=payload.display_name,
                role=payload.role,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return AdminUserInfo(**user.to_api(), created_at=user.created_at)

    @app.get("/health")
    def health() -> dict[str, Any]:
        preflight_status = str(startup_report.get("status", "warning"))
        return {
            "status": "ok" if preflight_status == "ready" else "degraded",
            "embedding": service.encoder_status(),
            "preflight": startup_report,
        }

    @app.get("/api/studies/study1")
    def study1_meta() -> dict[str, Any]:
        return study1.public_meta()

    @app.get("/api/studies/study1/studies/{study_id}")
    def study1_meta_for_study(study_id: str) -> dict[str, Any]:
        try:
            return study1.public_meta(study_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def study_client_key(request: Request) -> str:
        # Cloudflare Tunnel passes the original address in this header. For local
        # development it is absent and Starlette's peer address is the fallback.
        forwarded = request.headers.get("CF-Connecting-IP") or request.headers.get(
            "X-Forwarded-For"
        )
        if forwarded:
            return forwarded.split(",", 1)[0].strip()
        return request.client.host if request.client else "unknown"

    def enforce_study_rate_limit(
        request: Request, *, scope: str, maximum: int, window_seconds: float
    ) -> None:
        key = f"{scope}:{study_client_key(request)}"
        now = time.monotonic()
        with study_rate_lock:
            events = study_rate_events[key]
            while events and events[0] <= now - window_seconds:
                events.popleft()
            if len(events) >= maximum:
                raise HTTPException(
                    status_code=429,
                    detail="请求过于频繁，请稍后再试",
                )
            events.append(now)

    def require_study_session(
        session_id: str,
        x_study_session_token: str | None = Header(default=None),
    ) -> None:
        try:
            study1.authorize_session(session_id, x_study_session_token)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study session not found") from exc

    @app.post("/api/studies/study1/sessions")
    def start_study1(payload: StudyProfileRequest, request: Request) -> dict[str, Any]:
        try:
            enforce_study_rate_limit(
                request, scope="study1-start", maximum=30, window_seconds=600
            )
            return study1.start(payload)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/studies/study1/studies/{study_id}/sessions")
    def start_study1_for_study(
        study_id: str, payload: StudyProfileRequest, request: Request
    ) -> dict[str, Any]:
        try:
            enforce_study_rate_limit(
                request,
                scope=f"study1-start:{study_id}",
                maximum=30,
                window_seconds=600,
            )
            return study1.start(payload, study_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study not found") from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studies/study1/sessions/{session_id}")
    def study1_session(
        session_id: str, _access: None = Depends(require_study_session)
    ) -> dict[str, Any]:
        try:
            return study1.state(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study session not found") from exc

    @app.put("/api/studies/study1/sessions/{session_id}/pointwise/{trial_id}")
    def save_study1_pointwise(
        session_id: str,
        trial_id: str,
        payload: StudyPointwiseRequest,
        _access: None = Depends(require_study_session),
    ) -> dict[str, Any]:
        try:
            return study1.save_pointwise(session_id, trial_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study session not found") from exc

    @app.put("/api/studies/study1/sessions/{session_id}/pairwise/{trial_id}")
    def save_study1_pairwise(
        session_id: str,
        trial_id: str,
        payload: StudyPairwiseRequest,
        _access: None = Depends(require_study_session),
    ) -> dict[str, Any]:
        try:
            return study1.save_pairwise(session_id, trial_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study session not found") from exc

    @app.post("/api/studies/study1/sessions/{session_id}/navigate")
    def navigate_study1(
        session_id: str,
        payload: StudyNavigateRequest,
        _access: None = Depends(require_study_session),
    ) -> dict[str, Any]:
        try:
            if payload.direction == "previous":
                return study1.navigate_previous(session_id)
            raise ValueError("不支持的导航方向")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study session not found") from exc

    @app.post("/api/studies/study1/sessions/{session_id}/transition")
    def acknowledge_study1_transition(
        session_id: str,
        payload: StudyTransitionRequest,
        _access: None = Depends(require_study_session),
    ) -> dict[str, Any]:
        try:
            return study1.acknowledge_transition(session_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study session not found") from exc

    @app.put("/api/studies/study1/sessions/{session_id}/exposure")
    def save_study1_exposure(
        session_id: str,
        payload: StudyExposureRequest,
        _access: None = Depends(require_study_session),
    ) -> dict[str, Any]:
        try:
            return study1.save_exposure(session_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study session not found") from exc

    @app.put("/api/studies/study1/sessions/{session_id}/feedback")
    def save_study1_feedback(
        session_id: str,
        payload: StudyFeedbackRequest,
        _access: None = Depends(require_study_session),
    ) -> dict[str, Any]:
        try:
            return study1.save_feedback(session_id, payload.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study session not found") from exc

    @app.post("/api/studies/study1/sessions/{session_id}/demo/chat")
    def study1_demo_chat(
        session_id: str,
        payload: StudyDemoChatRequest,
        request: Request,
        _access: None = Depends(require_study_session),
    ) -> StreamingResponse:
        reservation: dict[str, str] | None = None
        try:
            enforce_study_rate_limit(
                request,
                scope=f"study1-demo:{session_id}",
                maximum=6,
                window_seconds=600,
            )
            reservation = study1.reserve_demo_turn(session_id, payload.query)
            persona = next(
                (item for item in service.list_personas() if item.author == reservation["author"]),
                None,
            )
            writer_prompt = "mrprompt" if persona and persona.narrative_schema_available else (
                "persona_pack" if persona and persona.persona_pack_available else "strong_identity"
            )
            request = ChatStreamRequest(
                author=reservation["author"],
                session_id=reservation["conversation_id"],
                query=payload.query,
                query_mode="grounded",
                writer_prompt=writer_prompt,
                parent_top_k=20,
                trace_capture="summary",
            )
            turn = _create_chat_turn(
                chat_manager, service, request, owner_id=reservation["owner_id"]
            )
            study1.attach_demo_turn(reservation["reservation_id"], turn.id)
        except (ValueError, FileNotFoundError) as exc:
            if reservation:
                study1.cancel_demo_turn(reservation["reservation_id"])
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            if reservation:
                study1.cancel_demo_turn(reservation["reservation_id"])
            raise
        return StreamingResponse(
            _persistent_chat_stream_events(chat_manager, turn.id, initial_turn=turn),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/studies/study1/admin/overview")
    def study1_admin_overview(
        study_id: str | None = None,
        _admin: AuthUser = Depends(current_admin),
    ) -> dict[str, Any]:
        try:
            return study1.overview(study_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study not found") from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studies/study1/admin/studies")
    def study1_admin_studies(
        _admin: AuthUser = Depends(current_admin),
    ) -> dict[str, Any]:
        return {"studies": study1.study_catalog()}

    @app.post("/api/studies/study1/admin/codes")
    def create_study1_codes(
        payload: StudyCodeCreateRequest, _admin: AuthUser = Depends(current_admin)
    ) -> dict[str, Any]:
        try:
            return {
                "study_id": payload.study_id or study1.public_meta()["study_id"],
                "codes": study1.create_codes(payload.count, payload.study_id),
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study not found") from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studies/study1/admin/sessions/{session_id}")
    def study1_admin_session(
        session_id: str, _admin: AuthUser = Depends(current_admin)
    ) -> dict[str, Any]:
        try:
            return study1.detail(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study session not found") from exc

    @app.get("/api/studies/study1/admin/export")
    def export_study1(
        format: str = "jsonl",
        study_id: str | None = None,
        _admin: AuthUser = Depends(current_admin),
    ) -> Response:
        try:
            content, media_type, filename = study1.export(format, study_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study not found") from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/studies/study1/admin/analysis-bundle")
    def export_study1_analysis_bundle(
        study_id: str | None = None,
        _admin: AuthUser = Depends(current_admin),
    ) -> Response:
        try:
            content, filename = study1.analysis_bundle(study_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Study not found") from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(
            content=content,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/memories", response_model=UserMemoriesResponse)
    def memories(user: AuthUser = Depends(current_user)) -> UserMemoriesResponse:
        return UserMemoriesResponse(
            memories=[
                UserMemoryInfo(**memory.to_api())
                for memory in service.user_memories.list_active(user.id)
            ]
        )

    @app.patch("/api/memories/{memory_id}", response_model=UserMemoryInfo)
    def patch_memory(
        memory_id: str,
        payload: UserMemoryPatchRequest,
        user: AuthUser = Depends(current_user),
    ) -> UserMemoryInfo:
        try:
            memory = service.user_memories.get(user.id, memory_id)
            if payload.content is not None and payload.content.strip() != memory.content:
                memory = service.user_memories.correct(user.id, memory_id, payload.content.strip())
                memory_id = memory.id
            if payload.pinned is not None and payload.pinned != memory.pinned:
                memory = service.user_memories.set_pinned(user.id, memory_id, payload.pinned)
            return UserMemoryInfo(**memory.to_api())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Memory not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/memories/{memory_id}")
    def delete_memory(memory_id: str, user: AuthUser = Depends(current_user)) -> dict[str, str]:
        try:
            service.user_memories.forget(user.id, memory_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Memory not found") from exc
        return {"status": "forgotten"}

    @app.delete("/api/memories")
    def clear_memories(user: AuthUser = Depends(current_user)) -> dict[str, Any]:
        return {"status": "forgotten", "count": service.user_memories.clear(user.id)}

    @app.get("/api/memory-settings", response_model=UserMemorySettingsResponse)
    def memory_settings(user: AuthUser = Depends(current_user)) -> UserMemorySettingsResponse:
        return UserMemorySettingsResponse(**service.user_memories.settings(user.id))

    @app.patch("/api/memory-settings", response_model=UserMemorySettingsResponse)
    def patch_memory_settings(
        payload: UserMemorySettingsPatchRequest,
        user: AuthUser = Depends(current_user),
    ) -> UserMemorySettingsResponse:
        return UserMemorySettingsResponse(
            **service.user_memories.update_settings(
                user.id,
                enabled=payload.enabled,
                auto_write=payload.auto_write,
            )
        )

    @app.get("/api/personas", response_model=PersonasResponse)
    def personas(_user: AuthUser = Depends(current_user)) -> PersonasResponse:
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
                narrative_schema_available=item.narrative_schema_available,
                profile_url=item.profile_url,
                last_synced_at=item.last_synced_at,
            )
            for item in service.list_personas()
        ]
        return PersonasResponse(personas=items, default_author=service.default_author())

    @app.post("/api/personas/preview", response_model=AuthorPreview)
    def preview_persona(
        request: AuthorPreviewRequest,
        _user: AuthUser = Depends(current_user),
    ) -> AuthorPreview:
        try:
            return AuthorPreview(**resolve_author_preview(config.data_dir, request.value))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/personas/{author}/avatar")
    def persona_avatar(author: str, _user: AuthUser = Depends(current_user)) -> FileResponse:
        try:
            safe_author = safe_author_token(author)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        path = local_avatar_path(config.data_dir, safe_author)
        if path is None:
            raise HTTPException(status_code=404, detail="Avatar not found")
        return FileResponse(path)

    @app.get("/api/author-jobs", response_model=AuthorJobsResponse)
    def author_jobs(_user: AuthUser = Depends(current_user)) -> AuthorJobsResponse:
        return AuthorJobsResponse(
            jobs=[AuthorJobResponse(**job.to_dict()) for job in job_manager.store.list()]
        )

    @app.get("/api/author-jobs/{job_id}", response_model=AuthorJobResponse)
    def author_job(job_id: str, _user: AuthUser = Depends(current_user)) -> AuthorJobResponse:
        try:
            return AuthorJobResponse(**job_manager.store.get(job_id).to_dict())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Author job not found") from exc

    @app.post("/api/author-jobs", response_model=AuthorJobResponse)
    def create_author_job(
        request: AuthorJobCreateRequest,
        _user: AuthUser = Depends(current_user),
    ) -> AuthorJobResponse:
        try:
            job = job_manager.create_job(
                author_input=request.author,
                kinds=request.kinds,
                max_items=request.max_items,
            )
            return AuthorJobResponse(**job.to_dict())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/author-jobs/{job_id}/cancel", response_model=AuthorJobResponse)
    def cancel_author_job(
        job_id: str,
        _user: AuthUser = Depends(current_user),
    ) -> AuthorJobResponse:
        try:
            return AuthorJobResponse(**job_manager.cancel(job_id).to_dict())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Author job not found") from exc

    @app.post("/api/author-jobs/{job_id}/retry", response_model=AuthorJobResponse)
    def retry_author_job(
        job_id: str,
        _user: AuthUser = Depends(current_user),
    ) -> AuthorJobResponse:
        try:
            return AuthorJobResponse(**job_manager.retry(job_id).to_dict())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Author job not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/personas/{author}/sessions", response_model=SessionsResponse)
    def sessions(author: str, user: AuthUser = Depends(current_user)) -> SessionsResponse:
        return SessionsResponse(sessions=service.list_sessions(author, owner_id=user.id))

    @app.get("/api/personas/{author}/suggestions", response_model=SuggestionsResponse)
    def suggestions(author: str, _user: AuthUser = Depends(current_user)) -> SuggestionsResponse:
        return SuggestionsResponse(suggestions=service.list_suggestions(author))

    @app.get("/api/personas/{author}/sessions/{session_id}", response_model=ChatSession)
    def session(
        author: str,
        session_id: str,
        user: AuthUser = Depends(current_user),
    ) -> ChatSession | JSONResponse:
        try:
            return ChatSession(**service.get_session(author, session_id, owner_id=user.id))
        except FileNotFoundError:
            return JSONResponse({"error": "Session not found"}, status_code=404)

    @app.delete("/api/personas/{author}/sessions/{session_id}")
    def delete_session(
        author: str,
        session_id: str,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, str]:
        service.delete_session(author, session_id, owner_id=user.id)
        return {"status": "ok"}

    @app.get("/api/personas/{author}/traces/{trace_id}", response_model=None)
    def trace(
        author: str,
        trace_id: str,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any] | JSONResponse:
        if not service.conversations.owner_has_trace(user.id, author, trace_id):
            return JSONResponse({"error": "Trace not found"}, status_code=404)
        try:
            return service.get_trace(author, trace_id)
        except FileNotFoundError:
            return JSONResponse({"error": "Trace not found"}, status_code=404)
        except ValueError:
            return JSONResponse({"error": "Trace is invalid"}, status_code=400)

    @app.post("/api/chat/stream")
    def chat_stream(
        request: ChatStreamRequest,
        user: AuthUser = Depends(current_user),
    ) -> StreamingResponse:
        turn = _create_chat_turn(
            chat_manager,
            service,
            request,
            owner_id=user.id,
            deployment_guard=deployment_guard,
        )
        return StreamingResponse(
            _persistent_chat_stream_events(chat_manager, turn.id, initial_turn=turn),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/chat/turns", response_model=TurnRunResponse)
    def create_chat_turn(
        request: ChatStreamRequest,
        user: AuthUser = Depends(current_user),
    ) -> TurnRunResponse:
        return TurnRunResponse(
            **_create_chat_turn(
                chat_manager,
                service,
                request,
                owner_id=user.id,
                deployment_guard=deployment_guard,
            ).to_dict()
        )

    @app.get("/api/chat/turns/{turn_id}", response_model=TurnRunResponse)
    def get_chat_turn(turn_id: str, user: AuthUser = Depends(current_user)) -> TurnRunResponse:
        try:
            return TurnRunResponse(**chat_manager.store.get_turn_for_owner(turn_id, user.id).to_dict())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Chat turn not found") from exc

    @app.post("/api/chat/turns/{turn_id}/retry", response_model=TurnRunResponse)
    def retry_chat_turn(turn_id: str, user: AuthUser = Depends(current_user)) -> TurnRunResponse:
        try:
            chat_manager.store.get_turn_for_owner(turn_id, user.id)
            deployment_guard.admit_chat(user.id, chat_manager.store)
            return TurnRunResponse(**chat_manager.retry(turn_id).to_dict())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Chat turn not found") from exc
        except DeploymentGuardError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": str(exc.retry_after)},
            ) from exc
        except (ValueError, ConversationBusyError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/chat/turns/{turn_id}/events")
    def chat_turn_events(
        turn_id: str,
        after: int = 0,
        user: AuthUser = Depends(current_user),
    ) -> StreamingResponse:
        try:
            chat_manager.store.get_turn_for_owner(turn_id, user.id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Chat turn not found") from exc
        return StreamingResponse(
            _persistent_chat_stream_events(chat_manager, turn_id, after_sequence=max(0, after)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/evaluations/retrieval/pools")
    def retrieval_pools(author: str | None = None, user: AuthUser = Depends(current_user)) -> dict[str, Any]:
        return {"pools": retrieval_evaluations.list_pools(user.id, author=author)}

    @app.get("/api/evaluations/retrieval/jobs")
    def retrieval_eval_jobs(user: AuthUser = Depends(current_user)) -> dict[str, Any]:
        return {"jobs": retrieval_eval_job_manager.list()}

    @app.post("/api/evaluations/retrieval/jobs")
    def create_retrieval_eval_job(
        payload: RetrievalEvalJobCreateRequest,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        if payload.labeler == "deepseek_api" and user.role != "admin":
            raise HTTPException(status_code=403, detail="Only administrators can spend the server API budget")
        try:
            return retrieval_eval_job_manager.create(
                author=payload.author,
                labeler=payload.labeler,
                split=payload.split,
                budget_cny=payload.budget_cny,
                owner_id=user.id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/evaluations/retrieval/jobs/{job_id}")
    def get_retrieval_eval_job(job_id: str, user: AuthUser = Depends(current_user)) -> dict[str, Any]:
        try:
            return retrieval_eval_job_manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Retrieval evaluation job not found") from exc

    @app.post("/api/evaluations/retrieval/jobs/{job_id}/resume")
    def resume_retrieval_eval_job(
        job_id: str,
        payload: RetrievalEvalJobResumeRequest,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            job = retrieval_eval_job_manager.get(job_id)
            if job["labeler"] == "deepseek_api" and user.role != "admin":
                raise HTTPException(status_code=403, detail="Only administrators can resume API jobs")
            return retrieval_eval_job_manager.resume(job_id, budget_cny=payload.budget_cny)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Retrieval evaluation job not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/evaluations/retrieval/jobs/{job_id}/handoff")
    def download_retrieval_eval_handoff(
        job_id: str,
        user: AuthUser = Depends(current_user),
    ) -> FileResponse:
        try:
            path = retrieval_eval_job_manager.handoff_zip(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type="application/zip", filename=path.name)

    @app.post("/api/evaluations/retrieval/jobs/{job_id}/codex-review")
    def import_retrieval_eval_codex_review(
        job_id: str,
        payload: dict[str, Any],
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return retrieval_eval_job_manager.import_codex_review(job_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/evaluations/retrieval/pools/{pool_id}")
    def retrieval_workspace(pool_id: str, user: AuthUser = Depends(current_user)) -> dict[str, Any]:
        try:
            return retrieval_evaluations.workspace(pool_id, user.id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/evaluations/retrieval/pools/{pool_id}/queries/{item_id}")
    def retrieval_query(pool_id: str, item_id: str, user: AuthUser = Depends(current_user)) -> dict[str, Any]:
        try:
            return retrieval_evaluations.query(pool_id, item_id, user.id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/evaluations/retrieval/pools/{pool_id}/llm-labels")
    def retrieval_llm_label_sets(pool_id: str, user: AuthUser = Depends(current_user)) -> dict[str, Any]:
        try:
            return {"label_sets": retrieval_evaluations.list_llm_label_sets(pool_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/evaluations/retrieval/pools/{pool_id}/llm-labels/{label_set}")
    def retrieval_llm_workspace(
        pool_id: str,
        label_set: str,
        axis: str | None = None,
        ranking: str | None = None,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return retrieval_evaluations.llm_workspace(pool_id, label_set, axis=axis, ranking_id=ranking)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/evaluations/retrieval/pools/{pool_id}/llm-labels/{label_set}/queries/{item_id}")
    def retrieval_llm_query(
        pool_id: str,
        label_set: str,
        item_id: str,
        axis: str | None = None,
        ranking: str | None = None,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return retrieval_evaluations.llm_query(pool_id, label_set, item_id, axis=axis, ranking_id=ranking)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/evaluations/retrieval/pools/{pool_id}/queries/{item_id}/candidates/{parent_id}")
    def label_retrieval_candidate(
        pool_id: str,
        item_id: str,
        parent_id: str,
        payload: RetrievalLabelRequest,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return retrieval_evaluations.set_label(pool_id, item_id, parent_id, user.id, payload.score)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/evaluations/retrieval/pools/{pool_id}/export")
    def export_retrieval_labels(
        pool_id: str,
        format: str = "jsonl",
        user: AuthUser = Depends(current_user),
    ) -> Response:
        try:
            content, media_type, filename = retrieval_evaluations.export(pool_id, user.id, format=format)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/evaluations/generation/systems")
    def generation_systems(author: str | None = None, user: AuthUser = Depends(current_user)) -> dict[str, Any]:
        return {"systems": generation_evaluations.list_systems(user.id, author=author)}

    @app.get("/api/evaluations/generation/systems/{system_id}")
    def generation_workspace(
        system_id: str,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return generation_evaluations.workspace(system_id, user.id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/evaluations/generation/systems/{system_id}/items/{item_id}")
    def generation_item(
        system_id: str,
        item_id: str,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return generation_evaluations.item(system_id, item_id, user.id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/evaluations/generation/systems/{system_id}/items/{item_id}/rubric")
    def label_generation_rubric(
        system_id: str,
        item_id: str,
        payload: GenerationRubricRequest,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return generation_evaluations.set_rubric(
                system_id, item_id, user.id, payload.scores, payload.note
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/evaluations/generation/comparisons/{left_id}/{right_id}")
    def generation_comparison(
        left_id: str,
        right_id: str,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return generation_evaluations.comparison(left_id, right_id, user.id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/evaluations/generation/comparisons/{left_id}/{right_id}/items/{item_id}")
    def generation_comparison_item(
        left_id: str,
        right_id: str,
        item_id: str,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return generation_evaluations.comparison_item(
                left_id, right_id, item_id, user.id
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/evaluations/generation/comparisons/{left_id}/{right_id}/items/{item_id}")
    def label_generation_comparison(
        left_id: str,
        right_id: str,
        item_id: str,
        payload: GenerationPairwiseRequest,
        user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return generation_evaluations.set_pair_vote(
                left_id, right_id, item_id, user.id, payload.choice
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/evaluations/generation/judge-jobs")
    def create_generation_judge_job(
        payload: GenerationJudgeJobRequest,
        _user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return generation_judge_manager.create(
                payload.system_id, repeats=payload.repeats
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/evaluations/generation/judge-jobs/{job_id}")
    def generation_judge_job(
        job_id: str,
        _user: AuthUser = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return generation_evaluations.public_judge_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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


def _create_chat_turn(
    manager: ChatTaskManager,
    service: PersonaChatService,
    request: ChatStreamRequest,
    *,
    owner_id: str | None = None,
    deployment_guard: DeploymentGuard | None = None,
):
    author = (request.author or service.default_author() or "").strip()
    if not author:
        raise HTTPException(status_code=400, detail="No local persona index found.")
    try:
        if deployment_guard is not None and owner_id is not None:
            deployment_guard.admit_chat(owner_id, manager.store)
        return manager.create_turn(
            author=author,
            conversation_id=request.session_id,
            query=request.query,
            query_mode=request.query_mode,
            writer_prompt=request.writer_prompt,
            parent_top_k=request.parent_top_k,
            trace_capture=request.trace_capture,
            owner_id=owner_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    except ConversationBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DeploymentGuardError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _persistent_chat_stream_events(
    manager: ChatTaskManager,
    turn_id: str,
    *,
    after_sequence: int = 0,
    initial_turn: TurnRun | None = None,
) -> Iterator[str]:
    sequence = after_sequence
    if initial_turn is not None:
        yield sse_event(
            "accepted",
            {
                "session_id": initial_turn.conversation_id,
                "turn_id": initial_turn.id,
                "status": initial_turn.status,
                "stage": initial_turn.stage,
                "label": initial_turn.label,
            },
        )
    while True:
        events = manager.store.list_events(turn_id, after_sequence=sequence)
        for event in events:
            sequence = int(event["sequence"])
            yield sse_event(str(event["event"]), dict(event["payload"]))
            if event["event"] in {"done", "error"}:
                return
        turn = manager.store.get_turn(turn_id)
        if turn.status in {"failed", "interrupted"} and not events:
            yield sse_event(
                "error",
                {
                    "error": str((turn.error or {}).get("message") or turn.label),
                    "turn_id": turn.id,
                },
            )
            return
        if turn.status == "completed" and not events:
            yield sse_event(
                "done",
                {
                    "session_id": turn.conversation_id,
                    "turn_id": turn.id,
                    "trace_id": turn.trace_id,
                    "answer": turn.partial_answer,
                    "sources": [],
                },
            )
            return
        time.sleep(0.1)


def _frontend_dist_dir() -> Path:
    configured_path = os.getenv("PERSONAFORGE_WEB_DIST")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "web" / "dist"


def _set_auth_cookie(
    response: JSONResponse,
    request: Request,
    token: str,
    config: WebConfig,
) -> None:
    secure = config.secure_cookies
    if secure is None:
        secure = request.url.scheme == "https"
    response.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        max_age=max(1, config.session_days) * 24 * 60 * 60,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )


def _client_key(request: Request) -> str:
    """Use the direct peer address for the local/private deployment.

    We intentionally do not trust arbitrary forwarded headers here. A named
    reverse proxy can add a trusted proxy integration later.
    """

    return str(request.client.host if request.client else "unknown")


def run_web(config: WebConfig) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - missing optional dependency.
        raise RuntimeError('Web server requires optional dependencies: pip install -e ".[web]"') from exc

    from personaforge.web.startup_checks import format_startup_report

    startup_report = run_startup_checks(
        data_dir=config.data_dir,
        model_name=config.model_name,
    )
    print(format_startup_report(startup_report), flush=True)
    service = PersonaChatService(config)
    if service.list_personas():
        print("Preparing BGE-M3 runtime before starting Web workers...", flush=True)
        service.prepare_encoder_runtime()
    app = create_app(config, service=service, startup_report=startup_report)
    uvicorn.run(app, host=config.host, port=config.port)
