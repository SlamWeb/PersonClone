"""Persistent background jobs for creating and refreshing local personas."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.request import Request, urlopen
from uuid import uuid4

from personaforge.crawler.exceptions import CrawlError
from personaforge.crawler.models import CreatorProfile
from personaforge.crawler.zhihu import ZhihuPublicCrawler, parse_user_token
from personaforge.crawler.zhihu_browser import ZhihuBrowserCrawler

JobStatus = Literal["queued", "running", "ready", "failed", "cancelled", "interrupted"]
JobStage = Literal[
    "queued",
    "resolving_profile",
    "crawling",
    "building",
    "indexing",
    "activating",
    "ready",
    "failed",
    "cancelled",
    "interrupted",
]

ACTIVE_STATUSES = ("queued", "running")
RETRYABLE_STATUSES = ("failed", "cancelled", "interrupted")
DEFAULT_KINDS = ("answer", "article", "pin")


@dataclass(slots=True)
class AuthorJob:
    id: str
    source: str
    author_input: str
    author: str
    operation: str
    status: JobStatus
    stage: JobStage
    label: str
    kinds: tuple[str, ...]
    max_items: int | None
    display_name: str
    avatar_url: str | None
    headline: str
    profile_url: str
    item_count: int | None
    parent_count: int | None
    node_count: int | None
    error_message: str | None
    cancel_requested: bool
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AuthorJobConfig:
    data_dir: Path
    model_name: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    use_fp16: bool = True
    batch_size: int = 12
    delay_seconds: float = 1.5
    max_api_pages: int = 100
    working_dir: Path = Path.cwd()

    def __post_init__(self) -> None:
        self.data_dir = self.data_dir.expanduser().resolve()
        self.working_dir = self.working_dir.expanduser().resolve()

    @property
    def database_path(self) -> Path:
        return self.data_dir / "system" / "personaforge.sqlite3"

    @property
    def storage_state_path(self) -> Path:
        return self.data_dir / "auth" / "zhihu_storage_state.json"


class AuthorJobStore:
    """Small SQLite repository shared by API requests and the worker thread."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create(
        self,
        *,
        author_input: str,
        author: str,
        operation: str,
        kinds: Iterable[str] = DEFAULT_KINDS,
        max_items: int | None = None,
        profile: dict[str, Any] | None = None,
    ) -> AuthorJob:
        now = utc_now_iso()
        profile = profile or {}
        job_id = f"job-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        values = {
            "id": job_id,
            "source": "zhihu",
            "author_input": author_input,
            "author": author,
            "operation": operation,
            "status": "queued",
            "stage": "queued",
            "label": "等待后台处理",
            "kinds_json": json.dumps(tuple(kinds), ensure_ascii=False),
            "max_items": max_items,
            "display_name": str(profile.get("display_name") or profile.get("nickname") or author),
            "avatar_url": profile.get("avatar_url"),
            "headline": str(profile.get("headline") or ""),
            "profile_url": str(profile.get("profile_url") or f"https://www.zhihu.com/people/{author}"),
            "item_count": None,
            "parent_count": None,
            "node_count": None,
            "error_message": None,
            "cancel_requested": 0,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO author_jobs ({columns}) VALUES ({placeholders})",
                tuple(values.values()),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> AuthorJob:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM author_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return _row_to_job(row)

    def list(self, *, limit: int = 100) -> list[AuthorJob]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM author_jobs
                ORDER BY
                    CASE WHEN status IN ('queued', 'running') THEN 0 ELSE 1 END,
                    updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_job(row) for row in rows]

    def find_active(self, author: str) -> AuthorJob | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM author_jobs
                WHERE author = ? AND status IN ('queued', 'running')
                ORDER BY created_at DESC LIMIT 1
                """,
                (author,),
            ).fetchone()
        return _row_to_job(row) if row is not None else None

    def claim_next(self) -> AuthorJob | None:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id FROM author_jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                """
                UPDATE author_jobs
                SET status = 'running', stage = 'resolving_profile',
                    label = '正在读取作者资料', started_at = COALESCE(started_at, ?),
                    updated_at = ?, error_message = NULL
                WHERE id = ?
                """,
                (now, now, row["id"]),
            )
            connection.commit()
        return self.get(str(row["id"]))

    def update(self, job_id: str, **values: Any) -> AuthorJob:
        allowed = {
            "status",
            "stage",
            "label",
            "display_name",
            "avatar_url",
            "headline",
            "profile_url",
            "item_count",
            "parent_count",
            "node_count",
            "error_message",
            "cancel_requested",
            "started_at",
            "completed_at",
        }
        unexpected = set(values) - allowed
        if unexpected:
            raise ValueError(f"Unsupported author job fields: {sorted(unexpected)}")
        values["updated_at"] = utc_now_iso()
        assignments = ", ".join(f"{key} = ?" for key in values)
        params = [int(value) if isinstance(value, bool) else value for value in values.values()]
        params.append(job_id)
        with self._connect() as connection:
            connection.execute(f"UPDATE author_jobs SET {assignments} WHERE id = ?", params)
        return self.get(job_id)

    def request_cancel(self, job_id: str) -> AuthorJob:
        job = self.get(job_id)
        if job.status == "queued":
            return self.update(
                job_id,
                status="cancelled",
                stage="cancelled",
                label="任务已取消",
                cancel_requested=True,
                completed_at=utc_now_iso(),
            )
        if job.status == "running":
            return self.update(job_id, cancel_requested=True, label="正在安全取消")
        return job

    def retry(self, job_id: str) -> AuthorJob:
        job = self.get(job_id)
        if job.status not in RETRYABLE_STATUSES:
            raise ValueError("Only failed, cancelled, or interrupted jobs can be retried.")
        return self.update(
            job_id,
            status="queued",
            stage="queued",
            label="等待后台重试",
            error_message=None,
            cancel_requested=False,
            started_at=None,
            completed_at=None,
        )

    def mark_running_interrupted(self) -> int:
        now = utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE author_jobs
                SET status = 'interrupted', stage = 'interrupted',
                    label = '服务中断，请重试',
                    error_message = '后台服务在任务完成前停止。',
                    updated_at = ?, completed_at = ?
                WHERE status = 'running'
                """,
                (now, now),
            )
            return cursor.rowcount

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS author_jobs (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    author_input TEXT NOT NULL,
                    author TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    label TEXT NOT NULL,
                    kinds_json TEXT NOT NULL,
                    max_items INTEGER,
                    display_name TEXT NOT NULL,
                    avatar_url TEXT,
                    headline TEXT NOT NULL,
                    profile_url TEXT NOT NULL,
                    item_count INTEGER,
                    parent_count INTEGER,
                    node_count INTEGER,
                    error_message TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_author_jobs_status_created
                ON author_jobs(status, created_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_author_jobs_author_updated
                ON author_jobs(author, updated_at)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection


CommandRunner = Callable[[list[str], Path, Path, Callable[[], bool]], None]


class AuthorJobManager:
    def __init__(
        self,
        config: AuthorJobConfig,
        *,
        store: AuthorJobStore | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.store = store or AuthorJobStore(config.database_path)
        self.command_runner = command_runner or self._run_subprocess
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.store.mark_running_interrupted()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="personaforge-author-worker", daemon=True)
        self._thread.start()
        self._wake.set()

    def stop(self, *, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def create_job(
        self,
        *,
        author_input: str,
        kinds: Iterable[str] = DEFAULT_KINDS,
        max_items: int | None = None,
        profile: dict[str, Any] | None = None,
    ) -> AuthorJob:
        author = safe_author_token(author_input)
        active = self.store.find_active(author)
        if active is not None:
            return active
        ready = persona_is_ready(self.config.data_dir, author)
        job = self.store.create(
            author_input=author_input,
            author=author,
            operation="sync" if ready else "create",
            kinds=validate_kinds(kinds),
            max_items=max_items,
            profile=profile,
        )
        self._wake.set()
        return job

    def cancel(self, job_id: str) -> AuthorJob:
        job = self.store.request_cancel(job_id)
        self._wake.set()
        return job

    def retry(self, job_id: str) -> AuthorJob:
        job = self.store.retry(job_id)
        self._wake.set()
        return job

    def run_once(self) -> bool:
        job = self.store.claim_next()
        if job is None:
            return False
        self._execute(job)
        return True

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            if self.run_once():
                continue
            self._wake.wait(timeout=1.0)
            self._wake.clear()

    def _execute(self, job: AuthorJob) -> None:
        author_dir = self.config.data_dir / "authors" / "zhihu" / job.author
        stage_root = author_dir / "staging" / job.id
        raw_dir = stage_root / "raw"
        index_dir = stage_root / "index"
        log_path = stage_root / "job.log"
        previous_manifest = read_jsonl(author_dir / "raw" / "manifest.jsonl")

        try:
            if stage_root.exists():
                shutil.rmtree(stage_root)
            stage_root.mkdir(parents=True, exist_ok=True)
            if (author_dir / "raw").exists():
                shutil.copytree(author_dir / "raw", raw_dir)

            self._check_cancel(job.id)
            self.store.update(job.id, stage="crawling", label="正在抓取公开内容")
            self.command_runner(
                self._crawl_command(job, raw_dir),
                self.config.working_dir,
                log_path,
                lambda: self._should_stop(job.id),
            )
            merge_manifest(raw_dir / "manifest.jsonl", previous_manifest)
            item_count = count_jsonl(raw_dir / "manifest.jsonl")
            if item_count < 1:
                raise PipelineError("没有抓取到可入库的公开内容。")
            profile = read_json(raw_dir / "profile.json")
            cache_profile_avatar(raw_dir, profile)
            self.store.update(
                job.id,
                item_count=item_count,
                display_name=str(profile.get("nickname") or job.author),
                avatar_url=profile.get("avatar_url"),
                headline=str(profile.get("headline") or ""),
                profile_url=str(profile.get("profile_url") or f"https://www.zhihu.com/people/{job.author}"),
                label=f"已抓取 {item_count} 篇内容",
            )

            self._check_cancel(job.id)
            self.store.update(job.id, stage="building", label=f"正在解析 {item_count} 篇内容")
            self.command_runner(
                self._build_command(job, raw_dir, index_dir),
                self.config.working_dir,
                log_path,
                lambda: self._should_stop(job.id),
            )
            build_manifest = read_json(index_dir / "build_manifest.json")
            self.store.update(
                job.id,
                parent_count=int(build_manifest.get("parent_count") or 0),
                node_count=int(build_manifest.get("node_count") or 0),
            )

            self._check_cancel(job.id)
            self.store.update(job.id, stage="indexing", label="正在创建向量索引")
            self.command_runner(
                self._index_command(job, index_dir),
                self.config.working_dir,
                log_path,
                lambda: self._should_stop(job.id),
            )
            validate_staged_persona(raw_dir, index_dir)

            self._check_cancel(job.id)
            self.store.update(job.id, stage="activating", label="正在启用作者")
            activate_persona(author_dir, stage_root)
            completed_at = utc_now_iso()
            self.store.update(
                job.id,
                status="ready",
                stage="ready",
                label="作者已就绪",
                completed_at=completed_at,
                cancel_requested=False,
            )
            if stage_root.exists():
                shutil.rmtree(stage_root, ignore_errors=True)
        except JobCancelled:
            self.store.update(
                job.id,
                status="cancelled",
                stage="cancelled",
                label="任务已取消",
                completed_at=utc_now_iso(),
                error_message=None,
            )
        except JobInterrupted:
            self.store.update(
                job.id,
                status="interrupted",
                stage="interrupted",
                label="服务中断，请重试",
                completed_at=utc_now_iso(),
                error_message="后台服务在任务完成前停止。",
            )
        except Exception as exc:
            self.store.update(
                job.id,
                status="failed",
                stage="failed",
                label="构建失败",
                completed_at=utc_now_iso(),
                error_message=friendly_error(exc),
            )

    def _crawl_command(self, job: AuthorJob, raw_dir: Path) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "personaforge.cli",
            "crawl",
            "zhihu",
            job.author,
            "--out-dir",
            str(raw_dir),
            "--delay-seconds",
            str(self.config.delay_seconds),
            "--max-api-pages",
            str(self.config.max_api_pages),
            "--quiet",
        ]
        if job.max_items is None:
            command.append("--all")
        else:
            command.extend(["--max-items", str(job.max_items)])
        for kind in job.kinds:
            command.extend(["--kind", kind])
        if self.config.storage_state_path.exists():
            command.extend(["--storage-state", str(self.config.storage_state_path)])
        return command

    def _build_command(self, job: AuthorJob, raw_dir: Path, index_dir: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "personaforge.cli",
            "build",
            job.author,
            "--raw-dir",
            str(raw_dir),
            "--index-dir",
            str(index_dir),
            "--quality",
            "fast",
        ]

    def _index_command(self, job: AuthorJob, index_dir: Path) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "personaforge.cli",
            "index",
            job.author,
            "--index-dir",
            str(index_dir),
            "--qdrant-path",
            str(index_dir / "qdrant"),
            "--model-name",
            self.config.model_name,
            "--embedding-device",
            self.config.embedding_device,
            "--batch-size",
            str(self.config.batch_size),
        ]
        if not self.config.use_fp16:
            command.append("--no-fp16")
        return command

    def _run_subprocess(
        self,
        command: list[str],
        cwd: Path,
        log_path: Path,
        should_stop: Callable[[], bool],
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            log.write(f"\n$ {redacted_command(command, self.config.storage_state_path)}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creationflags,
            )
            while process.poll() is None:
                if should_stop():
                    process.terminate()
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    if self._stop.is_set():
                        raise JobInterrupted
                    raise JobCancelled
                time.sleep(0.25)
            if process.returncode:
                raise PipelineCommandError(command_stage(command), process.returncode)

    def _should_stop(self, job_id: str) -> bool:
        return self._stop.is_set() or self.store.get(job_id).cancel_requested

    def _check_cancel(self, job_id: str) -> None:
        if self._stop.is_set():
            raise JobInterrupted
        if self.store.get(job_id).cancel_requested:
            raise JobCancelled


class PipelineError(RuntimeError):
    pass


class PipelineCommandError(PipelineError):
    def __init__(self, stage: str, return_code: int) -> None:
        super().__init__(f"{stage} exited with code {return_code}")
        self.stage = stage
        self.return_code = return_code


class JobCancelled(RuntimeError):
    pass


class JobInterrupted(RuntimeError):
    pass


def resolve_author_preview(data_dir: Path, value: str) -> dict[str, Any]:
    author = safe_author_token(value)
    author_dir = data_dir / "authors" / "zhihu" / author
    local_profile = read_profile(author_dir)
    if local_profile:
        return preview_payload(data_dir, author, local_profile)

    errors: list[str] = []
    profile: CreatorProfile | None = None
    try:
        profile = ZhihuPublicCrawler(delay_seconds=0, max_api_pages=1).crawl_profile(author)
    except CrawlError as exc:
        errors.append(str(exc))

    storage_state = data_dir / "auth" / "zhihu_storage_state.json"
    if profile is None and storage_state.exists():
        try:
            profile = ZhihuBrowserCrawler(
                headless=True,
                storage_state=storage_state,
                delay_seconds=0,
                max_api_pages=1,
            ).crawl_profile(author)
        except (CrawlError, RuntimeError) as exc:
            errors.append(str(exc))

    if profile is None:
        detail = "; ".join(errors[-2:])
        raise PipelineError(f"无法读取这个知乎用户的公开资料。{detail}".strip())
    return preview_payload(data_dir, author, profile.to_dict())


def preview_payload(data_dir: Path, author: str, profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "author": author,
        "display_name": str(profile.get("display_name") or profile.get("nickname") or author),
        "avatar_url": local_avatar_url(data_dir, author) or profile.get("avatar_url"),
        "headline": str(profile.get("headline") or ""),
        "profile_url": str(profile.get("profile_url") or f"https://www.zhihu.com/people/{author}"),
        "exists": (data_dir / "authors" / "zhihu" / author).exists(),
        "ready": persona_is_ready(data_dir, author),
    }


def safe_author_token(value: str) -> str:
    token = parse_user_token(value)
    if token in {".", ".."} or len(token) > 160 or not re.fullmatch(r"[\w-]+", token, flags=re.UNICODE):
        raise ValueError("知乎用户名格式无效。")
    return token


def validate_kinds(kinds: Iterable[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(kinds))
    if not result:
        raise ValueError("至少选择一种内容类型。")
    invalid = set(result) - set(DEFAULT_KINDS)
    if invalid:
        raise ValueError(f"不支持的内容类型：{', '.join(sorted(invalid))}")
    return result


def persona_is_ready(data_dir: Path, author: str) -> bool:
    index_dir = data_dir / "authors" / "zhihu" / author / "index"
    return (
        (index_dir / "parents.jsonl").exists()
        and (index_dir / "nodes.jsonl").exists()
        and (index_dir / "qdrant_manifest.json").exists()
        and (index_dir / "qdrant").exists()
    )


def validate_staged_persona(raw_dir: Path, index_dir: Path) -> None:
    required = [
        raw_dir / "profile.json",
        raw_dir / "manifest.jsonl",
        index_dir / "parents.jsonl",
        index_dir / "nodes.jsonl",
        index_dir / "build_manifest.json",
        index_dir / "qdrant_manifest.json",
        index_dir / "qdrant",
    ]
    missing = [str(path.name) for path in required if not path.exists()]
    if missing:
        raise PipelineError(f"构建产物不完整：{', '.join(missing)}")
    if count_jsonl(index_dir / "parents.jsonl") < 1:
        raise PipelineError("没有生成可用的父文档。")


def activate_persona(author_dir: Path, stage_root: Path) -> None:
    staged_raw = stage_root / "raw"
    staged_index = stage_root / "index"
    final_raw = author_dir / "raw"
    final_index = author_dir / "index"
    backup_raw = stage_root / "previous-raw"
    backup_index = stage_root / "previous-index"

    moved_raw = False
    moved_index = False
    try:
        if final_raw.exists():
            final_raw.replace(backup_raw)
        staged_raw.replace(final_raw)
        moved_raw = True

        if final_index.exists():
            final_index.replace(backup_index)
        staged_index.replace(final_index)
        moved_index = True
    except Exception:
        if moved_index and final_index.exists():
            shutil.rmtree(final_index, ignore_errors=True)
        if backup_index.exists():
            backup_index.replace(final_index)
        if moved_raw and final_raw.exists():
            shutil.rmtree(final_raw, ignore_errors=True)
        if backup_raw.exists():
            backup_raw.replace(final_raw)
        raise

    shutil.rmtree(backup_raw, ignore_errors=True)
    shutil.rmtree(backup_index, ignore_errors=True)


def merge_manifest(path: Path, previous_rows: list[dict[str, Any]]) -> None:
    current_rows = read_jsonl(path)
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in [*previous_rows, *current_rows]:
        key = (str(row.get("kind") or ""), str(row.get("id") or ""))
        if all(key):
            merged[key] = row
    if not merged:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in merged.values():
            relative = Path(str(row.get("path") or ""))
            if relative and (path.parent / relative).exists():
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")


def cache_profile_avatar(raw_dir: Path, profile: dict[str, Any]) -> Path | None:
    avatar_url = str(profile.get("avatar_url") or "")
    if not avatar_url.startswith(("http://", "https://")):
        return None
    try:
        request = Request(avatar_url, headers={"User-Agent": "Mozilla/5.0 PersonaForge/0.1"})
        with urlopen(request, timeout=15) as response:
            content = response.read(5 * 1024 * 1024)
            content_type = response.headers.get_content_type()
        extension = mimetypes.guess_extension(content_type) or Path(avatar_url.split("?", 1)[0]).suffix or ".jpg"
        if extension not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            extension = ".jpg"
        assets_dir = raw_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        path = assets_dir / f"avatar{extension}"
        path.write_bytes(content)
        profile["avatar_local_path"] = path.relative_to(raw_dir).as_posix()
        (raw_dir / "profile.json").write_text(
            json.dumps(profile, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return path
    except (OSError, ValueError):
        return None


def local_avatar_path(data_dir: Path, author: str) -> Path | None:
    assets_dir = data_dir / "authors" / "zhihu" / author / "raw" / "assets"
    if not assets_dir.exists():
        return None
    for path in sorted(assets_dir.glob("avatar.*")):
        if path.is_file():
            return path
    return None


def local_avatar_url(data_dir: Path, author: str) -> str | None:
    return f"/api/personas/{author}/avatar" if local_avatar_path(data_dir, author) else None


def read_profile(author_dir: Path) -> dict[str, Any]:
    for path in (author_dir / "profile.json", author_dir / "raw" / "profile.json"):
        if path.exists():
            return read_json(path)
    return {}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, PipelineCommandError):
        if exc.stage == "crawl":
            return "未能抓取到公开内容。知乎可能要求重新登录，请管理员检查服务端登录态后重试。"
        if exc.stage == "build":
            return "材料解析失败。已保留旧作者数据和本次排错目录。"
        if exc.stage == "index":
            return "向量索引构建失败。请检查 BGE-M3 模型、显存和运行环境。"
    return str(exc) or exc.__class__.__name__


def command_stage(command: list[str]) -> str:
    for value in ("crawl", "build", "index"):
        if value in command:
            return value
    return "pipeline"


def redacted_command(command: list[str], storage_state: Path) -> str:
    redacted: list[str] = []
    hide_next = False
    for part in command:
        if hide_next:
            redacted.append("<server-login-state>")
            hide_next = False
            continue
        redacted.append(part)
        if part == "--storage-state":
            hide_next = True
    return subprocess.list2cmdline(redacted)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _row_to_job(row: sqlite3.Row) -> AuthorJob:
    return AuthorJob(
        id=str(row["id"]),
        source=str(row["source"]),
        author_input=str(row["author_input"]),
        author=str(row["author"]),
        operation=str(row["operation"]),
        status=str(row["status"]),  # type: ignore[arg-type]
        stage=str(row["stage"]),  # type: ignore[arg-type]
        label=str(row["label"]),
        kinds=tuple(json.loads(row["kinds_json"])),
        max_items=row["max_items"],
        display_name=str(row["display_name"]),
        avatar_url=row["avatar_url"],
        headline=str(row["headline"]),
        profile_url=str(row["profile_url"]),
        item_count=row["item_count"],
        parent_count=row["parent_count"],
        node_count=row["node_count"],
        error_message=row["error_message"],
        cancel_requested=bool(row["cancel_requested"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )
