"""SQLite-backed Study 1 participant workflow for the integrated Web app."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import secrets
import sqlite3
import string
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


AI_SOURCES = ("rag_identity", "persona_pack", "codex")
ALL_SOURCES = ("gold", "rag_identity", "persona_pack", "codex", "other_human")
PROTOCOL_VERSION = "study1-v2"
ASSIGNMENT_SCHEMA_VERSION = "personaforge.study1.assignment.v2"
TOTAL_TRIALS = 4
POINTWISE_TRIALS = 2
PAIRWISE_TRIALS = 2


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StudyProfileRequest(BaseModel):
    participant_code: str = Field(min_length=3, max_length=40)
    follow_duration: Literal[
        "less_than_3_months", "3_to_12_months", "1_to_3_years", "more_than_3_years"
    ]
    reading_frequency: Literal["rarely", "monthly", "weekly", "almost_daily"]
    familiarity: Literal["somewhat", "familiar", "very_familiar"]
    ai_frequency: Literal["rarely", "monthly", "weekly", "almost_daily"]
    consent: bool

    @field_validator("participant_code")
    @classmethod
    def clean_code(cls, value: str) -> str:
        return value.strip().upper()


class StudyHighlight(BaseModel):
    annotation_id: str = Field(min_length=8, max_length=100)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    selected_text: str = Field(min_length=1)
    impact: int = Field(ge=-2, le=2)
    reason: str = Field(min_length=1, max_length=500)


class StudyPointwiseRequest(BaseModel):
    overall_score: int | None = Field(default=None, ge=-2, le=2)
    highlights: list[StudyHighlight] = Field(default_factory=list, max_length=6)
    primary_reason: str = Field(default="", max_length=1000)
    elapsed_ms: int = Field(default=0, ge=0)
    submit: bool = False


class StudyPairwiseRequest(BaseModel):
    choice: Literal["left", "right"] | None = None
    confidence: Literal["close", "fairly_sure", "very_sure"] | None = None
    selected_reason: str = Field(default="", max_length=1000)
    rejected_reason: str = Field(default="", max_length=1000)
    elapsed_ms: int = Field(default=0, ge=0)
    submit: bool = False


class StudyExposureRequest(BaseModel):
    value: Literal["no", "unsure", "yes"]


class StudyNavigateRequest(BaseModel):
    direction: Literal["previous"]


class StudyTransitionRequest(BaseModel):
    acknowledge: Literal[True]


class StudyCodeCreateRequest(BaseModel):
    count: int = Field(default=10, ge=1, le=200)
    study_id: str | None = Field(default=None, min_length=1, max_length=200)


class StudyFeedbackRequest(BaseModel):
    text: str = Field(default="", max_length=3000)


class StudyDemoChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class Study1Store:
    """Owns blinded assignment, drafts, exposure checks and admin export."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.db_path = self.data_dir / "system" / "personaforge.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.execute("PRAGMA synchronous = NORMAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS study1_participant_codes (
                    study_id TEXT NOT NULL,
                    code TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'available',
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(study_id, code)
                );

                CREATE TABLE IF NOT EXISTS study1_sessions (
                    id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    participant_code TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    assignment_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    current_index INTEGER NOT NULL DEFAULT 0,
                    prior_exposure TEXT,
                    exploratory_feedback TEXT NOT NULL DEFAULT '',
                    session_token_hash TEXT,
                    protocol_version TEXT NOT NULL DEFAULT 'legacy-v1',
                    phase2_started_at TEXT,
                    UNIQUE(study_id, participant_code)
                );

                CREATE TABLE IF NOT EXISTS study1_material_freezes (
                    study_id TEXT PRIMARY KEY,
                    material_sha256 TEXT NOT NULL,
                    item_count INTEGER NOT NULL,
                    frozen_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS study1_pointwise_responses (
                    session_id TEXT NOT NULL,
                    trial_id TEXT NOT NULL,
                    verdict TEXT,
                    highlights_json TEXT NOT NULL DEFAULT '[]',
                    global_note TEXT NOT NULL DEFAULT '',
                    elapsed_ms INTEGER NOT NULL DEFAULT 0,
                    revision_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'draft',
                    exposure TEXT,
                    overall_score INTEGER,
                    primary_reason TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    submitted_at TEXT,
                    PRIMARY KEY(session_id, trial_id),
                    FOREIGN KEY(session_id) REFERENCES study1_sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS study1_pairwise_responses (
                    session_id TEXT NOT NULL,
                    trial_id TEXT NOT NULL,
                    choice TEXT,
                    confidence TEXT,
                    left_highlights_json TEXT NOT NULL DEFAULT '[]',
                    right_highlights_json TEXT NOT NULL DEFAULT '[]',
                    global_note TEXT NOT NULL DEFAULT '',
                    elapsed_ms INTEGER NOT NULL DEFAULT 0,
                    revision_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'draft',
                    exposure TEXT,
                    selected_reason TEXT NOT NULL DEFAULT '',
                    rejected_reason TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    submitted_at TEXT,
                    PRIMARY KEY(session_id, trial_id),
                    FOREIGN KEY(session_id) REFERENCES study1_sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS study1_demo_turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_id TEXT,
                    query TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES study1_sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS study1_span_annotations (
                    annotation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    trial_id TEXT NOT NULL,
                    annotation_order INTEGER NOT NULL,
                    start INTEGER NOT NULL,
                    end INTEGER NOT NULL,
                    selected_text TEXT NOT NULL,
                    impact INTEGER NOT NULL CHECK(impact BETWEEN -2 AND 2),
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(session_id, trial_id, annotation_order),
                    FOREIGN KEY(session_id, trial_id)
                        REFERENCES study1_pointwise_responses(session_id, trial_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS study1_events (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    trial_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    client_elapsed_ms INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES study1_sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_study1_spans_session_trial
                    ON study1_span_annotations(session_id, trial_id);
                CREATE INDEX IF NOT EXISTS idx_study1_events_session
                    ON study1_events(session_id, created_at);
                """
            )
            columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(study1_sessions)").fetchall()
            }
            added_current_index = "current_index" not in columns
            if added_current_index:
                db.execute(
                    "ALTER TABLE study1_sessions ADD COLUMN current_index INTEGER NOT NULL DEFAULT 0"
                )
            if "prior_exposure" not in columns:
                db.execute("ALTER TABLE study1_sessions ADD COLUMN prior_exposure TEXT")
            if "session_token_hash" not in columns:
                db.execute(
                    "ALTER TABLE study1_sessions ADD COLUMN session_token_hash TEXT"
                )
            if "protocol_version" not in columns:
                db.execute(
                    "ALTER TABLE study1_sessions ADD COLUMN protocol_version TEXT NOT NULL DEFAULT 'legacy-v1'"
                )
            if "phase2_started_at" not in columns:
                db.execute("ALTER TABLE study1_sessions ADD COLUMN phase2_started_at TEXT")
            pointwise_columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(study1_pointwise_responses)").fetchall()
            }
            if "overall_score" not in pointwise_columns:
                db.execute("ALTER TABLE study1_pointwise_responses ADD COLUMN overall_score INTEGER")
            if "primary_reason" not in pointwise_columns:
                db.execute(
                    "ALTER TABLE study1_pointwise_responses ADD COLUMN primary_reason TEXT NOT NULL DEFAULT ''"
                )
            pairwise_columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(study1_pairwise_responses)").fetchall()
            }
            if "selected_reason" not in pairwise_columns:
                db.execute(
                    "ALTER TABLE study1_pairwise_responses ADD COLUMN selected_reason TEXT NOT NULL DEFAULT ''"
                )
            if "rejected_reason" not in pairwise_columns:
                db.execute(
                    "ALTER TABLE study1_pairwise_responses ADD COLUMN rejected_reason TEXT NOT NULL DEFAULT ''"
                )
            if added_current_index:
                sessions = db.execute(
                    "SELECT id, completed_at FROM study1_sessions"
                ).fetchall()
                for session in sessions:
                    if session["completed_at"]:
                        db.execute(
                            "UPDATE study1_sessions SET current_index = 5 WHERE id = ?",
                            (session["id"],),
                        )
                        continue
                    completed = 0
                    for table in (
                        "study1_pointwise_responses",
                        "study1_pairwise_responses",
                    ):
                        completed += int(
                            db.execute(
                                f"SELECT COUNT(*) FROM {table} WHERE session_id = ? AND status = 'submitted'",
                                (session["id"],),
                            ).fetchone()[0]
                        )
                    db.execute(
                        "UPDATE study1_sessions SET current_index = ? WHERE id = ?",
                        (min(completed, 5), session["id"]),
                    )

    def material_banks(self) -> list[Path]:
        return sorted((self.data_dir / "studies").glob("*/material_bank.json"))

    def _bank_validation_error(self, payload: dict[str, Any]) -> str:
        items = payload.get("items", [])
        if len(items) < TOTAL_TRIALS:
            return f"Study 1 至少需要 {TOTAL_TRIALS} 道完整问题"
        incomplete = []
        for item in items:
            responses = item.get("responses", {})
            if any(
                not str(responses.get(source, {}).get("text") or "").strip()
                for source in ALL_SOURCES
            ):
                incomplete.append(str(item.get("item_id") or "unknown"))
        if incomplete:
            preview = "、".join(incomplete[:3])
            suffix = "等" if len(incomplete) > 3 else ""
            return f"{len(incomplete)} 道题的五来源材料不完整：{preview}{suffix}"
        return ""

    def study_catalog(self) -> list[dict[str, Any]]:
        studies: list[dict[str, Any]] = []
        for path in self.material_banks():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                studies.append(
                    {
                        "study_id": path.parent.name,
                        "author": "",
                        "author_label": path.parent.name,
                        "item_count": 0,
                        "available": False,
                        "error": f"材料库无法读取：{exc}",
                    }
                )
                continue
            study_id = str(payload.get("study_id") or "").strip()
            protocol_version = str(payload.get("protocol_version") or "legacy-v1")
            author = str(payload.get("author", {}).get("token") or "").strip()
            stored_label = str(payload.get("author", {}).get("label") or "").strip()
            item_count = len(payload.get("items", []))
            error = self._bank_validation_error(payload)
            if not study_id:
                study_id = path.parent.name
                error = "材料库缺少 study_id"
            studies.append(
                {
                    "study_id": study_id,
                    "author": author,
                    "author_label": (
                        self._author_label(author, fallback=stored_label)
                        if author
                        else stored_label or path.parent.name
                    ),
                    "item_count": item_count,
                    "available": not error,
                    "protocol_version": protocol_version,
                    "recruitable": protocol_version == PROTOCOL_VERSION and not error,
                    "error": error or None,
                    "participant_path": f"/experiment/{study_id}",
                }
            )
        counts: dict[str, int] = {}
        for study in studies:
            study_id = str(study["study_id"])
            counts[study_id] = counts.get(study_id, 0) + 1
        for study in studies:
            if counts[str(study["study_id"])] > 1:
                study["available"] = False
                study["error"] = "study_id 与另一份材料库重复"
        return studies

    def load_bank(self, study_id: str | None = None) -> dict[str, Any]:
        banks = self.material_banks()
        if not banks:
            raise FileNotFoundError("未发现 Study 1 材料库")
        matches: list[dict[str, Any]] = []
        for path in banks:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if study_id is None or payload.get("study_id") == study_id:
                matches.append(payload)
                if study_id is None:
                    break
        if not matches:
            raise KeyError(study_id)
        if study_id is not None and len(matches) > 1:
            raise ValueError(f"存在重复的 study_id：{study_id}")
        payload = matches[0]
        validation_error = self._bank_validation_error(payload)
        if validation_error:
            raise ValueError(validation_error)
        return payload

    def public_meta(self, study_id: str | None = None) -> dict[str, Any]:
        try:
            bank = self.load_bank(study_id)
        except FileNotFoundError:
            return {"available": False, "title": "作者辨识实验"}
        author = str(bank.get("author", {}).get("token") or "")
        stored_label = str(bank.get("author", {}).get("label") or "").strip()
        avatar_url = self._author_avatar(author)
        return {
            "available": True,
            "study_id": bank["study_id"],
            "title": "作者辨识实验",
            "author": author,
            "author_label": self._author_label(author, fallback=stored_label),
            "avatar_url": avatar_url,
            "protocol_version": str(bank.get("protocol_version") or "legacy-v1"),
            "recruitable": str(bank.get("protocol_version") or "legacy-v1") == PROTOCOL_VERSION,
            "pointwise_count": POINTWISE_TRIALS,
            "pairwise_count": PAIRWISE_TRIALS,
            "total": TOTAL_TRIALS,
            "participant_path": f"/experiment/{bank['study_id']}",
        }

    def _author_label(self, author: str, *, fallback: str = "") -> str:
        for profile_path in (
            self.data_dir / "authors" / "zhihu" / author / "profile.json",
            self.data_dir / "authors" / "zhihu" / author / "raw" / "profile.json",
        ):
            if profile_path.exists():
                try:
                    profile = json.loads(profile_path.read_text(encoding="utf-8"))
                    return str(profile.get("nickname") or profile.get("name") or author)
                except (OSError, json.JSONDecodeError):
                    pass
        return fallback or author

    def _author_avatar(self, author: str) -> str | None:
        for profile_path in (
            self.data_dir / "authors" / "zhihu" / author / "profile.json",
            self.data_dir / "authors" / "zhihu" / author / "raw" / "profile.json",
        ):
            if profile_path.exists():
                try:
                    profile = json.loads(profile_path.read_text(encoding="utf-8"))
                    avatar = str(profile.get("avatar_url") or "").strip()
                    if avatar:
                        return avatar
                except (OSError, json.JSONDecodeError):
                    pass
        return None

    @staticmethod
    def _new_session_token() -> str:
        return secrets.token_urlsafe(32)

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _material_sha256(bank: dict[str, Any]) -> str:
        canonical = json.dumps(
            bank,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _freeze_material_bank(self, bank: dict[str, Any]) -> str:
        """Freeze a study's first recruitable material version permanently."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            return self._freeze_material_bank_in(db, bank)

    def _freeze_material_bank_in(
        self, db: sqlite3.Connection, bank: dict[str, Any]
    ) -> str:
        study_id = str(bank["study_id"])
        material_sha256 = self._material_sha256(bank)
        frozen = db.execute(
            "SELECT material_sha256 FROM study1_material_freezes WHERE study_id = ?",
            (study_id,),
        ).fetchone()
        if frozen is None:
            db.execute(
                "INSERT INTO study1_material_freezes(study_id, material_sha256, item_count, frozen_at) VALUES (?, ?, ?, ?)",
                (study_id, material_sha256, len(bank["items"]), now_iso()),
            )
            return material_sha256
        if not secrets.compare_digest(str(frozen["material_sha256"]), material_sha256):
            raise ValueError(
                "该 study_id 的材料已冻结且内容已变化；请创建新的 study_id 后再招募"
            )
        return material_sha256

    def create_codes(self, count: int, study_id: str | None = None) -> list[str]:
        bank = self.load_bank(study_id)
        if str(bank.get("protocol_version") or "legacy-v1") != PROTOCOL_VERSION:
            raise ValueError("当前材料库是旧协议，仅支持管理员回放和导出；请选择 V2 实验")
        self._freeze_material_bank(bank)
        alphabet = string.ascii_uppercase + string.digits
        created: list[str] = []
        with self.connect() as db:
            legacy = db.execute(
                "SELECT 1 FROM study1_sessions WHERE study_id = ? AND protocol_version != ? LIMIT 1",
                (bank["study_id"], PROTOCOL_VERSION),
            ).fetchone()
            if legacy is not None:
                raise ValueError("该 study_id 已包含旧协议数据；请复制材料并使用新的 study_id")
            while len(created) < count:
                code = "PF-" + "".join(secrets.choice(alphabet) for _ in range(4)) + "-" + "".join(
                    secrets.choice(alphabet) for _ in range(4)
                )
                try:
                    db.execute(
                        "INSERT INTO study1_participant_codes(study_id, code, created_at) VALUES (?, ?, ?)",
                        (bank["study_id"], code, now_iso()),
                    )
                except sqlite3.IntegrityError:
                    continue
                created.append(code)
        return created

    def list_codes(self, study_id: str | None = None) -> list[dict[str, Any]]:
        bank = self.load_bank(study_id)
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM study1_participant_codes WHERE study_id = ? ORDER BY created_at DESC",
                (bank["study_id"],),
            ).fetchall()
        return [dict(row) for row in rows]

    def start(
        self, profile: StudyProfileRequest, study_id: str | None = None
    ) -> dict[str, Any]:
        if not profile.consent:
            raise ValueError("请先确认知情同意")
        bank = self.load_bank(study_id)
        if str(bank.get("protocol_version") or "legacy-v1") != PROTOCOL_VERSION:
            raise ValueError("当前实验使用旧协议，不能开始新的参与者会话")
        code = profile.participant_code.strip().upper()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            code_row = db.execute(
                "SELECT * FROM study1_participant_codes WHERE study_id = ? AND code = ?",
                (bank["study_id"], code),
            ).fetchone()
            if code_row is None:
                raise ValueError("参与码不存在，请联系研究者")
            self._freeze_material_bank_in(db, bank)
            existing = db.execute(
                "SELECT id FROM study1_sessions WHERE study_id = ? AND participant_code = ?",
                (bank["study_id"], code),
            ).fetchone()
            if existing is not None:
                # A participant code is the recovery credential. Rotating the browser
                # token here lets the same participant resume from another device while
                # invalidating a stale token copied from an old browser session.
                session_id = str(existing["id"])
                existing_session = self._session_in(db, session_id)
                if str(existing_session["protocol_version"]) != PROTOCOL_VERSION:
                    raise ValueError("旧协议会话只支持管理员回放和导出，不能继续参与")
                resume_token = self._new_session_token()
                db.execute(
                    "UPDATE study1_sessions SET session_token_hash = ?, updated_at = ? WHERE id = ?",
                    (self._token_hash(resume_token), now_iso(), session_id),
                )
            else:
                legacy = db.execute(
                    "SELECT 1 FROM study1_sessions WHERE study_id = ? AND protocol_version != ? LIMIT 1",
                    (bank["study_id"], PROTOCOL_VERSION),
                ).fetchone()
                if legacy is not None:
                    raise ValueError("该 study_id 已包含旧协议数据；请使用新的 study_id")
                session_id = f"study1-{uuid4().hex}"
                resume_token = self._new_session_token()
                assignment = build_assignment(bank, code)
                assignment["material_sha256"] = self._material_sha256(bank)
                assignment["author_label"] = self._author_label(
                    assignment["author"],
                    fallback=str(bank.get("author", {}).get("label") or "").strip(),
                )
                now = now_iso()
                db.execute(
                    """
                    INSERT INTO study1_sessions(
                        id, study_id, participant_code, profile_json, assignment_json,
                        created_at, updated_at, session_token_hash, protocol_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        bank["study_id"],
                        code,
                        profile.model_dump_json(),
                        json.dumps(assignment, ensure_ascii=False),
                        now,
                        now,
                        self._token_hash(resume_token),
                        PROTOCOL_VERSION,
                    ),
                )
                db.execute(
                    "UPDATE study1_participant_codes SET status = 'started', session_id = ? WHERE study_id = ? AND code = ?",
                    (session_id, bank["study_id"], code),
                )
        result = self.state(session_id)
        result["resume_token"] = resume_token
        return result

    def authorize_session(self, session_id: str, resume_token: str | None) -> None:
        """Verify the participant-specific credential before exposing a blind session."""
        if not resume_token:
            raise PermissionError("实验会话已过期，请使用参与码重新进入")
        session = self._session(session_id)
        stored_hash = str(session["session_token_hash"] or "")
        actual_hash = self._token_hash(resume_token)
        if not stored_hash or not secrets.compare_digest(stored_hash, actual_hash):
            raise PermissionError("实验会话凭据无效，请使用参与码重新进入")

    def state(self, session_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        assignment = json.loads(str(session["assignment_json"]))
        protocol_version = str(session["protocol_version"] or "legacy-v1")
        total = len(self._ordered_trials(assignment))
        if session["completed_at"]:
            return {
                "session_id": session_id,
                "study_id": session["study_id"],
                "phase": "completed",
                "protocol_version": protocol_version,
                "progress": {"completed": total, "total": total},
                "author": assignment["author"],
                "author_label": assignment["author_label"],
                "prior_exposure": session["prior_exposure"],
                "exploratory_feedback": session["exploratory_feedback"],
                "demo_turn_count": self.demo_turn_count(session_id),
                "demo_turn_limit": 3,
            }
        index = max(0, min(int(session["current_index"]), total))
        if (
            protocol_version == PROTOCOL_VERSION
            and index == POINTWISE_TRIALS
            and not session["phase2_started_at"]
        ):
            return {
                "session_id": session_id,
                "study_id": session["study_id"],
                "protocol_version": protocol_version,
                "phase": "transition",
                "progress": {"completed": index, "total": total},
                "can_previous": True,
                "author": assignment["author"],
                "author_label": assignment["author_label"],
            }
        if index == total:
            return {
                "session_id": session_id,
                "study_id": session["study_id"],
                "phase": "exposure",
                "protocol_version": protocol_version,
                "progress": {"completed": total, "total": total},
                "can_previous": True,
                "author": assignment["author"],
                "author_label": assignment["author_label"],
            }
        kind, trial = self._ordered_trials(assignment)[index]
        row = self._response(f"study1_{kind}_responses", session_id, trial["trial_id"])
        return self._state_payload(session, assignment, kind, trial, row, index)

    def acknowledge_transition(
        self, session_id: str, payload: StudyTransitionRequest
    ) -> dict[str, Any]:
        del payload
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            session = self._session_in(db, session_id)
            if str(session["protocol_version"]) != PROTOCOL_VERSION:
                raise ValueError("旧协议会话没有阶段过渡页")
            if int(session["current_index"]) != POINTWISE_TRIALS:
                raise ValueError("当前不在阶段过渡位置")
            now = now_iso()
            if not session["phase2_started_at"]:
                db.execute(
                    "UPDATE study1_sessions SET phase2_started_at = ?, updated_at = ? WHERE id = ?",
                    (now, now, session_id),
                )
                self._record_event_in(
                    db,
                    session_id=session_id,
                    trial_id=None,
                    event_type="phase2_entered",
                    payload={},
                    elapsed_ms=0,
                )
        return self.state(session_id)

    def reserve_demo_turn(self, session_id: str, query: str) -> dict[str, str]:
        session = self._session(session_id)
        if not session["completed_at"]:
            raise ValueError("正式任务完成后才能自由体验")
        assignment = json.loads(str(session["assignment_json"]))
        reservation_id = f"study-demo-{uuid4().hex}"
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = int(
                db.execute(
                    "SELECT COUNT(*) FROM study1_demo_turns WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            )
            if count >= 3:
                raise ValueError("自由体验最多进行三轮")
            db.execute(
                "INSERT INTO study1_demo_turns(id, session_id, query, created_at) VALUES (?, ?, ?, ?)",
                (reservation_id, session_id, query.strip(), now_iso()),
            )
        return {
            "reservation_id": reservation_id,
            "author": assignment["author"],
            "conversation_id": f"study-demo-conversation-{session_id}",
            "owner_id": f"study-participant:{session_id}",
        }

    def attach_demo_turn(self, reservation_id: str, turn_id: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE study1_demo_turns SET turn_id = ? WHERE id = ?",
                (turn_id, reservation_id),
            )

    def cancel_demo_turn(self, reservation_id: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM study1_demo_turns WHERE id = ?", (reservation_id,))

    def demo_turn_count(self, session_id: str) -> int:
        with self.connect() as db:
            return int(
                db.execute(
                    "SELECT COUNT(*) FROM study1_demo_turns WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            )

    def _state_payload(
        self,
        session: sqlite3.Row,
        assignment: dict[str, Any],
        kind: str,
        trial: dict[str, Any],
        row: sqlite3.Row | None,
        index: int,
    ) -> dict[str, Any]:
        public = {
            "kind": kind,
            "trial_id": trial["trial_id"],
            "question": trial["question"],
        }
        if kind == "pointwise":
            public["answer"] = trial["answer"]
            draft = {
                "overall_score": row["overall_score"] if row else None,
                "highlights": self._annotations(session["id"], trial["trial_id"]),
                "primary_reason": row["primary_reason"] if row else "",
                "elapsed_ms": int(row["elapsed_ms"]) if row else 0,
            }
        else:
            public.update(left_answer=trial["left"]["text"], right_answer=trial["right"]["text"])
            draft = {
                "choice": row["choice"] if row else None,
                "confidence": row["confidence"] if row else None,
                "selected_reason": row["selected_reason"] if row else "",
                "rejected_reason": row["rejected_reason"] if row else "",
                "elapsed_ms": int(row["elapsed_ms"]) if row else 0,
            }
        total = len(self._ordered_trials(assignment))
        pointwise_total = len(assignment["pointwise"])
        pairwise_total = len(assignment["pairwise"])
        return {
            "session_id": session["id"],
            "study_id": session["study_id"],
            "protocol_version": session["protocol_version"],
            "phase": kind,
            "progress": {"completed": index, "total": total},
            "phase_progress": {
                "completed": index if kind == "pointwise" else index - pointwise_total,
                "total": pointwise_total if kind == "pointwise" else pairwise_total,
            },
            "can_previous": index > 0,
            "final_trial": index == total - 1,
            "author": assignment["author"],
            "author_label": assignment["author_label"],
            "trial": public,
            "draft": draft,
        }

    def save_pointwise(
        self, session_id: str, trial_id: str, payload: StudyPointwiseRequest
    ) -> dict[str, Any]:
        trial = self._current_trial(session_id, "pointwise", trial_id)
        highlights = [item.model_dump() for item in payload.highlights]
        validate_highlights(trial["answer"], highlights, maximum=6)
        reason = payload.primary_reason.strip()
        if payload.submit and payload.overall_score is None:
            raise ValueError("提交前请完成整篇作者相似度评分")
        if payload.submit and not highlights:
            raise ValueError("提交前请至少标注一处影响判断的文字")
        if payload.submit and not reason:
            raise ValueError("提交前请填写整篇最关键的判断理由")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            session, index = self._assert_current_in(db, session_id, "pointwise", trial_id)
            current = db.execute(
                "SELECT * FROM study1_pointwise_responses WHERE session_id = ? AND trial_id = ?",
                (session_id, trial_id),
            ).fetchone()
            old_score = current["overall_score"] if current else None
            old_annotations = {
                str(row["annotation_id"]): dict(row)
                for row in db.execute(
                    "SELECT * FROM study1_span_annotations WHERE session_id = ? AND trial_id = ?",
                    (session_id, trial_id),
                ).fetchall()
            }
            now = now_iso()
            revision = int(current["revision_count"]) + 1 if current else 1
            db.execute(
                """
                INSERT INTO study1_pointwise_responses(
                    session_id, trial_id, verdict, highlights_json, global_note,
                    elapsed_ms, revision_count, status, exposure, overall_score,
                    primary_reason, updated_at, submitted_at
                ) VALUES (?, ?, NULL, ?, '', ?, ?, ?, NULL, ?, ?, ?, ?)
                ON CONFLICT(session_id, trial_id) DO UPDATE SET
                    highlights_json=excluded.highlights_json,
                    elapsed_ms=excluded.elapsed_ms,
                    revision_count=excluded.revision_count,
                    status=excluded.status,
                    overall_score=excluded.overall_score,
                    primary_reason=excluded.primary_reason,
                    updated_at=excluded.updated_at,
                    submitted_at=excluded.submitted_at
                """,
                (
                    session_id,
                    trial_id,
                    json.dumps(highlights, ensure_ascii=False),
                    payload.elapsed_ms,
                    revision,
                    "submitted" if payload.submit else "draft",
                    payload.overall_score,
                    reason,
                    now,
                    now if payload.submit else None,
                ),
            )
            db.execute(
                "DELETE FROM study1_span_annotations WHERE session_id = ? AND trial_id = ?",
                (session_id, trial_id),
            )
            for order, item in enumerate(highlights):
                annotation_id = str(item["annotation_id"])
                created_at = str(old_annotations.get(annotation_id, {}).get("created_at") or now)
                db.execute(
                    """
                    INSERT INTO study1_span_annotations(
                        annotation_id, session_id, trial_id, annotation_order,
                        start, end, selected_text, impact, reason, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        annotation_id,
                        session_id,
                        trial_id,
                        order,
                        item["start"],
                        item["end"],
                        item["selected_text"],
                        item["impact"],
                        str(item["reason"]).strip(),
                        created_at,
                        now,
                    ),
                )
            self._record_pointwise_events_in(
                db,
                session_id=session_id,
                trial_id=trial_id,
                old_score=old_score,
                new_score=payload.overall_score,
                old_annotations=old_annotations,
                new_annotations=highlights,
                elapsed_ms=payload.elapsed_ms,
            )
            if payload.submit:
                self._record_event_in(
                    db,
                    session_id=session_id,
                    trial_id=trial_id,
                    event_type="trial_submitted",
                    payload={"kind": "pointwise"},
                    elapsed_ms=payload.elapsed_ms,
                )
            self._finish_save_in(db, session, index, payload.submit, now)
        return self.state(session_id)

    def save_pairwise(
        self, session_id: str, trial_id: str, payload: StudyPairwiseRequest
    ) -> dict[str, Any]:
        self._current_trial(session_id, "pairwise", trial_id)
        selected_reason = payload.selected_reason.strip()
        rejected_reason = payload.rejected_reason.strip()
        if payload.submit and (not payload.choice or not payload.confidence):
            raise ValueError("提交前请选择更像作者的回答并报告判断信心")
        if payload.submit and (not selected_reason or not rejected_reason):
            raise ValueError("提交前请分别填写选择理由和不选择理由")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            session, index = self._assert_current_in(db, session_id, "pairwise", trial_id)
            current = db.execute(
                "SELECT * FROM study1_pairwise_responses WHERE session_id = ? AND trial_id = ?",
                (session_id, trial_id),
            ).fetchone()
            old_choice = current["choice"] if current else None
            old_confidence = current["confidence"] if current else None
            revision = int(current["revision_count"]) + 1 if current else 1
            now = now_iso()
            db.execute(
                """
                INSERT INTO study1_pairwise_responses(
                    session_id, trial_id, choice, confidence,
                    left_highlights_json, right_highlights_json, global_note,
                    elapsed_ms, revision_count, status, exposure,
                    selected_reason, rejected_reason, updated_at, submitted_at
                ) VALUES (?, ?, ?, ?, '[]', '[]', '', ?, ?, ?, NULL, ?, ?, ?, ?)
                ON CONFLICT(session_id, trial_id) DO UPDATE SET
                    choice=excluded.choice,
                    confidence=excluded.confidence,
                    elapsed_ms=excluded.elapsed_ms,
                    revision_count=excluded.revision_count,
                    status=excluded.status,
                    selected_reason=excluded.selected_reason,
                    rejected_reason=excluded.rejected_reason,
                    updated_at=excluded.updated_at,
                    submitted_at=excluded.submitted_at
                """,
                (
                    session_id,
                    trial_id,
                    payload.choice,
                    payload.confidence,
                    payload.elapsed_ms,
                    revision,
                    "submitted" if payload.submit else "draft",
                    selected_reason,
                    rejected_reason,
                    now,
                    now if payload.submit else None,
                ),
            )
            self._record_pairwise_events_in(
                db,
                session_id=session_id,
                trial_id=trial_id,
                old_choice=old_choice,
                new_choice=payload.choice,
                old_confidence=old_confidence,
                new_confidence=payload.confidence,
                elapsed_ms=payload.elapsed_ms,
            )
            if payload.submit:
                self._record_event_in(
                    db,
                    session_id=session_id,
                    trial_id=trial_id,
                    event_type="trial_submitted",
                    payload={"kind": "pairwise"},
                    elapsed_ms=payload.elapsed_ms,
                )
            self._finish_save_in(db, session, index, payload.submit, now)
        return self.state(session_id)

    def save_exposure(self, session_id: str, payload: StudyExposureRequest) -> dict[str, Any]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            session = self._session_in(db, session_id)
            if session["completed_at"]:
                return self.state(session_id)
            assignment = json.loads(str(session["assignment_json"]))
            total = len(self._ordered_trials(assignment))
            if int(session["current_index"]) != total:
                raise ValueError("请先完成全部题目")
            for kind, trial in self._ordered_trials(assignment):
                row = db.execute(
                    f"SELECT status FROM study1_{kind}_responses WHERE session_id = ? AND trial_id = ?",
                    (session_id, trial["trial_id"]),
                ).fetchone()
                if row is None or row["status"] != "submitted":
                    raise ValueError("仍有题目尚未完成")
            now = now_iso()
            db.execute(
                "UPDATE study1_sessions SET prior_exposure = ?, completed_at = ?, updated_at = ? WHERE id = ?",
                (payload.value, now, now, session_id),
            )
            db.execute(
                "UPDATE study1_participant_codes SET status = 'completed' WHERE session_id = ?",
                (session_id,),
            )
            self._record_event_in(
                db,
                session_id=session_id,
                trial_id=None,
                event_type="study_submitted",
                payload={"prior_exposure": payload.value},
                elapsed_ms=0,
            )
        return self.state(session_id)

    def navigate_previous(self, session_id: str) -> dict[str, Any]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            session = self._session_in(db, session_id)
            if session["completed_at"]:
                raise ValueError("实验已经提交，不能返回修改")
            index = int(session["current_index"])
            if index <= 0:
                raise ValueError("已经是第一题")
            db.execute(
                "UPDATE study1_sessions SET current_index = ?, updated_at = ? WHERE id = ?",
                (index - 1, now_iso(), session_id),
            )
        return self.state(session_id)

    def save_feedback(self, session_id: str, text: str) -> dict[str, Any]:
        session = self._session(session_id)
        if not session["completed_at"]:
            raise ValueError("正式任务完成后才能填写体验反馈")
        with self.connect() as db:
            db.execute(
                "UPDATE study1_sessions SET exploratory_feedback = ?, updated_at = ? WHERE id = ?",
                (text.strip(), now_iso(), session_id),
            )
        return {"saved": True}

    def overview(self, study_id: str | None = None) -> dict[str, Any]:
        bank = self.load_bank(study_id)
        with self.connect() as db:
            sessions = db.execute(
                "SELECT * FROM study1_sessions WHERE study_id = ? ORDER BY created_at DESC",
                (bank["study_id"],),
            ).fetchall()
        items = []
        for row in sessions:
            state = self.state(str(row["id"]))
            items.append(
                {
                    "session_id": row["id"],
                    "participant_code": row["participant_code"],
                    "completed": state["progress"]["completed"],
                    "total": state["progress"]["total"],
                    "protocol_version": row["protocol_version"],
                    "status": "completed" if row["completed_at"] else "in_progress",
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return {
            "study": self.public_meta(str(bank["study_id"])),
            "study_id": bank["study_id"],
            "codes": self.list_codes(str(bank["study_id"])),
            "sessions": items,
        }

    def detail(self, session_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        assignment = json.loads(str(session["assignment_json"]))
        responses: dict[str, Any] = {"pointwise": [], "pairwise": []}
        for kind in responses:
            table = f"study1_{kind}_responses"
            for trial in assignment[kind]:
                row = self._response(table, session_id, trial["trial_id"])
                item = {"trial": trial, "response": dict(row) if row else None}
                if kind == "pointwise":
                    item["annotations"] = self._annotations(session_id, trial["trial_id"])
                responses[kind].append(item)
        with self.connect() as db:
            events = [
                {**dict(row), "payload": json.loads(str(row["payload_json"]))}
                for row in db.execute(
                    "SELECT * FROM study1_events WHERE session_id = ? ORDER BY created_at, id",
                    (session_id,),
                ).fetchall()
            ]
        return {
            "session_id": session_id,
            "study_id": session["study_id"],
            "author": assignment["author"],
            "author_label": assignment["author_label"],
            "participant_code": session["participant_code"],
            "profile": json.loads(str(session["profile_json"])),
            "created_at": session["created_at"],
            "completed_at": session["completed_at"],
            "prior_exposure": session["prior_exposure"],
            "exploratory_feedback": session["exploratory_feedback"],
            "material_sha256": assignment.get("material_sha256"),
            "protocol_version": session["protocol_version"],
            "phase2_started_at": session["phase2_started_at"],
            "events": events,
            **responses,
        }

    def export(
        self, format: str, study_id: str | None = None
    ) -> tuple[str, str, str]:
        if format not in {"jsonl", "csv"}:
            raise ValueError("format 必须是 jsonl 或 csv")
        overview = self.overview(study_id)
        records: list[dict[str, Any]] = []
        for session in overview["sessions"]:
            detail = self.detail(session["session_id"])
            for kind in ("pointwise", "pairwise"):
                for item in detail[kind]:
                    records.append(
                        {
                            "session_id": detail["session_id"],
                            "study_id": detail["study_id"],
                            "author": detail["author"],
                            "participant_code": detail["participant_code"],
                            "material_sha256": detail["material_sha256"],
                            "kind": kind,
                            "trial": item["trial"],
                            "response": item["response"],
                            "profile": detail["profile"],
                            "prior_exposure": detail["prior_exposure"],
                        }
                    )
        safe_study_id = "".join(
            character
            for character in str(overview["study_id"])
            if character.isalnum() or character in {"-", "_", "."}
        ) or "study"
        if format == "jsonl":
            return (
                "\n".join(json.dumps(item, ensure_ascii=False) for item in records),
                "application/x-ndjson",
                f"study1-{safe_study_id}-responses.jsonl",
            )
        output = io.StringIO()
        fields = ["study_id", "author", "material_sha256", "session_id", "participant_code", "kind", "trial_id", "question_id", "hidden_sources", "response_json", "profile_json", "prior_exposure"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for item in records:
            trial = item["trial"]
            if item["kind"] == "pointwise":
                sources = trial["hidden_source"]
            else:
                sources = f"{trial['left']['hidden_source']}|{trial['right']['hidden_source']}"
            writer.writerow(
                {
                    "study_id": item["study_id"],
                    "author": item["author"],
                    "material_sha256": item["material_sha256"],
                    "session_id": item["session_id"],
                    "participant_code": item["participant_code"],
                    "kind": item["kind"],
                    "trial_id": trial["trial_id"],
                    "question_id": trial["question_id"],
                    "hidden_sources": sources,
                    "response_json": json.dumps(item["response"], ensure_ascii=False),
                    "profile_json": json.dumps(item["profile"], ensure_ascii=False),
                    "prior_exposure": item["prior_exposure"],
                }
            )
        return (
            output.getvalue(),
            "text/csv; charset=utf-8",
            f"study1-{safe_study_id}-responses.csv",
        )

    def analysis_bundle(self, study_id: str | None = None) -> tuple[bytes, str]:
        overview = self.overview(study_id)
        sessions_rows: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []
        span_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        raw_records: list[dict[str, Any]] = []
        coding_rows: list[dict[str, Any]] = []
        for session_summary in overview["sessions"]:
            detail = self.detail(str(session_summary["session_id"]))
            profile = detail["profile"]
            sessions_rows.append(
                {
                    "study_id": detail["study_id"],
                    "session_id": detail["session_id"],
                    "participant_code": detail["participant_code"],
                    "author": detail["author"],
                    "protocol_version": detail["protocol_version"],
                    "material_sha256": detail["material_sha256"],
                    "follow_duration": profile.get("follow_duration"),
                    "reading_frequency": profile.get("reading_frequency"),
                    "familiarity": profile.get("familiarity"),
                    "ai_frequency": profile.get("ai_frequency"),
                    "prior_exposure": detail["prior_exposure"],
                    "created_at": detail["created_at"],
                    "phase2_started_at": detail["phase2_started_at"],
                    "completed_at": detail["completed_at"],
                }
            )
            for kind in ("pointwise", "pairwise"):
                for item in detail[kind]:
                    trial = item["trial"]
                    response = item["response"] or {}
                    base = {
                        "study_id": detail["study_id"],
                        "session_id": detail["session_id"],
                        "participant_code": detail["participant_code"],
                        "author": detail["author"],
                        "protocol_version": detail["protocol_version"],
                        "material_sha256": detail["material_sha256"],
                        "kind": kind,
                        "trial_id": trial["trial_id"],
                        "question_id": trial["question_id"],
                        "question": trial["question"],
                        "status": response.get("status"),
                        "elapsed_ms": response.get("elapsed_ms"),
                        "revision_count": response.get("revision_count"),
                        "submitted_at": response.get("submitted_at"),
                    }
                    if kind == "pointwise":
                        row = {
                            **base,
                            "pair_type": "",
                            "response_id": trial.get("response_id", ""),
                            "hidden_source": trial.get("hidden_source", ""),
                            "left_response_id": "",
                            "left_hidden_source": "",
                            "right_response_id": "",
                            "right_hidden_source": "",
                            "overall_score": response.get("overall_score"),
                            "choice": "",
                            "chosen_source": "",
                            "confidence": "",
                            "primary_reason": response.get("primary_reason", ""),
                            "selected_reason": "",
                            "rejected_reason": "",
                        }
                        evidence_id = f"reason:{detail['session_id']}:{trial['trial_id']}:primary"
                        if row["primary_reason"]:
                            coding_rows.append(
                                self._coding_template_row(
                                    evidence_id,
                                    detail,
                                    trial,
                                    "pointwise_primary_reason",
                                    str(row["primary_reason"]),
                                )
                            )
                        for annotation in item.get("annotations", []):
                            span_rows.append(
                                {
                                    **{key: base[key] for key in (
                                        "study_id", "session_id", "participant_code", "author",
                                        "protocol_version", "material_sha256", "trial_id", "question_id"
                                    )},
                                    **annotation,
                                }
                            )
                            coding_rows.append(
                                self._coding_template_row(
                                    str(annotation["annotation_id"]),
                                    detail,
                                    trial,
                                    "pointwise_span",
                                    str(annotation["reason"]),
                                )
                            )
                    else:
                        choice = str(response.get("choice") or "")
                        chosen_side = trial.get(choice) if choice in {"left", "right"} else None
                        row = {
                            **base,
                            "pair_type": trial.get("hidden_pair_type", ""),
                            "response_id": "",
                            "hidden_source": "",
                            "left_response_id": trial["left"].get("response_id", ""),
                            "left_hidden_source": trial["left"].get("hidden_source", ""),
                            "right_response_id": trial["right"].get("response_id", ""),
                            "right_hidden_source": trial["right"].get("hidden_source", ""),
                            "overall_score": "",
                            "choice": choice,
                            "chosen_source": chosen_side.get("hidden_source", "") if chosen_side else "",
                            "confidence": response.get("confidence", ""),
                            "primary_reason": "",
                            "selected_reason": response.get("selected_reason", ""),
                            "rejected_reason": response.get("rejected_reason", ""),
                        }
                        for role in ("selected_reason", "rejected_reason"):
                            if row[role]:
                                coding_rows.append(
                                    self._coding_template_row(
                                        f"reason:{detail['session_id']}:{trial['trial_id']}:{role}",
                                        detail,
                                        trial,
                                        f"pairwise_{role}",
                                        str(row[role]),
                                    )
                                )
                    trial_rows.append(row)
                    raw_records.append({**base, "trial": trial, "response": response, "annotations": item.get("annotations", [])})
            for event in detail["events"]:
                event_rows.append(
                    {
                        "study_id": detail["study_id"],
                        "participant_code": detail["participant_code"],
                        **event,
                    }
                )

        safe_id = "".join(
            character for character in str(overview["study_id"])
            if character.isalnum() or character in {"-", "_", "."}
        ) or "study"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("sessions.csv", self._dicts_to_csv(sessions_rows))
            archive.writestr("trials.csv", self._dicts_to_csv(trial_rows))
            archive.writestr("span_annotations.csv", self._dicts_to_csv(span_rows))
            archive.writestr("feature_coding_template.csv", self._dicts_to_csv(coding_rows))
            archive.writestr(
                "events.jsonl",
                "\n".join(json.dumps(row, ensure_ascii=False) for row in event_rows),
            )
            archive.writestr(
                "raw.jsonl",
                "\n".join(json.dumps(row, ensure_ascii=False) for row in raw_records),
            )
            archive.writestr("data_dictionary.md", self._data_dictionary())
        return buffer.getvalue(), f"study1-{safe_id}-analysis.zip"

    @staticmethod
    def _dicts_to_csv(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()

    @staticmethod
    def _coding_template_row(
        evidence_id: str,
        detail: dict[str, Any],
        trial: dict[str, Any],
        evidence_role: str,
        evidence_text: str,
    ) -> dict[str, Any]:
        return {
            "evidence_id": evidence_id,
            "study_id": detail["study_id"],
            "session_id": detail["session_id"],
            "participant_code": detail["participant_code"],
            "author": detail["author"],
            "trial_id": trial["trial_id"],
            "question_id": trial["question_id"],
            "evidence_role": evidence_role,
            "evidence_text": evidence_text,
            "coder_id": "",
            "feature_dimension": "",
            "feature_realization": "",
            "is_primary": "",
            "coder_note": "",
        }

    @staticmethod
    def _data_dictionary() -> str:
        return """# Study 1 V2 数据字典

## 文件

- `sessions.csv`：一行一个参与者会话；人口学字段仅包含实验约定的阅读与 AI 使用背景。
- `trials.csv`：一行一个单篇或配对 trial，包含盲化来源、最终判断、理由与耗时。
- `span_annotations.csv`：一行一处单篇划线；`impact` 范围为 -2～+2。
- `feature_coding_template.csv`：研究者人工编码模板；可为同一 evidence 复制多行进行多标签编码。
- `events.jsonl`：评分修改、划线增删改、阶段进入和提交事件，仅用于过程质量检查。
- `raw.jsonl`：完整 trial 快照，用于审计和复现。

## 关键边界

- `overall_score` 是整篇结果变量；`impact` 是局部证据方向，不得相乘。
- `hidden_source`、`left_hidden_source`、`right_hidden_source` 只供研究者分析，从不返回参与者界面。
- 空字符串表示该字段不适用于当前 trial；缺失 response 表示尚未保存。
- Study 1 结果是探索性关联和辨别证据，不单独证明 feature 的因果贡献。
"""

    def _session(self, session_id: str) -> sqlite3.Row:
        with self.connect() as db:
            return self._session_in(db, session_id)

    @staticmethod
    def _session_in(db: sqlite3.Connection, session_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM study1_sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        return row

    def _response(self, table: str, session_id: str, trial_id: str) -> sqlite3.Row | None:
        with self.connect() as db:
            return db.execute(
                f"SELECT * FROM {table} WHERE session_id = ? AND trial_id = ?",
                (session_id, trial_id),
            ).fetchone()

    def _annotations(self, session_id: str, trial_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT annotation_id, start, end, selected_text, impact, reason
                FROM study1_span_annotations
                WHERE session_id = ? AND trial_id = ?
                ORDER BY annotation_order
                """,
                (session_id, trial_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def _assert_current_in(
        self,
        db: sqlite3.Connection,
        session_id: str,
        kind: Literal["pointwise", "pairwise"],
        trial_id: str,
    ) -> tuple[sqlite3.Row, int]:
        session = self._session_in(db, session_id)
        if session["completed_at"]:
            raise ValueError("实验已经提交")
        assignment = json.loads(str(session["assignment_json"]))
        index = int(session["current_index"])
        ordered = self._ordered_trials(assignment)
        if index >= len(ordered):
            raise ValueError("全部题目已经完成")
        expected_kind, expected_trial = ordered[index]
        if expected_kind != kind or expected_trial["trial_id"] != trial_id:
            raise ValueError("题目状态已变化，请刷新后继续")
        if kind == "pairwise" and not session["phase2_started_at"]:
            raise ValueError("请先确认第二阶段说明")
        return session, index

    @staticmethod
    def _finish_save_in(
        db: sqlite3.Connection,
        session: sqlite3.Row,
        index: int,
        submit: bool,
        now: str,
    ) -> None:
        if submit:
            db.execute(
                "UPDATE study1_sessions SET current_index = ?, updated_at = ? WHERE id = ?",
                (index + 1, now, session["id"]),
            )
        else:
            db.execute(
                "UPDATE study1_sessions SET updated_at = ? WHERE id = ?",
                (now, session["id"]),
            )

    @staticmethod
    def _record_event_in(
        db: sqlite3.Connection,
        *,
        session_id: str,
        trial_id: str | None,
        event_type: str,
        payload: dict[str, Any],
        elapsed_ms: int,
    ) -> None:
        db.execute(
            """
            INSERT INTO study1_events(
                id, session_id, trial_id, event_type, payload_json,
                client_elapsed_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"event-{uuid4().hex}",
                session_id,
                trial_id,
                event_type,
                json.dumps(payload, ensure_ascii=False),
                elapsed_ms,
                now_iso(),
            ),
        )

    def _record_pointwise_events_in(
        self,
        db: sqlite3.Connection,
        *,
        session_id: str,
        trial_id: str,
        old_score: int | None,
        new_score: int | None,
        old_annotations: dict[str, dict[str, Any]],
        new_annotations: list[dict[str, Any]],
        elapsed_ms: int,
    ) -> None:
        if old_score != new_score and new_score is not None:
            self._record_event_in(
                db,
                session_id=session_id,
                trial_id=trial_id,
                event_type="score_selected" if old_score is None else "score_changed",
                payload={"from": old_score, "to": new_score},
                elapsed_ms=elapsed_ms,
            )
        new_by_id = {str(item["annotation_id"]): item for item in new_annotations}
        for annotation_id in old_annotations.keys() - new_by_id.keys():
            self._record_event_in(
                db,
                session_id=session_id,
                trial_id=trial_id,
                event_type="annotation_deleted",
                payload={"annotation_id": annotation_id},
                elapsed_ms=elapsed_ms,
            )
        for annotation_id, item in new_by_id.items():
            old = old_annotations.get(annotation_id)
            comparable = {
                "start": item["start"],
                "end": item["end"],
                "selected_text": item["selected_text"],
                "impact": item["impact"],
                "reason": str(item["reason"]).strip(),
            }
            if old is None:
                event_type = "annotation_added"
            else:
                previous = {key: old[key] for key in comparable}
                if previous == comparable:
                    continue
                event_type = "annotation_updated"
            self._record_event_in(
                db,
                session_id=session_id,
                trial_id=trial_id,
                event_type=event_type,
                payload={"annotation_id": annotation_id, **comparable},
                elapsed_ms=elapsed_ms,
            )

    def _record_pairwise_events_in(
        self,
        db: sqlite3.Connection,
        *,
        session_id: str,
        trial_id: str,
        old_choice: str | None,
        new_choice: str | None,
        old_confidence: str | None,
        new_confidence: str | None,
        elapsed_ms: int,
    ) -> None:
        if old_choice != new_choice:
            self._record_event_in(
                db,
                session_id=session_id,
                trial_id=trial_id,
                event_type="pair_choice_selected" if old_choice is None else "pair_choice_changed",
                payload={"old": old_choice, "new": new_choice},
                elapsed_ms=elapsed_ms,
            )
        if old_confidence != new_confidence:
            self._record_event_in(
                db,
                session_id=session_id,
                trial_id=trial_id,
                event_type="pair_confidence_selected" if old_confidence is None else "pair_confidence_changed",
                payload={"old": old_confidence, "new": new_confidence},
                elapsed_ms=elapsed_ms,
            )

    @staticmethod
    def _ordered_trials(assignment: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        return [
            *[("pointwise", trial) for trial in assignment["pointwise"]],
            *[("pairwise", trial) for trial in assignment["pairwise"]],
        ]

    def _current_trial(self, session_id: str, kind: str, trial_id: str) -> dict[str, Any]:
        state = self.state(session_id)
        if state.get("phase") != kind or state.get("trial", {}).get("trial_id") != trial_id:
            raise ValueError("只能保存当前题目")
        assignment = json.loads(str(self._session(session_id)["assignment_json"]))
        return next(item for item in assignment[kind] if item["trial_id"] == trial_id)

    def _save_response(
        self,
        table: str,
        kind: Literal["pointwise", "pairwise"],
        session_id: str,
        trial_id: str,
        fields: dict[str, Any],
        submit: bool,
    ) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            session = self._session_in(db, session_id)
            if session["completed_at"]:
                raise ValueError("实验已经提交")
            assignment = json.loads(str(session["assignment_json"]))
            index = int(session["current_index"])
            ordered = self._ordered_trials(assignment)
            if index >= len(ordered):
                raise ValueError("全部题目已经完成")
            expected_kind, expected_trial = ordered[index]
            if expected_kind != kind or expected_trial["trial_id"] != trial_id:
                raise ValueError("题目状态已变化，请刷新后继续")
            current = db.execute(
                f"SELECT revision_count FROM {table} WHERE session_id = ? AND trial_id = ?",
                (session_id, trial_id),
            ).fetchone()
            now = now_iso()
            revision = int(current["revision_count"]) + 1 if current else 1
            columns = ["session_id", "trial_id", *fields, "revision_count", "status", "updated_at", "submitted_at"]
            values = [session_id, trial_id, *fields.values(), revision, "submitted" if submit else "draft", now, now if submit else None]
            updates = ", ".join(f"{column}=excluded.{column}" for column in columns[2:])
            placeholders = ",".join("?" for _ in columns)
            db.execute(
                f"INSERT INTO {table}({','.join(columns)}) VALUES ({placeholders}) ON CONFLICT(session_id, trial_id) DO UPDATE SET {updates}",
                values,
            )
            if submit:
                db.execute(
                    "UPDATE study1_sessions SET current_index = ?, updated_at = ? WHERE id = ?",
                    (index + 1, now, session_id),
                )
            else:
                db.execute("UPDATE study1_sessions SET updated_at = ? WHERE id = ?", (now, session_id))


def stable_int(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_assignment(bank: dict[str, Any], participant_code: str) -> dict[str, Any]:
    study_id = str(bank["study_id"])
    cohort = stable_int(study_id, participant_code, "cohort") % 15
    rng = random.Random(stable_int(study_id, participant_code, "assignment"))
    items = list(bank["items"])
    rng.shuffle(items)
    chosen = items[:TOTAL_TRIALS]
    author = str(bank.get("author", {}).get("token") or "")

    pointwise = []
    for index, item in enumerate(chosen[:2]):
        source = ALL_SOURCES[(cohort + index) % len(ALL_SOURCES)]
        response = item["responses"][source]
        pointwise.append(
            {
                "trial_id": f"pw-{item['item_id']}",
                "question_id": item["question_id"],
                "question": item["question"],
                "response_id": f"{item['item_id']}:{source}",
                "answer": response["text"],
                "hidden_source": source,
            }
        )
    rng.shuffle(pointwise)

    gold_ai_index = cohort % len(AI_SOURCES)
    gold_ai_source = AI_SOURCES[gold_ai_index]
    remaining_ai = [source for source in AI_SOURCES if source != gold_ai_source]
    pair_design = (
        ("gold_vs_ai", "gold", gold_ai_source),
        ("ai_vs_ai", remaining_ai[0], remaining_ai[1]),
    )
    pairwise = []
    for index, (pair_type, source_a, source_b) in enumerate(pair_design):
        item = chosen[index + POINTWISE_TRIALS]
        sides = [
            {
                "response_id": f"{item['item_id']}:{source_a}",
                "text": item["responses"][source_a]["text"],
                "hidden_source": source_a,
            },
            {
                "response_id": f"{item['item_id']}:{source_b}",
                "text": item["responses"][source_b]["text"],
                "hidden_source": source_b,
            },
        ]
        rng.shuffle(sides)
        pairwise.append(
            {
                "trial_id": f"pair-{pair_type}-{item['item_id']}",
                "question_id": item["question_id"],
                "question": item["question"],
                "hidden_pair_type": pair_type,
                "left": sides[0],
                "right": sides[1],
            }
        )
    rng.shuffle(pairwise)
    return {
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "study_id": study_id,
        "cohort": cohort,
        "author": author,
        "author_label": author,
        "pointwise": pointwise,
        "pairwise": pairwise,
    }


def validate_highlights(
    text: str,
    highlights: list[dict[str, Any]],
    *,
    maximum: int,
) -> None:
    if len(highlights) > maximum:
        raise ValueError(f"最多保留 {maximum} 处划线")
    annotation_ids = [str(item["annotation_id"]) for item in highlights]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("同一题内的划线 ID 不能重复")
    ordered = sorted(highlights, key=lambda item: item["start"])
    previous_end = -1
    for item in ordered:
        start, end = int(item["start"]), int(item["end"])
        if start < 0 or end <= start or end > len(text):
            raise ValueError("划线位置超出正文范围")
        if start < previous_end:
            raise ValueError("划线不能重叠")
        if text[start:end] != item["selected_text"]:
            raise ValueError("划线文字与正文位置不一致")
        previous_end = end
