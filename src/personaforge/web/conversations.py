"""Persistent conversations, messages, and chat turn runs."""

from __future__ import annotations

import json
import sqlite3
from array import array
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal
from uuid import uuid4

ConversationRole = Literal["user", "assistant", "error"]
TurnStatus = Literal["queued", "running", "completed", "failed", "interrupted"]
ACTIVE_TURN_STATUSES = ("queued", "running")


class ConversationBusyError(RuntimeError):
    """Raised when another turn is already running in the same conversation."""


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    id: str
    conversation_id: str
    sequence: int
    role: ConversationRole
    text: str
    status: str
    sources: list[dict[str, Any]]
    trace_id: str | None
    turn_id: str | None
    created_at: str
    updated_at: str

    def to_api(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "text": self.text,
            "status": self.status,
            "sources": self.sources or None,
            "trace_id": self.trace_id,
            "turn_id": self.turn_id,
        }


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    id: str
    conversation_id: str
    author: str
    user_message_id: str
    assistant_message_id: str
    query: str
    assistant_text: str
    assistant_status: str
    trace_id: str | None
    parent_ids: list[str]
    sequence: int

    @property
    def memory_text(self) -> str:
        return f"用户：{self.query}\n回答：{self.assistant_text}".strip()


@dataclass(frozen=True, slots=True)
class TurnRun:
    id: str
    conversation_id: str
    author: str
    user_message_id: str
    assistant_message_id: str
    selected_attempt_id: str
    query: str
    query_mode: str
    writer_prompt: str
    parent_top_k: int
    trace_capture: str
    status: TurnStatus
    stage: str
    label: str
    partial_answer: str
    error: dict[str, Any] | None
    planner: dict[str, Any] | None
    parent_ids: list[str]
    response_depth: str | None
    trace_id: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConversationStore:
    """Small SQLite repository shared by chat requests and worker threads."""

    def __init__(self, data_dir: Path, *, owner_id: str = "local-user") -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.owner_id = owner_id
        self.path = self.data_dir / "system" / "personaforge.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.mark_running_interrupted()
        self.migrate_legacy_sessions()

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
        selected_owner_id = owner_id or self.owner_id
        now = utc_now_iso()
        conversation_id = conversation_id or new_conversation_id()
        turn_id = new_turn_id()
        user_message_id = new_message_id()
        assistant_message_id = new_message_id()
        attempt_id = new_attempt_id()
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT author, owner_id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                connection.execute(
                    """
                    INSERT INTO conversations (
                        id, owner_id, author, title, summary_json,
                        summary_through_sequence, summary_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, '{}', 0, 0, ?, ?)
                    """,
                    (
                        conversation_id,
                        selected_owner_id,
                        author,
                        conversation_title(query),
                        now,
                        now,
                    ),
                )
            elif str(conversation["owner_id"]) != selected_owner_id:
                raise FileNotFoundError(f"Conversation not found: {conversation_id}")
            elif str(conversation["author"]) != author:
                raise ValueError("Conversation belongs to a different author.")

            active = connection.execute(
                """
                SELECT id FROM turn_runs
                WHERE conversation_id = ? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            if active is not None:
                raise ConversationBusyError("This conversation already has a running response.")

            next_sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, sequence, role, text, status,
                    sources_json, trace_id, turn_id, created_at, updated_at
                ) VALUES (?, ?, ?, 'user', ?, 'completed', '[]', NULL, ?, ?, ?)
                """,
                (user_message_id, conversation_id, next_sequence, query, turn_id, now, now),
            )
            connection.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, sequence, role, text, status,
                    sources_json, trace_id, turn_id, created_at, updated_at
                ) VALUES (?, ?, ?, 'assistant', '', 'queued', '[]', NULL, ?, ?, ?)
                """,
                (assistant_message_id, conversation_id, next_sequence + 1, turn_id, now, now),
            )
            connection.execute(
                """
                INSERT INTO turn_runs (
                    id, conversation_id, author, user_message_id, assistant_message_id,
                    selected_attempt_id, query, query_mode, writer_prompt, parent_top_k,
                    trace_capture, status, stage, label, partial_answer, error_json,
                    planner_json, parent_ids_json, response_depth, trace_id,
                    created_at, updated_at, started_at, completed_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    'queued', 'queued', '等待处理', '', NULL, NULL, '[]', NULL, NULL,
                    ?, ?, NULL, NULL
                )
                """,
                (
                    turn_id,
                    conversation_id,
                    author,
                    user_message_id,
                    assistant_message_id,
                    attempt_id,
                    query,
                    query_mode,
                    writer_prompt,
                    parent_top_k,
                    trace_capture,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO generation_attempts (
                    id, turn_id, attempt_index, assistant_message_id, trace_id,
                    status, created_at, updated_at
                ) VALUES (?, ?, 1, ?, NULL, 'queued', ?, ?)
                """,
                (attempt_id, turn_id, assistant_message_id, now, now),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        self.append_event(
            turn_id,
            "meta",
            {
                "session_id": conversation_id,
                "turn_id": turn_id,
                "author": author,
            },
        )
        self.append_event(turn_id, "status", {"stage": "queued", "label": "等待处理"})
        return self.get_turn(turn_id)

    def get_turn(self, turn_id: str) -> TurnRun:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM turn_runs WHERE id = ?", (turn_id,)).fetchone()
        if row is None:
            raise KeyError(turn_id)
        return _row_to_turn_run(row)

    def get_turn_for_owner(self, turn_id: str, owner_id: str) -> TurnRun:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT tr.*
                FROM turn_runs tr
                JOIN conversations c ON c.id = tr.conversation_id
                WHERE tr.id = ? AND c.owner_id = ?
                """,
                (turn_id, owner_id),
            ).fetchone()
        if row is None:
            raise KeyError(turn_id)
        return _row_to_turn_run(row)

    def list_queued_turns(self, *, limit: int = 100) -> list[TurnRun]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM turn_runs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_turn_run(row) for row in rows]

    def retry_turn(self, turn_id: str) -> TurnRun:
        now = utc_now_iso()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM turn_runs WHERE id = ?", (turn_id,)).fetchone()
            if row is None:
                raise KeyError(turn_id)
            if str(row["status"]) not in {"failed", "interrupted"}:
                raise ValueError("Only failed or interrupted turns can be retried.")
            active = connection.execute(
                """
                SELECT id FROM turn_runs
                WHERE conversation_id = ? AND status IN ('queued', 'running') AND id != ?
                LIMIT 1
                """,
                (str(row["conversation_id"]), turn_id),
            ).fetchone()
            if active is not None:
                raise ConversationBusyError("This conversation already has a running response.")
            next_attempt_index = int(
                connection.execute(
                    "SELECT COALESCE(MAX(attempt_index), 0) + 1 FROM generation_attempts WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()[0]
            )
            attempt_id = new_attempt_id()
            connection.execute(
                """
                INSERT INTO generation_attempts (
                    id, turn_id, attempt_index, assistant_message_id, trace_id,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, NULL, 'queued', ?, ?)
                """,
                (
                    attempt_id,
                    turn_id,
                    next_attempt_index,
                    str(row["assistant_message_id"]),
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE turn_runs
                SET selected_attempt_id = ?, status = 'queued', stage = 'queued',
                    label = '等待重试', partial_answer = '', error_json = NULL,
                    planner_json = NULL, parent_ids_json = '[]', response_depth = NULL,
                    trace_id = NULL, updated_at = ?, started_at = NULL, completed_at = NULL
                WHERE id = ?
                """,
                (attempt_id, now, turn_id),
            )
            connection.execute(
                """
                UPDATE messages
                SET role = 'assistant', text = '', status = 'queued', sources_json = '[]',
                    trace_id = NULL, updated_at = ?
                WHERE id = ?
                """,
                (now, str(row["assistant_message_id"])),
            )
            connection.execute("DELETE FROM turn_events WHERE turn_id = ?", (turn_id,))
        self.append_event(
            turn_id,
            "meta",
            {
                "session_id": str(row["conversation_id"]),
                "turn_id": turn_id,
                "author": str(row["author"]),
            },
        )
        self.append_event(turn_id, "status", {"stage": "queued", "label": "等待重试"})
        return self.get_turn(turn_id)

    def claim_turn(self, turn_id: str) -> bool:
        now = utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE turn_runs
                SET status = 'running', stage = 'context_load', label = '正在读取对话',
                    started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (now, now, turn_id),
            )
            if cursor.rowcount != 1:
                return False
            connection.execute(
                """
                UPDATE generation_attempts
                SET status = 'running', updated_at = ?
                WHERE id = (SELECT selected_attempt_id FROM turn_runs WHERE id = ?)
                """,
                (now, turn_id),
            )
            connection.execute(
                "UPDATE messages SET status = 'running', updated_at = ? WHERE id = "
                "(SELECT assistant_message_id FROM turn_runs WHERE id = ?)",
                (now, turn_id),
            )
        return True

    def update_turn_stage(self, turn_id: str, stage: str, label: str) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                "UPDATE turn_runs SET stage = ?, label = ?, updated_at = ? WHERE id = ?",
                (stage, label, now, turn_id),
            )
        self.append_event(turn_id, "status", {"stage": stage, "label": label})

    def set_turn_plan(self, turn_id: str, planner: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE turn_runs
                SET planner_json = ?, response_depth = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(planner, ensure_ascii=False),
                    str(planner.get("response_depth") or "") or None,
                    utc_now_iso(),
                    turn_id,
                ),
            )

    def set_turn_evidence(self, turn_id: str, parent_ids: Iterable[str], trace_id: str | None) -> None:
        parent_ids = [str(item) for item in parent_ids if str(item)]
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE turn_runs
                SET parent_ids_json = ?, trace_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(parent_ids, ensure_ascii=False), trace_id, now, turn_id),
            )
            connection.execute(
                "UPDATE messages SET trace_id = ?, updated_at = ? WHERE id = "
                "(SELECT assistant_message_id FROM turn_runs WHERE id = ?)",
                (trace_id, now, turn_id),
            )
            connection.execute(
                "UPDATE generation_attempts SET trace_id = ?, updated_at = ? WHERE id = "
                "(SELECT selected_attempt_id FROM turn_runs WHERE id = ?)",
                (trace_id, now, turn_id),
            )

    def update_partial_answer(self, turn_id: str, answer: str) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE turn_runs
                SET partial_answer = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (answer, now, turn_id),
            )
            connection.execute(
                """
                UPDATE messages
                SET text = ?, updated_at = ?
                WHERE id = (SELECT assistant_message_id FROM turn_runs WHERE id = ?)
                """,
                (answer, now, turn_id),
            )

    def complete_turn(
        self,
        turn_id: str,
        *,
        answer: str,
        sources: list[dict[str, Any]],
        trace_id: str | None,
    ) -> TurnRun:
        now = utc_now_iso()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT conversation_id, assistant_message_id, selected_attempt_id FROM turn_runs WHERE id = ?",
                (turn_id,),
            ).fetchone()
            if row is None:
                raise KeyError(turn_id)
            connection.execute(
                """
                UPDATE turn_runs
                SET status = 'completed', stage = 'completed', label = '回答完成',
                    partial_answer = ?, trace_id = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (answer, trace_id, now, now, turn_id),
            )
            connection.execute(
                """
                UPDATE messages
                SET role = 'assistant', text = ?, status = 'completed',
                    sources_json = ?, trace_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    answer,
                    json.dumps(sources, ensure_ascii=False),
                    trace_id,
                    now,
                    str(row["assistant_message_id"]),
                ),
            )
            connection.execute(
                """
                UPDATE generation_attempts
                SET status = 'completed', trace_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (trace_id, now, str(row["selected_attempt_id"])),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, str(row["conversation_id"])),
            )
        return self.get_turn(turn_id)

    def fail_turn(self, turn_id: str, error: dict[str, Any]) -> TurnRun:
        now = utc_now_iso()
        error_json = json.dumps(error, ensure_ascii=False)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status, assistant_message_id, selected_attempt_id FROM turn_runs WHERE id = ?",
                (turn_id,),
            ).fetchone()
            if row is None:
                raise KeyError(turn_id)
            if str(row["status"]) == "completed":
                return self.get_turn(turn_id)
            connection.execute(
                """
                UPDATE turn_runs
                SET status = 'failed', stage = 'failed', label = '生成失败',
                    error_json = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (error_json, now, now, turn_id),
            )
            connection.execute(
                """
                UPDATE messages
                SET role = 'error', status = 'failed', text = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(error.get("message") or "生成失败"), now, str(row["assistant_message_id"])),
            )
            connection.execute(
                "UPDATE generation_attempts SET status = 'failed', updated_at = ? WHERE id = ?",
                (now, str(row["selected_attempt_id"])),
            )
        return self.get_turn(turn_id)

    def append_event(self, turn_id: str, event_type: str, payload: dict[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO turn_events (turn_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (turn_id, event_type, json.dumps(payload, ensure_ascii=False), utc_now_iso()),
            )
            return int(cursor.lastrowid)

    def list_events(self, turn_id: str, *, after_sequence: int = 0) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_type, payload_json, created_at
                FROM turn_events
                WHERE turn_id = ? AND sequence > ?
                ORDER BY sequence ASC
                """,
                (turn_id, after_sequence),
            ).fetchall()
        return [
            {
                "sequence": int(row["sequence"]),
                "event": str(row["event_type"]),
                "payload": _json_object(row["payload_json"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def list_conversations(self, author: str, *, owner_id: str | None = None) -> list[dict[str, Any]]:
        selected_owner_id = owner_id or self.owner_id
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.author = ? AND c.owner_id = ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                """,
                (author, selected_owner_id),
            ).fetchall()
        return [
            {
                "id": str(row["id"]),
                "author": str(row["author"]),
                "title": str(row["title"]),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
                "message_count": int(row["message_count"]),
            }
            for row in rows
        ]

    def get_conversation(
        self,
        author: str,
        conversation_id: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        selected_owner_id = owner_id or self.owner_id
        with self._connect() as connection:
            conversation = connection.execute(
                """
                SELECT * FROM conversations
                WHERE id = ? AND author = ? AND owner_id = ?
                """,
                (conversation_id, author, selected_owner_id),
            ).fetchone()
            if conversation is None:
                raise FileNotFoundError(f"Session not found: {conversation_id}")
            message_rows = connection.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY sequence ASC",
                (conversation_id,),
            ).fetchall()
        messages = [_row_to_message(row).to_api() for row in message_rows]
        return {
            "id": str(conversation["id"]),
            "author": str(conversation["author"]),
            "title": str(conversation["title"]),
            "created_at": str(conversation["created_at"]),
            "updated_at": str(conversation["updated_at"]),
            "messages": messages,
        }

    def delete_conversation(
        self,
        author: str,
        conversation_id: str,
        *,
        owner_id: str | None = None,
    ) -> list[str]:
        selected_owner_id = owner_id or self.owner_id
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM conversations WHERE id = ? AND author = ? AND owner_id = ?",
                (conversation_id, author, selected_owner_id),
            ).fetchone()
            if row is None:
                return []
            trace_rows = connection.execute(
                "SELECT trace_id FROM messages WHERE conversation_id = ? AND trace_id IS NOT NULL",
                (conversation_id,),
            ).fetchall()
            connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return [str(item["trace_id"]) for item in trace_rows]

    def owner_has_trace(self, owner_id: str, author: str, trace_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM conversations c
                JOIN messages m ON m.conversation_id = c.id
                WHERE c.owner_id = ? AND c.author = ? AND m.trace_id = ?
                LIMIT 1
                """,
                (owner_id, author, trace_id),
            ).fetchone()
        return row is not None

    def get_conversation_owner(self, conversation_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT owner_id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"Conversation not found: {conversation_id}")
        return str(row["owner_id"])

    def get_completed_turns(self, conversation_id: str) -> list[ConversationTurn]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    t.id, t.conversation_id, t.author, t.user_message_id,
                    t.assistant_message_id, t.trace_id, t.parent_ids_json,
                    u.text AS query, u.sequence AS user_sequence,
                    a.text AS assistant_text, a.status AS assistant_status
                FROM turn_runs t
                JOIN messages u ON u.id = t.user_message_id
                JOIN messages a ON a.id = t.assistant_message_id
                WHERE t.conversation_id = ? AND t.status = 'completed'
                ORDER BY u.sequence ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [
            ConversationTurn(
                id=str(row["id"]),
                conversation_id=str(row["conversation_id"]),
                author=str(row["author"]),
                user_message_id=str(row["user_message_id"]),
                assistant_message_id=str(row["assistant_message_id"]),
                query=str(row["query"]),
                assistant_text=str(row["assistant_text"]),
                assistant_status=str(row["assistant_status"]),
                trace_id=str(row["trace_id"]) if row["trace_id"] else None,
                parent_ids=_json_string_list(row["parent_ids_json"]),
                sequence=int(row["user_sequence"]),
            )
            for row in rows
        ]

    def get_summary(self, conversation_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT summary_json, summary_through_sequence, summary_version
                FROM conversations WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"Session not found: {conversation_id}")
        return {
            "summary": _json_object(row["summary_json"]),
            "through_sequence": int(row["summary_through_sequence"]),
            "version": int(row["summary_version"]),
        }

    def save_summary(
        self,
        conversation_id: str,
        summary: dict[str, Any],
        *,
        through_sequence: int,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations
                SET summary_json = ?, summary_through_sequence = ?,
                    summary_version = summary_version + 1, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(summary, ensure_ascii=False), through_sequence, now, conversation_id),
            )
        return self.get_summary(conversation_id)

    def get_turn_embedding(self, turn_id: str, *, model: str, version: str) -> list[float] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT dense_blob, dimensions FROM turn_embeddings
                WHERE turn_id = ? AND embedding_model = ? AND embedding_version = ?
                """,
                (turn_id, model, version),
            ).fetchone()
        if row is None:
            return None
        values = array("f")
        values.frombytes(bytes(row["dense_blob"]))
        dense = [float(item) for item in values]
        if len(dense) != int(row["dimensions"]):
            return None
        return dense

    def save_turn_embedding(
        self,
        turn_id: str,
        dense: list[float],
        *,
        model: str,
        version: str,
    ) -> None:
        values = array("f", dense)
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO turn_embeddings (
                    turn_id, dense_blob, dimensions, embedding_model,
                    embedding_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    dense_blob = excluded.dense_blob,
                    dimensions = excluded.dimensions,
                    embedding_model = excluded.embedding_model,
                    embedding_version = excluded.embedding_version,
                    updated_at = excluded.updated_at
                """,
                (turn_id, values.tobytes(), len(dense), model, version, now, now),
            )

    def save_completed_turn(
        self,
        *,
        conversation_id: str,
        author: str,
        query: str,
        answer: str,
        sources: list[dict[str, Any]],
        trace_id: str | None,
        query_mode: str = "raw",
        writer_prompt: str = "strong_identity",
        parent_top_k: int = 20,
        trace_capture: str = "summary",
        owner_id: str | None = None,
    ) -> TurnRun:
        run = self.create_turn(
            author=author,
            conversation_id=conversation_id,
            query=query,
            query_mode=query_mode,
            writer_prompt=writer_prompt,
            parent_top_k=parent_top_k,
            trace_capture=trace_capture,
            owner_id=owner_id,
        )
        self.claim_turn(run.id)
        self.set_turn_evidence(
            run.id,
            [str(item.get("parent_id")) for item in sources if item.get("parent_id")],
            trace_id,
        )
        return self.complete_turn(run.id, answer=answer, sources=sources, trace_id=trace_id)

    def mark_running_interrupted(self) -> int:
        now = utc_now_iso()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, assistant_message_id, selected_attempt_id FROM turn_runs WHERE status = 'running'"
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE turn_runs
                    SET status = 'interrupted', stage = 'interrupted',
                        label = '服务重启，任务已中断', updated_at = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (now, now, str(row["id"])),
                )
                connection.execute(
                    "UPDATE messages SET status = 'interrupted', updated_at = ? WHERE id = ?",
                    (now, str(row["assistant_message_id"])),
                )
                connection.execute(
                    "UPDATE generation_attempts SET status = 'interrupted', updated_at = ? WHERE id = ?",
                    (now, str(row["selected_attempt_id"])),
                )
        return len(rows)

    def migrate_legacy_sessions(self) -> int:
        root = self.data_dir / "authors" / "zhihu"
        if not root.exists():
            return 0
        imported = 0
        for path in root.glob("*/sessions/*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            conversation_id = str(payload.get("id") or path.stem)
            author = str(payload.get("author") or path.parent.parent.name)
            with self._connect() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if exists is not None:
                    continue
                self._import_legacy_payload(connection, author, conversation_id, payload, path)
            imported += 1
        return imported

    def _import_legacy_payload(
        self,
        connection: sqlite3.Connection,
        author: str,
        conversation_id: str,
        payload: dict[str, Any],
        path: Path,
    ) -> None:
        created_at = str(payload.get("created_at") or _file_time_iso(path))
        updated_at = str(payload.get("updated_at") or created_at)
        connection.execute(
            """
            INSERT INTO conversations (
                id, owner_id, author, title, summary_json,
                summary_through_sequence, summary_version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, '{}', 0, 0, ?, ?)
            """,
            (
                conversation_id,
                self.owner_id,
                author,
                str(payload.get("title") or "未命名对话"),
                created_at,
                updated_at,
            ),
        )
        pending_user: tuple[str, str, int] | None = None
        messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
        for sequence, raw in enumerate(messages, start=1):
            if not isinstance(raw, dict):
                continue
            role = str(raw.get("role") or "")
            if role not in {"user", "assistant", "error"}:
                continue
            message_id = f"legacy-message-{uuid4().hex}"
            text = str(raw.get("text") or "")
            sources = raw.get("sources") if isinstance(raw.get("sources"), list) else []
            trace_id = str(raw.get("trace_id") or "") or None
            status = "completed" if role != "error" else "failed"
            turn_id: str | None = None
            if role == "user":
                pending_user = (message_id, text, sequence)
            elif pending_user is not None:
                turn_id = f"legacy-turn-{uuid4().hex}"
            connection.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, sequence, role, text, status,
                    sources_json, trace_id, turn_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    sequence,
                    role,
                    text,
                    status,
                    json.dumps(sources, ensure_ascii=False),
                    trace_id,
                    turn_id,
                    created_at,
                    updated_at,
                ),
            )
            if turn_id and pending_user is not None:
                user_message_id, query, _ = pending_user
                attempt_id = f"legacy-attempt-{uuid4().hex}"
                connection.execute(
                    "UPDATE messages SET turn_id = ? WHERE id = ?",
                    (turn_id, user_message_id),
                )
                parent_ids = [
                    str(item.get("parent_id"))
                    for item in sources
                    if isinstance(item, dict) and item.get("parent_id")
                ]
                connection.execute(
                    """
                    INSERT INTO turn_runs (
                        id, conversation_id, author, user_message_id, assistant_message_id,
                        selected_attempt_id, query, query_mode, writer_prompt, parent_top_k,
                        trace_capture, status, stage, label, partial_answer, error_json,
                        planner_json, parent_ids_json, response_depth, trace_id,
                        created_at, updated_at, started_at, completed_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, 'legacy', 'unknown', ?, 'summary',
                        ?, ?, ?, ?, NULL, NULL, ?, NULL, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        turn_id,
                        conversation_id,
                        author,
                        user_message_id,
                        message_id,
                        attempt_id,
                        query,
                        len(parent_ids) or 20,
                        "completed" if role == "assistant" else "failed",
                        "completed" if role == "assistant" else "failed",
                        "历史记录",
                        text,
                        json.dumps(parent_ids, ensure_ascii=False),
                        trace_id,
                        created_at,
                        updated_at,
                        created_at,
                        updated_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO generation_attempts (
                        id, turn_id, attempt_index, assistant_message_id, trace_id,
                        status, created_at, updated_at
                    ) VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt_id,
                        turn_id,
                        message_id,
                        trace_id,
                        "completed" if role == "assistant" else "failed",
                        created_at,
                        updated_at,
                    ),
                )
                pending_user = None

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    author TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    summary_through_sequence INTEGER NOT NULL DEFAULT 0,
                    summary_version INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    trace_id TEXT,
                    turn_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(conversation_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS turn_runs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    author TEXT NOT NULL,
                    user_message_id TEXT NOT NULL,
                    assistant_message_id TEXT NOT NULL,
                    selected_attempt_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    query_mode TEXT NOT NULL,
                    writer_prompt TEXT NOT NULL,
                    parent_top_k INTEGER NOT NULL,
                    trace_capture TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    label TEXT NOT NULL,
                    partial_answer TEXT NOT NULL DEFAULT '',
                    error_json TEXT,
                    planner_json TEXT,
                    parent_ids_json TEXT NOT NULL DEFAULT '[]',
                    response_depth TEXT,
                    trace_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS generation_attempts (
                    id TEXT PRIMARY KEY,
                    turn_id TEXT NOT NULL REFERENCES turn_runs(id) ON DELETE CASCADE,
                    attempt_index INTEGER NOT NULL,
                    assistant_message_id TEXT NOT NULL,
                    trace_id TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(turn_id, attempt_index)
                );

                CREATE TABLE IF NOT EXISTS turn_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_id TEXT NOT NULL REFERENCES turn_runs(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS turn_embeddings (
                    turn_id TEXT PRIMARY KEY REFERENCES turn_runs(id) ON DELETE CASCADE,
                    dense_blob BLOB NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_author_updated
                    ON conversations(author, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_conversations_owner_author_updated
                    ON conversations(owner_id, author, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_sequence
                    ON messages(conversation_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_turn_runs_conversation_status
                    ON turn_runs(conversation_id, status);
                CREATE INDEX IF NOT EXISTS idx_turn_events_turn_sequence
                    ON turn_events(turn_id, sequence);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


def _row_to_message(row: sqlite3.Row) -> ConversationMessage:
    sources = json.loads(str(row["sources_json"]) or "[]")
    if not isinstance(sources, list):
        sources = []
    return ConversationMessage(
        id=str(row["id"]),
        conversation_id=str(row["conversation_id"]),
        sequence=int(row["sequence"]),
        role=str(row["role"]),  # type: ignore[arg-type]
        text=str(row["text"]),
        status=str(row["status"]),
        sources=[item for item in sources if isinstance(item, dict)],
        trace_id=str(row["trace_id"]) if row["trace_id"] else None,
        turn_id=str(row["turn_id"]) if row["turn_id"] else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_turn_run(row: sqlite3.Row) -> TurnRun:
    return TurnRun(
        id=str(row["id"]),
        conversation_id=str(row["conversation_id"]),
        author=str(row["author"]),
        user_message_id=str(row["user_message_id"]),
        assistant_message_id=str(row["assistant_message_id"]),
        selected_attempt_id=str(row["selected_attempt_id"]),
        query=str(row["query"]),
        query_mode=str(row["query_mode"]),
        writer_prompt=str(row["writer_prompt"]),
        parent_top_k=int(row["parent_top_k"]),
        trace_capture=str(row["trace_capture"]),
        status=str(row["status"]),  # type: ignore[arg-type]
        stage=str(row["stage"]),
        label=str(row["label"]),
        partial_answer=str(row["partial_answer"]),
        error=_json_object(row["error_json"]) if row["error_json"] else None,
        planner=_json_object(row["planner_json"]) if row["planner_json"] else None,
        parent_ids=_json_string_list(row["parent_ids_json"]),
        response_depth=str(row["response_depth"]) if row["response_depth"] else None,
        trace_id=str(row["trace_id"]) if row["trace_id"] else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        started_at=str(row["started_at"]) if row["started_at"] else None,
        completed_at=str(row["completed_at"]) if row["completed_at"] else None,
    )


def _json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if str(item)]


def new_conversation_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid4().hex[:8]}"


def new_turn_id() -> str:
    return f"turn-{uuid4().hex}"


def new_message_id() -> str:
    return f"message-{uuid4().hex}"


def new_attempt_id() -> str:
    return f"attempt-{uuid4().hex}"


def conversation_title(query: str) -> str:
    title = " ".join(query.strip().split())
    return title[:32] if title else "未命名对话"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_time_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
