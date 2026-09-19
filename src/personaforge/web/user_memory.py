"""Cross-session user memory with evidence, revisions, and hybrid recall."""

from __future__ import annotations

import json
from contextlib import nullcontext
import math
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from personaforge.ingest.embeddings import SparseEmbedding, TextEmbedding, TextEncoder
from personaforge.llm import JsonChatClient
from personaforge.web.conversations import ConversationTurn, utc_now_iso

from personaforge.web.memory_evidence import EVIDENCE_SCHEMA, EvidenceStoreMixin

MemoryKind = Literal["semantic", "episodic", "procedural"]
MemorySensitivity = Literal["normal", "private", "restricted"]
MemoryEventStatus = Literal["ongoing", "historical", "stable"]

MEMORY_KINDS = {"semantic", "episodic", "procedural"}
SENSITIVITIES = {"normal", "private", "restricted"}
EVENT_STATUSES = {"ongoing", "historical", "stable"}
QUESTION_MARKERS = ("?", "？", "是不是", "会不会", "难道", "是否", "为什么", "怎么")
BELIEF_MARKERS = ("用户认为", "用户相信", "用户认定", "用户喜欢", "用户偏好")
SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_ -]?key|password|密码|cookie|token)\s*[:=：]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)
EXACT_FINANCE_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:\d+(?:\.\d+)?\s*(?:万|千|百|元|块|%|％|成|倍|个点|[wW])|"
    r"[一二三四五六七八九十百千万两]+倍|几(?:百|千|万)(?:元|块)?|\d{4,})(?![A-Za-z])"
)
THIRD_PARTY_MARKERS = ("我哥", "哥哥", "我弟", "弟弟", "姐姐", "妹妹", "朋友", "同事", "父亲", "母亲", "家人")
FINANCE_MARKERS = ("亏", "贷款", "杠杆", "充值", "余额", "炒股", "短线", "借贷", "负债", "钱")


@dataclass(frozen=True, slots=True)
class UserMemory:
    id: str
    owner_id: str
    kind: MemoryKind
    memory_key: str
    content: str
    status: str
    pinned: bool
    sensitivity: MemorySensitivity
    importance: int
    confidence: float
    event_status: MemoryEventStatus
    source_author: str | None
    source_conversation_id: str | None
    source_message_ids: list[str]
    evidence_quotes: list[str]
    supersedes_id: str | None
    created_at: str
    updated_at: str
    last_accessed_at: str | None
    access_count: int

    def to_api(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_line(self) -> str:
        return f"[{self.id}] ({self.kind}) {self.content}"


@dataclass(frozen=True, slots=True)
class MemoryRecallHit:
    memory: UserMemory
    rank: int
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float

    def trace_payload(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory.id,
            "rank": self.rank,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "rrf_score": round(self.rrf_score, 8),
            "kind": self.memory.kind,
            "pinned": self.memory.pinned,
        }


class UserMemoryStore(EvidenceStoreMixin):
    """SQLite repository for user-owned long-term memory."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir.expanduser().resolve() / "system" / "personaforge.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def settings(self, owner_id: str) -> dict[str, bool]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT enabled, auto_write FROM user_memory_settings WHERE owner_id = ?",
                (owner_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO user_memory_settings (owner_id, enabled, auto_write, updated_at) VALUES (?, 1, 1, ?)",
                    (owner_id, utc_now_iso()),
                )
                return {"enabled": True, "auto_write": True}
        return {"enabled": bool(row["enabled"]), "auto_write": bool(row["auto_write"])}

    def update_settings(
        self,
        owner_id: str,
        *,
        enabled: bool | None = None,
        auto_write: bool | None = None,
    ) -> dict[str, bool]:
        current = self.settings(owner_id)
        resolved_enabled = current["enabled"] if enabled is None else enabled
        resolved_auto_write = current["auto_write"] if auto_write is None else auto_write
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_memory_settings (owner_id, enabled, auto_write, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(owner_id) DO UPDATE SET
                    enabled=excluded.enabled,
                    auto_write=excluded.auto_write,
                    updated_at=excluded.updated_at
                """,
                (owner_id, int(resolved_enabled), int(resolved_auto_write), utc_now_iso()),
            )
        return {"enabled": resolved_enabled, "auto_write": resolved_auto_write}

    def list_active(self, owner_id: str) -> list[UserMemory]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM user_memories
                WHERE owner_id = ? AND status = 'active'
                ORDER BY pinned DESC, importance DESC, updated_at DESC
                """,
                (owner_id,),
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def get(self, owner_id: str, memory_id: str) -> UserMemory:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM user_memories WHERE owner_id = ? AND id = ?",
                (owner_id, memory_id),
            ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return _row_to_memory(row)

    def save_revision(
        self,
        owner_id: str,
        *,
        kind: str,
        memory_key: str,
        content: str,
        sensitivity: str,
        importance: int,
        confidence: float,
        event_status: str,
        source_author: str | None,
        source_conversation_id: str | None,
        source_message_ids: list[str],
        evidence_quotes: list[str],
        pinned: bool = False,
        supersedes_id: str | None = None,
        preserve_previous: bool = True,
        _connection: sqlite3.Connection | None = None,
    ) -> UserMemory:
        kind = kind if kind in MEMORY_KINDS else "episodic"
        sensitivity = sensitivity if sensitivity in SENSITIVITIES else "private"
        event_status = event_status if event_status in EVENT_STATUSES else "ongoing"
        memory_key = canonical_memory_key(memory_key)
        content = redact_sensitive_text(content.strip(), force_finance=sensitivity == 'restricted')
        evidence_quotes = [] if sensitivity == 'restricted' else [redact_sensitive_text(q) for q in evidence_quotes]
        if not content:
            raise ValueError("Memory content cannot be empty.")
        now = utc_now_iso()
        memory_id = f"mem-{uuid4().hex}"
        with (nullcontext(_connection) if _connection is not None else self._connect()) as connection:
            if _connection is None:
                connection.execute("BEGIN IMMEDIATE")
            previous = None
            if supersedes_id:
                previous = connection.execute(
                    "SELECT * FROM user_memories WHERE id = ? AND owner_id = ? AND status = 'active'",
                    (supersedes_id, owner_id),
                ).fetchone()
            if previous is None:
                previous = connection.execute(
                    """
                    SELECT * FROM user_memories
                    WHERE owner_id = ? AND memory_key = ? AND status = 'active'
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (owner_id, memory_key),
                ).fetchone()
            previous_id = str(previous["id"]) if previous is not None else None
            if previous is not None and preserve_previous:
                content = _merge_revision_content(str(previous["content"]), content)
                previous_source_ids = [
                    str(item) for item in json.loads(str(previous["source_message_ids_json"]))
                ]
                source_message_ids = list(dict.fromkeys([*previous_source_ids, *source_message_ids]))
                if sensitivity != "restricted":
                    previous_quotes = [
                        str(item) for item in json.loads(str(previous["evidence_quotes_json"]))
                    ]
                    evidence_quotes = list(dict.fromkeys([*previous_quotes, *evidence_quotes]))
            if previous is not None and str(previous["content"]).strip() == content:
                connection.execute(
                    "UPDATE user_memories SET updated_at = ?, confidence = MAX(confidence, ?) WHERE id = ?",
                    (now, confidence, previous_id),
                )
                return _row_to_memory(connection.execute("SELECT * FROM user_memories WHERE id=?", (previous_id,)).fetchone())
            if previous_id:
                connection.execute(
                    "UPDATE user_memories SET status = 'superseded', updated_at = ? WHERE id = ?",
                    (now, previous_id),
                )
            connection.execute(
                """
                INSERT INTO user_memories (
                    id, owner_id, kind, memory_key, content, status, pinned,
                    sensitivity, importance, confidence, event_status,
                    source_author, source_conversation_id, source_message_ids_json,
                    evidence_quotes_json, supersedes_id, created_at, updated_at,
                    last_accessed_at, access_count
                ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)
                """,
                (
                    memory_id, owner_id, kind, memory_key, content, int(pinned), sensitivity,
                    max(1, min(5, int(importance))), max(0.0, min(1.0, float(confidence))),
                    event_status, source_author, source_conversation_id,
                    json.dumps(source_message_ids, ensure_ascii=False),
                    json.dumps(evidence_quotes, ensure_ascii=False), previous_id, now, now,
                ),
            )
            return _row_to_memory(connection.execute("SELECT * FROM user_memories WHERE id=?", (memory_id,)).fetchone())

    def correct(self, owner_id: str, memory_id: str, content: str) -> UserMemory:
        previous = self.get(owner_id, memory_id)
        if previous.status != "active":
            raise ValueError("Only active memories can be corrected.")
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            active = connection.execute("SELECT status FROM user_memories WHERE id=? AND owner_id=?",
                                        (memory_id, owner_id)).fetchone()
            if active is None or active[0] != 'active':
                raise ValueError('Memory changed during correction; refresh and retry.')
            connection.execute("""UPDATE user_memory_evidence SET status='superseded', updated_at=?
                WHERE owner_id=? AND topic_key=? AND status='pending'""",
                (utc_now_iso(), owner_id, canonical_memory_key(previous.memory_key)))
            return self.save_revision(
                owner_id, kind=previous.kind, memory_key=previous.memory_key, content=content,
                sensitivity=previous.sensitivity, importance=previous.importance, confidence=1.0,
                event_status=previous.event_status, source_author=previous.source_author,
                source_conversation_id=previous.source_conversation_id,
                source_message_ids=previous.source_message_ids, evidence_quotes=previous.evidence_quotes,
                pinned=previous.pinned, supersedes_id=previous.id, preserve_previous=False,
                _connection=connection,
            )

    def set_pinned(self, owner_id: str, memory_id: str, pinned: bool) -> UserMemory:
        self.get(owner_id, memory_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE user_memories SET pinned = ?, updated_at = ? WHERE owner_id = ? AND id = ?",
                (int(pinned), utc_now_iso(), owner_id, memory_id),
            )
        return self.get(owner_id, memory_id)

    def forget(self, owner_id: str, memory_id: str) -> None:
        previous = self.get(owner_id, memory_id)
        with self._connect() as connection:
            self.suppress_evidence(connection, owner_id, previous.memory_key)
            connection.execute(
                "UPDATE user_memories SET status = 'forgotten', pinned = 0, updated_at = ? WHERE owner_id = ? AND memory_key = ?",
                (utc_now_iso(), owner_id, previous.memory_key),
            )

    def clear(self, owner_id: str) -> int:
        with self._connect() as connection:
            self.suppress_evidence(connection, owner_id)
            cursor = connection.execute(
                "UPDATE user_memories SET status = 'forgotten', pinned = 0, updated_at = ? WHERE owner_id = ? AND status = 'active'",
                (utc_now_iso(), owner_id),
            )
        return int(cursor.rowcount)

    def load_embedding(self, memory_id: str, *, model: str, version: str) -> TextEmbedding | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM user_memory_embeddings WHERE memory_id = ? AND embedding_model = ? AND embedding_version = ?",
                (memory_id, model, version),
            ).fetchone()
        if row is None:
            return None
        return TextEmbedding(
            dense=[float(item) for item in json.loads(str(row["dense_json"]))],
            sparse=SparseEmbedding(
                indices=[int(item) for item in json.loads(str(row["sparse_indices_json"]))],
                values=[float(item) for item in json.loads(str(row["sparse_values_json"]))],
            ),
        )

    def save_embedding(
        self,
        memory_id: str,
        embedding: TextEmbedding,
        *,
        model: str,
        version: str,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_memory_embeddings (
                    memory_id, dense_json, sparse_indices_json, sparse_values_json,
                    embedding_model, embedding_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    dense_json=excluded.dense_json,
                    sparse_indices_json=excluded.sparse_indices_json,
                    sparse_values_json=excluded.sparse_values_json,
                    embedding_model=excluded.embedding_model,
                    embedding_version=excluded.embedding_version,
                    updated_at=excluded.updated_at
                """,
                (
                    memory_id,
                    json.dumps(embedding.dense),
                    json.dumps(embedding.sparse.indices),
                    json.dumps(embedding.sparse.values),
                    model,
                    version,
                    now,
                    now,
                ),
            )

    def mark_accessed(self, owner_id: str, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as connection:
            connection.execute(
                f"""UPDATE user_memories SET last_accessed_at = ?, access_count = access_count + 1
                    WHERE owner_id = ? AND id IN ({placeholders})""",
                (utc_now_iso(), owner_id, *memory_ids),
            )

    def window_checkpoint(self, owner_id: str, conversation_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT through_sequence FROM user_memory_checkpoints
                   WHERE owner_id = ? AND conversation_id = ?""",
                (owner_id, conversation_id),
            ).fetchone()
        return int(row["through_sequence"]) if row is not None else 0

    def advance_window_checkpoint(
        self,
        owner_id: str,
        conversation_id: str,
        through_sequence: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_memory_checkpoints (
                    owner_id, conversation_id, through_sequence, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(owner_id, conversation_id) DO UPDATE SET
                    through_sequence = MAX(
                        user_memory_checkpoints.through_sequence,
                        excluded.through_sequence
                    ),
                    updated_at = excluded.updated_at
                """,
                (owner_id, conversation_id, max(0, int(through_sequence)), utc_now_iso()),
            )

    def delete_window_checkpoint(self, owner_id: str, conversation_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM user_memory_checkpoints WHERE owner_id = ? AND conversation_id = ?",
                (owner_id, conversation_id),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(EVIDENCE_SCHEMA)
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    sensitivity TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    event_status TEXT NOT NULL,
                    source_author TEXT,
                    source_conversation_id TEXT,
                    source_message_ids_json TEXT NOT NULL DEFAULT '[]',
                    evidence_quotes_json TEXT NOT NULL DEFAULT '[]',
                    supersedes_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS user_memory_embeddings (
                    memory_id TEXT PRIMARY KEY REFERENCES user_memories(id) ON DELETE CASCADE,
                    dense_json TEXT NOT NULL,
                    sparse_indices_json TEXT NOT NULL,
                    sparse_values_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_memory_settings (
                    owner_id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    auto_write INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_memory_checkpoints (
                    owner_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    through_sequence INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id, conversation_id)
                );
                CREATE INDEX IF NOT EXISTS idx_user_memories_owner_status
                    ON user_memories(owner_id, status, pinned DESC, importance DESC);
                CREATE INDEX IF NOT EXISTS idx_user_memories_owner_key
                    ON user_memories(owner_id, memory_key, status);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


def recall_user_memories(
    store: UserMemoryStore,
    owner_id: str,
    query: str,
    *,
    encoder: TextEncoder,
    model: str,
    version: str = "user-memory-v1",
    limit: int = 8,
) -> list[MemoryRecallHit]:
    if not query.strip() or not store.settings(owner_id)["enabled"]:
        return []
    memories = store.list_active(owner_id)
    if not memories:
        return []
    cached: dict[str, TextEmbedding] = {}
    missing: list[UserMemory] = []
    for memory in memories:
        embedding = store.load_embedding(memory.id, model=model, version=version)
        if embedding is None:
            missing.append(memory)
        else:
            cached[memory.id] = embedding
    query_embedding = encoder.encode_texts([query], batch_size=1)[0]
    if missing:
        encoded = encoder.encode_texts([memory.content for memory in missing])
        for memory, embedding in zip(missing, encoded, strict=True):
            cached[memory.id] = embedding
            store.save_embedding(memory.id, embedding, model=model, version=version)

    dense_order = sorted(
        memories,
        key=lambda memory: (-_cosine(query_embedding.dense, cached[memory.id].dense), memory.id),
    )
    sparse_order = sorted(
        memories,
        key=lambda memory: (-_sparse_dot(query_embedding.sparse, cached[memory.id].sparse), memory.id),
    )
    dense_ranks = {memory.id: index + 1 for index, memory in enumerate(dense_order)}
    sparse_ranks = {memory.id: index + 1 for index, memory in enumerate(sparse_order)}
    scored = []
    for memory in memories:
        score = 1.0 / (60 + dense_ranks[memory.id]) + 1.0 / (60 + sparse_ranks[memory.id])
        if memory.pinned:
            score += 1.0 / 60
        scored.append((score, memory))
    scored.sort(key=lambda item: (-item[0], -item[1].importance, item[1].id))
    hits = [
        MemoryRecallHit(
            memory=memory,
            rank=index + 1,
            dense_rank=dense_ranks[memory.id],
            sparse_rank=sparse_ranks[memory.id],
            rrf_score=score,
        )
        for index, (score, memory) in enumerate(scored[: max(1, limit)])
    ]
    return hits


def update_user_memories(
    store: UserMemoryStore, owner_id: str, *, author: str, conversation_id: str,
    user_turns: list[ConversationTurn], llm: JsonChatClient,
    related_memories: list[UserMemory] | None = None, force_flush: bool = False,
    window_size: int = 3,
) -> dict[str, Any]:
    """Consume oldest complete windows; extraction and consolidation have separate durability."""
    started = time.perf_counter()
    settings = store.settings(owner_id)
    result: dict[str, Any] = {"status": "skipped", "operations": [], "rejections": [],
                              "batches": [], "extracted_evidence_ids": [], "critic_skipped": True}
    if not settings['enabled'] or not settings['auto_write']:
        return {**result, "duration_ms": 0}
    checkpoint = store.window_checkpoint(owner_id, conversation_id)
    pending = sorted((t for t in user_turns if t.sequence > checkpoint), key=lambda t: t.sequence)
    window_size = max(1, int(window_size))
    explicit = any(_is_explicit_remember(t.query) or _is_explicit_correction(t.query)
                   or _is_explicit_forget(t.query) for t in pending)
    flush = force_flush or explicit
    result.update(reason="no_pending_turns" if not pending else "window_not_full",
                  pending_turns=len(pending), window_size=window_size)
    while len(pending) >= window_size or (flush and pending):
        turns = pending[:window_size]
        source = {t.user_message_id: t.query for t in turns}
        joined = '\n'.join(source.values())
        finance = any(m in joined for m in THIRD_PARTY_MARKERS) and any(m in joined for m in FINANCE_MARKERS)
        sanitized = {key: redact_sensitive_text(value, force_finance=finance) for key, value in source.items()}
        context = [message for t in turns for message in (
            {"role": "user", "message_id": t.user_message_id, "content": sanitized[t.user_message_id]},
            {"role": "assistant", "message_id": t.assistant_message_id,
             "content": redact_sensitive_text(t.assistant_text, force_finance=finance)})]
        forget = any(_is_explicit_forget(t.query) for t in turns)
        # A selected target is required. Never guess which private facts to delete.
        if forget and related_memories:
            for memory in related_memories:
                store.forget(owner_id, memory.id)
                result['operations'].append({"operation": "forget", "memory_id": memory.id,
                                             "memory_key": memory.memory_key})
        extraction_turns = [t for t in turns if not _is_explicit_forget(t.query)]
        atoms = []
        if extraction_turns:
            eligible = {t.user_message_id: sanitized[t.user_message_id] for t in extraction_turns}
            payload = llm.complete_json([
                {"role": "system", "content": MEMORY_EXTRACTOR_PROMPT},
                {"role": "user", "content": _memory_write_prompt(
                    context_messages=context, source_messages=eligible,
                    existing=store.list_active(owner_id),
                    existing_topics=list(dict.fromkeys(row['topic_key'] for row in
                        store.list_evidence(owner_id, status='pending'))))},
            ], temperature=0.0, max_tokens=1400)
            if not isinstance(payload, dict) or not isinstance(payload.get('candidates'), list):
                raise ValueError('Invalid atom extraction response; checkpoint unchanged.')
            for candidate in _canonicalize_extracted_candidates(payload['candidates'], eligible):
                atom, reason = _validated_candidate(candidate, source_text={k: source[k] for k in eligible},
                                                    sanitized=eligible)
                if atom is None:
                    result['rejections'].append({"reason": reason})
                    continue
                atom['memory_key'] = canonical_memory_key(atom['memory_key'])
                atom['source_author'] = author
                atom['source_conversation_id'] = conversation_id
                atom['source_messages'] = {k: eligible[k] for k in atom['source_message_ids']}
                atom['source_timestamps'] = {t.user_message_id: t.created_at for t in turns
                                             if t.user_message_id in atom['source_message_ids']}
                atom['remember'] = any(_is_explicit_remember(source[k]) for k in atom['source_message_ids'])
                atom['explicit'] = any(_is_explicit_remember(source[k]) or _is_explicit_correction(source[k])
                                       for k in atom['source_message_ids'])
                atom['correction'] = any(_is_explicit_correction(source[k]) for k in atom['source_message_ids'])
                atoms.append(atom)
        ids = store.commit_evidence_batch(owner_id, conversation_id, checkpoint, turns[-1].sequence, atoms)
        checkpoint = turns[-1].sequence
        result['batches'].append({"source_turn_ids": [t.id for t in turns],
                                  "source_message_ids": list(source), "evidence_ids": ids,
                                  "through_sequence": checkpoint})
        result['extracted_evidence_ids'].extend(ids)
        result.update(status='completed', through_sequence=checkpoint, source_message_ids=list(source),
                      flush_reason='explicit' if explicit else 'planner' if force_flush else 'window_full')
        pending = pending[len(turns):]
    consolidation = consolidate_user_memories(store, owner_id, llm=llm, force=flush)
    result['operations'].extend(consolidation['operations'])
    result['rejections'].extend(consolidation['rejections'])
    result['consolidation'] = consolidation
    result['critic_skipped'] = not consolidation['attempted_topics']
    if consolidation['attempted_topics']:
        result['status'] = 'completed'
    elif not result['batches'] and pending:
        result['status'] = 'deferred'
    result['pending_turns'] = len(pending)
    result['duration_ms'] = round((time.perf_counter() - started) * 1000)
    return result


def canonical_memory_key(key: str) -> str:
    """Small alias vocabulary, open-ended topics; no fuzzy merging of different people."""
    key = _normalize_key(key)
    aliases = {'user.preference.response_length': 'user.preference.answer_length',
               'user.communication.conciseness': 'user.preference.answer_length'}
    return aliases.get(key, key)


def consolidate_user_memories(store: UserMemoryStore, owner_id: str, *, llm: JsonChatClient,
                              force: bool = False) -> dict[str, Any]:
    """Dream only changed topics. Atom status and memory revision commit together."""
    result: dict[str, Any] = {'operations': [], 'rejections': [], 'attempted_topics': []}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in store.list_evidence(owner_id, status='pending'):
        groups.setdefault(row['topic_key'], []).append(row)
    for topic, all_rows in groups.items():
        rows = all_rows[:24]  # Bounded topic work; remaining evidence is durable for the next run.
        atoms = [json.loads(row['payload_json']) for row in rows]
        existing = [m for m in store.list_active(owner_id) if canonical_memory_key(m.memory_key) == topic]
        source = {k: v for atom in atoms for k, v in atom['source_messages'].items()}
        important = any(a['importance'] >= 5 or a['explicit'] for a in atoms)
        state_change = bool(existing) and any(a['event_status'] != existing[0].event_status for a in atoms)
        if not (force or important or state_change or len(source) >= 3):
            continue
        result['attempted_topics'].append(topic)
        payload = llm.complete_json([
            {'role': 'system', 'content': MEMORY_CRITIC_PROMPT},
            {'role': 'user', 'content': json.dumps({
                'topic_key': topic, 'source_messages': source,
                'candidates': atoms,
                'evidence': [{'id': row['id'], 'timestamp': row['created_at']} for row in rows],
                'existing_memories': [m.to_api() for m in existing],
            }, ensure_ascii=False)},
        ], temperature=0.0, max_tokens=1600)
        if not isinstance(payload, dict) or not isinstance(payload.get('memories'), list):
            raise ValueError('Invalid consolidation response; evidence remains pending.')
        approved = [item for item in payload['memories'] if isinstance(item, dict)
                    and item.get('decision') == 'approve']
        if len(approved) > 1:
            raise ValueError('Consolidation must produce at most one memory per topic.')
        normalized = None
        reason = 'critic_rejected'
        if approved:
            normalized, reason = _validated_candidate(approved[0], source_text=source, sanitized=source)
            if normalized:
                levels = {'normal': 0, 'private': 1, 'restricted': 2}
                normalized['sensitivity'] = max(
                    [normalized['sensitivity'], *(a['sensitivity'] for a in atoms),
                     *(m.sensitivity for m in existing)], key=levels.__getitem__)
                if normalized['sensitivity'] == 'restricted':
                    normalized['content'] = _generalize_finance_numbers(normalized['content'])
                    normalized['evidence_quotes'] = []
        ids = [row['id'] for row in rows]
        with store._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            marks = ','.join('?' for _ in ids)
            pending_count = connection.execute(
                f"SELECT COUNT(*) FROM user_memory_evidence WHERE owner_id=? AND status='pending' AND id IN ({marks})",
                (owner_id, *ids),
            ).fetchone()[0]
            if pending_count != len(ids):
                continue  # Concurrent forget / consolidation won; never resurrect its evidence.
            active_ids = {row[0] for row in connection.execute(
                "SELECT id FROM user_memories WHERE owner_id=? AND status='active' AND memory_key IN (?,?)",
                (owner_id, topic, existing[0].memory_key if existing else topic),
            )}
            if active_ids != {m.id for m in existing}:
                raise RuntimeError('Memory revision changed during consolidation; retry.')
            memory_id = None
            operation = 'no-op'
            mode = str(approved[0].get('revision_mode') or 'extend') if approved else 'extend'
            if normalized:
                normalized['memory_key'] = topic
                previous = existing[0] if existing else None
                correction = any(a['correction'] for a in atoms)
                # A local brevity request cannot change a stable preference, even if the LLM approves it.
                durable = any(marker in '\n'.join(source.values()) for marker in
                              ('以后', '一直', '通常', '喜欢', '偏好', '请记住', '总是'))
                if previous and previous.event_status == 'stable' and mode == 'replace' and not (correction or durable):
                    normalized = None
                    reason = 'temporary_request_cannot_replace_stable_memory'
                else:
                    normalized['pinned'] = bool(normalized['pinned']) or any(a.get('remember') for a in atoms)
                    if previous:
                        normalized['pinned'] |= previous.pinned
                    if previous and mode != 'replace' and not correction:
                        normalized['source_message_ids'] = list(dict.fromkeys(
                            previous.source_message_ids + normalized['source_message_ids']))
                        if normalized['sensitivity'] != 'restricted':
                            normalized['evidence_quotes'] = list(dict.fromkeys(
                                previous.evidence_quotes + normalized['evidence_quotes']))
                    memory = store.save_revision(owner_id, **normalized,
                        source_author=atoms[-1]['source_author'],
                        source_conversation_id=atoms[-1]['source_conversation_id'],
                        supersedes_id=previous.id if previous else None,
                        preserve_previous=False, _connection=connection)
                    memory_id = memory.id
                    operation = 'no-op' if previous and previous.id == memory.id else 'supersede' if previous else 'create'
            connection.execute(
                f"UPDATE user_memory_evidence SET status=?, memory_id=?, updated_at=? WHERE owner_id=? AND id IN ({marks})",
                ('consolidated' if normalized else 'rejected', memory_id, utc_now_iso(), owner_id, *ids),
            )
        if not normalized:
            result['rejections'].append({'reason': reason})
        result['operations'].append({'operation': operation, 'memory_id': memory_id,
            'memory_key': topic, 'topic_key': topic, 'evidence_ids': ids,
            'superseded_memory_id': existing[0].id if existing and operation == 'supersede' else None,
            'revision_mode': mode if existing else 'new'})
    return result


def memory_retrieval_text(query: str, recent_turns: list[ConversationTurn],
                          summary: dict[str, Any] | None = None) -> str:
    """Resolve lightweight follow-up recall without an additional model call."""
    dependent = bool(re.search(r'^(那|然后|这种|这样|这时|他|她|它|继续|所以|what next|then)', query.strip(), re.I))
    if not dependent:
        return query
    recent = '\n'.join(f'用户：{t.query[:400]}\n助手上下文：{t.assistant_text[:200]}' for t in recent_turns[-3:])
    background = json.dumps(summary, ensure_ascii=False)[:400] if summary and not recent else ''
    return redact_sensitive_text(f'Recent:\n{recent}\nSummary:\n{background}\nCurrent:\n{query}')



def redact_sensitive_text(text: str, *, force_finance: bool = False) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[已隐藏敏感凭据]", redacted)
    if force_finance or (any(marker in text for marker in THIRD_PARTY_MARKERS) and any(
        marker in text for marker in FINANCE_MARKERS
    )):
        redacted = EXACT_FINANCE_PATTERN.sub("[已泛化的财务数值]", redacted)
    return redacted


def _validated_candidate(
    item: dict[str, Any],
    *,
    source_text: dict[str, str],
    sanitized: dict[str, str],
) -> tuple[dict[str, Any] | None, str]:
    content = str(item.get("content") or "").strip()
    confidence = _float(item.get("confidence"), 0.0)
    source_ids = [
        str(value) for value in item.get("source_message_ids", [])
        if str(value) in source_text
    ] if isinstance(item.get("source_message_ids"), list) else []
    if isinstance(item.get('source_message_ids'), list) and any(
        str(value) not in source_text for value in item['source_message_ids']
    ):
        return None, "non_user_evidence_id"
    if not content or confidence < 0.8 or not source_ids:
        return None, "missing_content_evidence_or_confidence"
    if any(pattern.search(content) for pattern in SECRET_PATTERNS):
        return None, "secret_detected"
    sensitivity = str(item.get("sensitivity") or "private")
    if sensitivity not in SENSITIVITIES:
        sensitivity = "private"
    source_joined = "\n".join(source_text[message_id] for message_id in source_ids)
    candidate_context = f"{source_joined}\n{content}"
    if (
        any(marker in candidate_context for marker in THIRD_PARTY_MARKERS)
        and any(marker in candidate_context for marker in FINANCE_MARKERS)
    ):
        sensitivity = "restricted"
        if EXACT_FINANCE_PATTERN.search(content):
            content = _generalize_finance_numbers(content)
    if any(marker in content for marker in BELIEF_MARKERS) and all(
        any(marker in source_text[message_id] for marker in QUESTION_MARKERS)
        for message_id in source_ids
    ):
        return None, "question_promoted_to_belief"
    if all(any(marker in source_text[mid] for marker in QUESTION_MARKERS)
           and not any(marker in source_text[mid] for marker in ("我", "以后", "请记住"))
           for mid in source_ids):
        return None, "general_knowledge_question"
    if any(marker in source_joined for marker in ("简单说", "简短说", "一句话")) and not any(
        marker in source_joined for marker in ("以后", "一直", "通常", "喜欢", "偏好", "请记住")
    ):
        return None, "temporary_request_not_memory"
    source_relation = _third_party_relation(source_joined)
    if source_relation and _third_party_relation(content) != source_relation and not any(
        marker in content for marker in ("用户担心", "用户希望")
    ):
        return None, "third_party_attribution_mismatch"
    if "未提供资金支持" in content and not any(
        marker in source_joined for marker in ("没给钱", "没有给钱", "未提供资金", "没出钱")
    ):
        content = re.sub(r"用户未提供资金支持[，,、；;]?", "", content).strip()
        if not content:
            return None, "unsupported_literalization"
    evidence_quotes = [
        str(value).strip() for value in item.get("evidence_quotes", []) if str(value).strip()
    ] if isinstance(item.get("evidence_quotes"), list) else []
    if sensitivity == "restricted":
        content = _generalize_finance_numbers(content)
        evidence_quotes = []
    elif not evidence_quotes or any(
        quote not in "\n".join(sanitized[message_id] for message_id in source_ids)
        for quote in evidence_quotes
    ):
        return None, "invalid_evidence_quote"
    kind = str(item.get("kind") or "episodic")
    if kind not in MEMORY_KINDS:
        kind = "episodic"
    event_status = str(item.get("event_status") or "ongoing")
    if event_status not in EVENT_STATUSES:
        event_status = "ongoing"
    if kind == "procedural" and not any(
        marker in source_joined for marker in ("以后", "请记住", "总是", "不要再", "希望你")
    ):
        return None, "procedural_not_explicit"
    relation = _third_party_relation(candidate_context)
    if relation and any(marker in candidate_context for marker in FINANCE_MARKERS):
        kind = "episodic"
        sensitivity = "restricted"
        event_status = "ongoing"
        memory_key = f"family.{relation}.high_risk_finance"
        importance = 5
    else:
        memory_key = _normalize_key(str(item.get("memory_key") or "user.context"))
        importance = max(1, min(5, int(_float(item.get("importance"), 3))))
    return {
        "kind": kind,
        "memory_key": memory_key,
        "content": content,
        "sensitivity": sensitivity,
        "importance": importance,
        "confidence": confidence,
        "event_status": event_status,
        "source_message_ids": source_ids,
        "evidence_quotes": evidence_quotes,
        "pinned": bool(item.get("pinned")),
    }, ""


def _memory_write_prompt(
    *,
    context_messages: list[dict[str, str]],
    source_messages: dict[str, str],
    existing: list[UserMemory],
    existing_topics: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "context_messages": context_messages,
            "eligible_evidence_message_ids": list(source_messages),
            "source_messages": source_messages,
            "existing_topics": existing_topics or [],
            "existing_memories": [
                {
                    "id": memory.id,
                    "memory_key": memory.memory_key,
                    "content": memory.content,
                    "updated_at": memory.updated_at,
                }
                for memory in existing
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def _canonicalize_extracted_candidates(
    candidates: list[Any],
    source_messages: dict[str, str],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for value in candidates:
        if not isinstance(value, dict):
            continue
        item = dict(value)
        source_ids = item.get("source_message_ids") if isinstance(item.get("source_message_ids"), list) else []
        source = "\n".join(source_messages.get(str(message_id), "") for message_id in source_ids)
        candidate_context = f"{source}\n{str(item.get('content') or '')}"
        relation = _third_party_relation(candidate_context)
        if relation and any(marker in candidate_context for marker in FINANCE_MARKERS):
            item["memory_key"] = f"family.{relation}.high_risk_finance"
            item["kind"] = "episodic"
            item["sensitivity"] = "restricted"
            item["importance"] = 5
            item["event_status"] = "ongoing"
        normalized.append(item)
    return normalized


def _third_party_relation(text: str) -> str | None:
    relations = (
        (("我哥", "哥哥"), "brother"),
        (("我弟", "弟弟"), "brother"),
        (("姐姐", "妹妹"), "sibling"),
        (("父亲", "爸爸"), "father"),
        (("母亲", "妈妈"), "mother"),
        (("朋友",), "friend"),
        (("同事",), "colleague"),
        (("家人",), "family_member"),
    )
    for markers, relation in relations:
        if any(marker in text for marker in markers):
            return relation
    return None


def _row_to_memory(row: sqlite3.Row) -> UserMemory:
    return UserMemory(
        id=str(row["id"]), owner_id=str(row["owner_id"]), kind=str(row["kind"]),
        memory_key=str(row["memory_key"]), content=str(row["content"]), status=str(row["status"]),
        pinned=bool(row["pinned"]), sensitivity=str(row["sensitivity"]),
        importance=int(row["importance"]), confidence=float(row["confidence"]),
        event_status=str(row["event_status"]), source_author=row["source_author"],
        source_conversation_id=row["source_conversation_id"],
        source_message_ids=[str(item) for item in json.loads(str(row["source_message_ids_json"]))],
        evidence_quotes=[str(item) for item in json.loads(str(row["evidence_quotes_json"]))],
        supersedes_id=row["supersedes_id"], created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]), last_accessed_at=row["last_accessed_at"],
        access_count=int(row["access_count"]),
    )


def _normalize_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.-]+", ".", value.strip().lower()).strip(".")
    return normalized[:120] or "user.context"


def _generalize_finance_numbers(text: str) -> str:
    generalized = re.sub(r"\d+(?:\.\d+)?\s*(?:%|％|成|个点)", "较大幅度", text)
    generalized = re.sub(r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万两]+)\s*倍", "高倍", generalized)
    generalized = re.sub(
        r"(?<![A-Za-z])(?:\d+(?:\.\d+)?\s*(?:万|千|百|元|块|[wW])|几(?:百|千|万)(?:元|块)?|\d{4,})(?![A-Za-z])",
        "一笔资金",
        generalized,
    )
    return generalized


def _merge_revision_content(previous: str, current: str) -> str:
    previous = previous.strip()
    current = current.strip()
    if not previous or previous in current:
        return current
    if current in previous:
        return previous
    sentences: list[str] = []
    for text in (previous, current):
        for sentence in re.split(r"(?<=[。！？!?；;])", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            replacement_index = next(
                (index for index, existing in enumerate(sentences) if existing in sentence),
                None,
            )
            if replacement_index is not None:
                sentences[replacement_index] = sentence
                continue
            if any(sentence in existing for existing in sentences):
                continue
            sentences.append(sentence)
    return "".join(sentences)[:1000]


def _is_explicit_forget(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return any(marker in compact for marker in ("忘掉这件事", "忘记这件事", "不要记住这个", "删除这条记忆"))


def _is_explicit_remember(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return any(marker in compact for marker in ("请记住", "帮我记住", "以后记得", "以后都"))


def _is_explicit_correction(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return any(marker in compact for marker in ("前面说错了", "刚才记错了", "纠正一下")) or (
        "不是" in compact and "是" in compact and not any(m in compact for m in ("是不是", "？", "?"))
    )


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / denominator if denominator else 0.0


def _sparse_dot(left: SparseEmbedding, right: SparseEmbedding) -> float:
    right_values = dict(zip(right.indices, right.values, strict=True))
    return sum(value * right_values.get(index, 0.0) for index, value in zip(left.indices, left.values, strict=True))


MEMORY_EXTRACTOR_PROMPT = """你是 PersonaForge 的用户长期记忆候选提取器。

V2：提取最小可追溯的 Atomic Evidence，不重写整个用户画像。
优先使用 existing_memories / existing_topics 中语义一致的 memory_key；仅在人物和主题
明显不同的时候创建新 key。不要因为措辞差异拆分同一个主题，也不能合并不同亲友。

context_messages 包含完整的用户与助手对话，只用于理解代词、指代和用户在回应什么。
只有 source_messages 和 eligible_evidence_message_ids 中的用户原话可以作为记忆事实依据。
助手回答不能独立支持任何记忆，也不能因为助手提出了某个说法就推断用户认可。
可提取 semantic（稳定事实/偏好）、episodic（仍可能延续的重要事件）、procedural（用户明确要求以后如何协作）。

严格约束：
- 问题、假设、担忧不是用户信念；“是不是一定亏光”不能写成“用户认为一定亏光”。
- 必须区分用户本人和哥哥、朋友等第三方，不能把第三方经历归到用户本人。
- 不保存闲聊、一次性任务、普通知识问题、模型生成的建议。
- 凭据、密码、Cookie、API key 永不提取。
- 财务、健康、关系等敏感事件只保留完成未来帮助所需的概括，不恢复被隐藏的具体数值。
- procedural 只有用户明确表达长期要求时才允许。

只能输出 JSON：
{"candidates":[{"kind":"semantic|episodic|procedural","memory_key":"稳定英文点号键","content":"第三人称中文事实","sensitivity":"normal|private|restricted","importance":1,"confidence":0.0,"event_status":"ongoing|historical|stable","source_message_ids":[],"context_message_ids":[],"evidence_quotes":[],"supersedes_id":null,"revision_mode":"new|extend|replace","pinned":false}]}
没有值得长期保存的信息时输出 {"candidates":[]}。
"""


MEMORY_CRITIC_PROMPT = """你是用户长期记忆的保守审查器。逐条核对候选是否被用户原话直接支持、主语是否正确、是否值得跨会话保存，以及是否与已有记忆重复或修订。context_messages 中的助手消息只能帮助消解指代，不能作为事实证据。

特别拒绝：把疑问当信念；把亲友经历当用户经历；从助手建议推断用户接受；精确敏感财务数值；普通提问；低置信度推断。敏感事件应概括但保留关键人物关系与用户真正担忧。若是同一事实的新状态，填写已有 memory id 为 supersedes_id。
同一人物、同一持续事件的多个候选应合并成一条完整但克制的 episodic 记忆，不要把
“亏损事实、风险行为、用户担忧”机械拆成多条。最终内容不得恢复或保留金额、比例、
余额、杠杆倍数等精确数值。只要批准，confidence 应表示经过本轮审查后的置信度。
若多个候选具有相同 memory_key，只能输出一条。若已有相同 key，将已有 id 写入
supersedes_id：新证据与旧内容兼容时 revision_mode=extend；出现冲突、状态改变或用户纠正
时区分：时态状态变化和用户明确纠错可 revision_mode=replace；稳定偏好不能统一 latest-wins。
V2 中 candidates 是已存储的原子证据。按 topic_key 合并成一个稳定归纳，保留来源；
不要直接逐条 append。extend 的 content 必须包含仍成立的旧记忆和新的兼容信息，形成完整新归纳。
考虑独立用户消息数量、source_timestamps 所示原始证据时间跨度、表达强度以及以后/通常等长期意图。
一次“简单说”只是局部要求；不足以替换过去反复明确的详细解释偏好。
新状态替代旧 active 状态，旧 evidence 保留。证据不足可 decision=reject，不得编造。
不得把隐喻扩写成具体事实，例如
“没递拐杖”本身不能推出“没有提供资金支持”。

只能输出 JSON：
{"memories":[{"decision":"approve|reject","reason":"简短原因","kind":"semantic|episodic|procedural","memory_key":"...","content":"...","sensitivity":"normal|private|restricted","importance":1,"confidence":0.0,"event_status":"ongoing|historical|stable","source_message_ids":[],"context_message_ids":[],"evidence_quotes":[],"supersedes_id":null,"revision_mode":"new|extend|replace","pinned":false}]}
"""
