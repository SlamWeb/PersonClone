"""Durable atomic evidence; never recalled directly into Writer context."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from personaforge.web.conversations import utc_now_iso


EVIDENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_memory_evidence (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    topic_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    source_conversation_id TEXT NOT NULL,
    source_message_ids_json TEXT NOT NULL,
    evidence_quotes_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    sensitivity TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    payload_json TEXT NOT NULL,
    memory_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_evidence_pending
    ON user_memory_evidence(owner_id, status, topic_key);
"""


class EvidenceStoreMixin:
    def list_evidence(self, owner_id: str, *, status: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM user_memory_evidence WHERE owner_id = ?"
                + (" AND status = ?" if status else "") + " ORDER BY created_at, rowid",
                (owner_id, status) if status else (owner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def commit_evidence_batch(self, owner_id, conversation_id, checkpoint, through_sequence, atoms):
        """CAS checkpoint and atom inserts share one transaction, including empty batches."""
        ids = []
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT through_sequence FROM user_memory_checkpoints WHERE owner_id=? AND conversation_id=?",
                (owner_id, conversation_id),
            ).fetchone()
            actual = int(row[0]) if row else 0
            if actual != checkpoint:
                raise RuntimeError("Memory extraction checkpoint changed; retry from durable state.")
            for atom in atoms:
                identity = json.dumps([owner_id, conversation_id, sorted(atom['source_message_ids']),
                                       atom['memory_key'], atom['content']], ensure_ascii=False)
                evidence_id = 'atom-' + hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]
                connection.execute(
                    """INSERT OR IGNORE INTO user_memory_evidence
                    (id, owner_id, topic_key, kind, content, source_conversation_id,
                     source_message_ids_json, evidence_quotes_json, confidence, sensitivity,
                     payload_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (evidence_id, owner_id, atom['memory_key'], atom['kind'], atom['content'], conversation_id,
                     json.dumps(atom['source_message_ids']), json.dumps(atom['evidence_quotes'], ensure_ascii=False),
                     atom['confidence'], atom['sensitivity'], json.dumps(atom, ensure_ascii=False), now, now),
                )
                ids.append(evidence_id)
            connection.execute(
                """INSERT INTO user_memory_checkpoints VALUES (?,?,?,?)
                ON CONFLICT(owner_id,conversation_id) DO UPDATE SET
                through_sequence=excluded.through_sequence, updated_at=excluded.updated_at""",
                (owner_id, conversation_id, through_sequence, now),
            )
        return ids

    def suppress_evidence(self, connection, owner_id, topic_key=None):
        connection.execute(
            "UPDATE user_memory_evidence SET status='forgotten', updated_at=? WHERE owner_id=?"
            + (" AND topic_key=?" if topic_key else ""),
            (utc_now_iso(), owner_id, topic_key) if topic_key else (utc_now_iso(), owner_id),
        )
