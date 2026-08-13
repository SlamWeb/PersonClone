"""Generation-system discovery, human labels, pairwise votes, and Judge jobs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from personaforge.eval.gold_judge import (
    DIMENSION_KEYS,
    GROUP_LABELS,
    JudgeConfig,
    PROMPT_VERSION,
    evaluate_item,
    prompt_hash,
    rubric_payload,
    summarize_system,
)
from personaforge.llm import DEFAULT_DEEPSEEK_MODEL, DeepSeekJsonClient


SUPPORTED_RUN_SCHEMAS = {
    "personaforge.eval.run.v0",
    "personaforge.eval.writer-replay.v0",
}


@dataclass(frozen=True, slots=True)
class GenerationSystemFiles:
    system_id: str
    run_dir: Path
    manifest_path: Path
    runs_path: Path
    manifest: dict[str, Any]
    dataset_manifest: dict[str, Any]
    items: tuple[dict[str, Any], ...]


class GenerationEvaluationStore:
    """Discover immutable generation runs and persist per-user judgments."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.eval_root = self.data_dir / "eval"
        self.database_path = self.data_dir / "system" / "personaforge.sqlite3"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def list_systems(self, user_id: str, author: str | None = None) -> list[dict[str, Any]]:
        systems = []
        for files in self._discover().values():
            public_system = self._public_system(files)
            if author is not None and public_system["author"] != author:
                continue
            rubric_count = self._rubric_count(files.system_id, user_id)
            job = self.latest_judge_job(files.system_id)
            systems.append(
                {
                    **public_system,
                    "human_progress": {
                        "completed": rubric_count,
                        "total": len(files.items),
                    },
                    "judge": self._public_job(job, include_result=True) if job else None,
                }
            )
        return sorted(
            systems,
            key=lambda row: (row.get("created_at") or "", row["display_name"]),
            reverse=True,
        )

    def workspace(self, system_id: str, user_id: str) -> dict[str, Any]:
        files = self.system(system_id)
        labels = self._rubric_labels(system_id, user_id)
        return {
            "system": self._public_system(files),
            "rubric": rubric_payload(),
            "groups": GROUP_LABELS,
            "progress": {
                "completed": sum(_rubric_complete(labels.get(item["item_id"], {})) for item in files.items),
                "total": len(files.items),
            },
            "items": [
                {
                    "item_id": item["item_id"],
                    "ordinal": index,
                    "question": item["question"],
                    "completed": _rubric_complete(labels.get(item["item_id"], {})),
                }
                for index, item in enumerate(files.items, start=1)
            ],
            "judge": self._public_job(self.latest_judge_job(system_id), include_result=True),
        }

    def item(self, system_id: str, item_id: str, user_id: str) -> dict[str, Any]:
        files = self.system(system_id)
        item = self._item(files, item_id)
        label = self._rubric_labels(system_id, user_id).get(item_id, {})
        judge_item = self._judge_item(system_id, item_id)
        return {
            "system": self._public_system(files),
            **item,
            "human_scores": label.get("scores", {}),
            "human_note": label.get("note", ""),
            "human_completed": _rubric_complete(label),
            "judge": judge_item,
        }

    def set_rubric(
        self,
        system_id: str,
        item_id: str,
        user_id: str,
        scores: dict[str, int | None],
        note: str = "",
    ) -> dict[str, Any]:
        files = self.system(system_id)
        self._item(files, item_id)
        unknown = set(scores) - set(DIMENSION_KEYS)
        if unknown:
            raise ValueError(f"Unknown dimensions: {', '.join(sorted(unknown))}")
        clean: dict[str, int] = {}
        for key, value in scores.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"{key} must be an integer from 1 to 5")
            clean[key] = value
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO generation_rubric_labels (
                    system_id, item_id, user_id, scores_json, note, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(system_id, item_id, user_id)
                DO UPDATE SET scores_json=excluded.scores_json, note=excluded.note,
                              updated_at=excluded.updated_at
                """,
                (system_id, item_id, user_id, json.dumps(clean, sort_keys=True), note.strip(), now, now),
            )
        return {
            "system_id": system_id,
            "item_id": item_id,
            "scores": clean,
            "note": note.strip(),
            "completed": set(clean) == set(DIMENSION_KEYS),
            "saved_at": now,
        }

    def comparison(self, left_id: str, right_id: str, user_id: str) -> dict[str, Any]:
        left, right = self._compatible_pair(left_id, right_id)
        votes = self._pair_votes(left.system_id, right.system_id, user_id)
        return {
            "comparison_id": _pair_id(left.system_id, right.system_id),
            "systems": [self._public_system(left), self._public_system(right)],
            "progress": {"completed": len(votes), "total": len(left.items)},
            "items": [
                {
                    "item_id": item["item_id"],
                    "ordinal": index,
                    "question": item["question"],
                    "completed": item["item_id"] in votes,
                }
                for index, item in enumerate(left.items, start=1)
            ],
            "result": self._pair_result(left, right, votes),
        }

    def comparison_item(
        self, left_id: str, right_id: str, item_id: str, user_id: str
    ) -> dict[str, Any]:
        left, right = self._compatible_pair(left_id, right_id)
        left_item = self._item(left, item_id)
        right_item = self._item(right, item_id)
        a_system, b_system = self._ab_order(left, right, item_id, user_id)
        a_item = left_item if a_system.system_id == left.system_id else right_item
        b_item = right_item if b_system.system_id == right.system_id else left_item
        vote = self._pair_votes(left.system_id, right.system_id, user_id).get(item_id)
        payload = {
            "comparison_id": _pair_id(left.system_id, right.system_id),
            "item_id": item_id,
            "question": left_item["question"],
            "gold_answer": left_item["gold_answer"],
            "candidate_a": a_item["candidate_answer"],
            "candidate_b": b_item["candidate_answer"],
            "choice": None,
            "revealed": None,
        }
        if vote:
            choice = "A" if vote == a_system.system_id else "B"
            payload["choice"] = choice
            payload["revealed"] = {
                "A": self._public_system(a_system),
                "B": self._public_system(b_system),
            }
        return payload

    def set_pair_vote(
        self,
        left_id: str,
        right_id: str,
        item_id: str,
        user_id: str,
        choice: str,
    ) -> dict[str, Any]:
        left, right = self._compatible_pair(left_id, right_id)
        self._item(left, item_id)
        a_system, b_system = self._ab_order(left, right, item_id, user_id)
        normalized = choice.upper()
        if normalized not in {"A", "B"}:
            raise ValueError("choice must be A or B")
        winner = a_system if normalized == "A" else b_system
        now = _utc_now()
        pair_id = _pair_id(left.system_id, right.system_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO generation_pairwise_labels (
                    pair_id, item_id, user_id, winner_system_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pair_id, item_id, user_id)
                DO UPDATE SET winner_system_id=excluded.winner_system_id,
                              updated_at=excluded.updated_at
                """,
                (pair_id, item_id, user_id, winner.system_id, now, now),
            )
        return {
            "comparison_id": pair_id,
            "item_id": item_id,
            "choice": normalized,
            "winner": self._public_system(winner),
            "revealed": {
                "A": self._public_system(a_system),
                "B": self._public_system(b_system),
            },
            "saved_at": now,
        }

    def system(self, system_id: str) -> GenerationSystemFiles:
        try:
            return self._discover()[system_id]
        except KeyError as exc:
            raise KeyError(f"Unknown complete generation system: {system_id}") from exc

    def latest_judge_job(self, system_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM generation_judge_jobs
                WHERE system_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (system_id,),
            ).fetchone()
        return dict(row) if row else None

    def judge_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generation_judge_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError("Unknown generation Judge job")
        return dict(row)

    def public_judge_job(self, job_id: str) -> dict[str, Any]:
        return self._public_job(self.judge_job(job_id), include_result=True) or {}

    def _discover(self) -> dict[str, GenerationSystemFiles]:
        systems: dict[str, GenerationSystemFiles] = {}
        if not self.eval_root.exists():
            return systems
        for manifest_path in self.eval_root.glob("*/runs/*/manifest.json"):
            files = self._load_system(manifest_path)
            if files:
                systems[files.system_id] = files
        return systems

    def _load_system(self, manifest_path: Path) -> GenerationSystemFiles | None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") not in SUPPORTED_RUN_SCHEMAS:
                return None
            if manifest.get("status") != "completed":
                return None
            run_dir = manifest_path.parent
            runs_path = run_dir / "runs.jsonl"
            if not runs_path.exists():
                return None
            dataset_manifest_path = run_dir.parent.parent / "dataset_manifest.json"
            dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            if manifest.get("dataset_sha256") != dataset_manifest.get("dataset_sha256"):
                return None
            config = manifest.get("config") or {}
            split = str(config.get("split") or "dev")
            if split not in {"dev", "test"}:
                return None
            expected_count = int((dataset_manifest.get("counts") or {}).get(split) or 0)
            if expected_count <= 0:
                expected_count = 10 if split == "dev" else 20
            if int(manifest.get("item_count") or 0) != expected_count:
                return None
            rows = _read_jsonl(runs_path)
            items: list[dict[str, Any]] = []
            for row in rows:
                if str(row.get("split") or "") != split or row.get("status") != "completed":
                    continue
                item_id = str(row.get("item_id") or "")
                question = str(row.get("query") or row.get("question") or "").strip()
                gold = str(row.get("gold_answer") or "").strip()
                candidate = str(
                    row.get("candidate_answer")
                    or row.get("generated_answer")
                    or row.get("answer")
                    or ""
                ).strip()
                if item_id and question and gold and candidate:
                    items.append(
                        {
                            "item_id": item_id,
                            "question": question,
                            "gold_answer": gold,
                            "candidate_answer": candidate,
                        }
                    )
            items.sort(key=lambda row: row["item_id"])
            if len(items) != expected_count or len({row["item_id"] for row in items}) != expected_count:
                return None
            run_sha = str(manifest.get("run_sha256") or _sha256(runs_path))
            return GenerationSystemFiles(
                system_id=run_sha,
                run_dir=run_dir,
                manifest_path=manifest_path,
                runs_path=runs_path,
                manifest=manifest,
                dataset_manifest=dataset_manifest,
                items=tuple(items),
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _public_system(self, files: GenerationSystemFiles) -> dict[str, Any]:
        config = files.manifest.get("config") or {}
        name = str(config.get("run_name") or files.run_dir.name)
        parameters = {
            key: config.get(key)
            for key in (
                "split",
                "query_mode",
                "writer_prompt",
                "child_top_k",
                "per_query_parent_k",
                "parent_top_k",
                "writer_context_top_k",
                "max_search_results",
                "temperature",
                "max_tokens",
            )
            if config.get(key) is not None
        }
        git_info = files.manifest.get("git") or {}
        return {
            "system_id": files.system_id,
            "run_name": name,
            "method_id": str(files.manifest.get("method_id") or config.get("method_id") or name),
            "display_name": str(files.manifest.get("display_name") or config.get("display_name") or name),
            "description": str(files.manifest.get("description") or ""),
            "parent_method": files.manifest.get("parent_method") or config.get("parent_method"),
            "author": str(config.get("author") or files.dataset_manifest.get("author") or ""),
            "author_status": "assigned" if str(config.get("author") or files.dataset_manifest.get("author") or "").strip() else "unassigned",
            "dataset_id": str(
                files.dataset_manifest.get("dataset_id")
                or ("temporal_test20_v0" if str(config.get("split") or "dev") == "test" else "temporal_dev10_v0")
            ),
            "dataset_sha256": str(files.manifest.get("dataset_sha256") or ""),
            "split": str(config.get("split") or "dev"),
            "item_count": len(files.items),
            "writer_prompt": str(config.get("writer_prompt") or ""),
            "prompt_version": str(files.manifest.get("prompt_version") or config.get("prompt_version") or config.get("writer_prompt") or ""),
            "prompt_sha256": str(files.manifest.get("prompt_sha256") or ""),
            "parameters": parameters,
            "git_revision": str(git_info.get("revision") or ""),
            "model": str(files.manifest.get("writer_model") or ""),
            "created_at": str(files.manifest.get("finished_at") or files.manifest.get("started_at") or ""),
        }

    @staticmethod
    def _item(files: GenerationSystemFiles, item_id: str) -> dict[str, Any]:
        item = next((row for row in files.items if row["item_id"] == item_id), None)
        if item is None:
            raise KeyError(f"Unknown generation item: {item_id}")
        return item

    def _compatible_pair(
        self, left_id: str, right_id: str
    ) -> tuple[GenerationSystemFiles, GenerationSystemFiles]:
        if left_id == right_id:
            raise ValueError("Choose two different systems")
        left, right = self.system(left_id), self.system(right_id)
        if left.manifest.get("dataset_sha256") != right.manifest.get("dataset_sha256"):
            raise ValueError("Systems use different frozen datasets")
        left_author = str(self._public_system(left).get("author") or "").strip()
        right_author = str(self._public_system(right).get("author") or "").strip()
        if not left_author or not right_author:
            raise ValueError("Cannot compare systems without an assigned author")
        if left_author != right_author:
            raise ValueError("Systems belong to different authors")
        if [row["item_id"] for row in left.items] != [row["item_id"] for row in right.items]:
            raise ValueError("Systems do not contain the same evaluation items")
        return left, right

    @staticmethod
    def _ab_order(
        left: GenerationSystemFiles,
        right: GenerationSystemFiles,
        item_id: str,
        user_id: str,
    ) -> tuple[GenerationSystemFiles, GenerationSystemFiles]:
        digest = hashlib.sha256(
            f"{_pair_id(left.system_id, right.system_id)}|{item_id}|{user_id}".encode("utf-8")
        ).digest()
        return (left, right) if digest[0] % 2 == 0 else (right, left)

    def _pair_result(
        self,
        left: GenerationSystemFiles,
        right: GenerationSystemFiles,
        votes: dict[str, str],
    ) -> dict[str, Any]:
        counts = {left.system_id: 0, right.system_id: 0}
        for winner in votes.values():
            if winner in counts:
                counts[winner] += 1
        return {
            "votes": len(votes),
            "wins": [
                {"system": self._public_system(left), "count": counts[left.system_id]},
                {"system": self._public_system(right), "count": counts[right.system_id]},
            ],
        }

    def _rubric_labels(self, system_id: str, user_id: str) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT item_id, scores_json, note FROM generation_rubric_labels
                WHERE system_id = ? AND user_id = ?
                """,
                (system_id, user_id),
            ).fetchall()
        return {
            str(row["item_id"]): {
                "scores": json.loads(row["scores_json"]),
                "note": str(row["note"] or ""),
            }
            for row in rows
        }

    def _rubric_count(self, system_id: str, user_id: str) -> int:
        return sum(_rubric_complete(row) for row in self._rubric_labels(system_id, user_id).values())

    def _pair_votes(self, left_id: str, right_id: str, user_id: str) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT item_id, winner_system_id FROM generation_pairwise_labels
                WHERE pair_id = ? AND user_id = ?
                """,
                (_pair_id(left_id, right_id), user_id),
            ).fetchall()
        return {str(row["item_id"]): str(row["winner_system_id"]) for row in rows}

    def _judge_item(self, system_id: str, item_id: str) -> dict[str, Any] | None:
        job = self.latest_judge_job(system_id)
        if not job or job.get("status") != "completed" or not job.get("output_path"):
            return None
        payload = _read_json(Path(str(job["output_path"])))
        return next((row for row in payload.get("items", []) if row.get("item_id") == item_id), None)

    def _public_job(
        self, job: dict[str, Any] | None, *, include_result: bool = False
    ) -> dict[str, Any] | None:
        if not job:
            return None
        payload = {
            key: job.get(key)
            for key in (
                "id", "system_id", "status", "stage", "label", "model", "repeats",
                "completed_items", "total_items", "error_message", "created_at",
                "started_at", "completed_at",
            )
        }
        if include_result and job.get("status") == "completed" and job.get("output_path"):
            output = _read_json(Path(str(job["output_path"])))
            payload["result"] = output.get("summary")
            payload["prompt_version"] = output.get("config", {}).get("prompt_version") or PROMPT_VERSION
        elif job.get("status") in {"queued", "running"}:
            payload["prompt_version"] = PROMPT_VERSION
        return payload

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS generation_rubric_labels (
                    system_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    scores_json TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(system_id, item_id, user_id)
                );
                CREATE TABLE IF NOT EXISTS generation_pairwise_labels (
                    pair_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    winner_system_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(pair_id, item_id, user_id)
                );
                CREATE TABLE IF NOT EXISTS generation_judge_jobs (
                    id TEXT PRIMARY KEY,
                    system_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    label TEXT NOT NULL,
                    model TEXT NOT NULL,
                    repeats INTEGER NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    completed_items INTEGER NOT NULL DEFAULT 0,
                    total_items INTEGER NOT NULL DEFAULT 10,
                    output_path TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_generation_judge_system
                ON generation_judge_jobs(system_id, created_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


class GenerationJudgeManager:
    """One persistent local worker for long-running Gold Judge jobs."""

    def __init__(
        self,
        store: GenerationEvaluationStore,
        *,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.store = store
        self.client_factory = client_factory or self._default_client
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self.store._connect() as connection:
            connection.execute(
                """
                UPDATE generation_judge_jobs
                SET status='queued', stage='queued', label='等待继续评估',
                    error_message=NULL, updated_at=?
                WHERE status='running'
                """,
                (_utc_now(),),
            )
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="generation-judge", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=3)

    def create(self, system_id: str, *, repeats: int = 3) -> dict[str, Any]:
        files = self.store.system(system_id)
        if repeats != 3:
            raise ValueError("Gold Judge V1 currently requires exactly 3 repeats")
        model = os.getenv("PERSONAFORGE_JUDGE_MODEL") or os.getenv("PERSONAFORGE_QUERY_MODEL") or DEFAULT_DEEPSEEK_MODEL
        retry_job_id: str | None = None
        with self.store._connect() as connection:
            active = connection.execute(
                """
                SELECT * FROM generation_judge_jobs
                WHERE system_id=? AND status IN ('queued','running')
                ORDER BY created_at DESC LIMIT 1
                """,
                (system_id,),
            ).fetchone()
            if active:
                return self.store._public_job(dict(active), include_result=True) or {}
            completed = connection.execute(
                """
                SELECT * FROM generation_judge_jobs
                WHERE system_id=? AND status='completed' AND model=? AND repeats=? AND prompt_hash=?
                ORDER BY created_at DESC LIMIT 1
                """,
                (system_id, model, repeats, prompt_hash()),
            ).fetchone()
            if completed:
                return self.store._public_job(dict(completed), include_result=True) or {}
            failed = connection.execute(
                """
                SELECT * FROM generation_judge_jobs
                WHERE system_id=? AND status='failed' AND model=? AND repeats=? AND prompt_hash=?
                ORDER BY created_at DESC LIMIT 1
                """,
                (system_id, model, repeats, prompt_hash()),
            ).fetchone()
            if failed:
                connection.execute(
                    """
                    UPDATE generation_judge_jobs
                    SET status='queued', stage='queued', label='等待继续评估',
                        error_message=NULL, completed_at=NULL, updated_at=?
                    WHERE id=?
                    """,
                    (_utc_now(), failed["id"]),
                )
                retry_job_id = str(failed["id"])
            else:
                job_id = uuid.uuid4().hex
                now = _utc_now()
                output_path = files.run_dir / "judges" / "gold-v1" / job_id / "result.json"
                connection.execute(
                    """
                    INSERT INTO generation_judge_jobs (
                        id, system_id, status, stage, label, model, repeats, prompt_hash,
                        completed_items, total_items, output_path, created_at, updated_at
                    ) VALUES (?, ?, 'queued', 'queued', '等待评估', ?, ?, ?, 0, ?, ?, ?, ?)
                    """,
                    (job_id, system_id, model, repeats, prompt_hash(), len(files.items), str(output_path), now, now),
                )
        self._wake.set()
        if retry_job_id is not None:
            return self.store.public_judge_job(retry_job_id)
        return self.store.public_judge_job(job_id)

    def run_once(self) -> bool:
        with self.store._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generation_judge_jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return False
            now = _utc_now()
            connection.execute(
                """
                UPDATE generation_judge_jobs
                SET status='running', stage='judging', label='正在运行 Gold Judge',
                    started_at=COALESCE(started_at, ?), updated_at=?
                WHERE id=?
                """,
                (now, now, row["id"]),
            )
        self._execute(str(row["id"]))
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.run_once():
                continue
            self._wake.wait(timeout=1)
            self._wake.clear()

    def _execute(self, job_id: str) -> None:
        job = self.store.judge_job(job_id)
        files = self.store.system(str(job["system_id"]))
        output_path = Path(str(job["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = _read_json(output_path)
        finished = {str(row["item_id"]): row for row in existing.get("items", [])}
        config = JudgeConfig(model=str(job["model"]), repeats=int(job["repeats"]))
        try:
            for index, item in enumerate(files.items, start=1):
                if self._stop.is_set():
                    raise RuntimeError("服务停止，任务将在下次启动后继续")
                if item["item_id"] not in finished:
                    result = evaluate_item(
                        client_factory=self.client_factory,
                        question=item["question"],
                        gold_answer=item["gold_answer"],
                        candidate_answer=item["candidate_answer"],
                        config=config,
                    )
                    finished[item["item_id"]] = {"item_id": item["item_id"], **result}
                rows = [finished[key] for key in sorted(finished)]
                _write_json(
                    output_path,
                    {
                        "schema_version": "personaforge.eval.gold_judge_result.v1",
                        "system_id": files.system_id,
                        "config": config.to_dict(),
                        "items": rows,
                        "summary": summarize_system(rows),
                    },
                )
                self._update_job(
                    job_id,
                    completed_items=len(rows),
                    label=f"已评估 {len(rows)} / {len(files.items)} 题",
                )
            self._update_job(
                job_id,
                status="completed",
                stage="completed",
                label="Gold Judge 已完成",
                completed_at=_utc_now(),
                error_message=None,
            )
        except Exception as exc:
            status = "queued" if self._stop.is_set() else "failed"
            self._update_job(
                job_id,
                status=status,
                stage=status,
                label="等待继续评估" if status == "queued" else "Gold Judge 失败",
                error_message=str(exc),
                completed_at=None if status == "queued" else _utc_now(),
            )

    def _update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {
            "status",
            "stage",
            "label",
            "completed_items",
            "error_message",
            "completed_at",
        }
        unexpected = set(fields) - allowed
        if unexpected:
            raise ValueError(f"Unsupported generation judge fields: {sorted(unexpected)}")
        fields["updated_at"] = _utc_now()
        assignments = ", ".join(f"{key}=?" for key in fields)
        with self.store._connect() as connection:
            connection.execute(
                f"UPDATE generation_judge_jobs SET {assignments} WHERE id=?",
                (*fields.values(), job_id),
            )

    @staticmethod
    def _default_client() -> DeepSeekJsonClient:
        client = DeepSeekJsonClient.from_env()
        client.model = os.getenv("PERSONAFORGE_JUDGE_MODEL") or client.model
        client.thinking = "disabled"
        client.timeout_seconds = 180
        return client


def _rubric_complete(label: dict[str, Any]) -> bool:
    return set(label.get("scores", {})) == set(DIMENSION_KEYS)


def _pair_id(left_id: str, right_id: str) -> str:
    ordered = sorted((left_id, right_id))
    return hashlib.sha256(f"{ordered[0]}|{ordered[1]}".encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
