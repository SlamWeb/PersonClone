"""Persistent background jobs for initializing author retrieval evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from personaforge.eval.dataset import (
    freeze_corpus_snapshot,
    is_eligible_answer,
    load_jsonl,
    prepare_temporal_dataset,
)
from personaforge.eval.retrieval_gold_qrels import (
    export_codex_gold_handoff,
    extract_gold_units,
    label_gold_aware_pool,
    materialize_codex_gold_labels,
)
from personaforge.eval.retrieval_pool import build_exhaustive_retrieval_pool
from personaforge.llm import DeepSeekJsonClient


ACTIVE_STATUSES = ("queued", "running")


@dataclass(slots=True)
class RetrievalEvalJobConfig:
    data_dir: Path
    model_name: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    use_fp16: bool = True
    working_dir: Path = Path.cwd()

    def __post_init__(self) -> None:
        self.data_dir = self.data_dir.expanduser().resolve()
        self.working_dir = self.working_dir.expanduser().resolve()


class RetrievalEvalJobManager:
    """One restart-safe worker for expensive multi-author retrieval eval setup."""

    def __init__(
        self,
        config: RetrievalEvalJobConfig,
        *,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.database_path = config.data_dir / "system" / "personaforge.sqlite3"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.client_factory = client_factory or self._default_client
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._initialize()

    def start(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE retrieval_eval_jobs
                SET status='queued', stage='queued', label='等待继续评估',
                    error_message=NULL, updated_at=?
                WHERE status='running'
                """,
                (_utc_now(),),
            )
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="retrieval-eval-jobs", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=3)

    def create(
        self,
        *,
        author: str,
        labeler: str,
        split: str = "dev",
        budget_cny: float = 5.0,
        owner_id: str,
    ) -> dict[str, Any]:
        if labeler not in {"deepseek_api", "codex_handoff", "manual_import"}:
            raise ValueError("Unknown retrieval evaluation labeler")
        if split not in {"dev", "test"}:
            raise ValueError("split must be dev or test")
        if budget_cny <= 0:
            raise ValueError("budget_cny must be positive")
        index_dir = self._index_dir(author)
        parents_path = index_dir / "parents.jsonl"
        nodes_path = index_dir / "nodes.jsonl"
        if not parents_path.exists() or not nodes_path.exists() or not (index_dir / "qdrant").exists():
            raise ValueError("Author must finish crawl, build, and index before evaluation initialization")
        source_sha = hashlib.sha256(parents_path.read_bytes()).hexdigest()
        label_set_base = {
            "deepseek_api": "gold_aware_candidate_first_deepseek_v1",
            "codex_handoff": "codex_gold_aware_dual_axis_v1",
            "manual_import": "manual_gold_aware_dual_axis_v1",
        }[labeler]
        label_set = f"{label_set_base}_{split}"
        with self._connect() as connection:
            active = connection.execute(
                """
                SELECT * FROM retrieval_eval_jobs
                WHERE author=? AND source_parents_sha256=? AND labeler=? AND split=?
                  AND status IN ('queued','running','awaiting_codex','paused_budget')
                ORDER BY created_at DESC LIMIT 1
                """,
                (author, source_sha, labeler, split),
            ).fetchone()
            if active:
                return self._public(dict(active))
            job_id = f"reval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            dataset_id = f"{author}-temporal-{source_sha[:10]}"
            eval_dir = self.config.data_dir / "eval" / dataset_id
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO retrieval_eval_jobs (
                    id, author, owner_id, labeler, split, status, stage, label,
                    source_parents_sha256, dataset_id, eval_dir, budget_cny,
                    label_set, completed_items, total_items, estimated_cost_cny, usage_json,
                    error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', 'queued', '等待初始化评估',
                          ?, ?, ?, ?, ?, 0, 0, 0, '{}', NULL, ?, ?)
                """,
                (
                    job_id,
                    author,
                    owner_id,
                    labeler,
                    split,
                    source_sha,
                    dataset_id,
                    str(eval_dir),
                    budget_cny,
                    label_set,
                    now,
                    now,
                ),
            )
        self._wake.set()
        return self.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM retrieval_eval_jobs
                ORDER BY CASE WHEN status IN ('queued','running','awaiting_codex','paused_budget')
                              THEN 0 ELSE 1 END, updated_at DESC
                """
            ).fetchall()
        return [self._public(dict(row)) for row in rows]

    def get(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM retrieval_eval_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._public(dict(row))

    def resume(self, job_id: str, *, budget_cny: float | None = None) -> dict[str, Any]:
        raw = self._raw(job_id)
        if raw["status"] not in {"failed", "interrupted", "paused_budget"}:
            raise ValueError("Only failed, interrupted, or budget-paused jobs can resume")
        fields: dict[str, Any] = {
            "status": "queued",
            "stage": "queued",
            "label": "等待继续评估",
            "error_message": None,
            "completed_at": None,
        }
        if budget_cny is not None:
            if budget_cny <= float(raw["budget_cny"]):
                raise ValueError("New budget must be greater than the current budget")
            fields["budget_cny"] = budget_cny
        self._update(job_id, **fields)
        self._wake.set()
        return self.get(job_id)

    def handoff_zip(self, job_id: str) -> Path:
        job = self._raw(job_id)
        path = Path(str(job.get("handoff_zip_path") or ""))
        if job["labeler"] not in {"codex_handoff", "manual_import"} or not path.exists():
            raise KeyError("Codex handoff package is not ready")
        return path

    def import_codex_review(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        job = self._raw(job_id)
        if job["labeler"] not in {"codex_handoff", "manual_import"} or job["status"] != "awaiting_codex":
            raise ValueError("Job is not waiting for a Codex review")
        paths = self._paths(job)
        review_path = paths["eval_dir"] / "retrieval_jobs" / job_id / "codex_review.json"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(review_path, payload)
        result = materialize_codex_gold_labels(
            paths["full_pool_manifest"],
            review_path,
            dataset_path=paths["dataset"],
            gold_units_path=paths["gold_units"],
            label_set=str(job["label_set"]),
            labeler=str(job["labeler"]),
            splits=[str(job["split"])],
        )
        self._update(
            job_id,
            status="completed",
            stage="completed",
            label="Codex 双轴 Qrels 已发布",
            completed_items=int(result["manifest"]["completed"]),
            total_items=int(result["manifest"]["total"]),
            label_manifest_path=str(result["manifest_path"]),
            completed_at=_utc_now(),
            error_message=None,
        )
        return self.get(job_id)

    def run_once(self) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM retrieval_eval_jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return False
            now = _utc_now()
            connection.execute(
                """
                UPDATE retrieval_eval_jobs
                SET status='running', stage='preparing_dataset', label='正在准备时间切分',
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
        try:
            job = self._raw(job_id)
            paths = self._paths(job)
            paths["eval_dir"].mkdir(parents=True, exist_ok=True)

            self._stage(job_id, "preparing_dataset", "正在准备时间切分数据集")
            if not paths["dataset_manifest"].exists():
                dataset_options = self._dataset_options(
                    paths["index_dir"],
                    split=str(job["split"]),
                )
                prepare_temporal_dataset(
                    author=str(job["author"]),
                    index_dir=paths["index_dir"],
                    out_dir=paths["eval_dir"],
                    **dataset_options,
                )
            self._stage(job_id, "freezing_snapshot", "正在冻结语料快照")
            freeze_corpus_snapshot(index_dir=paths["index_dir"], dataset_path=paths["dataset"])

            self._stage(job_id, "building_candidate_pool", "正在构建六路候选池")
            if not paths["six_route_manifest"].exists():
                self._run_cli(
                    "eval",
                    "retrieval-pool",
                    str(job["author"]),
                    "--dataset",
                    str(paths["dataset"]),
                    "--dataset-id",
                    str(job["dataset_id"]),
                    "--index-dir",
                    str(paths["index_dir"]),
                    "--qdrant-path",
                    str(paths["index_dir"] / "qdrant"),
                    "--out-dir",
                    str(paths["six_route_dir"]),
                    "--split",
                    "all",
                    "--model-name",
                    self.config.model_name,
                    "--embedding-device",
                    self.config.embedding_device,
                    *([] if self.config.use_fp16 else ["--no-fp16"]),
                )

            self._stage(job_id, "freezing_exhaustive_pool", "正在冻结 cutoff 前完整材料池")
            if not paths["full_pool_manifest"].exists():
                build_exhaustive_retrieval_pool(
                    paths["six_route_manifest"],
                    dataset_path=paths["dataset"],
                    index_dir=paths["index_dir"],
                    out_dir=paths["full_pool_dir"],
                )

            self._stage(job_id, "extracting_gold_units", "正在提取 Gold 原子单元")
            if not self._gold_units_complete(paths["gold_units"], paths["dataset"], str(job["split"])):
                extract_gold_units(
                    paths["dataset"],
                    client=self.client_factory(),
                    out_path=paths["gold_units"],
                    splits=[str(job["split"])],
                )

            labeler = str(job["labeler"])
            if labeler in {"codex_handoff", "manual_import"}:
                self._stage(job_id, "exporting_codex", "正在生成双轴标注 handoff")
                handoff = export_codex_gold_handoff(
                    paths["full_pool_manifest"],
                    dataset_path=paths["dataset"],
                    gold_units_path=paths["gold_units"],
                    out_dir=paths["eval_dir"] / "retrieval_jobs" / job_id / "handoff",
                    label_set=str(job["label_set"]),
                    labeler=labeler,
                    splits=[str(job["split"])],
                )
                self._update(
                    job_id,
                    status="awaiting_codex",
                    stage="awaiting_codex",
                    label="等待导入 Codex 双轴标注" if labeler == "codex_handoff" else "等待导入人工双轴标注",
                    handoff_zip_path=str(handoff["zip_path"]),
                    total_items=int(handoff["manifest"]["candidate_count"]),
                )
                return

            self._stage(job_id, "labeling", "正在运行候选前缀 Gold-aware 标注")

            def report_label_progress(phase: str, current: int, total: int) -> None:
                pass_label = {
                    "pass-1": "正在进行首轮双轴判断",
                    "pass-2": "正在进行第 2 轮稳定性复核",
                    "pass-3": "正在进行冲突样本复核",
                }.get(phase, "正在运行候选前缀 Gold-aware 标注")
                runtime_manifest = (
                    paths["full_pool_dir"]
                    / "llm_labels"
                    / str(job["label_set"])
                    / "manifest.json"
                )
                fields: dict[str, Any] = {
                    "stage": f"labeling_{phase.replace('-', '_')}",
                    "label": pass_label,
                    "completed_items": current,
                    "total_items": total,
                }
                if runtime_manifest.exists():
                    try:
                        runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        runtime = {}
                    fields.update(
                        estimated_cost_cny=float(runtime.get("estimated_cost_cny") or 0.0),
                        usage_json=json.dumps(runtime.get("usage") or {}, ensure_ascii=False),
                    )
                self._update(job_id, **fields)

            result = label_gold_aware_pool(
                paths["full_pool_manifest"],
                dataset_path=paths["dataset"],
                gold_units_path=paths["gold_units"],
                client=self.client_factory(),
                label_set=str(job["label_set"]),
                batch_size=10,
                max_concurrency=4,
                stability_sample_rate=0.05,
                budget_cny=float(job["budget_cny"]),
                candidate_warmup_count=2,
                splits=[str(job["split"])],
                progress=report_label_progress,
            )
            manifest = result["manifest"]
            status = str(manifest["status"])
            public_status = "paused_budget" if status == "paused_budget" else ("completed" if status == "completed" else "failed")
            self._update(
                job_id,
                status=public_status,
                stage=public_status,
                label=(
                    "达到预算，已安全暂停"
                    if public_status == "paused_budget"
                    else ("检索评估已完成" if public_status == "completed" else "检索评估未完整")
                ),
                completed_items=int(manifest.get("completed") or 0),
                total_items=int(manifest.get("total") or 0),
                estimated_cost_cny=float(manifest.get("estimated_cost_cny") or 0),
                usage_json=json.dumps(manifest.get("usage") or {}, ensure_ascii=False),
                label_manifest_path=str(result["manifest_path"]),
                completed_at=_utc_now() if public_status == "completed" else None,
                error_message=None if public_status != "failed" else "Qrels 未完整，请查看标注 manifest",
            )
        except Exception as exc:  # noqa: BLE001 - persisted for repair/resume
            status = "interrupted" if self._stop.is_set() else "failed"
            self._update(
                job_id,
                status=status,
                stage=status,
                label="任务已中断" if status == "interrupted" else "评估初始化失败",
                error_message=str(exc),
                completed_at=None if status == "interrupted" else _utc_now(),
            )

    @staticmethod
    def _dataset_options(index_dir: Path, *, split: str) -> dict[str, Any]:
        """Select the standard or sparse-author protocol without treating articles as queries."""

        parents = load_jsonl(index_dir / "parents.jsonl")
        standard_answers = [
            row for row in parents if is_eligible_answer(row, min_answer_characters=200)
        ]
        all_answers = [
            row for row in parents if is_eligible_answer(row, min_answer_characters=0)
        ]
        if len(standard_answers) >= 30:
            return {}
        if split != "test":
            raise ValueError(
                "This author has fewer than 30 long answers; create a Test-only evaluation task."
            )
        if not all_answers:
            raise ValueError(
                "This author has no timestamped answers; the answer-based retrieval benchmark cannot be built."
            )
        return {
            "dev_size": 0,
            "test_size": len(all_answers),
            "test_only": True,
        }

    def _paths(self, job: dict[str, Any]) -> dict[str, Path]:
        eval_dir = Path(str(job["eval_dir"]))
        index_dir = self._index_dir(str(job["author"]))
        six_route_dir = eval_dir / "retrieval_pool" / "all30_six_route_v1"
        full_pool_dir = eval_dir / "retrieval_pool" / "all30_exhaustive_qrels_v2"
        return {
            "eval_dir": eval_dir,
            "index_dir": index_dir,
            "dataset": eval_dir / "dataset.jsonl",
            "dataset_manifest": eval_dir / "dataset_manifest.json",
            "six_route_dir": six_route_dir,
            "six_route_manifest": six_route_dir / "manifest.json",
            "full_pool_dir": full_pool_dir,
            "full_pool_manifest": full_pool_dir / "manifest.json",
            "gold_units": eval_dir / f"gold_units_{job['split']}_v2.jsonl",
        }

    def _run_cli(self, *args: str) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.config.working_dir / "src")
        completed = subprocess.run(
            [sys.executable, "-m", "personaforge.cli", *args],
            cwd=self.config.working_dir,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "Evaluation subprocess failed").strip()
            raise RuntimeError(message[-4000:])

    def _stage(self, job_id: str, stage: str, label: str) -> None:
        if self._stop.is_set():
            raise RuntimeError("服务停止，任务将在下次启动后继续")
        self._update(job_id, stage=stage, label=label)

    def _index_dir(self, author: str) -> Path:
        return self.config.data_dir / "authors" / "zhihu" / author / "index"

    @staticmethod
    def _gold_units_complete(gold_units_path: Path, dataset_path: Path, split: str) -> bool:
        """Reject stale partial Gold-unit caches before pool/label alignment."""
        if not gold_units_path.exists() or not dataset_path.exists():
            return False
        try:
            expected = {
                str(row.get("item_id") or "")
                for row in load_jsonl(dataset_path)
                if str(row.get("split") or split) == split
            }
            actual_rows = load_jsonl(gold_units_path)
            actual = {str(row.get("item_id") or "") for row in actual_rows}
            if not expected or actual != expected:
                return False
            required = {"stance", "reasoning", "example", "expression"}
            return all(
                required.issubset(set((row.get("units") or {}).keys()))
                and all((row.get("units") or {}).get(category) for category in required)
                for row in actual_rows
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def _raw(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM retrieval_eval_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def _public(self, row: dict[str, Any]) -> dict[str, Any]:
        try:
            usage = json.loads(str(row.get("usage_json") or "{}"))
        except json.JSONDecodeError:
            usage = {}
        return {
            key: row.get(key)
            for key in (
                "id", "author", "owner_id", "labeler", "split", "status", "stage", "label",
                "dataset_id", "label_set", "budget_cny", "completed_items", "total_items", "estimated_cost_cny",
                "error_message", "created_at", "updated_at", "started_at", "completed_at",
            )
        } | {"usage": usage, "handoff_ready": bool(row.get("handoff_zip_path"))}

    def _update(self, job_id: str, **fields: Any) -> None:
        allowed = {
            "status",
            "stage",
            "label",
            "budget_cny",
            "completed_items",
            "total_items",
            "estimated_cost_cny",
            "usage_json",
            "handoff_zip_path",
            "label_manifest_path",
            "error_message",
            "started_at",
            "completed_at",
        }
        unexpected = set(fields) - allowed
        if unexpected:
            raise ValueError(f"Unsupported retrieval evaluation fields: {sorted(unexpected)}")
        fields["updated_at"] = _utc_now()
        assignments = ", ".join(f"{key}=?" for key in fields)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE retrieval_eval_jobs SET {assignments} WHERE id=?",
                (*fields.values(), job_id),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS retrieval_eval_jobs (
                    id TEXT PRIMARY KEY,
                    author TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    labeler TEXT NOT NULL,
                    split TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    label TEXT NOT NULL,
                    source_parents_sha256 TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    eval_dir TEXT NOT NULL,
                    label_set TEXT,
                    budget_cny REAL NOT NULL,
                    completed_items INTEGER NOT NULL DEFAULT 0,
                    total_items INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_cny REAL NOT NULL DEFAULT 0,
                    usage_json TEXT NOT NULL DEFAULT '{}',
                    handoff_zip_path TEXT,
                    label_manifest_path TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_retrieval_eval_jobs_author
                ON retrieval_eval_jobs(author, created_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @staticmethod
    def _default_client() -> DeepSeekJsonClient:
        client = DeepSeekJsonClient.from_env()
        client.thinking = "disabled"
        client.timeout_seconds = 180
        return client


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
