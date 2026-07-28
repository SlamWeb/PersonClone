"""Replay only the writer stage from a completed evaluation run."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from personaforge.eval.dataset import load_dataset_manifest, sha256_json
from personaforge.eval.runner import (
    assert_persona_pack_no_leak,
    git_revision,
    utc_now,
    write_item_markdown,
    write_json,
)
from personaforge.ingest.retrieve import ChildHit, ParentHit, load_parents
from personaforge.llm import JsonChatClient
from personaforge.persona.pack import load_persona_pack
from personaforge.persona.writer import generate_answer


WRITER_REPLAY_SCHEMA_VERSION = "personaforge.eval.writer-replay.v0"


@dataclass(frozen=True, slots=True)
class WriterReplayConfig:
    source_runs_path: Path
    parent_store_path: Path
    persona_pack_path: Path
    run_name: str
    out_dir: Path
    writer_prompt: str = "persona_pack"
    temperature: float = 0.85
    max_tokens: int = 1600
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class WriterReplayResult:
    run_dir: Path
    manifest_path: Path
    runs_path: Path
    summary_path: Path
    item_count: int


def run_writer_replay(
    config: WriterReplayConfig,
    *,
    llm: JsonChatClient,
) -> WriterReplayResult:
    """Generate new answers while preserving each source item's writer inputs."""

    source_records = _load_jsonl(config.source_runs_path)
    if config.limit is not None:
        source_records = source_records[: config.limit]
    if not source_records:
        raise ValueError(f"No source records found in {config.source_runs_path}.")

    source_manifest_path = config.source_runs_path.parent / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "completed":
        raise ValueError(f"Source run is not completed: {source_manifest_path}")

    dataset_path = _resolve_repo_path(
        str(source_manifest["dataset_path"]),
        anchor=config.source_runs_path,
    )
    dataset_manifest = load_dataset_manifest(dataset_path)
    _assert_matching_dataset(source_manifest, dataset_manifest)
    excluded_parent_ids = {
        str(value) for value in dataset_manifest.get("excluded_parent_ids", [])
    }

    persona_pack = load_persona_pack(
        config.persona_pack_path,
        parent_store_path=config.parent_store_path,
        verify_evidence=True,
    )
    assert_persona_pack_no_leak(persona_pack, excluded_parent_ids)
    parents_by_id = load_parents(config.parent_store_path)

    run_dir = config.out_dir / "runs" / config.run_name
    if run_dir.exists():
        raise FileExistsError(f"Writer replay already exists: {run_dir}")
    items_dir = run_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=False)
    runs_path = run_dir / "runs.jsonl"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.md"
    shutil.copyfile(config.persona_pack_path, run_dir / "frozen_persona_pack.json")

    manifest: dict[str, Any] = {
        "schema_version": WRITER_REPLAY_SCHEMA_VERSION,
        "status": "running",
        "started_at": utc_now(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_manifest.get("dataset_sha256"),
        "excluded_parent_ids_sha256": dataset_manifest.get(
            "excluded_parent_ids_sha256"
        ),
        "excluded_parent_count": len(excluded_parent_ids),
        "config": _config_to_dict(config),
        "git": git_revision(),
        "writer_model": str(getattr(llm, "model", type(llm).__name__)),
        "persona_pack": {
            "pack_id": persona_pack.pack_id,
            "sha256": persona_pack.sha256,
            "claim_count": persona_pack.claim_count,
        },
        "replay_source": {
            "runs_path": str(config.source_runs_path),
            "runs_sha256": _sha256_file(config.source_runs_path),
            "manifest_path": str(source_manifest_path),
            "run_sha256": source_manifest.get("run_sha256"),
            "writer_prompt": source_manifest.get("config", {}).get("writer_prompt"),
            "temperature": source_manifest.get("config", {}).get("temperature"),
            "max_tokens": source_manifest.get("config", {}).get("max_tokens"),
        },
        "frozen_inputs": [
            "query",
            "objective_background",
            "retrieval.parents",
            "retrieval parent order",
        ],
        "skipped_stages": [
            "query_understanding",
            "query_transform",
            "embedding",
            "qdrant_retrieval",
        ],
    }
    write_json(manifest, manifest_path)

    records: list[dict[str, Any]] = []
    try:
        for source_record in source_records:
            record = replay_writer_item(
                source_record,
                config=config,
                parents_by_id=parents_by_id,
                excluded_parent_ids=excluded_parent_ids,
                persona_pack=persona_pack,
                llm=llm,
            )
            records.append(record)
            _append_jsonl(record, runs_path)
            write_item_markdown(
                record,
                items_dir / f"{source_record['item_id']}.md",
            )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
            }
        )
        write_json(manifest, manifest_path)
        raise

    manifest.update(
        {
            "status": "completed",
            "finished_at": utc_now(),
            "item_count": len(records),
            "run_sha256": sha256_json(records),
        }
    )
    write_json(manifest, manifest_path)
    _write_summary(config, records, summary_path)
    return WriterReplayResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        runs_path=runs_path,
        summary_path=summary_path,
        item_count=len(records),
    )


def replay_writer_item(
    source_record: dict[str, Any],
    *,
    config: WriterReplayConfig,
    parents_by_id: dict[str, dict[str, Any]],
    excluded_parent_ids: set[str],
    persona_pack: Any,
    llm: JsonChatClient,
) -> dict[str, Any]:
    """Replay a single writer call from its frozen trace."""

    source_trace = source_record["trace"]
    retrieval_trace = source_trace["retrieval"]
    parent_hits = rebuild_parent_hits(
        retrieval_trace["parents"],
        parents_by_id=parents_by_id,
    )
    leaked = {
        hit.parent_id for hit in parent_hits if hit.parent_id in excluded_parent_ids
    }
    if leaked:
        raise RuntimeError(
            f"Writer replay contains excluded parent(s): {sorted(leaked)}"
        )

    generation_started_at = perf_counter()
    answer_result = generate_answer(
        query=str(source_record["query"]),
        parent_hits=parent_hits,
        llm=llm,
        objective_background=str(source_trace.get("objective_background") or ""),
        writer_prompt=config.writer_prompt,
        persona_pack=persona_pack,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    generation_ms = round((perf_counter() - generation_started_at) * 1000)
    expected_titles = [hit.title for hit in parent_hits]
    if answer_result.parent_titles != expected_titles:
        raise RuntimeError("Writer replay changed parent order before prompt construction.")

    return {
        "item_id": source_record["item_id"],
        "split": source_record["split"],
        "parent_id": source_record["parent_id"],
        "created_at": source_record["created_at"],
        "query": source_record["query"],
        "gold_answer": source_record["gold_answer"],
        "answer": answer_result.answer,
        "status": "completed",
        "trace": {
            "query_understanding": source_trace.get("query_understanding"),
            "objective_background": source_trace.get("objective_background", ""),
            "retrieval": retrieval_trace,
            "writer": {
                "variant": answer_result.writer_prompt,
                "persona_pack_id": answer_result.persona_pack_id,
                "persona_pack_sha256": answer_result.persona_pack_sha256,
                "context_parent_titles": answer_result.parent_titles,
                "message_characters": [
                    {
                        "role": message["role"],
                        "characters": len(message["content"]),
                    }
                    for message in answer_result.messages
                ],
                "replayed_from_item_id": source_record["item_id"],
            },
            "timing": {
                "query_understanding_ms": 0,
                "retrieval_ms": 0,
                "generation_ms": generation_ms,
                "total_ms": generation_ms,
            },
            "replay": {
                "source_item_id": source_record["item_id"],
                "parent_count": len(parent_hits),
                "frozen_objective_background": True,
                "frozen_parent_order": True,
            },
        },
    }


def rebuild_parent_hits(
    serialized_hits: list[dict[str, Any]],
    *,
    parents_by_id: dict[str, dict[str, Any]],
) -> list[ParentHit]:
    """Rehydrate ranked parent hits without querying an embedding model or Qdrant."""

    parent_hits: list[ParentHit] = []
    seen: set[str] = set()
    for position, value in enumerate(serialized_hits, start=1):
        parent_id = str(value["parent_id"])
        if parent_id in seen:
            raise ValueError(f"Duplicate parent in replay trace: {parent_id}")
        seen.add(parent_id)
        parent = parents_by_id.get(parent_id)
        if parent is None:
            raise ValueError(f"Replay parent is missing from local store: {parent_id}")
        rank = int(value["rank"])
        if rank != position:
            raise ValueError(
                f"Replay parent rank/order mismatch for {parent_id}: "
                f"rank={rank}, position={position}"
            )
        parent_hits.append(
            ParentHit(
                rank=rank,
                parent_id=parent_id,
                score=float(value.get("score", 0)),
                title=str(value.get("title") or parent.get("title") or ""),
                path=str(value.get("path") or parent.get("path") or ""),
                first_hits=[
                    ChildHit(
                        rank=int(child["rank"]),
                        score=float(child.get("score", 0)),
                        node_id=str(child["node_id"]),
                        parent_id=str(child["parent_id"]),
                        node_type=str(child.get("node_type") or ""),
                        title=str(child.get("title") or ""),
                        path=str(child.get("path") or ""),
                        route=str(child.get("route") or ""),
                    )
                    for child in value.get("first_hits", [])
                ],
                parent=parent,
            )
        )
    return parent_hits


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_repo_path(value: str, *, anchor: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for ancestor in (anchor.resolve(), *anchor.resolve().parents):
        candidate = ancestor / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve replay dataset path: {value}")


def _assert_matching_dataset(
    source_manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
) -> None:
    for key in ("dataset_sha256", "excluded_parent_ids_sha256"):
        if source_manifest.get(key) != dataset_manifest.get(key):
            raise ValueError(
                f"Writer replay dataset mismatch for {key}: "
                f"{source_manifest.get(key)!r} != {dataset_manifest.get(key)!r}"
            )


def _config_to_dict(config: WriterReplayConfig) -> dict[str, Any]:
    value = asdict(config)
    for key in ("source_runs_path", "parent_store_path", "persona_pack_path", "out_dir"):
        value[key] = str(value[key])
    return value


def _append_jsonl(record: dict[str, Any], path: Path) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_summary(
    config: WriterReplayConfig,
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    average_ms = round(
        sum(record["trace"]["timing"]["generation_ms"] for record in records)
        / len(records)
    )
    content = f"""# {config.run_name}

- 题目数：{len(records)}
- writer：{config.writer_prompt}
- 平均 Writer 耗时：{average_ms} ms
- 来源 run：{config.source_runs_path}

本轮只重放 Writer。每题的原问题、客观背景、20 篇 parent 全文及顺序均来自来源
run；没有重新调用 query understanding、query transform、embedding 或 Qdrant。
"""
    path.write_text(content, encoding="utf-8", newline="\n")
