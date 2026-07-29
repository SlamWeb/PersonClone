from __future__ import annotations

import json
from pathlib import Path

from personaforge.web.author_jobs import (
    AuthorJobConfig,
    AuthorJobManager,
    AuthorJobStore,
    command_stage,
    persona_is_ready,
    resolve_author_preview,
    safe_author_token,
)


def test_author_job_store_persists_and_cancels_queued_job(tmp_path: Path) -> None:
    store = AuthorJobStore(tmp_path / "system" / "personaforge.sqlite3")
    job = store.create(author_input="alice", author="alice", operation="create")

    reloaded = AuthorJobStore(store.path).get(job.id)
    cancelled = store.request_cancel(job.id)

    assert reloaded.author == "alice"
    assert cancelled.status == "cancelled"
    assert store.find_active("alice") is None


def test_resolve_author_preview_reuses_local_profile(tmp_path: Path) -> None:
    author_dir = tmp_path / "authors" / "zhihu" / "alice"
    raw_dir = author_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "profile.json").write_text(
        json.dumps(
            {
                "nickname": "Alice",
                "profile_url": "https://www.zhihu.com/people/alice",
                "headline": "简介",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    preview = resolve_author_preview(tmp_path, "https://www.zhihu.com/people/alice")

    assert preview["author"] == "alice"
    assert preview["display_name"] == "Alice"
    assert preview["exists"] is True
    assert preview["ready"] is False


def test_author_job_manager_runs_existing_pipeline_contract(tmp_path: Path) -> None:
    commands: list[str] = []

    def fake_runner(command: list[str], _cwd: Path, _log: Path, should_stop) -> None:
        assert should_stop() is False
        stage = command_stage(command)
        commands.append(stage)
        if stage == "crawl":
            raw_dir = Path(argument_after(command, "--out-dir"))
            answer_dir = raw_dir / "answer"
            answer_dir.mkdir(parents=True)
            (raw_dir / "profile.json").write_text(
                json.dumps(
                    {
                        "nickname": "Alice",
                        "profile_url": "https://www.zhihu.com/people/alice",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (answer_dir / "answer-1.md").write_text("# 问题\n\n回答", encoding="utf-8")
            (raw_dir / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "source": "zhihu",
                        "kind": "answer",
                        "id": "1",
                        "path": "answer/answer-1.md",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        elif stage == "build":
            index_dir = Path(argument_after(command, "--index-dir"))
            index_dir.mkdir(parents=True)
            (index_dir / "parents.jsonl").write_text("{}\n", encoding="utf-8")
            (index_dir / "nodes.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            (index_dir / "build_manifest.json").write_text(
                json.dumps({"parent_count": 1, "node_count": 2}),
                encoding="utf-8",
            )
        elif stage == "index":
            index_dir = Path(argument_after(command, "--index-dir"))
            (index_dir / "qdrant").mkdir(parents=True)
            (index_dir / "qdrant_manifest.json").write_text(
                json.dumps({"indexed_at": "2026-01-01T00:00:00+00:00"}),
                encoding="utf-8",
            )

    manager = AuthorJobManager(
        AuthorJobConfig(data_dir=tmp_path, model_name="local-bge", embedding_device="cpu"),
        command_runner=fake_runner,
    )
    created = manager.create_job(author_input="https://www.zhihu.com/people/alice")

    assert manager.run_once() is True

    completed = manager.store.get(created.id)
    assert commands == ["crawl", "build", "index"]
    assert completed.status == "ready"
    assert completed.item_count == 1
    assert completed.parent_count == 1
    assert completed.node_count == 2
    assert persona_is_ready(tmp_path, "alice") is True
    assert not (tmp_path / "authors" / "zhihu" / "alice" / "staging" / created.id).exists()


def test_safe_author_token_accepts_profile_url() -> None:
    assert safe_author_token("https://www.zhihu.com/people/wu-ren-jun-28") == "wu-ren-jun-28"


def argument_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]
