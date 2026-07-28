from __future__ import annotations

from pathlib import Path

import pytest

from personaforge import cli


def test_forge_runs_existing_stages_in_order(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    def stage(name: str):
        def run(args):
            calls.append((name, args))
            return 0

        return run

    monkeypatch.setattr(cli, "_run_crawl", stage("crawl"))
    monkeypatch.setattr(cli, "_run_build", stage("build"))
    monkeypatch.setattr(cli, "_run_index", stage("index"))
    monkeypatch.setattr(cli, "_run_web", stage("web"))

    code = cli.main(
        [
            "forge",
            "zhihu",
            "alice",
            "--data-dir",
            str(tmp_path / "data"),
            "--embedding-device",
            "cpu",
            "--host",
            "0.0.0.0",
            "--port",
            "8012",
        ]
    )

    assert code == 0
    assert [name for name, _ in calls] == ["crawl", "build", "index", "web"]
    crawl_args = calls[0][1]
    build_args = calls[1][1]
    index_args = calls[2][1]
    web_args = calls[3][1]
    author_dir = tmp_path / "data" / "authors" / "zhihu" / "alice"
    assert Path(crawl_args.out_dir) == author_dir / "raw"
    assert crawl_args.all is True
    assert Path(build_args.raw_dir) == author_dir / "raw"
    assert Path(build_args.index_dir) == author_dir / "index"
    assert Path(index_args.qdrant_path) == author_dir / "index" / "qdrant"
    assert index_args.embedding_device == "cpu"
    assert web_args.host == "0.0.0.0"
    assert web_args.port == 8012


def test_forge_stops_after_failed_stage(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def crawl(_args):
        calls.append("crawl")
        return 2

    monkeypatch.setattr(cli, "_run_crawl", crawl)
    monkeypatch.setattr(
        cli,
        "_run_build",
        lambda _args: calls.append("build") or 0,
    )

    code = cli.main(
        [
            "forge",
            "zhihu",
            "alice",
            "--data-dir",
            str(tmp_path / "data"),
            "--no-web",
        ]
    )

    assert code == 2
    assert calls == ["crawl"]


def test_forge_can_reuse_all_existing_artifacts(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    author_dir = data_dir / "authors" / "zhihu" / "alice"
    (author_dir / "raw").mkdir(parents=True)
    index_dir = author_dir / "index"
    (index_dir / "qdrant").mkdir(parents=True)
    (index_dir / "parents.jsonl").write_text("{}\n", encoding="utf-8")
    (index_dir / "nodes.jsonl").write_text("{}\n", encoding="utf-8")
    (index_dir / "qdrant_manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "_run_crawl",
        lambda _args: pytest.fail("crawl should be skipped"),
    )
    monkeypatch.setattr(
        cli,
        "_run_build",
        lambda _args: pytest.fail("build should be skipped"),
    )
    monkeypatch.setattr(
        cli,
        "_run_index",
        lambda _args: pytest.fail("index should be skipped"),
    )

    code = cli.main(
        [
            "forge",
            "zhihu",
            "alice",
            "--data-dir",
            str(data_dir),
            "--skip-crawl",
            "--skip-build",
            "--skip-index",
            "--no-web",
        ]
    )

    assert code == 0


def test_forge_rejects_skip_when_artifact_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--skip-crawl"):
        cli.main(
            [
                "forge",
                "zhihu",
                "alice",
                "--data-dir",
                str(tmp_path / "data"),
                "--skip-crawl",
                "--no-web",
            ]
        )


def test_web_cli_passes_bind_host_to_config(monkeypatch) -> None:
    captured = {}

    def fake_run_web(config) -> None:
        captured["config"] = config

    monkeypatch.setattr("personaforge.web.app.run_web", fake_run_web)

    code = cli.main(["web", "alice", "--host", "0.0.0.0", "--port", "8020"])

    assert code == 0
    assert captured["config"].host == "0.0.0.0"
    assert captured["config"].port == 8020
