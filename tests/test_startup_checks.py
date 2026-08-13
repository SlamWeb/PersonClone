from __future__ import annotations

from personaforge.web.startup_checks import format_startup_report, run_startup_checks


def test_startup_checks_are_ready_with_key_and_local_model(tmp_path) -> None:
    model_dir = tmp_path / "bge-m3"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    report = run_startup_checks(
        data_dir=tmp_path / "data",
        model_name=str(model_dir),
        env={"DEEPSEEK_API_KEY": "test-only", "TAVILY_API_KEY": "test-only"},
    )

    assert report["status"] == "ready"
    assert all(item["status"] == "ready" for item in report["checks"])


def test_remote_model_without_cache_is_an_actionable_warning(tmp_path) -> None:
    report = run_startup_checks(
        data_dir=tmp_path / "data",
        model_name="BAAI/bge-m3",
        env={"DEEPSEEK_API_KEY": "test-only", "HF_HOME": str(tmp_path / "empty-cache")},
    )

    assert report["status"] == "warning"
    model_check = next(item for item in report["checks"] if item["check_id"] == "embedding_model")
    assert "first indexing or retrieval operation will download it" in model_check["message"]
    assert "[WARN] BGE-M3 model" in format_startup_report(report)
