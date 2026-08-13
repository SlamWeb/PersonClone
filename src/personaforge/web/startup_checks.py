"""Non-destructive startup checks for local and Docker Web deployments."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from personaforge.env import load_env_file


@dataclass(frozen=True, slots=True)
class StartupCheck:
    check_id: str
    status: str
    title: str
    message: str
    action: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def run_startup_checks(
    *,
    data_dir: Path,
    model_name: str,
    env: Mapping[str, str] | None = None,
    env_file: Path = Path(".env"),
) -> dict[str, object]:
    """Inspect configuration without loading BGE-M3 or calling an external API."""

    if env is None:
        load_env_file(env_file)
        env = os.environ

    checks = [
        _check_llm_key(env),
        _check_embedding_model(model_name, env),
        _check_data_directory(data_dir),
        _check_tavily_key(env),
    ]
    statuses = {item.status for item in checks}
    status = "error" if "error" in statuses else "warning" if "warning" in statuses else "ready"
    return {"status": status, "checks": [item.as_dict() for item in checks]}


def format_startup_report(report: Mapping[str, object]) -> str:
    """Render a short terminal report without exposing secret values."""

    icons = {"ready": "OK", "warning": "WARN", "error": "ERROR"}
    lines = ["[PersonaForge startup check]"]
    for raw_item in report.get("checks", []):
        if not isinstance(raw_item, Mapping):
            continue
        status = str(raw_item.get("status", "warning"))
        lines.append(
            f"  [{icons.get(status, status.upper())}] "
            f"{raw_item.get('title', '')}: {raw_item.get('message', '')}"
        )
        action = raw_item.get("action")
        if action:
            lines.append(f"          Fix: {action}")
    if report.get("status") != "ready":
        lines.append("  Web UI will still start; unavailable capabilities will fail with the message above.")
    return "\n".join(lines)


def _check_llm_key(env: Mapping[str, str]) -> StartupCheck:
    if env.get("DEEPSEEK_API_KEY", "").strip():
        return StartupCheck("llm_api_key", "ready", "DeepSeek API key", "DEEPSEEK_API_KEY is configured.")
    return StartupCheck(
        "llm_api_key",
        "error",
        "DeepSeek API key",
        "DEEPSEEK_API_KEY is missing; Chat, query understanding, and LLM Judge are unavailable.",
        "Copy .env.example to .env and set DEEPSEEK_API_KEY, then restart the service.",
    )


def _check_embedding_model(model_name: str, env: Mapping[str, str]) -> StartupCheck:
    model_name = model_name.strip()
    if _looks_like_local_path(model_name):
        path = Path(model_name).expanduser()
        if path.is_dir() and any(path.iterdir()):
            return StartupCheck(
                "embedding_model", "ready", "BGE-M3 model", f"Local model directory is available: {path}"
            )
        return StartupCheck(
            "embedding_model",
            "error",
            "BGE-M3 model",
            f"Configured local model directory is missing or empty: {path}",
            "Download BAAI/bge-m3 there or use --model-name BAAI/bge-m3 for automatic download.",
        )

    if _huggingface_cache_has_model(model_name, env):
        return StartupCheck(
            "embedding_model",
            "ready",
            "BGE-M3 model",
            f"A local Hugging Face cache entry was found for {model_name}.",
        )
    return StartupCheck(
        "embedding_model",
        "warning",
        "BGE-M3 model",
        f"No local cache was found for {model_name}; the first indexing or retrieval operation will download it.",
        "Keep enough disk space and network access, or mount a pre-downloaded model and pass --model-name <path>.",
    )


def _check_data_directory(data_dir: Path) -> StartupCheck:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".personaforge-write-check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return StartupCheck(
            "data_directory",
            "error",
            "Data directory",
            f"Data directory is not writable: {data_dir} ({exc})",
            "Grant write permission or mount a writable directory at the configured data path.",
        )
    return StartupCheck("data_directory", "ready", "Data directory", f"Writable data directory: {data_dir}")


def _check_tavily_key(env: Mapping[str, str]) -> StartupCheck:
    if env.get("TAVILY_API_KEY", "").strip():
        return StartupCheck("tavily_api_key", "ready", "Tavily API key", "TAVILY_API_KEY is configured.")
    return StartupCheck(
        "tavily_api_key",
        "warning",
        "Tavily API key",
        "TAVILY_API_KEY is missing; web background search is disabled, while local RAG remains available.",
        "Set TAVILY_API_KEY in .env only when web-grounded query understanding is needed.",
    )


def _looks_like_local_path(value: str) -> bool:
    return bool(value.startswith((".", "~", "/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value))


def _huggingface_cache_has_model(model_name: str, env: Mapping[str, str]) -> bool:
    cache_root = env.get("HF_HOME", "").strip()
    hub_root = Path(cache_root).expanduser() / "hub" if cache_root else Path.home() / ".cache" / "huggingface" / "hub"
    model_root = hub_root / ("models--" + model_name.replace("/", "--"))
    return model_root.is_dir() and any(model_root.iterdir())
