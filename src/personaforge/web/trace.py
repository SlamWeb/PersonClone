"""Local, structured trace storage for PersonaForge Web runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any
from uuid import uuid4


TRACE_SCHEMA_VERSION = "personaforge.web.trace.v2"
DEFAULT_TRACE_RETENTION = 200


def new_trace_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"trace-{stamp}-{uuid4().hex[:8]}"


def trace_directory(data_dir: Path, author: str) -> Path:
    return data_dir / "authors" / "zhihu" / _safe_segment(author) / "traces"


def trace_path(data_dir: Path, author: str, trace_id: str) -> Path:
    safe_id = _safe_segment(trace_id)
    if not safe_id or safe_id != trace_id:
        raise FileNotFoundError(f"Trace not found: {trace_id}")
    return trace_directory(data_dir, author) / f"{safe_id}.json"


def write_trace(
    data_dir: Path,
    author: str,
    trace_id: str,
    payload: dict[str, Any],
    *,
    retention: int = DEFAULT_TRACE_RETENTION,
) -> Path:
    path = trace_path(data_dir, author, trace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    temporary.replace(path)
    prune_traces(data_dir, author, keep=retention, protected_trace_id=trace_id)
    return path


def read_trace(data_dir: Path, author: str, trace_id: str) -> dict[str, Any]:
    path = trace_path(data_dir, author, trace_id)
    if not path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid trace payload: {path}")
    return payload


def new_stage(
    *,
    stage_id: str,
    label: str,
    started_at: float,
    duration_ms: int,
    status: str = "completed",
    details: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one displayable, non-CoT trace stage."""

    payload: dict[str, Any] = {
        "id": stage_id,
        "label": label,
        "status": status,
        "started_offset_ms": round(started_at * 1000),
        "duration_ms": duration_ms,
    }
    if details:
        payload["details"] = details
    if usage:
        payload["usage"] = usage
    return payload


def estimated_usage_for_text(*texts: str) -> dict[str, Any]:
    """Give an intentionally labelled fallback when a provider omits usage."""

    characters = sum(len(text) for text in texts)
    cjk = sum(1 for text in texts for char in text if "\u4e00" <= char <= "\u9fff")
    other = max(0, characters - cjk)
    # CJK tokenizers often split near character granularity; for other text,
    # four characters is a conservative, explainable estimate.
    estimated = max(1, ceil(cjk * 0.8 + other / 4)) if characters else 0
    return {
        "source": "estimated",
        "estimated_tokens": estimated,
        "characters": characters,
        "note": "接口未返回 usage，按中文约 0.8 token/字、其他文本约 1 token/4 字估算。",
    }


def provider_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "as_dict"):
        values = usage.as_dict()
    elif isinstance(usage, dict):
        values = usage
    else:
        return None
    if not any(value is not None for value in values.values()):
        return None
    return {"source": "provider", **values}


def prune_traces(data_dir: Path, author: str, *, keep: int, protected_trace_id: str | None = None) -> None:
    """Keep normal Web traces bounded without touching eval artifacts elsewhere."""

    if keep < 1:
        return
    directory = trace_directory(data_dir, author)
    if not directory.exists():
        return
    paths = sorted(directory.glob("trace-*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    retained = 0
    for path in paths:
        if path.stem == protected_trace_id or retained < keep:
            retained += 1
            continue
        try:
            path.unlink()
        except OSError:
            continue


def _safe_segment(value: str) -> str:
    return "".join(character for character in value if character.isalnum() or character in {"-", "_"})
