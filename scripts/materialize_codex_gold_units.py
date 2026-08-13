"""Materialize local Codex Gold-unit notes into the project v2 schema.

This script deliberately has no LLM or network dependency.  The input is the
JSONL emitted by a local Codex handoff; the script only normalizes it, adds
stable IDs and hashes, and writes the audit manifest used by retrieval qrels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "personaforge.eval.retrieval_gold_units.v2"
PROMPT_VERSION = "retrieval-gold-units-v2.0"
CATEGORIES = ("stance", "reasoning", "example", "expression")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset = _read_jsonl(args.dataset)
    raw_rows = _read_jsonl(args.raw)
    expected = {str(row.get("item_id") or ""): row for row in dataset}
    actual = {str(row.get("item_id") or ""): row for row in raw_rows}
    if set(expected) != set(actual) or "" in expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise SystemExit(f"Gold-unit coverage mismatch; missing={missing}, extra={extra}")

    records: list[dict[str, Any]] = []
    for item in dataset:
        item_id = str(item["item_id"])
        raw_units = actual[item_id].get("units")
        if not isinstance(raw_units, dict):
            raise SystemExit(f"units must be an object for {item_id}")
        normalized: dict[str, list[dict[str, str]]] = {}
        fallback = str(item.get("gold_answer") or "").strip()[:300]
        for category in CATEGORIES:
            values = raw_units.get(category)
            if not isinstance(values, list):
                values = []
            seen: set[str] = set()
            texts: list[str] = []
            for value in values:
                text = str(value or "").strip()[:300]
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
            if category in {"stance", "reasoning", "expression"} and not texts:
                if not fallback:
                    raise SystemExit(f"missing required Gold unit category {category} for {item_id}")
                texts = [fallback]
            normalized[category] = [
                {"id": f"{category}-{index}", "text": text}
                for index, text in enumerate(texts[:10], start=1)
            ]

        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "item_id": item_id,
                "split": str(item.get("split") or ""),
                "question": str(item.get("query") or ""),
                "gold_answer_sha256": _sha256_text(str(item.get("gold_answer") or "")),
                "units": normalized,
                "model": "codex-gpt-5.4-local",
                "created_at": _utc_now(),
            }
        )

    _write_jsonl(args.out, records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": _sha256_file(args.dataset),
        "raw_source": str(args.raw.resolve()),
        "raw_source_sha256": _sha256_file(args.raw),
        "gold_units_file": args.out.name,
        "gold_units_sha256": _sha256_file(args.out),
        "prompt_version": PROMPT_VERSION,
        "model": "codex-gpt-5.4-local",
        "count": len(records),
        "splits": sorted({str(row.get("split") or "") for row in records}),
        "updated_at": _utc_now(),
    }
    _write_json(args.out.with_name(args.out.stem + ".manifest.json"), manifest)
    print(json.dumps({"count": len(records), "path": str(args.out), "manifest": manifest}, ensure_ascii=False))
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
