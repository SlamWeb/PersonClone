"""Resume oversized Codex Gold handoffs by reviewing candidates in chunks.

This runner is intentionally separate from the paid project labeler. It uses
local Codex processes only, writes one file per chunk, and replaces an item
review only after every candidate in that item has been covered.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from run_codex_gold_handoff import (
    _load_valid_output,
    _prompt_for,
    _read_jsonl,
    _write_merged_review,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--codex-exe", default=r"C:\Program Files\nodejs\codex.cmd")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--run-id",
        default="v1",
        help="Versioned suffix for generated chunk request/review directories.",
    )
    parser.add_argument("--item", action="append", dest="items")
    args = parser.parse_args()
    if args.chunk_size < 1 or args.workers < 1:
        raise SystemExit("--chunk-size and --workers must be >= 1")

    handoff_dir = args.handoff_dir.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    requests = _read_jsonl(handoff_dir / "requests.jsonl")
    request_by_item = {str(row["item_id"]): row for row in requests}
    selected = sorted(set(args.items or request_by_item))
    unknown = set(selected).difference(request_by_item)
    if unknown:
        raise SystemExit(f"Unknown item IDs: {sorted(unknown)}")

    run_id = "".join(ch for ch in str(args.run_id) if ch.isalnum() or ch in "-_" )
    if not run_id:
        raise SystemExit("--run-id must contain at least one letter or digit")
    request_dir = handoff_dir / f"codex_chunk_requests_{run_id}"
    review_dir = handoff_dir / f"codex_chunk_reviews_{run_id}"
    request_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for item_id in selected:
        candidates = list(request_by_item[item_id].get("candidates") or [])
        for chunk_index, start in enumerate(range(0, len(candidates), args.chunk_size)):
            chunk_candidates = candidates[start : start + args.chunk_size]
            chunk_id = f"{item_id}.chunk-{chunk_index:03d}"
            request = dict(request_by_item[item_id])
            request["candidates"] = chunk_candidates
            request["chunk_id"] = chunk_id
            request_path = request_dir / f"{chunk_id}.json"
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            jobs.append(
                {
                    "item_id": item_id,
                    "chunk_id": chunk_id,
                    "request_path": request_path,
                    "review_path": review_dir / f"{chunk_id}.json",
                    "candidate_ids": {str(row["candidate_id"]) for row in chunk_candidates},
                }
            )

    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs) or 1)) as pool:
        futures = {
            pool.submit(
                _run_chunk,
                job=job,
                repo=repo,
                codex_exe=args.codex_exe,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            future.result()

    raw_dir = handoff_dir / "codex_item_reviews"
    for item_id in selected:
        item_jobs = [job for job in jobs if job["item_id"] == item_id]
        labels: list[dict[str, Any]] = []
        for job in item_jobs:
            payload = _load_valid_output(job["review_path"], item_id=item_id, expected_ids=job["candidate_ids"])
            if payload is None:
                raise SystemExit(f"invalid chunk output: {job['review_path']}")
            labels.extend(payload["labels"])
        expected = {str(row["candidate_id"]) for row in request_by_item[item_id].get("candidates") or []}
        actual = {str(row.get("candidate_id") or "") for row in labels}
        if actual != expected or len(labels) != len(expected):
            raise SystemExit(f"coverage mismatch for {item_id}: {len(actual)}/{len(expected)}")
        item_payload = {"item_id": item_id, "review_complete": True, "labels": labels}
        (raw_dir / f"{item_id}.json").write_text(
            json.dumps(item_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    merged = _write_merged_review(handoff_dir, request_by_item, raw_dir)
    print(json.dumps(merged, ensure_ascii=False))
    return 0


def _run_chunk(
    *,
    job: dict[str, Any],
    repo: Path,
    codex_exe: str,
    model: str,
    timeout_seconds: int,
) -> None:
    payload = _load_valid_output(job["review_path"], item_id=job["item_id"], expected_ids=job["candidate_ids"])
    if payload is not None:
        print(f"skip {job['chunk_id']}", flush=True)
        return
    started = time.perf_counter()
    print(f"review {job['chunk_id']} ({len(job['candidate_ids'])} candidates)", flush=True)
    result = subprocess.run(
        [
            codex_exe,
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "-C",
            str(repo),
            "-s",
            "read-only",
            "-m",
            model,
            "-o",
            str(job["review_path"]),
            "-",
        ],
        input=_prompt_for(job["item_id"], job["request_path"]),
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(repo),
        timeout=timeout_seconds,
        check=False,
    )
    payload = _load_valid_output(job["review_path"], item_id=job["item_id"], expected_ids=job["candidate_ids"])
    if payload is None:
        raise RuntimeError(f"invalid Codex chunk output for {job['chunk_id']} (exit={result.returncode})")
    print(f"completed {job['chunk_id']} in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
