"""Run a no-API, per-question Codex reviewer for a Gold-aware handoff.

The project API labeler is intentionally not used here.  Each question is
reviewed in its own Codex process, so a large handoff can resume by skipping
validated item outputs.  The merged review remains incomplete until every
question has a complete candidate-coverage response.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--codex-exe", default=r"C:\Program Files\nodejs\codex.cmd")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行启动的本地 Codex 进程数；每道题仍独立落盘，可断点恢复。",
    )
    parser.add_argument("--item", action="append", dest="items")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    handoff_dir = args.handoff_dir.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    requests_path = handoff_dir / "requests.jsonl"
    manifest_path = handoff_dir / "manifest.json"
    template_path = handoff_dir / "review_template.json"
    if not requests_path.exists() or not manifest_path.exists() or not template_path.exists():
        raise SystemExit("handoff-dir must contain requests.jsonl, manifest.json and review_template.json")

    requests = _read_jsonl(requests_path)
    request_by_item = {str(row.get("item_id") or ""): row for row in requests}
    selected = set(args.items or request_by_item)
    unknown = selected.difference(request_by_item)
    if unknown:
        raise SystemExit(f"Unknown item IDs: {sorted(unknown)}")

    raw_dir = handoff_dir / "codex_item_reviews"
    raw_dir.mkdir(parents=True, exist_ok=True)
    request_dir = handoff_dir / "codex_item_requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    for item_id, row in request_by_item.items():
        request_path = request_dir / f"{item_id}.json"
        if not request_path.exists():
            request_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    selected_items = sorted(selected)
    if args.workers == 1 or len(selected_items) == 1:
        for index, item_id in enumerate(selected_items, start=1):
            _review_one(
                index=index,
                total=len(selected_items),
                item_id=item_id,
                request_by_item=request_by_item,
                request_dir=request_dir,
                raw_dir=raw_dir,
                repo=repo,
                codex_exe=args.codex_exe,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
            )
            _write_merged_review(handoff_dir, request_by_item, raw_dir)
    else:
        # Each task writes a distinct item file. The merged review is written
        # only after all workers finish, so readers never see a half-written
        # aggregate during a parallel batch.
        with ThreadPoolExecutor(max_workers=min(args.workers, len(selected_items))) as pool:
            futures = {
                pool.submit(
                    _review_one,
                    index=index,
                    total=len(selected_items),
                    item_id=item_id,
                    request_by_item=request_by_item,
                    request_dir=request_dir,
                    raw_dir=raw_dir,
                    repo=repo,
                    codex_exe=args.codex_exe,
                    model=args.model,
                    timeout_seconds=args.timeout_seconds,
                ): item_id
                for index, item_id in enumerate(selected_items, start=1)
            }
            for future in as_completed(futures):
                future.result()
        _write_merged_review(handoff_dir, request_by_item, raw_dir)

    merged = _write_merged_review(handoff_dir, request_by_item, raw_dir)
    print(json.dumps(merged, ensure_ascii=False))
    return 0


def _review_one(
    *,
    index: int,
    total: int,
    item_id: str,
    request_by_item: dict[str, dict[str, Any]],
    request_dir: Path,
    raw_dir: Path,
    repo: Path,
    codex_exe: str,
    model: str,
    timeout_seconds: int,
) -> None:
    request = request_by_item[item_id]
    expected_ids = {str(candidate.get("candidate_id") or "") for candidate in request.get("candidates") or []}
    output_path = raw_dir / f"{item_id}.json"
    payload = _load_valid_output(output_path, item_id=item_id, expected_ids=expected_ids)
    if payload is not None:
        print(f"[{index}/{total}] skip completed {item_id}", flush=True)
        return

    prompt = _prompt_for(item_id, request_dir / f"{item_id}.json")
    started = time.perf_counter()
    print(f"[{index}/{total}] Codex reviewing {item_id} ({len(expected_ids)} candidates)", flush=True)
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
            str(output_path),
            "-",
        ],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(repo),
        timeout=timeout_seconds,
        check=False,
    )
    payload = _load_valid_output(output_path, item_id=item_id, expected_ids=expected_ids)
    if payload is None:
        raise RuntimeError(
            f"Codex output for {item_id} is invalid (exit={result.returncode}); "
            f"inspect {output_path} and rerun this item."
        )
    print(f"  completed {item_id} in {time.perf_counter() - started:.1f}s", flush=True)


def _prompt_for(item_id: str, request_path: Path) -> str:
    return f"""
你是离线检索评估标注员。不要调用任何项目 API、DeepSeek、Tavily、网络服务或外部工具；只读取本地文件并输出最终 JSON。
请读取文件：{request_path}
这个文件只有一道题，包含 question、gold_answer、gold_units 和 candidates。你要逐个判断全部 candidates，不能遗漏、不能修改 candidate_id。

对每个候选输出：
- content_support：0=不能帮助重建本题 Gold 的核心立场/机制/事实/例子；1=提供部分可迁移支撑；2=直接支撑 Gold 的核心内容。
- persona_expression_support：0=不能帮助重建本题 Gold 中实际出现的论证动作/语气/节奏/表达；1=有部分可迁移表达参考；2=清晰呈现与本题 Gold 对应的表达实现。
- confidence：low、medium 或 high。
- 非零内容轴必须提供 content_candidate_evidence（候选原文中的短片段）和 content_gold_unit_ids（只能使用输入 gold_units 中的 ID）。
- 非零表达轴必须提供 persona_candidate_evidence 和 persona_gold_unit_ids（只能使用输入 gold_units 中的 ID）。
- 任一轴非零时必须提供简短 reason。

严格只输出一个 JSON 对象，不要 Markdown 代码围栏、不要解释：
{{"item_id":"{item_id}","review_complete":true,"labels":[{{"candidate_id":"...","content_support":0,"persona_expression_support":0,"confidence":"medium","content_candidate_evidence":"","content_gold_unit_ids":[],"persona_candidate_evidence":"","persona_gold_unit_ids":[],"reason":""}}]}}
""".strip()


def _load_valid_output(path: Path, *, item_id: str, expected_ids: set[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("\n", 1)[0]
        payload = json.loads(raw)
        labels = payload.get("labels")
        actual_ids = {str(row.get("candidate_id") or "") for row in labels} if isinstance(labels, list) else set()
        if payload.get("item_id") != item_id or payload.get("review_complete") is not True or actual_ids != expected_ids:
            return None
        if any(not isinstance(row, dict) for row in labels):
            return None
        for row in labels:
            if row.get("content_support") not in {0, 1, 2} or row.get("persona_expression_support") not in {0, 1, 2}:
                return None
        return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _write_merged_review(
    handoff_dir: Path,
    request_by_item: dict[str, dict[str, Any]],
    raw_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads((handoff_dir / "manifest.json").read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    complete_count = 0
    for item_id in sorted(request_by_item):
        expected_ids = {str(candidate.get("candidate_id") or "") for candidate in request_by_item[item_id].get("candidates") or []}
        payload = _load_valid_output(raw_dir / f"{item_id}.json", item_id=item_id, expected_ids=expected_ids)
        if payload is None:
            items.append({"item_id": item_id, "review_complete": False, "labels": []})
        else:
            complete_count += 1
            items.append(payload)
    review = {
        "schema_version": "personaforge.eval.retrieval_gold_codex_review.v1",
        "handoff_id": manifest["handoff_id"],
        "pool_id": manifest["pool_id"],
        "pool_manifest_sha256": manifest["pool_manifest_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "gold_units_sha256": manifest["gold_units_sha256"],
        "reviewer": "codex-gpt-5.4-local",
        "complete_items": complete_count,
        "total_items": len(items),
        "items": items,
    }
    output_path = handoff_dir / "codex_review.json"
    output_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"review_file": str(output_path), "complete_items": complete_count, "total_items": len(items)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
