"""Build and audit the private five-source stimulus bank for Study 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

from personaforge.crawler.zhihu import html_to_text


DEFAULT_DATASET_DIR = Path("data/eval/wu-ren-jun-28-temporal-v0")
DEFAULT_OUT_DIR = Path("data/studies/wu-ren-jun-28-study1-dev10")
DEFAULT_PARENT_STORE = Path("data/authors/zhihu/wu-ren-jun-28/index/parents.jsonl")
DEFAULT_RAG_RUN = "baseline-dev-v0"
DEFAULT_PERSONA_RUN = "persona-pack-response-strategy-v3-writer-replay-dev-20260728"
DEFAULT_PERSONA_PACK = "persona_pack_response_strategy_v3_compact.json"
DEFAULT_AUTHOR = "wu-ren-jun-28"
DEFAULT_STUDY_ID = "wu-ren-jun-28-study1-dev10-v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp_path.replace(path)


def clone_bank_for_v2(
    *, source: Path, out_dir: Path, study_id: str
) -> dict[str, Any]:
    """Clone frozen stimuli under a fresh V2 identity without changing any text."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,199}", study_id):
        raise ValueError("study_id 只能包含字母、数字、点、下划线和连字符")
    source_text = source.read_text(encoding="utf-8")
    bank = json.loads(source_text)
    source_study_id = str(bank.get("study_id") or "")
    if len(bank.get("items") or []) < 5:
        raise ValueError("V2 材料库至少需要五道完整题目")
    bank["schema_version"] = "personaforge.study1.material-bank.v2"
    bank["protocol_version"] = "study1-v2"
    bank["study_id"] = study_id
    bank["cloned_from"] = {
        "path": str(source),
        "study_id": source_study_id,
        "material_sha256": sha256_text(source_text),
    }
    write_json(out_dir / "material_bank.json", bank)
    return bank


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def codex_runtime_metadata(*, model: str = "gpt-5.6-sol") -> dict[str, str | None]:
    package_path = Path(
        "data/tmp/codex-cli/node_modules/@openai/codex/package.json"
    ).resolve()
    version = None
    if package_path.exists():
        version = json.loads(package_path.read_text(encoding="utf-8")).get("version")
    return {
        "cli_version": f"codex-cli {version}" if version else None,
        "model": model,
        "network_policy": "web_search_disabled",
    }


def text_stats(text: str) -> dict[str, int]:
    stripped = text.strip()
    paragraphs = [part for part in re.split(r"\n\s*\n", stripped) if part.strip()]
    sentences = [part for part in re.split(r"[。！？!?]+", stripped) if part.strip()]
    return {
        "chars": len(stripped),
        "paragraphs": len(paragraphs),
        "sentences": len(sentences),
    }


def response_record(
    *,
    source: str,
    text: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    cleaned = text.strip()
    return {
        "source": source,
        "text": cleaned,
        "text_sha256": sha256_text(cleaned),
        "stats": text_stats(cleaned),
        "provenance": provenance,
    }


def stable_choice(
    candidates: list[dict[str, Any]],
    *,
    study_id: str,
    question_id: str,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("没有可供冻结的其他真人候选")
    ordered = sorted(candidates, key=lambda row: str(row["answer_id"]))
    digest = hashlib.sha256(f"{study_id}:{question_id}".encode("utf-8")).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    return ordered[generator.randrange(len(ordered))]


def _load_run(dataset_dir: Path, run_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = dataset_dir / "runs" / run_name
    rows = {row["item_id"]: row for row in read_jsonl(run_dir / "runs.jsonl")}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return rows, manifest


def prepare_bank(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    parent_store_path: Path = DEFAULT_PARENT_STORE,
    rag_run_name: str = DEFAULT_RAG_RUN,
    persona_run_name: str = DEFAULT_PERSONA_RUN,
    persona_pack_path: Path | None = None,
    author: str = DEFAULT_AUTHOR,
    author_label: str | None = None,
    study_id: str = DEFAULT_STUDY_ID,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,199}", study_id):
        raise ValueError("study_id 只能包含字母、数字、点、下划线和连字符")
    author = author.strip()
    if not author:
        raise ValueError("author 不能为空")
    resolved_persona_pack = persona_pack_path or (dataset_dir / DEFAULT_PERSONA_PACK)
    dataset_rows = [row for row in read_jsonl(dataset_dir / "dataset.jsonl") if row["split"] == "dev"]
    rag_rows, rag_manifest = _load_run(dataset_dir, rag_run_name)
    persona_rows, persona_manifest = _load_run(dataset_dir, persona_run_name)
    parents = {row["doc_id"]: row for row in read_jsonl(parent_store_path)}

    dataset_ids = [row["item_id"] for row in dataset_rows]
    if dataset_ids != sorted(dataset_ids):
        raise ValueError("Dev10 item_id 顺序异常")
    if set(dataset_ids) != set(rag_rows) or set(dataset_ids) != set(persona_rows):
        raise ValueError("数据集与两个冻结 run 的 item_id 未严格对齐")

    items: list[dict[str, Any]] = []
    for row in dataset_rows:
        item_id = row["item_id"]
        parent = parents.get(row["parent_id"])
        if parent is None:
            raise ValueError(f"Parent store 缺少 Gold parent: {row['parent_id']}")
        question_id = str(parent.get("metadata", {}).get("question_id") or "")
        if not question_id:
            raise ValueError(f"无法恢复知乎问题 ID: {item_id}")
        gold_answer_id = row["parent_id"].rsplit(":", 1)[-1]
        reference_parent_ids = [
            result["parent_id"]
            for result in persona_rows[item_id].get("trace", {}).get("retrieval", {}).get("parents", [])
        ]
        if row["parent_id"] in reference_parent_ids:
            raise ValueError(f"Codex 参考材料泄漏当前 Gold: {item_id}")

        items.append(
            {
                "item_id": item_id,
                "question": row["query"],
                "question_id": question_id,
                "question_url": f"https://www.zhihu.com/question/{question_id}",
                "gold_parent_id": row["parent_id"],
                "gold_answer_id": gold_answer_id,
                "created_at": row.get("created_at"),
                "reference_parent_ids": reference_parent_ids,
                "responses": {
                    "gold": response_record(
                        source="gold",
                        text=row["gold_answer"],
                        provenance={
                            "parent_id": row["parent_id"],
                            "answer_url": (
                                f"https://www.zhihu.com/question/{question_id}/answer/{gold_answer_id}"
                            ),
                        },
                    ),
                    "rag_identity": response_record(
                        source="rag_identity",
                        text=rag_rows[item_id]["answer"],
                        provenance={
                            "run_name": rag_run_name,
                            "writer_model": rag_manifest.get("writer_model"),
                            "writer_prompt": rag_manifest.get("config", {}).get("writer_prompt"),
                        },
                    ),
                    "persona_pack": response_record(
                        source="persona_pack",
                        text=persona_rows[item_id]["answer"],
                        provenance={
                            "run_name": persona_run_name,
                            "writer_model": persona_manifest.get("writer_model"),
                            "writer_prompt": persona_manifest.get("config", {}).get("writer_prompt"),
                            "persona_pack": persona_manifest.get("persona_pack"),
                        },
                    ),
                },
            }
        )

    bank = {
        "schema_version": "personaforge.study1.material-bank.v2",
        "protocol_version": "study1-v2",
        "study_id": study_id,
        "author": {
            "platform": "zhihu",
            "token": author,
            "label": (author_label or author).strip(),
        },
        "selection_policy": {
            "other_human": "stable_random_from_eligible_question_answers",
            "length_matching": False,
            "text_truncation": False,
        },
        "inputs": {
            "dataset": str(dataset_dir / "dataset.jsonl"),
            "parent_store": str(parent_store_path),
            "rag_identity_run": rag_run_name,
            "persona_pack_run": persona_run_name,
            "persona_pack_path": str(resolved_persona_pack),
        },
        "items": items,
    }
    write_json(out_dir / "material_bank.json", bank)
    return bank


def _answer_api_url(question_id: str, *, limit: int = 20, offset: int = 0) -> str:
    include = (
        "data[*].id,content,excerpt,voteup_count,comment_count,created_time,updated_time,"
        "author.id,author.url_token,author.name"
    )
    query = urlencode(
        {"include": include, "limit": limit, "offset": offset, "sort_by": "default"}
    )
    return f"https://www.zhihu.com/api/v4/questions/{question_id}/answers?{query}"


def collect_other_humans(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    storage_state_path: Path = Path("data/auth/zhihu_storage_state.json"),
    target_author_token: str | None = None,
    max_answers_per_question: int = 60,
    min_chars: int = 80,
    force: bool = False,
) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    bank_path = out_dir / "material_bank.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    target_author_token = target_author_token or str(
        bank.get("author", {}).get("token") or ""
    ).strip()
    if not target_author_token:
        raise ValueError("材料库缺少目标作者 token")
    if not storage_state_path.exists():
        raise FileNotFoundError(f"知乎登录态不存在: {storage_state_path}")

    all_candidates: list[dict[str, Any]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(storage_state=str(storage_state_path))
        try:
            for item in bank["items"]:
                if item["responses"].get("other_human") and not force:
                    continue
                question_id = item["question_id"]
                offset = 0
                candidates: list[dict[str, Any]] = []
                while offset < max_answers_per_question:
                    request_url = _answer_api_url(question_id, offset=offset)
                    response = context.request.get(
                        request_url,
                        headers={"Referer": item["question_url"]},
                        timeout=30_000,
                    )
                    if not response.ok:
                        raise RuntimeError(
                            f"知乎问题 {question_id} 抓取失败: HTTP {response.status}"
                        )
                    payload = response.json()
                    rows = payload.get("data") or []
                    for rank, answer in enumerate(rows, start=offset + 1):
                        answer_id = str(answer.get("id") or "")
                        author = answer.get("author") or {}
                        author_token = str(author.get("url_token") or "")
                        text = html_to_text(answer.get("content") or answer.get("excerpt") or "").strip()
                        eligible = (
                            bool(answer_id)
                            and answer_id != item["gold_answer_id"]
                            and author_token != target_author_token
                            and len(text) >= min_chars
                        )
                        candidate = {
                            "item_id": item["item_id"],
                            "question_id": question_id,
                            "answer_id": answer_id,
                            "answer_url": (
                                f"https://www.zhihu.com/question/{question_id}/answer/{answer_id}"
                            ),
                            "api_rank": rank,
                            "author": {
                                "id": author.get("id"),
                                "url_token": author_token,
                                "name": author.get("name"),
                            },
                            "voteup_count": answer.get("voteup_count"),
                            "comment_count": answer.get("comment_count"),
                            "created_time": answer.get("created_time"),
                            "text": text,
                            "text_sha256": sha256_text(text),
                            "stats": text_stats(text),
                            "eligible": eligible,
                            "exclusion_reason": (
                                None
                                if eligible
                                else _human_exclusion_reason(
                                    answer_id=answer_id,
                                    gold_answer_id=item["gold_answer_id"],
                                    author_token=author_token,
                                    target_author_token=target_author_token,
                                    text=text,
                                    min_chars=min_chars,
                                )
                            ),
                        }
                        all_candidates.append(candidate)
                        if eligible:
                            candidates.append(candidate)
                    if payload.get("paging", {}).get("is_end") or not rows:
                        break
                    offset += len(rows)

                if not candidates:
                    item.setdefault("audit_flags", []).append("missing_other_human")
                    continue
                selected = stable_choice(
                    candidates,
                    study_id=bank["study_id"],
                    question_id=question_id,
                )
                item["responses"]["other_human"] = response_record(
                    source="other_human",
                    text=selected["text"],
                    provenance={
                        "answer_id": selected["answer_id"],
                        "answer_url": selected["answer_url"],
                        "author": selected["author"],
                        "api_rank": selected["api_rank"],
                        "candidate_count": len(candidates),
                        "selection": "stable_random",
                    },
                )
        finally:
            context.close()
            browser.close()

    write_jsonl(out_dir / "other_human_candidates.jsonl", all_candidates)
    write_json(bank_path, bank)
    return bank


def _human_exclusion_reason(
    *,
    answer_id: str,
    gold_answer_id: str,
    author_token: str,
    target_author_token: str,
    text: str,
    min_chars: int,
) -> str:
    if not answer_id:
        return "missing_answer_id"
    if answer_id == gold_answer_id:
        return "gold_answer"
    if author_token == target_author_token:
        return "target_author"
    if len(text) < min_chars:
        return "too_short_or_empty"
    return "ineligible"


def build_codex_prompt(
    *,
    question: str,
    persona_pack: dict[str, Any],
    reference_parents: list[dict[str, Any]],
) -> str:
    references = []
    for index, parent in enumerate(reference_parents, start=1):
        references.append(
            f"### 历史表达 {index}\n标题：{parent['title']}\n\n{parent['text'].strip()}"
        )
    return "\n\n".join(
        [
            """你现在就是这些历史表达的作者本人。请直接回答最后的问题，不要分析自己如何模仿，
不要提到 Persona Pack、历史材料、检索、模型或实验。先从历史表达中理解作者稳定的立场、
判断方式、论证习惯、语气和篇幅选择，再自然作答。不要机械拼贴原句，不要求覆盖画像中的所有特点，
也不要为了显得像而编造作者经历或实时事实。禁止联网、搜索或使用任何工具，只能依据下面明确提供的
作者画像、历史表达和问题作答。只输出最终回答正文。""",
            "## 作者画像\n" + json.dumps(persona_pack, ensure_ascii=False, indent=2),
            "## 作者历史表达\n\n" + "\n\n".join(references),
            "## 当前问题\n\n" + question,
        ]
    )


def _run_codex_item(
    *,
    item: dict[str, Any],
    out_dir: Path,
    persona_pack: dict[str, Any],
    parents: dict[str, dict[str, Any]],
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
) -> tuple[str, dict[str, Any]]:
    reference_ids = item["reference_parent_ids"]
    if item["gold_parent_id"] in reference_ids:
        raise ValueError(f"隔离输入包含 Gold parent: {item['item_id']}")
    missing = [parent_id for parent_id in reference_ids if parent_id not in parents]
    if missing:
        raise ValueError(f"Parent store 缺少 {item['item_id']} 的参考材料: {missing[:3]}")
    reference_parents = [parents[parent_id] for parent_id in reference_ids]
    prompt = build_codex_prompt(
        question=item["question"],
        persona_pack=persona_pack,
        reference_parents=reference_parents,
    )

    item_dir = out_dir / "codex_inputs" / item["item_id"]
    item_dir.mkdir(parents=True, exist_ok=True)
    (item_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    write_json(
        item_dir / "input_manifest.json",
        {
            "item_id": item["item_id"],
            "question": item["question"],
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_sha256": sha256_text(prompt),
            "reference_parent_ids": reference_ids,
            "gold_parent_excluded": item["gold_parent_id"] not in reference_ids,
        },
    )

    codex = shutil.which("codex.exe") or shutil.which("codex") or shutil.which("codex.cmd")
    if not codex:
        raise FileNotFoundError("找不到 codex CLI")
    output_path = item_dir / "answer.txt"
    launcher = [codex]
    node = shutil.which("node.exe")
    local_codex_js = (
        Path("data/tmp/codex-cli/node_modules/@openai/codex/bin/codex.js").resolve()
    )
    codex_js = (
        local_codex_js
        if local_codex_js.exists()
        else Path(node).parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        if node
        else None
    )
    if codex_js and codex_js.exists():
        launcher = [node, str(codex_js)]
    command = [
        *launcher,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-c",
        'service_tier="fast"',
        "-c",
        'web_search="disabled"',
        "-C",
        str(item_dir),
        "-o",
        str(output_path),
        "-",
    ]
    if launcher == [codex] and codex.lower().endswith((".cmd", ".bat")):
        command = ["cmd.exe", "/d", "/s", "/c", subprocess.list2cmdline(command)]
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    (item_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (item_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"Codex 生成失败 {item['item_id']}，exit={completed.returncode}: "
            f"{completed.stderr[-500:]}"
        )
    answer = output_path.read_text(encoding="utf-8").strip()
    if not answer:
        raise RuntimeError(f"Codex 返回空回答: {item['item_id']}")
    return item["item_id"], response_record(
        source="codex",
        text=answer,
        provenance={
            "model": model,
            "reasoning_effort": reasoning_effort,
            "session_policy": "fresh_ephemeral_per_item",
            "network_policy": "web_search_disabled",
            "prompt_sha256": sha256_text(prompt),
            "reference_parent_ids": reference_ids,
            "input_manifest": str(item_dir / "input_manifest.json"),
        },
    )


def generate_codex(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    parent_store_path: Path = DEFAULT_PARENT_STORE,
    persona_pack_path: Path | None = None,
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "high",
    workers: int = 2,
    timeout_seconds: int = 900,
    force: bool = False,
    item_ids: set[str] | None = None,
) -> dict[str, Any]:
    bank_path = out_dir / "material_bank.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    configured_pack = bank.get("inputs", {}).get("persona_pack_path")
    resolved_persona_pack = (
        persona_pack_path
        or (Path(configured_pack) if configured_pack else None)
        or (dataset_dir / DEFAULT_PERSONA_PACK)
    )
    persona_pack = json.loads(resolved_persona_pack.read_text(encoding="utf-8"))
    parents = {row["doc_id"]: row for row in read_jsonl(parent_store_path)}
    pending = [
        item
        for item in bank["items"]
        if (item_ids is None or item["item_id"] in item_ids)
        and (force or "codex" not in item.get("responses", {}))
    ]
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _run_codex_item,
                item=item,
                out_dir=out_dir,
                persona_pack=persona_pack,
                parents=parents,
                model=model,
                reasoning_effort=reasoning_effort,
                timeout_seconds=timeout_seconds,
            ): item["item_id"]
            for item in pending
        }
        for future in as_completed(futures):
            item_id, record = future.result()
            results[item_id] = record
            print(f"Codex 完成 {item_id}: {record['stats']['chars']} 字符", flush=True)

    for item in bank["items"]:
        if item["item_id"] in results:
            item["responses"]["codex"] = results[item["item_id"]]
        if "codex" in item["responses"]:
            item["responses"]["codex"]["provenance"].update(
                codex_runtime_metadata(model=model)
            )
    bank["codex_runtime"] = codex_runtime_metadata(model=model)
    write_json(bank_path, bank)
    return bank


def audit_bank(*, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, Any]:
    bank = json.loads((out_dir / "material_bank.json").read_text(encoding="utf-8"))
    expected = ["gold", "rag_identity", "persona_pack", "codex", "other_human"]
    source_stats: dict[str, list[int]] = {source: [] for source in expected}
    item_reports = []
    for item in bank["items"]:
        missing = [source for source in expected if source not in item["responses"]]
        flags = list(item.get("audit_flags") or [])
        lengths: dict[str, int] = {}
        for source, response in item["responses"].items():
            chars = int(response["stats"]["chars"])
            lengths[source] = chars
            source_stats.setdefault(source, []).append(chars)
        gold_chars = max(1, lengths.get("gold", 1))
        extreme = [
            source
            for source, chars in lengths.items()
            if source != "gold" and (chars / gold_chars < 0.25 or chars / gold_chars > 4.0)
        ]
        if extreme:
            flags.append("extreme_length_ratio:" + ",".join(sorted(extreme)))
        if item["gold_parent_id"] in item.get("reference_parent_ids", []):
            flags.append("gold_reference_leak")
        prompt_path = out_dir / "codex_inputs" / item["item_id"] / "prompt.md"
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
            gold_text = item["responses"]["gold"]["text"].strip()
            if gold_text and gold_text in prompt:
                flags.append("gold_text_leak")
        stderr_path = out_dir / "codex_inputs" / item["item_id"] / "stderr.log"
        if stderr_path.exists() and "web search:" in stderr_path.read_text(
            encoding="utf-8", errors="replace"
        ).lower():
            flags.append("codex_web_search_used")
        item_reports.append(
            {
                "item_id": item["item_id"],
                "question": item["question"],
                "missing_sources": missing,
                "lengths": lengths,
                "flags": sorted(set(flags)),
            }
        )

    summary = {}
    for source, values in source_stats.items():
        ordered = sorted(values)
        summary[source] = {
            "count": len(values),
            "min_chars": min(values) if values else None,
            "median_chars": ordered[len(ordered) // 2] if ordered else None,
            "max_chars": max(values) if values else None,
        }
    report = {
        "schema_version": "personaforge.study1.material-audit.v1",
        "study_id": bank["study_id"],
        "generated_at": datetime.now().astimezone().isoformat(),
        "codex_runtime": bank.get("codex_runtime") or codex_runtime_metadata(),
        "complete": all(not item["missing_sources"] for item in item_reports),
        "source_summary": summary,
        "items": item_reports,
    }
    write_json(out_dir / "audit.json", report)

    lines = [
        "# Study 1 材料审计",
        "",
        f"- 五来源完整：{'是' if report['complete'] else '否'}",
        f"- 问题数：{len(item_reports)}",
        "- 长度仅用于审计，未用于匹配、截断或自动淘汰。",
        "",
        "## 来源长度",
        "",
        "| 来源 | 数量 | 最短 | 中位数 | 最长 |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in expected:
        stats = summary[source]
        lines.append(
            f"| {source} | {stats['count']} | {stats['min_chars']} | "
            f"{stats['median_chars']} | {stats['max_chars']} |"
        )
    lines.extend(["", "## 逐题状态", "", "| 题目 | 缺失来源 | 审计标记 |", "|---|---|---|"])
    for item in item_reports:
        lines.append(
            f"| {item['item_id']} | {', '.join(item['missing_sources']) or '无'} | "
            f"{', '.join(item['flags']) or '无'} |"
        )
    (out_dir / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_researcher_review(bank=bank, out_dir=out_dir)
    return report


def write_researcher_review(*, bank: dict[str, Any], out_dir: Path) -> None:
    labels = {
        "gold": "目标作者 Gold",
        "rag_identity": "RAG20 + 强身份 Prompt",
        "persona_pack": "Persona Pack 系统",
        "codex": "隔离 Codex",
        "other_human": "同题其他真人",
    }
    review_dir = out_dir / "researcher_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    for item in bank["items"]:
        lines = [
            f"# {item['item_id']} {item['question']}",
            "",
            f"- 知乎问题：{item['question_url']}",
            f"- Gold回答：{item['responses']['gold']['provenance']['answer_url']}",
            "- 本文件仅供研究者审计，正式参与者界面必须隐藏来源。",
            "",
        ]
        for source in ["gold", "rag_identity", "persona_pack", "codex", "other_human"]:
            response = item["responses"].get(source)
            if response is None:
                lines.extend([f"## {labels[source]}", "", "缺失", ""])
                continue
            stats = response["stats"]
            lines.extend(
                [
                    f"## {labels[source]}",
                    "",
                    f"字符数：{stats['chars']}；段落：{stats['paragraphs']}；句子：{stats['sentences']}",
                ]
            )
            answer_url = response.get("provenance", {}).get("answer_url")
            if answer_url:
                lines.append(f"原文：{answer_url}")
            lines.extend(["", response["text"], ""])
        (review_dir / f"{item['item_id']}.md").write_text(
            "\n".join(lines).rstrip() + "\n",
            encoding="utf-8",
        )


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建 Study 1 五来源 Dev10 材料池")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="汇总 Gold 与两个已有系统版本")
    prepare.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    prepare.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    prepare.add_argument("--parent-store", type=Path, default=DEFAULT_PARENT_STORE)
    prepare.add_argument("--author", default=DEFAULT_AUTHOR, help="知乎用户 token")
    prepare.add_argument("--author-label", help="参与者页面显示的作者名称")
    prepare.add_argument("--study-id", default=DEFAULT_STUDY_ID)
    prepare.add_argument("--rag-run", default=DEFAULT_RAG_RUN)
    prepare.add_argument("--persona-run", default=DEFAULT_PERSONA_RUN)
    prepare.add_argument("--persona-pack", type=Path)

    humans = subparsers.add_parser("collect-humans", help="抓取并稳定随机冻结同题真人回答")
    humans.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    humans.add_argument(
        "--storage-state",
        type=Path,
        default=Path("data/auth/zhihu_storage_state.json"),
    )
    humans.add_argument("--max-answers", type=int, default=60)
    humans.add_argument("--min-chars", type=int, default=80)
    humans.add_argument("--force", action="store_true")

    codex = subparsers.add_parser("generate-codex", help="用隔离 Codex 会话生成十篇回答")
    codex.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    codex.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    codex.add_argument("--parent-store", type=Path, default=DEFAULT_PARENT_STORE)
    codex.add_argument(
        "--persona-pack",
        type=Path,
        help="覆盖材料库记录的 Persona Pack 路径",
    )
    codex.add_argument("--model", default="gpt-5.6-sol")
    codex.add_argument("--reasoning-effort", default="high")
    codex.add_argument("--workers", type=int, default=2)
    codex.add_argument("--timeout-seconds", type=int, default=900)
    codex.add_argument("--item-id", action="append", dest="item_ids")
    codex.add_argument("--force", action="store_true")

    audit = subparsers.add_parser("audit", help="输出完整性与长度分布审计")
    audit.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    clone = subparsers.add_parser("clone-v2", help="把已审计刺激原样复制到新的 V2 study_id")
    clone.add_argument("--source", type=Path, required=True)
    clone.add_argument("--out-dir", type=Path, required=True)
    clone.add_argument("--study-id", required=True)
    return parser


def main() -> None:
    args = _common_parser().parse_args()
    if args.command == "prepare":
        bank = prepare_bank(
            dataset_dir=args.dataset_dir,
            out_dir=args.out_dir,
            parent_store_path=args.parent_store,
            rag_run_name=args.rag_run,
            persona_run_name=args.persona_run,
            persona_pack_path=args.persona_pack,
            author=args.author,
            author_label=args.author_label,
            study_id=args.study_id,
        )
        print(f"已准备 {len(bank['items'])} 道题的三来源材料")
    elif args.command == "collect-humans":
        bank = collect_other_humans(
            out_dir=args.out_dir,
            storage_state_path=args.storage_state,
            max_answers_per_question=args.max_answers,
            min_chars=args.min_chars,
            force=args.force,
        )
        count = sum("other_human" in item["responses"] for item in bank["items"])
        print(f"已冻结 {count}/{len(bank['items'])} 篇其他真人回答")
    elif args.command == "generate-codex":
        bank = generate_codex(
            dataset_dir=args.dataset_dir,
            out_dir=args.out_dir,
            parent_store_path=args.parent_store,
            persona_pack_path=args.persona_pack,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            workers=args.workers,
            timeout_seconds=args.timeout_seconds,
            force=args.force,
            item_ids=set(args.item_ids) if args.item_ids else None,
        )
        count = sum("codex" in item["responses"] for item in bank["items"])
        print(f"已生成 {count}/{len(bank['items'])} 篇 Codex 回答")
    elif args.command == "audit":
        report = audit_bank(out_dir=args.out_dir)
        print(f"材料五来源完整：{report['complete']}")
    elif args.command == "clone-v2":
        bank = clone_bank_for_v2(
            source=args.source,
            out_dir=args.out_dir,
            study_id=args.study_id,
        )
        print(f"已复制 V2 材料：{bank['study_id']}，共 {len(bank['items'])} 道题")


if __name__ == "__main__":
    main()
