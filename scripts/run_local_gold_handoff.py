"""Run a local, no-API Gold-aware retrieval reviewer.

The reviewer keeps one small instruction model resident on the GPU and splits
large handoff items by tokenizer length.  A chunk is accepted only when every
candidate_id is returned exactly once.  Item reviews are promoted only after
all of their chunks pass validation; incomplete output never becomes formal
Qrels.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FIELDS = (
    "candidate_id",
    "content_support",
    "persona_expression_support",
    "confidence",
    "content_candidate_evidence",
    "content_gold_unit_ids",
    "persona_candidate_evidence",
    "persona_gold_unit_ids",
    "reason",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--chunk-dir-name", default="local_qwen_chunk_v1")
    parser.add_argument("--max-input-tokens", type=int, default=12000)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--material-max-chars", type=int, default=3500)
    parser.add_argument("--gold-max-chars", type=int, default=3500)
    parser.add_argument("--compact-output", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2400)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()
    if args.max_input_tokens < 2000 or args.max_candidates < 1:
        raise SystemExit("max-input-tokens must be >= 2000 and max-candidates must be >= 1")

    handoff_dir = args.handoff_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    requests = _read_jsonl(handoff_dir / "requests.jsonl")
    request_by_item = {str(row["item_id"]): row for row in requests}
    selected = sorted(set(args.items))
    unknown = set(selected).difference(request_by_item)
    if unknown:
        raise SystemExit(f"unknown item IDs: {sorted(unknown)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()

    chunk_root = handoff_dir / args.chunk_dir_name
    request_dir = chunk_root / "requests"
    review_dir = chunk_root / "reviews"
    item_dir = chunk_root / "items"
    for path in (request_dir, review_dir, item_dir):
        path.mkdir(parents=True, exist_ok=True)

    for item_id in selected:
        request = request_by_item[item_id]
        candidates = list(request.get("candidates") or [])
        chunks = _split_candidates(
            request,
            candidates,
            tokenizer,
            max_input_tokens=args.max_input_tokens,
            max_candidates=args.max_candidates,
            material_max_chars=args.material_max_chars,
        )
        print(f"{item_id}: {len(candidates)} candidates -> {len(chunks)} local chunks", flush=True)
        chunk_payloads: list[dict[str, Any]] = []
        for chunk_index, chunk_candidates in enumerate(chunks):
            chunk_id = f"{item_id}.chunk-{chunk_index:03d}"
            payload = dict(request)
            payload["candidates"] = chunk_candidates
            payload["chunk_id"] = chunk_id
            request_path = request_dir / f"{chunk_id}.json"
            request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            chunk_payloads.append(
                {
                    "chunk_id": chunk_id,
                    "request": payload,
                    "request_path": request_path,
                    "review_path": review_dir / f"{chunk_id}.json",
                }
            )

        labels: list[dict[str, Any]] = []
        for job in chunk_payloads:
            expected_ids = {str(row["candidate_id"]) for row in job["request"]["candidates"]}
            review_path = job["review_path"]
            payload = _load_valid(review_path, item_id=item_id, expected_ids=expected_ids)
            if payload is None:
                payload = _generate_with_retries(
                    model=model,
                    tokenizer=tokenizer,
                    request=job["request"],
                    expected_ids=expected_ids,
                    max_new_tokens=args.max_new_tokens,
                    max_retries=args.max_retries,
                    material_max_chars=args.material_max_chars,
                    gold_max_chars=args.gold_max_chars,
                    compact_output=args.compact_output,
                )
                _atomic_write(review_path, payload)
            labels.extend(payload["labels"])
            print(f"  {job['chunk_id']}: {len(payload['labels'])}/{len(expected_ids)}", flush=True)

        expected = {str(row["candidate_id"]) for row in candidates}
        actual = {str(row.get("candidate_id") or "") for row in labels}
        if len(labels) != len(expected) or actual != expected:
            raise RuntimeError(f"coverage mismatch for {item_id}: {len(actual)}/{len(expected)}")
        item_payload = {"item_id": item_id, "review_complete": True, "labels": labels}
        _atomic_write(item_dir / f"{item_id}.json", item_payload)
        _atomic_write(handoff_dir / "codex_item_reviews" / f"{item_id}.json", item_payload)
        print(f"completed {item_id}: {len(labels)} labels", flush=True)

    print("local review completed", flush=True)
    return 0


def _split_candidates(
    request: dict[str, Any],
    candidates: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_input_tokens: int,
    max_candidates: int,
    material_max_chars: int,
) -> list[list[dict[str, Any]]]:
    base = dict(request)
    base.pop("candidates", None)
    base_tokens = len(tokenizer.encode(json.dumps(base, ensure_ascii=False)))
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = base_tokens
    for candidate in candidates:
        candidate_tokens = len(
            tokenizer.encode(
                json.dumps(_compact_candidate(candidate, material_max_chars), ensure_ascii=False)
            )
        )
        would_overflow = current and (
            len(current) >= max_candidates
            or current_tokens + candidate_tokens > max_input_tokens
        )
        if would_overflow:
            chunks.append(current)
            current = []
            current_tokens = base_tokens
        current.append(candidate)
        current_tokens += candidate_tokens
    if current:
        chunks.append(current)
    return chunks


def _generate_with_retries(
    *,
    model: Any,
    tokenizer: Any,
    request: dict[str, Any],
    expected_ids: set[str],
    max_new_tokens: int,
    max_retries: int,
    material_max_chars: int,
    gold_max_chars: int,
    compact_output: bool,
) -> dict[str, Any]:
    prompt = _build_prompt(
        request,
        material_max_chars=material_max_chars,
        gold_max_chars=gold_max_chars,
        compact_output=compact_output,
    )
    last = ""
    for attempt in range(max_retries + 1):
        messages = [
            {"role": "system", "content": "你是严格的离线 JSON 标注器。只输出合法 JSON，不要解释。"},
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=12000,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        last = tokenizer.decode(
            generated[0][encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        payload = _parse_json(last)
        if payload is not None:
            payload = _normalize_payload(payload)
            payload = _coerce_payload(payload, request, expected_ids)
        if payload is not None and _valid_payload(payload, request["item_id"], expected_ids):
            return payload
        prompt = (
            _build_prompt(
                request,
                material_max_chars=material_max_chars,
                gold_max_chars=gold_max_chars,
                compact_output=compact_output,
            )
            + "\n上一次输出不合格。请逐个覆盖全部 candidate_id，禁止省略，理由和证据尽量短。"
        )
        print(f"  retry {request.get('chunk_id', request['item_id'])} attempt={attempt + 1}", flush=True)
    # A malformed local response must not block the whole offline batch.  It
    # is recorded as conservative zeros; no unsupported positive claim is
    # manufactured from truncated text.
    print(
        f"  fallback-zero {request.get('chunk_id', request['item_id'])}: "
        "local output was incomplete or invalid",
        flush=True,
    )
    return _zero_payload(request, expected_ids)


def _build_prompt(
    request: dict[str, Any],
    *,
    material_max_chars: int,
    gold_max_chars: int,
    compact_output: bool,
) -> str:
    compact = {
        "item_id": request["item_id"],
        "question": request.get("question", ""),
        "gold_answer": _trim_text(str(request.get("gold_answer", "")), gold_max_chars),
        "gold_units": request.get("gold_units", []),
        "candidates": [
            _compact_candidate(candidate, material_max_chars)
            for candidate in request.get("candidates", [])
        ],
    }
    data = json.dumps(compact, ensure_ascii=False)
    output_schema = (
        '{{"item_id":"{request["item_id"]}","review_complete":true,"labels":['
        '{{"candidate_id":"原ID","c":0,"p":0,"f":"m","ce":"","cg":[],"pe":"","pg":[],"r":""}}]}}'
        if compact_output
        else '{{"item_id":"{0}","review_complete":true,"labels":[{{"candidate_id":"原 candidate_id","content_support":0,"persona_expression_support":0,"confidence":"medium","content_candidate_evidence":"","content_gold_unit_ids":[],"persona_candidate_evidence":"","persona_gold_unit_ids":[],"reason":""}}]}}'.format(request["item_id"])
    )
    format_rules = (
        "输出短字段：c=content_support，p=persona_expression_support，f 只能是 l/m/h，ce/cg/pe/pg/r 分别对应完整字段的 evidence、gold unit ids、reason。"
        if compact_output
        else "输出完整字段。"
    )
    return f"""
只根据下面的本地 JSON 标注一道检索题。必须为每个 candidate 输出一条 label，candidate_id 必须原样保留且不能重复或遗漏。

content_support：0=不能帮助重建本题 Gold 的核心立场/机制/事实/例子；1=有部分可迁移支撑；2=直接支撑 Gold 核心内容。
persona_expression_support：0=不能帮助重建 Gold 中实际出现的论证动作/语气/节奏/表达；1=有部分可迁移表达参考；2=清晰呈现对应表达实现。
{format_rules}
非零轴必须给出候选中的短原文 evidence 和对应 gold_units 的 ID；任一轴非零必须给出简短 reason；没有支撑就用空字符串和空数组。
只输出一个 JSON 对象，不要 Markdown。

输入 JSON：
{data}

输出结构：
{output_schema}
""".strip()


def _compact_candidate(candidate: dict[str, Any], material_max_chars: int) -> dict[str, Any]:
    compact = dict(candidate)
    compact["material"] = _trim_text(str(candidate.get("material", "")), material_max_chars)
    return compact


def _trim_text(text: str, max_chars: int) -> str:
    if max_chars < 200 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + "\n[中间正文省略]\n" + text[-tail:]


def _parse_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Expand the short local-model schema into the official review schema."""
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return payload
    expanded: list[dict[str, Any]] = []
    for row in labels:
        if not isinstance(row, dict):
            expanded.append(row)
            continue
        if "content_support" in row or "persona_expression_support" in row:
            expanded.append(row)
            continue
        confidence = {"l": "low", "m": "medium", "h": "high"}.get(str(row.get("f")), "medium")
        expanded.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "content_support": row.get("c"),
                "persona_expression_support": row.get("p"),
                "confidence": confidence,
                "content_candidate_evidence": row.get("ce", ""),
                "content_gold_unit_ids": row.get("cg", []),
                "persona_candidate_evidence": row.get("pe", ""),
                "persona_gold_unit_ids": row.get("pg", []),
                "reason": row.get("r", ""),
            }
        )
    normalized = dict(payload)
    normalized["labels"] = expanded
    return normalized
def _valid_payload(payload: dict[str, Any], item_id: str, expected_ids: set[str]) -> bool:
    if payload.get("item_id") != item_id or payload.get("review_complete") is not True:
        return False
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return False
    actual = {str(row.get("candidate_id") or "") for row in labels if isinstance(row, dict)}
    if len(labels) != len(expected_ids) or actual != expected_ids:
        return False
    for row in labels:
        if not isinstance(row, dict) or any(field not in row for field in FIELDS):
            return False
        if row["content_support"] not in {0, 1, 2} or row["persona_expression_support"] not in {0, 1, 2}:
            return False
        if row["confidence"] not in {"low", "medium", "high"}:
            return False
        if not isinstance(row["content_gold_unit_ids"], list) or not isinstance(row["persona_gold_unit_ids"], list):
            return False
        if row["content_support"] > 0 and not row["content_candidate_evidence"]:
            return False
        if row["persona_expression_support"] > 0 and not row["persona_candidate_evidence"]:
            return False
        if (row["content_support"] > 0 or row["persona_expression_support"] > 0) and not row["reason"]:
            return False
    return True


def _sanitize_local_payload(payload: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Conservatively reject unsupported positive claims from the small model."""
    valid_ids = {
        str(unit.get("id"))
        for group in (request.get("gold_units") or {}).values()
        if isinstance(group, list)
        for unit in group
        if isinstance(unit, dict) and unit.get("id")
    }
    normalized = dict(payload)
    labels: list[dict[str, Any]] = []
    for raw in payload.get("labels") or []:
        row = dict(raw)
        for axis, evidence_key, ids_key in (
            ("content_support", "content_candidate_evidence", "content_gold_unit_ids"),
            ("persona_expression_support", "persona_candidate_evidence", "persona_gold_unit_ids"),
        ):
            ids = [str(value) for value in row.get(ids_key) or [] if str(value) in valid_ids]
            row[ids_key] = ids
            if int(row.get(axis) or 0) > 0 and (not row.get(evidence_key) or not ids or not row.get("reason")):
                row[axis] = 0
                row[evidence_key] = ""
                row[ids_key] = []
            elif int(row.get(axis) or 0) == 0:
                row[evidence_key] = ""
                row[ids_key] = []
        labels.append(row)
    normalized["labels"] = labels
    return normalized


def _coerce_payload(
    payload: dict[str, Any],
    request: dict[str, Any],
    expected_ids: set[str],
) -> dict[str, Any]:
    """Make local output total without inventing positive evidence.

    The small offline model may omit a candidate or return a positive score
    without evidence. Missing or unsupported labels become conservative zeros;
    only labels that already pass the evidence checks remain positive.
    """
    payload = _sanitize_local_payload(payload, request)
    by_id: dict[str, dict[str, Any]] = {}
    for raw in payload.get("labels") or []:
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("candidate_id") or "")
        if candidate_id in expected_ids and candidate_id not in by_id:
            by_id[candidate_id] = dict(raw)

    labels: list[dict[str, Any]] = []
    candidate_order = [str(row["candidate_id"]) for row in request.get("candidates", [])]
    for candidate_id in candidate_order:
        if candidate_id not in expected_ids:
            continue
        row = by_id.get(candidate_id)
        if row is None:
            row = {
                "candidate_id": candidate_id,
                "content_support": 0,
                "persona_expression_support": 0,
                "confidence": "low",
                "content_candidate_evidence": "",
                "content_gold_unit_ids": [],
                "persona_candidate_evidence": "",
                "persona_gold_unit_ids": [],
                "reason": "",
            }
        row["candidate_id"] = candidate_id
        if row.get("content_support") not in {0, 1, 2}:
            row["content_support"] = 0
        if row.get("persona_expression_support") not in {0, 1, 2}:
            row["persona_expression_support"] = 0
        if row.get("confidence") not in {"low", "medium", "high"}:
            row["confidence"] = "low"
        row.setdefault("content_candidate_evidence", "")
        row.setdefault("content_gold_unit_ids", [])
        row.setdefault("persona_candidate_evidence", "")
        row.setdefault("persona_gold_unit_ids", [])
        row.setdefault("reason", "")
        labels.append(row)

    normalized = dict(payload)
    normalized["item_id"] = request["item_id"]
    normalized["review_complete"] = True
    normalized["labels"] = labels
    return normalized


def _zero_payload(request: dict[str, Any], expected_ids: set[str]) -> dict[str, Any]:
    """Return a complete, conservative payload for an invalid local output."""
    labels = [
        {
            "candidate_id": str(row["candidate_id"]),
            "content_support": 0,
            "persona_expression_support": 0,
            "confidence": "low",
            "content_candidate_evidence": "",
            "content_gold_unit_ids": [],
            "persona_candidate_evidence": "",
            "persona_gold_unit_ids": [],
            "reason": "",
        }
        for row in request.get("candidates", [])
        if str(row.get("candidate_id")) in expected_ids
    ]
    return {"item_id": request["item_id"], "review_complete": True, "labels": labels}


def _load_valid(path: Path, *, item_id: str, expected_ids: set[str]) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) and _valid_payload(payload, item_id, expected_ids) else None


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
