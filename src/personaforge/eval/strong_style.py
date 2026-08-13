"""Dual-context selection and planning for strong style experiments.

This module deliberately keeps the large retrieval pool unchanged. It adds a small,
auditable decision layer that separates evidence for the current answer from examples
chosen for the author's way of expressing an answer.
"""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from personaforge.ingest.retrieve import ParentHit
from personaforge.llm import JsonChatClient
from personaforge.persona.writer import pack_author_context


STYLE_SELECTOR_PROMPT_VERSION = "strong-style-selector-v1"
CONTENT_PLANNER_PROMPT_VERSION = "strong-style-content-plan-v1"


def select_expression_hits(
    *,
    query: str,
    objective_background: str,
    candidates: list[ParentHit],
    llm: JsonChatClient,
    top_k: int = 3,
) -> tuple[list[ParentHit], dict[str, Any]]:
    """Select style exemplars without using a separate embedding model.

    The selector is asked to judge response shape rather than topical usefulness. The
    returned IDs are validated against the candidate pool and fall back to rank order
    if the provider returns malformed or unknown IDs.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not candidates:
        return [], {
            "prompt_version": STYLE_SELECTOR_PROMPT_VERSION,
            "candidate_ids": [],
            "selected_ids": [],
            "fallback": True,
        }

    candidate_blocks = []
    for hit in candidates:
        candidate_blocks.append(
            f"[候选ID: {hit.parent_id}]\n标题：{hit.title}\n正文：\n{_style_excerpt(hit.parent)}"
        )
    messages = [
        {
            "role": "system",
            "content": """你是作者表达示范选择器。
你的任务不是选择最能回答当前问题的材料，而是从候选历史回答中选择最多三篇，作为最终写作者观察作者表达方式的示范。
优先观察：切入方式、论证推进、例子使用、句子和段落节奏、语气姿态、结束方式。
不要因为候选回答的观点正确、主题相近或信息更多而选择它。
只返回 JSON：{"selected_parent_ids": ["候选ID1", "候选ID2", "候选ID3"]}。""",
        },
        {
            "role": "user",
            "content": f"""当前问题：
{query}

题目客观背景：
{objective_background or '无额外背景。'}

候选历史回答：
{chr(10).join(candidate_blocks)}""",
        },
    ]
    started_at = perf_counter()
    fallback = False
    try:
        payload = llm.complete_json(messages, temperature=0.0, max_tokens=300)
        raw_ids = payload.get("selected_parent_ids", [])
        selected_ids = [str(value) for value in raw_ids] if isinstance(raw_ids, list) else []
    except Exception:
        selected_ids = []
        fallback = True

    candidate_map = {hit.parent_id: hit for hit in candidates}
    selected: list[ParentHit] = []
    seen: set[str] = set()
    for parent_id in selected_ids:
        if parent_id in candidate_map and parent_id not in seen:
            selected.append(candidate_map[parent_id])
            seen.add(parent_id)
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        fallback = True
        for hit in candidates:
            if hit.parent_id not in seen:
                selected.append(hit)
                seen.add(hit.parent_id)
            if len(selected) >= top_k:
                break
    return selected, {
        "prompt_version": STYLE_SELECTOR_PROMPT_VERSION,
        "candidate_ids": [hit.parent_id for hit in candidates],
        "selected_ids": [hit.parent_id for hit in selected],
        "fallback": fallback,
        "duration_ms": round((perf_counter() - started_at) * 1000),
    }


def build_content_plan(
    *,
    query: str,
    objective_background: str,
    content_hits: list[ParentHit],
    llm: JsonChatClient,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Extract a compact content boundary before the final style-conditioned write."""
    messages = [
        {
            "role": "system",
            "content": """你是回答内容规划器，不负责模仿作者。
根据当前问题和内容参考，提取一个可执行的回答边界，避免最终写作时把所有材料拼成标准答案。
只返回 JSON，字段必须是：
{"core_claim":"这次最想说的一件事","entry_angle":"适合从哪里切入","supporting_points":"最多两个支持点","avoid":"当前问题下最需要避免的内容偏移"}
不要写完整回答，不要创造材料中没有的事实。""",
        },
        {
            "role": "user",
            "content": f"""当前问题：
{query}

题目客观背景：
{objective_background or '无额外背景。'}

内容参考：
{pack_author_context(content_hits) or '无内容参考。'}""",
        },
    ]
    started_at = perf_counter()
    fallback = False
    try:
        payload = llm.complete_json(messages, temperature=0.0, max_tokens=500)
    except Exception:
        payload = {}
        fallback = True
    plan = {
        key: str(payload.get(key, "")).strip()
        for key in ("core_claim", "entry_angle", "supporting_points", "avoid")
    }
    if not any(plan.values()):
        fallback = True
        plan = {
            "core_claim": "",
            "entry_angle": "",
            "supporting_points": "",
            "avoid": "",
        }
    return plan, {
        "prompt_version": CONTENT_PLANNER_PROMPT_VERSION,
        "fallback": fallback,
        "duration_ms": round((perf_counter() - started_at) * 1000),
        "plan": plan,
    }


def _style_excerpt(parent: dict[str, Any] | None, max_chars: int = 2600) -> str:
    text = str((parent or {}).get("text") or "").strip()
    if len(text) <= max_chars:
        return text
    head = round(max_chars * 0.62)
    tail = max_chars - head
    return f"{text[:head]}\n……（中段省略）……\n{text[-tail:]}"


def stable_json(value: dict[str, Any]) -> str:
    """Render a diagnostic payload consistently in markdown traces and tests."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
