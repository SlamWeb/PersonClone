"""Evidence-backed narrative schemas for inference-time role playing.

The schema is a compact, auditable memory layer.  It is intentionally separate
from ``persona_pack.py``: the old pack remains available for compatibility, while
the narrative schema is organized around situations and enactment signals.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NARRATIVE_SCHEMA_VERSION = 1
NARRATIVE_SCHEMA_FILENAME = "narrative_schema.json"


class NarrativeSchemaError(ValueError):
    """Raised when a narrative schema is malformed or not evidence-backed."""


@dataclass(frozen=True, slots=True)
class NarrativeEvidence:
    claim_id: str
    doc_id: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class NarrativeFacet:
    facet_id: str
    title: str
    cue_keys: tuple[str, ...]
    situation: str
    thinking_pattern: str
    behavior_pattern: str
    expression_signals: tuple[str, ...]
    boundary_anchors: tuple[str, ...]
    source_evidence: tuple[NarrativeEvidence, ...]


@dataclass(frozen=True, slots=True)
class NarrativeSchema:
    schema_version: int
    schema_id: str
    author_id: str
    display_name: str
    source: dict[str, Any]
    corpus_snapshot: dict[str, Any]
    identity: dict[str, Any]
    global_summary: str
    core_traits: tuple[str, ...]
    scene_facets: tuple[NarrativeFacet, ...]
    generation_policy: dict[str, Any]
    sha256: str
    path: Path

    @property
    def facet_count(self) -> int:
        return len(self.scene_facets)

    @property
    def evidence_count(self) -> int:
        return sum(len(facet.source_evidence) for facet in self.scene_facets)


def load_narrative_schema(
    path: Path,
    *,
    parent_store_path: Path | None = None,
    verify_evidence: bool = True,
) -> NarrativeSchema:
    """Load a schema and optionally verify every excerpt against parents.jsonl."""

    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise NarrativeSchemaError(f"Invalid narrative schema JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise NarrativeSchemaError("Narrative schema root must be a JSON object.")
    if int(payload.get("schema_version", 0)) != NARRATIVE_SCHEMA_VERSION:
        raise NarrativeSchemaError(
            f"Unsupported narrative schema version: {payload.get('schema_version')!r}; "
            f"expected {NARRATIVE_SCHEMA_VERSION}."
        )

    facets_payload = payload.get("scene_facets")
    if not isinstance(facets_payload, list) or not facets_payload:
        raise NarrativeSchemaError("Narrative schema must contain non-empty scene_facets.")
    facets = tuple(_parse_facet(item) for item in facets_payload)
    facet_ids = [facet.facet_id for facet in facets]
    if len(facet_ids) != len(set(facet_ids)):
        raise NarrativeSchemaError("Narrative facet_id values must be unique.")

    schema = NarrativeSchema(
        schema_version=NARRATIVE_SCHEMA_VERSION,
        schema_id=_required_text(payload, "schema_id"),
        author_id=_required_text(payload, "author_id"),
        display_name=_required_text(payload, "display_name"),
        source=_object(payload, "source"),
        corpus_snapshot=_object(payload, "corpus_snapshot"),
        identity=_object(payload, "identity"),
        global_summary=_required_text(payload, "global_summary"),
        core_traits=_text_tuple(payload.get("core_traits"), "core_traits"),
        scene_facets=facets,
        generation_policy=_object(payload, "generation_policy"),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        path=path.resolve(),
    )
    if verify_evidence:
        if parent_store_path is None:
            raise NarrativeSchemaError("parent_store_path is required when verify_evidence=True.")
        verify_narrative_schema_evidence(schema, parent_store_path)
    return schema


def load_narrative_schema_for_index(
    index_dir: Path,
    *,
    required: bool = False,
) -> NarrativeSchema | None:
    """Load the author-level schema next to an index."""

    candidates = [
        index_dir.parent / NARRATIVE_SCHEMA_FILENAME,
        index_dir / NARRATIVE_SCHEMA_FILENAME,
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        if required:
            raise NarrativeSchemaError(
                "Narrative schema not found. Expected one of: "
                + ", ".join(str(candidate) for candidate in candidates)
            )
        return None
    return load_narrative_schema(
        path,
        parent_store_path=index_dir / "parents.jsonl",
        verify_evidence=True,
    )


def verify_narrative_schema_evidence(
    schema: NarrativeSchema,
    parent_store_path: Path,
) -> None:
    """Fail closed if a schema cites missing documents or altered excerpts."""

    parents: dict[str, str] = {}
    with parent_store_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise NarrativeSchemaError(
                    f"Invalid parent JSON at {parent_store_path}:{line_number}"
                ) from exc
            doc_id = str(row.get("doc_id") or row.get("parent_id") or "").strip()
            text = str(row.get("text") or row.get("markdown") or "")
            if doc_id:
                parents[doc_id] = text

    missing: list[str] = []
    mismatched: list[str] = []
    for facet in schema.scene_facets:
        for evidence in facet.source_evidence:
            text = parents.get(evidence.doc_id)
            if text is None:
                missing.append(evidence.doc_id)
            elif evidence.excerpt not in text:
                mismatched.append(f"{evidence.doc_id}:{evidence.excerpt}")
    if missing:
        raise NarrativeSchemaError(
            "Narrative schema cites missing parent document(s): " + ", ".join(sorted(set(missing)))
        )
    if mismatched:
        raise NarrativeSchemaError(
            "Narrative schema contains non-verbatim evidence: " + "; ".join(mismatched)
        )


def render_narrative_schema_prompt(schema: NarrativeSchema) -> str:
    """Render memory for the writer without exposing audit excerpts."""

    parts = [
        "## Narrative Schema（长期叙事记忆）",
        "",
        "这是一份从该创作者公开表达中归纳出的长期记忆，不是答案模板，也不是每次都要执行的清单。",
        "当前问题和本轮检索到的作者原文优先；只有相关的记忆才被激活。",
        "",
        f"身份锚点：{schema.identity.get('public_identity') or schema.display_name}",
        f"全局叙事概括：{schema.global_summary}",
        "",
        "核心稳定倾向：",
    ]
    parts.extend(f"- {trait}" for trait in schema.core_traits)
    parts.extend(["", "场景记忆（只选择与当前问题相符的少量部分）："])
    for facet in schema.scene_facets:
        parts.extend(
            [
                f"### {facet.title}",
                f"触发线索：{'、'.join(facet.cue_keys)}",
                f"适用情境：{facet.situation}",
                f"思考方式：{facet.thinking_pattern}",
                f"表达动作：{facet.behavior_pattern}",
                f"可表现的信号：{'；'.join(facet.expression_signals)}",
                f"边界：{'；'.join(facet.boundary_anchors)}",
            ]
        )
    parts.extend(
        [
            "",
            "### Magic-If 执行协议",
            "1. Anchoring：先根据当前问题、本轮作者原文和身份锚点判断‘我是谁’。",
            "2. Selecting：只选与当前情境相关的场景记忆，不平均融合所有记忆。",
            "3. Bounding：不超出记忆的适用主题、时间范围和已知边界；缺少证据时不要装作知道。",
            "4. Enacting：把选中的判断方式自然地写进回答，不解释记忆、不复述 schema、不拼贴证据摘录。",
            "",
            "审计证据只用于证明这份记忆有来源，不要把文档 ID、摘录或‘根据记忆’写进最终回答。",
        ]
    )
    return "\n".join(parts).strip()


def _parse_facet(payload: Any) -> NarrativeFacet:
    if not isinstance(payload, dict):
        raise NarrativeSchemaError("Each scene facet must be a JSON object.")
    evidence_payload = payload.get("source_evidence")
    if not isinstance(evidence_payload, list) or not evidence_payload:
        raise NarrativeSchemaError("Each scene facet needs non-empty source_evidence.")
    evidence = tuple(_parse_evidence(item) for item in evidence_payload)
    return NarrativeFacet(
        facet_id=_required_text(payload, "facet_id"),
        title=_required_text(payload, "title"),
        cue_keys=_text_tuple(payload.get("cue_keys"), "cue_keys"),
        situation=_required_text(payload, "situation"),
        thinking_pattern=_required_text(payload, "thinking_pattern"),
        behavior_pattern=_required_text(payload, "behavior_pattern"),
        expression_signals=_text_tuple(payload.get("expression_signals"), "expression_signals"),
        boundary_anchors=_text_tuple(payload.get("boundary_anchors"), "boundary_anchors"),
        source_evidence=evidence,
    )


def _parse_evidence(payload: Any) -> NarrativeEvidence:
    if not isinstance(payload, dict):
        raise NarrativeSchemaError("Narrative evidence must be a JSON object.")
    return NarrativeEvidence(
        claim_id=_required_text(payload, "claim_id"),
        doc_id=_required_text(payload, "doc_id"),
        excerpt=_required_text(payload, "excerpt"),
    )


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise NarrativeSchemaError(f"Narrative schema field {key!r} is required.")
    return value


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise NarrativeSchemaError(f"Narrative schema field {key!r} must be an object.")
    return value


def _text_tuple(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise NarrativeSchemaError(f"Narrative schema field {key!r} must be a non-empty list.")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    if not result:
        raise NarrativeSchemaError(f"Narrative schema field {key!r} must contain text.")
    return result
