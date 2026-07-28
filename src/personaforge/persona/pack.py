"""Evidence-backed persona pack loading, validation, and prompt rendering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PERSONA_PACK_SCHEMA_VERSION = 1
PERSONA_PACK_FILENAME = "persona_pack.json"
SECTION_ORDER = ("response_strategy", "worldview", "reasoning", "voice")
REQUIRED_SECTION_ORDER = ("worldview", "reasoning", "voice")


@dataclass(frozen=True, slots=True)
class PersonaEvidence:
    doc_id: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class PersonaClaim:
    claim_id: str
    claim: str
    confidence: float
    scopes: tuple[str, ...]
    activation_condition: str
    avoid_overapplication: str
    evidence: tuple[PersonaEvidence, ...]
    counterevidence: tuple[PersonaEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class PersonaPack:
    schema_version: int
    pack_id: str
    author_id: str
    display_name: str
    source: dict[str, Any]
    corpus_stats: dict[str, Any]
    response_strategy: tuple[PersonaClaim, ...]
    worldview: tuple[PersonaClaim, ...]
    reasoning: tuple[PersonaClaim, ...]
    voice: tuple[PersonaClaim, ...]
    generation_policy: dict[str, Any]
    research_basis: tuple[dict[str, str], ...]
    sha256: str
    path: Path

    @property
    def claim_count(self) -> int:
        return (
            len(self.response_strategy)
            + len(self.worldview)
            + len(self.reasoning)
            + len(self.voice)
        )

    @property
    def evidence_count(self) -> int:
        return sum(
            len(claim.evidence) + len(claim.counterevidence)
            for claims in (
                self.response_strategy,
                self.worldview,
                self.reasoning,
                self.voice,
            )
            for claim in claims
        )

    def claims_for(self, section: str) -> tuple[PersonaClaim, ...]:
        if section not in SECTION_ORDER:
            raise ValueError(f"Unknown persona pack section: {section}")
        return getattr(self, section)


def load_persona_pack(
    path: Path,
    *,
    parent_store_path: Path | None = None,
    verify_evidence: bool = True,
) -> PersonaPack:
    """Load a pack and optionally verify every excerpt against the local parent store."""

    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if int(payload.get("schema_version", 0)) != PERSONA_PACK_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported persona pack schema: {payload.get('schema_version')!r}; "
            f"expected {PERSONA_PACK_SCHEMA_VERSION}."
        )

    sections = payload.get("sections")
    if not isinstance(sections, dict):
        raise ValueError("Persona pack must contain an object named 'sections'.")

    parsed_sections = {
        section: (
            _parse_claims(section, sections.get(section))
            if section in REQUIRED_SECTION_ORDER
            else _parse_optional_claims(section, sections.get(section))
        )
        for section in SECTION_ORDER
    }
    claim_ids = [
        claim.claim_id
        for section in SECTION_ORDER
        for claim in parsed_sections[section]
    ]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Persona pack claim_id values must be unique.")

    pack = PersonaPack(
        schema_version=PERSONA_PACK_SCHEMA_VERSION,
        pack_id=_required_text(payload, "pack_id"),
        author_id=_required_text(payload, "author_id"),
        display_name=_required_text(payload, "display_name"),
        source=_required_object(payload, "source"),
        corpus_stats=_required_object(payload, "corpus_stats"),
        response_strategy=parsed_sections["response_strategy"],
        worldview=parsed_sections["worldview"],
        reasoning=parsed_sections["reasoning"],
        voice=parsed_sections["voice"],
        generation_policy=_required_object(payload, "generation_policy"),
        research_basis=tuple(_parse_research_basis(payload.get("research_basis"))),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        path=path.resolve(),
    )
    if verify_evidence:
        if parent_store_path is None:
            raise ValueError("parent_store_path is required when verify_evidence=True.")
        verify_persona_pack_evidence(pack, parent_store_path)
    return pack


def load_persona_pack_for_index(index_dir: Path, *, required: bool = False) -> PersonaPack | None:
    """Load the pack next to an author index, preferring the author-level asset."""

    candidates = [
        index_dir.parent / PERSONA_PACK_FILENAME,
        index_dir / PERSONA_PACK_FILENAME,
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        if required:
            rendered = ", ".join(str(candidate) for candidate in candidates)
            raise ValueError(f"Persona Pack not found. Expected one of: {rendered}")
        return None
    return load_persona_pack(
        path,
        parent_store_path=index_dir / "parents.jsonl",
        verify_evidence=True,
    )


def verify_persona_pack_evidence(pack: PersonaPack, parent_store_path: Path) -> None:
    """Fail closed when a claim cites a missing document or a non-verbatim excerpt."""

    parents: dict[str, str] = {}
    with parent_store_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid parent JSON at {parent_store_path}:{line_number}"
                ) from exc
            doc_id = str(row.get("doc_id") or row.get("parent_id") or "").strip()
            text = str(row.get("text") or row.get("markdown") or "").strip()
            if doc_id:
                parents[doc_id] = text

    for section in SECTION_ORDER:
        for claim in pack.claims_for(section):
            for evidence in (*claim.evidence, *claim.counterevidence):
                text = parents.get(evidence.doc_id)
                if text is None:
                    raise ValueError(
                        f"Persona claim {claim.claim_id} cites missing document {evidence.doc_id}."
                    )
                if evidence.excerpt not in text:
                    raise ValueError(
                        f"Persona claim {claim.claim_id} contains a non-verbatim excerpt "
                        f"for {evidence.doc_id}: {evidence.excerpt!r}"
                    )


def render_persona_pack_prompt(pack: PersonaPack) -> str:
    """Render the full auditable pack into a compact, non-checklist writer context."""

    labels = {
        "response_strategy": "回应策略",
        "worldview": "稳定判断框架",
        "reasoning": "常见论证动作",
        "voice": "表达声音",
    }
    parts = [
        "## 证据化 Persona Pack",
        "",
        "这份 Pack 是从该创作者训练期原文中归纳的概率性倾向，不是每篇回答都要完成的清单。",
        "当前问题相关的 RAG 原文仍是具体观点的主要依据；Pack 只提供跨问题较稳定的身份先验。",
        "只激活与当前问题和回答形态相符的少量规则。不要为了展示画像而堆口癖、复制证据句或强套无关观点。",
    ]
    if pack.response_strategy:
        parts.extend(
            [
                "",
                "### 回应边界",
                "是否直接回答、是否转向、是否给建议以及何时结束，只服从当前 RAG 中最相似的作者原文。",
                "不要因为 AI 助手身份额外补充完整总结或行动建议；作者原文若借题发挥可以照做，但不得自行发明新的转向。",
            ]
        )
    for section in SECTION_ORDER:
        if not pack.claims_for(section):
            continue
        parts.extend(["", f"### {labels[section]}"])
        for claim in pack.claims_for(section):
            if section == "response_strategy":
                parts.append(f"- {claim.claim} 边界：{claim.avoid_overapplication}")
                continue
            scopes = "、".join(claim.scopes) if claim.scopes else "跨主题"
            evidence = claim.evidence[0].excerpt
            parts.extend(
                [
                    f"- {claim.claim}",
                    f"  - 适用：{claim.activation_condition}（范围：{scopes}）",
                    f"  - 边界：{claim.avoid_overapplication}",
                    f"  - 作者原文证据：{evidence}",
                ]
            )

    selection_rule = str(pack.generation_policy.get("selection_rule") or "").strip()
    forbidden = [
        str(item).strip()
        for item in pack.generation_policy.get("forbidden_overfit", [])
        if str(item).strip()
    ]
    if selection_rule or forbidden:
        parts.extend(["", "### 使用边界"])
    if selection_rule:
        parts.append(f"- {selection_rule}")
    parts.extend(f"- {item}" for item in forbidden)
    return "\n".join(parts).strip()


def _parse_claims(section: str, value: Any) -> tuple[PersonaClaim, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Persona pack section {section!r} must be a non-empty list.")
    claims: list[PersonaClaim] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Persona pack {section}[{index}] must be an object.")
        confidence = float(item.get("confidence", 0))
        if not 0 <= confidence <= 1:
            raise ValueError(f"Persona claim confidence must be within [0, 1]: {confidence}")
        evidence = _parse_evidence(item.get("evidence"), required=True)
        counterevidence = _parse_evidence(item.get("counterevidence"), required=False)
        scopes = tuple(
            str(scope).strip()
            for scope in item.get("scopes", [])
            if str(scope).strip()
        )
        claims.append(
            PersonaClaim(
                claim_id=_required_text(item, "claim_id"),
                claim=_required_text(item, "claim"),
                confidence=confidence,
                scopes=scopes,
                activation_condition=_required_text(item, "activation_condition"),
                avoid_overapplication=_required_text(item, "avoid_overapplication"),
                evidence=evidence,
                counterevidence=counterevidence,
            )
        )
    return tuple(claims)


def _parse_optional_claims(section: str, value: Any) -> tuple[PersonaClaim, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Persona pack section {section!r} must be a list.")
    if not value:
        return ()
    return _parse_claims(section, value)


def _parse_evidence(value: Any, *, required: bool) -> tuple[PersonaEvidence, ...]:
    if value is None and not required:
        return ()
    if not isinstance(value, list) or (required and not value):
        raise ValueError("Persona claim evidence must be a non-empty list.")
    return tuple(
        PersonaEvidence(
            doc_id=_required_text(item, "doc_id"),
            excerpt=_required_text(item, "excerpt"),
        )
        for item in value
        if isinstance(item, dict)
    )


def _parse_research_basis(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("research_basis must be a list.")
    return [
        {
            "title": _required_text(item, "title"),
            "url": _required_text(item, "url"),
            "applied_principle": _required_text(item, "applied_principle"),
        }
        for item in value
        if isinstance(item, dict)
    ]


def _required_text(value: dict[str, Any], key: str) -> str:
    text = str(value.get(key) or "").strip()
    if not text:
        raise ValueError(f"Persona pack field {key!r} must be a non-empty string.")
    return text


def _required_object(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"Persona pack field {key!r} must be an object.")
    return dict(item)
