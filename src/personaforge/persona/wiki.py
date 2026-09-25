"""Compile existing evidence-backed author assets into one inspectable Wiki file.

This first migration step preserves the source records instead of guessing that
similarly worded Pack claims and Narrative facets mean the same thing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from personaforge.persona.narrative import load_narrative_schema_for_index
from personaforge.persona.pack import SECTION_ORDER, load_persona_pack_for_index


PERSONA_WIKI_FILENAME = "persona_wiki.json"
PERSONA_WIKI_VERSION = 1


def build_persona_wiki(index_dir: Path) -> Path:
    """Publish a deterministic, source-linked Wiki beside an author's index.

    Both legacy assets are migration inputs. Their existing loaders verify every
    excerpt against parents.jsonl before any Wiki file is replaced.
    """

    index_dir = Path(index_dir)
    pack = load_persona_pack_for_index(index_dir)
    schema = load_narrative_schema_for_index(index_dir)
    if pack is None and schema is None:
        raise ValueError("Persona Wiki needs a validated Persona Pack or Narrative Schema.")
    author_ids = {asset.author_id for asset in (pack, schema) if asset is not None}
    if len(author_ids) != 1:
        raise ValueError("Persona Pack and Narrative Schema belong to different authors.")

    cards: list[dict[str, Any]] = []
    if pack is not None:
        for section in SECTION_ORDER:
            for claim in pack.claims_for(section):
                cards.append(
                    {
                        "node_id": f"pack:{claim.claim_id}",
                        "kind": "claim",
                        "section": section,
                        "title": claim.claim,
                        "text": claim.claim,
                        "keywords": list(claim.scopes),
                        "condition": claim.activation_condition,
                        "boundary": claim.avoid_overapplication,
                        "confidence": claim.confidence,
                        "source_asset": "persona_pack",
                        "source_id": claim.claim_id,
                        "evidence": [
                            {"doc_id": item.doc_id, "excerpt": item.excerpt}
                            for item in claim.evidence
                        ],
                        "counterevidence": [
                            {"doc_id": item.doc_id, "excerpt": item.excerpt}
                            for item in claim.counterevidence
                        ],
                    }
                )
    if schema is not None:
        for facet in schema.scene_facets:
            cards.append(
                {
                    "node_id": f"narrative:{facet.facet_id}",
                    "kind": "scene_facet",
                    "section": "situation",
                    "title": facet.title,
                    "text": "；".join((facet.thinking_pattern, facet.behavior_pattern)),
                    "keywords": list(facet.cue_keys),
                    "condition": facet.situation,
                    "boundary": "；".join(facet.boundary_anchors),
                    "expression_signals": list(facet.expression_signals),
                    "source_asset": "narrative_schema",
                    "source_id": facet.facet_id,
                    "evidence": [
                        {"claim_id": item.claim_id, "doc_id": item.doc_id, "excerpt": item.excerpt}
                        for item in facet.source_evidence
                    ],
                    "counterevidence": [],
                }
            )

    # These are navigation links only. Shared source does not prove that two
    # observations support each other or are semantically equivalent.
    by_doc: dict[str, set[str]] = {}
    for card in cards:
        for item in (*card["evidence"], *card["counterevidence"]):
            by_doc.setdefault(item["doc_id"], set()).add(card["node_id"])
    for card in cards:
        linked = set().union(
            *(by_doc[item["doc_id"]] for item in (*card["evidence"], *card["counterevidence"]))
        )
        linked.discard(card["node_id"])
        card["shared_source_links"] = sorted(linked)

    payload = {
        "schema_version": PERSONA_WIKI_VERSION,
        "author_id": next(iter(author_ids)),
        "display_name": schema.display_name if schema is not None else pack.display_name,
        "source_hashes": {
            **({"persona_pack": pack.sha256} if pack is not None else {}),
            **({"narrative_schema": schema.sha256} if schema is not None else {}),
        },
        "core": {
            "public_identity": (
                str(schema.identity.get("public_identity") or schema.display_name)
                if schema is not None else pack.display_name
            ),
            "global_summary": schema.global_summary if schema is not None else "",
            "core_traits": list(schema.core_traits) if schema is not None else [],
            "source_asset": "narrative_schema" if schema is not None else "persona_pack",
        },
        "cards": cards,
    }
    destination = index_dir.parent / PERSONA_WIKI_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Validate the new representation against the current raw source before
    # publishing; a failed migration leaves the previous Wiki untouched.
    _validate_wiki(payload, index_dir / "parents.jsonl")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=f".{PERSONA_WIKI_FILENAME}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_persona_wiki(index_dir: Path) -> dict[str, Any]:
    """Load a published Wiki, failing closed if its cited raw text changed."""

    index_dir = Path(index_dir)
    path = index_dir.parent / PERSONA_WIKI_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_wiki(payload, index_dir / "parents.jsonl")
    return payload


def render_persona_wiki_core(wiki: dict[str, Any]) -> str:
    core = wiki["core"]
    lines = ["## Persona Wiki：作者总纲", f"公开身份：{core['public_identity']}"]
    if core.get("global_summary"):
        lines.append(f"整体倾向：{core['global_summary']}")
    lines.extend(f"- {trait}" for trait in core.get("core_traits", []))
    lines.append("总纲是概率性作者画像；当前问题及作者原文优先，不把它当作回答清单。")
    return "\n".join(lines)


def render_persona_wiki_cards(cards: list[dict[str, Any]]) -> str:
    lines = ["Persona Wiki 记录（有条件的作者倾向，不是每轮都要使用的清单）："]
    for card in cards:
        lines.extend(
            (f"- {card['title']}", f"  内容：{card['text']}",
             f"  适用：{card['condition']}", f"  边界：{card['boundary']}")
        )
        if card.get("expression_signals"):
            lines.append(f"  表达信号：{'；'.join(card['expression_signals'])}")
    lines.append("审计摘录与文档 ID 不进入回答；具体观点以本轮作者原文为准。")
    return "\n".join(lines)


def _validate_wiki(payload: Any, parent_store_path: Path) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != PERSONA_WIKI_VERSION:
        raise ValueError("Unsupported Persona Wiki schema version.")
    if not isinstance(payload.get("author_id"), str) or not payload["author_id"].strip():
        raise ValueError("Persona Wiki needs an author_id.")
    if not isinstance(payload.get("core"), dict):
        raise ValueError("Persona Wiki needs a core projection.")
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Persona Wiki needs source asset hashes.")
    for asset, filename in (
        ("persona_pack", "persona_pack.json"),
        ("narrative_schema", "narrative_schema.json"),
    ):
        expected = source_hashes.get(asset)
        if expected is None:
            continue
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Persona Wiki has an invalid {asset} source hash.")
        source_path = next(
            (path for path in (parent_store_path.parent.parent / filename,
                              parent_store_path.parent / filename) if path.exists()),
            None,
        )
        if source_path is not None and hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Persona Wiki {asset} source changed; rebuild the Wiki.")
    cards = payload.get("cards")
    if not isinstance(cards, list) or not cards:
        raise ValueError("Persona Wiki needs evidence-backed cards.")
    ids = [card.get("node_id") for card in cards if isinstance(card, dict)]
    if len(ids) != len(cards) or len(ids) != len(set(ids)):
        raise ValueError("Persona Wiki card IDs must be unique.")
    parents: dict[str, str] = {}
    with parent_store_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                doc_id = str(row.get("doc_id") or row.get("parent_id") or "")
                parents[doc_id] = str(row.get("text") or row.get("markdown") or "")
    id_set = set(ids)
    for card in cards:
        evidence = card.get("evidence")
        counterevidence = card.get("counterevidence")
        if not isinstance(evidence, list) or not evidence or not isinstance(counterevidence, list):
            raise ValueError(f"Persona Wiki card {card['node_id']} needs evidence lists.")
        for item in (*evidence, *counterevidence):
            if not isinstance(item, dict):
                raise ValueError(f"Persona Wiki card {card['node_id']} has invalid evidence.")
            doc_id, excerpt = item.get("doc_id"), item.get("excerpt")
            if not isinstance(doc_id, str) or not isinstance(excerpt, str) or not excerpt:
                raise ValueError(f"Persona Wiki card {card['node_id']} has invalid evidence.")
            if excerpt not in parents.get(doc_id, ""):
                raise ValueError(f"Persona Wiki card {card['node_id']} cites missing or altered raw text: {doc_id}")
        links = card.get("shared_source_links")
        if not isinstance(links, list) or any(link not in id_set for link in links):
            raise ValueError(f"Persona Wiki card {card['node_id']} has invalid links.")
