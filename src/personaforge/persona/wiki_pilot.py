"""Validate and render a source-grounded Persona Wiki pilot.

The language model writes candidate observations and a draft. This module is
deliberately model-agnostic: it checks the frozen source boundary and renders a
readable view without copying internal source pointers into Writer context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from personaforge.persona.context_budget import estimate_tokens


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("documents", "selected", "sample"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError("sample_manifest.json must list selected documents")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_and_render_pilot(
    *, index_dir: Path, pilot_dir: Path, eval_manifest: Path, require_full_train: bool = False,
) -> dict[str, Any]:
    """Validate a Luna-authored pilot and write a source-free readable view.

    Validation failures are reported and leave any previous readable view alone.
    This is structural/source validation, not proof that a paraphrase entails
    the cited excerpt; the latter needs the separate semantic review.
    """

    index_dir, pilot_dir = Path(index_dir), Path(pilot_dir)
    parent_rows = _read_jsonl(index_dir / "parents.jsonl")
    parents = {row["doc_id"]: row for row in parent_rows}
    evaluation = json.loads(Path(eval_manifest).read_text(encoding="utf-8"))
    excluded = set(evaluation["excluded_parent_ids"])
    sample_manifest = json.loads((pilot_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    selected = _sample_rows(sample_manifest)
    outcomes = _read_jsonl(pilot_dir / "source_outcomes.jsonl")
    observations = _read_jsonl(pilot_dir / "observations.jsonl")
    draft = json.loads((pilot_dir / "wiki_draft.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if require_full_train and draft.get("source_observations_sha256") != _sha256_json(observations):
        errors.append("full Wiki draft is not bound to current reviewed observations")

    sample_ids = [str(row.get("doc_id") or "") for row in selected]
    train_ids = set(parents) - excluded
    if len(sample_ids) != len(set(sample_ids)) or not sample_ids:
        errors.append("sample document IDs must be nonempty and unique")
    if require_full_train:
        if not isinstance(sample_manifest, dict) or sample_manifest.get("author") != evaluation.get("author"):
            errors.append("full Wiki author differs from frozen evaluation author")
        if set(sample_ids) != train_ids:
            errors.append(f"full Wiki must cover every frozen training document: expected {len(train_ids)}, got {len(set(sample_ids))}")
        if evaluation.get("source_parents_sha256") != _sha256_json(parent_rows):
            errors.append("source parent index differs from frozen evaluation manifest")
        if evaluation.get("excluded_parent_ids_sha256") != _sha256_json(sorted(excluded)):
            errors.append("excluded document IDs differ from frozen evaluation manifest")
        if evaluation.get("counts", {}).get("train_parents") != len(train_ids):
            errors.append("frozen evaluation training count differs from current index")
    for row in selected:
        doc_id = str(row.get("doc_id") or "")
        parent = parents.get(doc_id)
        if parent is None or doc_id in excluded:
            errors.append(f"sample is absent or excluded from training: {doc_id}")
            continue
        if row.get("kind") != parent.get("kind"):
            errors.append(f"sample kind differs from source: {doc_id}")
        source_hash = hashlib.sha256(str(parent.get("text") or "").encode("utf-8")).hexdigest()
        if (row.get("text_sha256") or row.get("sha256")) != source_hash:
            errors.append(f"sample text hash differs from source: {doc_id}")

    outcome_ids = [str(row.get("doc_id") or "") for row in outcomes]
    outcome_status_by_doc = {str(row.get("doc_id") or ""): row.get("status") for row in outcomes}
    if set(outcome_ids) != set(sample_ids) or len(outcome_ids) != len(sample_ids):
        errors.append("source outcomes must cover each sampled document exactly once")
    unreviewed_outcome_ids = [str(row.get("doc_id") or "") for row in outcomes if row.get("reviewed") is not True]
    if require_full_train and unreviewed_outcome_ids:
        errors.append(f"full Wiki has {len(unreviewed_outcome_ids)} sources without completed semantic review")
    valid_statuses = {"observed", "no_persona_signal", "uncertain"}
    for row in outcomes:
        if row.get("status") not in valid_statuses:
            errors.append(f"invalid source outcome: {row.get('doc_id')}")

    observation_ids: set[str] = set()
    accepted: dict[str, dict[str, Any]] = {}
    observations_by_doc: dict[str, int] = {}
    for row in observations:
        observation_id = str(row.get("observation_id") or "")
        doc_id = str(row.get("doc_id") or "")
        if not observation_id or observation_id in observation_ids:
            errors.append(f"duplicate or empty observation ID: {observation_id}")
        observation_ids.add(observation_id)
        if doc_id not in sample_ids or doc_id not in parents:
            errors.append(f"observation cites a non-sampled document: {observation_id}")
        excerpt = row.get("excerpt")
        if not isinstance(excerpt, str) or not excerpt or excerpt not in str(parents.get(doc_id, {}).get("text") or ""):
            errors.append(f"observation excerpt is not verbatim source text: {observation_id}")
        if row.get("type") not in {"stance", "style", "experience", "uncertain"}:
            errors.append(f"invalid observation type: {observation_id}")
        if row.get("attribution") not in {"self", "quoted", "uncertain"}:
            errors.append(f"invalid attribution: {observation_id}")
        if not isinstance(row.get("observation"), str) or not row["observation"].strip():
            errors.append(f"empty observation: {observation_id}")
        observations_by_doc[doc_id] = observations_by_doc.get(doc_id, 0) + 1
        accepted[observation_id] = row
    for row in outcomes:
        doc_id = str(row.get("doc_id") or "")
        if row.get("status") == "observed" and not observations_by_doc.get(doc_id):
            errors.append(f"observed source has no observations: {doc_id}")
        if row.get("status") == "no_persona_signal" and observations_by_doc.get(doc_id):
            errors.append(f"no-signal source has observations: {doc_id}")

    used_observation_ids: set[str] = set()

    def check_item(item: Any, label: str) -> None:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            errors.append(f"{label} needs nonempty text")
            return
        refs = item.get("observation_ids")
        if not isinstance(refs, list) or not refs:
            errors.append(f"{label} needs observation IDs")
            return
        for ref in refs:
            observation = accepted.get(ref)
            if observation is None:
                errors.append(f"{label} cites unknown observation: {ref}")
            elif observation.get("type") == "uncertain":
                errors.append(f"{label} promotes an uncertain observation: {ref}")
            elif observation.get("attribution") != "self":
                errors.append(f"{label} promotes a non-self observation: {ref}")
            elif outcome_status_by_doc.get(str(observation.get("doc_id") or "")) != "observed":
                errors.append(f"{label} promotes an observation from an uncertain source: {ref}")
            else:
                used_observation_ids.add(ref)

    for i, item in enumerate(draft.get("overview", [])):
        check_item(item, f"overview {i}")
    topic_ids: list[str] = []
    for topic in draft.get("topics", []):
        topic_id = str(topic.get("topic_id") or "")
        topic_ids.append(topic_id)
        if not topic_id or not str(topic.get("title") or "").strip():
            errors.append("topic needs an ID and title")
        section_path = topic.get("section_path", [])
        if not isinstance(section_path, list) or any(not isinstance(part, str) or not part.strip() for part in section_path) or len(section_path) > 3:
            errors.append(f"topic {topic_id} has an invalid section path")
        for i, claim in enumerate(topic.get("claims", [])):
            check_item(claim, f"topic {topic_id} claim {i}")
    if not topic_ids or len(topic_ids) != len(set(topic_ids)):
        errors.append("topic IDs must be nonempty and unique")
    for i, item in enumerate(draft.get("style", [])):
        check_item(item, f"style {i}")
    if not isinstance(draft.get("subject_name"), str) or not draft["subject_name"].strip():
        errors.append("draft needs a subject_name")

    report = {
        "schema_version": "persona_wiki_v2_pilot_validation.v1",
        "sample_count": len(sample_ids),
        "train_pool_count": len(train_ids),
        "full_train_coverage": set(sample_ids) == train_ids,
        "frozen_dataset_sha256": evaluation.get("dataset_sha256"),
        "source_parents_sha256": _sha256_json(parent_rows),
        "wiki_draft_sha256": _sha256_json(draft),
        "wiki_draft_file_sha256": hashlib.sha256((pilot_dir / "wiki_draft.json").read_bytes()).hexdigest(),
        "observations_sha256": _sha256_json(observations),
        "observation_count": len(observations),
        "eligible_observation_count": sum(
            row.get("attribution") == "self" and row.get("type") != "uncertain"
            for row in observations
        ),
        "referenced_observation_count": len(used_observation_ids),
        "referenced_source_count": len({accepted[ref]["doc_id"] for ref in used_observation_ids}),
        "unreferenced_eligible_observation_count": sum(
            row.get("attribution") == "self" and row.get("type") != "uncertain"
            and row.get("observation_id") not in used_observation_ids
            for row in observations
        ),
        "topic_count": len(topic_ids),
        "source_outcome_counts": {status: sum(row.get("status") == status for row in outcomes) for status in sorted(valid_statuses)},
        "reviewed_source_count": len(outcomes) - len(unreviewed_outcome_ids),
        "unreviewed_source_count": len(unreviewed_outcome_ids),
        "errors": errors,
        "note": "Source and structure checks cannot prove semantic entailment; review the changed claims separately.",
    }
    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if errors:
        return report

    lines = [f"# Persona Wiki：{draft['subject_name']}", "", "## 人物概览"]
    lines.extend(f"- {item['text']}" for item in draft["overview"])
    current_path: list[str] = []
    for topic in draft["topics"]:
        section_path = topic.get("section_path", [])
        shared = 0
        while shared < min(len(current_path), len(section_path)) and current_path[shared] == section_path[shared]:
            shared += 1
        for depth, name in enumerate(section_path[shared:], start=shared):
            lines.extend(("", f"{'#' * (depth + 2)} {name}"))
        current_path = section_path
        lines.extend(("", f"{'#' * (len(section_path) + 2)} {topic['title']}"))
        if topic.get("scope"):
            lines.append(str(topic["scope"]))
        for claim in topic["claims"]:
            lines.append(f"- {claim['text']}")
            if claim.get("condition"):
                lines.append(f"  - 适用情境：{claim['condition']}")
    lines.extend(("", "## 表达风格"))
    lines.extend(f"- {item['text']}" for item in draft["style"])
    readable = "\n".join(lines) + "\n"
    if any(doc_id in readable for doc_id in sample_ids) or re.search(r"\bo\d{3,}\b", readable):
        report["errors"].append("Writer view contains an internal source or observation ID")
        (pilot_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    report["writer_estimated_tokens"] = estimate_tokens(readable)
    report["writer_view_sha256"] = hashlib.sha256(readable.encode("utf-8")).hexdigest()
    (pilot_dir / "persona_wiki_readable.md").write_text(readable, encoding="utf-8")
    (pilot_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and render a Persona Wiki pilot")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--require-full-train", action="store_true", help="Reject incomplete or mismatched frozen training coverage")
    args = parser.parse_args(argv)
    report = validate_and_render_pilot(
        index_dir=args.index_dir, pilot_dir=args.pilot_dir, eval_manifest=args.eval_manifest,
        require_full_train=args.require_full_train,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
