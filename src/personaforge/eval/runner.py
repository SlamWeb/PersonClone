"""Run leak-safe persona generations over a prepared temporal dataset."""

from __future__ import annotations

import json
import hashlib
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from personaforge.eval.dataset import load_dataset, load_dataset_manifest, sha256_json
from personaforge.eval.strong_style import build_content_plan, select_expression_hits
from personaforge.ingest.embeddings import TextEncoder
from personaforge.ingest.query_understanding import build_grounded_query_plan, plan_to_trace
from personaforge.ingest.retrieve import ParentHit, RetrieveResult, retrieve_parents, retrieve_parents_for_queries
from personaforge.llm import JsonChatClient
from personaforge.persona.narrative import (
    NarrativeSchema,
    load_narrative_schema,
    load_narrative_schema_for_index,
)
from personaforge.persona.pack import PersonaPack, load_persona_pack, load_persona_pack_for_index
from personaforge.persona.writer import (
    RAG_MAGIC_IF_PROMPT_VERSION,
    RAG_MAGIC_IF_V2_PROMPT_VERSION,
    PURE_ROLE_RAG10_PROMPT_VERSION,
    generate_answer,
    writer_system_prompt,
)


RUN_SCHEMA_VERSION = "personaforge.eval.run.v0"


@dataclass(frozen=True, slots=True)
class EvalRunConfig:
    author: str
    dataset_path: Path
    split: str
    run_name: str
    out_dir: Path
    query_mode: str = "grounded"
    writer_prompt: str = "strong_identity"
    child_top_k: int = 100
    per_query_parent_k: int = 30
    parent_top_k: int = 20
    writer_context_top_k: int = 20
    content_context_top_k: int = 5
    style_context_top_k: int = 3
    max_search_results: int = 5
    temperature: float = 0.85
    max_tokens: int = 1600
    limit: int | None = None
    persona_pack_path: Path | None = None
    narrative_schema_path: Path | None = None
    method_id: str | None = None
    display_name: str | None = None
    description: str | None = None
    parent_method: str | None = None
    prompt_version: str | None = None


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    run_dir: Path
    manifest_path: Path
    runs_path: Path
    summary_path: Path
    item_count: int


def run_temporal_eval(
    config: EvalRunConfig,
    *,
    index_dir: Path,
    qdrant_path: Path,
    encoder: TextEncoder,
    llm: JsonChatClient,
) -> EvalRunResult:
    if config.parent_top_k <= 0:
        raise ValueError("parent_top_k must be positive.")
    if config.writer_context_top_k <= 0:
        raise ValueError("writer_context_top_k must be positive.")
    if config.writer_context_top_k > config.parent_top_k:
        raise ValueError(
            "writer_context_top_k cannot exceed parent_top_k; retrieve more parents first."
        )
    if config.writer_prompt in {"strong_style_v1", "strong_style_2pass_v1"}:
        if config.content_context_top_k <= 0 or config.style_context_top_k <= 0:
            raise ValueError("strong style context sizes must be positive.")
        if config.content_context_top_k + config.style_context_top_k > config.parent_top_k:
            raise ValueError(
                "strong style content and expression contexts cannot exceed parent_top_k."
            )
    dataset = [row for row in load_dataset(config.dataset_path) if row.get("split") == config.split]
    if config.limit is not None:
        dataset = dataset[: config.limit]
    if not dataset:
        raise ValueError(f"No {config.split!r} items found in {config.dataset_path}.")

    dataset_manifest = load_dataset_manifest(config.dataset_path)
    excluded_parent_ids = set(str(value) for value in dataset_manifest.get("excluded_parent_ids", []))
    if config.writer_prompt == "persona_pack" and config.persona_pack_path is not None:
        persona_pack = load_persona_pack(
            config.persona_pack_path,
            parent_store_path=index_dir / "parents.jsonl",
            verify_evidence=True,
        )
    elif config.writer_prompt == "persona_pack":
        persona_pack = load_persona_pack_for_index(index_dir, required=True)
    else:
        persona_pack = None
    if persona_pack is not None:
        assert_persona_pack_no_leak(persona_pack, excluded_parent_ids)
    if config.writer_prompt == "mrprompt" and config.narrative_schema_path is not None:
        narrative_schema = load_narrative_schema(
            config.narrative_schema_path,
            parent_store_path=index_dir / "parents.jsonl",
            verify_evidence=True,
        )
    elif config.writer_prompt == "mrprompt":
        narrative_schema = load_narrative_schema_for_index(index_dir, required=True)
    else:
        narrative_schema = None
    if narrative_schema is not None:
        assert_narrative_schema_no_leak(narrative_schema, excluded_parent_ids)
    run_dir = config.out_dir / "runs" / config.run_name
    if run_dir.exists():
        raise FileExistsError(f"Eval run already exists: {run_dir}. Choose a new --run-name.")
    items_dir = run_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=False)
    runs_path = run_dir / "runs.jsonl"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.md"
    run_started_at = utc_now()
    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "running",
        "started_at": run_started_at,
        "dataset_path": str(config.dataset_path),
        "dataset_sha256": dataset_manifest.get("dataset_sha256"),
        "excluded_parent_ids_sha256": dataset_manifest.get("excluded_parent_ids_sha256"),
        "excluded_parent_count": len(excluded_parent_ids),
        "config": config_to_dict(config),
        "method_id": config.method_id or config.run_name,
        "display_name": config.display_name or config.run_name,
        "description": config.description or "",
        "parent_method": config.parent_method,
        "prompt_version": config.prompt_version or prompt_version_for(config.writer_prompt),
        "prompt_sha256": prompt_sha256_for(config.writer_prompt),
        "git": git_revision(),
        "writer_model": str(getattr(llm, "model", type(llm).__name__)),
        "embedding_model": type(encoder).__name__,
        "persona_pack": {
            "pack_id": persona_pack.pack_id,
            "sha256": persona_pack.sha256,
            "claim_count": persona_pack.claim_count,
        } if persona_pack else None,
        "narrative_schema": {
            "schema_id": narrative_schema.schema_id,
            "sha256": narrative_schema.sha256,
            "facet_count": narrative_schema.facet_count,
            "evidence_count": narrative_schema.evidence_count,
        } if narrative_schema else None,
    }
    write_json(run_manifest, manifest_path)

    records: list[dict[str, Any]] = []
    try:
        for item in dataset:
            record = run_eval_item(
                item,
                config=config,
                index_dir=index_dir,
                qdrant_path=qdrant_path,
                excluded_parent_ids=excluded_parent_ids,
                encoder=encoder,
                llm=llm,
                persona_pack=persona_pack,
                narrative_schema=narrative_schema,
            )
            records.append(record)
            append_jsonl(record, runs_path)
            write_item_markdown(record, items_dir / f"{item['item_id']}.md")
    except Exception as exc:
        run_manifest.update({"status": "failed", "finished_at": utc_now(), "error": error_payload(exc)})
        write_json(run_manifest, manifest_path)
        raise

    run_manifest.update(
        {
            "status": "completed",
            "finished_at": utc_now(),
            "item_count": len(records),
            "run_sha256": sha256_json(records),
        }
    )
    write_json(run_manifest, manifest_path)
    write_summary(config, records, summary_path)
    return EvalRunResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        runs_path=runs_path,
        summary_path=summary_path,
        item_count=len(records),
    )


def run_eval_item(
    item: dict[str, Any],
    *,
    config: EvalRunConfig,
    index_dir: Path,
    qdrant_path: Path,
    excluded_parent_ids: set[str],
    encoder: TextEncoder,
    llm: JsonChatClient,
    persona_pack: PersonaPack | None = None,
    narrative_schema: NarrativeSchema | None = None,
) -> dict[str, Any]:
    query = str(item["query"])
    started_at = perf_counter()
    query_trace: dict[str, Any] | None = None
    objective_background = ""
    if config.query_mode == "grounded":
        understanding_started_at = perf_counter()
        plan = build_grounded_query_plan(
            query,
            llm=llm,
            max_results_per_query=config.max_search_results,
        )
        query_trace = plan_to_trace(plan)
        objective_background = plan.transform.objective_background
        understanding_duration_ms = elapsed_ms(understanding_started_at)
        retrieval_started_at = perf_counter()
        retrieve_result = retrieve_parents_for_queries(
            query,
            plan.transform.retrieval_queries,
            author=config.author,
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            child_top_k=config.child_top_k,
            per_query_parent_k=config.per_query_parent_k,
            parent_top_k=config.parent_top_k,
            exclude_parent_ids=excluded_parent_ids,
        )
    elif config.query_mode == "raw":
        retrieval_started_at = perf_counter()
        retrieve_result = retrieve_parents(
            query,
            author=config.author,
            index_dir=index_dir,
            qdrant_path=qdrant_path,
            encoder=encoder,
            child_top_k=config.child_top_k,
            parent_top_k=config.parent_top_k,
            exclude_parent_ids=excluded_parent_ids,
        )
        understanding_duration_ms = 0
    else:
        raise ValueError(f"Unknown query mode: {config.query_mode}")

    assert_no_leak(retrieve_result, excluded_parent_ids)
    retrieval_duration_ms = elapsed_ms(retrieval_started_at)
    content_hits: list[ParentHit] | None = None
    style_hits: list[ParentHit] | None = None
    content_plan: dict[str, str] | None = None
    style_selection_trace: dict[str, Any] | None = None
    content_plan_trace: dict[str, Any] | None = None
    if config.writer_prompt in {"strong_style_v1", "strong_style_2pass_v1"}:
        content_hits = retrieve_result.parents[: config.content_context_top_k]
        style_candidates = retrieve_result.parents[config.content_context_top_k :]
        style_hits, style_selection_trace = select_expression_hits(
            query=query,
            objective_background=objective_background,
            candidates=style_candidates,
            llm=llm,
            top_k=config.style_context_top_k,
        )
        if config.writer_prompt == "strong_style_2pass_v1":
            content_plan, content_plan_trace = build_content_plan(
                query=query,
                objective_background=objective_background,
                content_hits=content_hits,
                llm=llm,
            )
        writer_parent_hits = _merge_hits(content_hits, style_hits)
    else:
        writer_parent_hits = retrieve_result.parents[: config.writer_context_top_k]
    if not writer_parent_hits:
        raise RuntimeError("Retrieval returned no parent documents for the writer context.")
    generation_started_at = perf_counter()
    answer_result = generate_answer(
        query=query,
        parent_hits=writer_parent_hits,
        llm=llm,
        objective_background=objective_background,
        writer_prompt=config.writer_prompt,
        persona_pack=persona_pack,
        narrative_schema=narrative_schema,
        content_hits=content_hits,
        style_hits=style_hits,
        content_plan=content_plan,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    generation_duration_ms = elapsed_ms(generation_started_at)
    return {
        "item_id": item["item_id"],
        "split": item["split"],
        "parent_id": item["parent_id"],
        "created_at": item["created_at"],
        "query": query,
        "gold_answer": item["gold_answer"],
        "answer": answer_result.answer,
        "status": "completed",
        "trace": {
            "query_understanding": query_trace,
            "objective_background": objective_background,
            "retrieval": serialize_retrieve_result(retrieve_result),
            "writer": {
                "variant": answer_result.writer_prompt,
                "persona_pack_id": answer_result.persona_pack_id,
                "persona_pack_sha256": answer_result.persona_pack_sha256,
                "narrative_schema_id": answer_result.narrative_schema_id,
                "narrative_schema_sha256": answer_result.narrative_schema_sha256,
                "retrieved_parent_top_k": config.parent_top_k,
                "context_parent_top_k": config.writer_context_top_k,
                "context_parent_titles": answer_result.parent_titles,
                "content_context_parent_ids": [hit.parent_id for hit in content_hits or []],
                "style_context_parent_ids": [hit.parent_id for hit in style_hits or []],
                "style_selection": style_selection_trace,
                "content_plan": content_plan_trace,
                "message_characters": [
                    {"role": message["role"], "characters": len(message["content"])}
                    for message in answer_result.messages
                ],
            },
            "timing": {
                "query_understanding_ms": understanding_duration_ms,
                "retrieval_ms": retrieval_duration_ms,
                "generation_ms": generation_duration_ms,
                "style_selection_ms": (style_selection_trace or {}).get("duration_ms", 0),
                "content_plan_ms": (content_plan_trace or {}).get("duration_ms", 0),
                "total_ms": elapsed_ms(started_at),
            },
        },
    }


def assert_no_leak(result: RetrieveResult, excluded_parent_ids: set[str]) -> None:
    leaked = {
        hit.parent_id
        for hits in result.routes.values()
        for hit in hits
        if hit.parent_id in excluded_parent_ids
    }
    leaked.update(hit.parent_id for hit in result.parents if hit.parent_id in excluded_parent_ids)
    if leaked:
        raise RuntimeError(f"Evaluation leakage: excluded parent(s) retrieved: {sorted(leaked)}")


def _merge_hits(*groups: list[ParentHit] | None) -> list[ParentHit]:
    merged: list[ParentHit] = []
    seen: set[str] = set()
    for group in groups:
        for hit in group or []:
            if hit.parent_id in seen:
                continue
            seen.add(hit.parent_id)
            merged.append(hit)
    return merged


def assert_persona_pack_no_leak(pack: PersonaPack, excluded_parent_ids: set[str]) -> None:
    evidence_ids = {
        evidence.doc_id
        for section in ("response_strategy", "worldview", "reasoning", "voice")
        for claim in pack.claims_for(section)
        for evidence in (*claim.evidence, *claim.counterevidence)
    }
    leaked = evidence_ids & excluded_parent_ids
    if leaked:
        raise RuntimeError(
            f"Evaluation leakage: Persona Pack cites excluded parent(s): {sorted(leaked)}"
        )


def assert_narrative_schema_no_leak(
    schema: NarrativeSchema,
    excluded_parent_ids: set[str],
) -> None:
    evidence_ids = {
        evidence.doc_id
        for facet in schema.scene_facets
        for evidence in facet.source_evidence
    }
    leaked = evidence_ids & excluded_parent_ids
    if leaked:
        raise RuntimeError(
            f"Evaluation leakage: Narrative Schema cites excluded parent(s): {sorted(leaked)}"
        )


def serialize_retrieve_result(result: RetrieveResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "collection_name": result.collection_name,
        "child_top_k": result.child_top_k,
        "parent_top_k": result.parent_top_k,
        "retrieval_queries": [
            {"route": item.route, "query": item.query}
            for item in result.retrieval_queries
        ],
        "routes": {
            route: [serialize_child_hit(hit) for hit in hits]
            for route, hits in result.routes.items()
        },
        "parents": [serialize_parent_hit(hit) for hit in result.parents],
    }


def serialize_child_hit(hit: Any) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "score": hit.score,
        "node_id": hit.node_id,
        "parent_id": hit.parent_id,
        "node_type": hit.node_type,
        "title": hit.title,
        "path": hit.path,
        "route": hit.route,
    }


def serialize_parent_hit(hit: ParentHit) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "score": hit.score,
        "parent_id": hit.parent_id,
        "title": hit.title,
        "path": hit.path,
        "first_hits": [serialize_child_hit(child) for child in hit.first_hits],
    }


def write_item_markdown(record: dict[str, Any], path: Path) -> None:
    sources = record["trace"]["retrieval"]["parents"]
    source_lines = "\n".join(
        f"{source['rank']}. {source['title'] or source['parent_id']}\n   - 命中："
        + "；".join(f"{hit['route']} #{hit['rank']} {hit['node_type']}" for hit in source["first_hits"])
        for source in sources
    )
    content = f"""# {record['item_id']}\n\n## 原问题\n\n{record['query']}\n\n## 原作者回答（Gold）\n\n{record['gold_answer']}\n\n## PersonaForge 回答\n\n{record['answer']}\n\n## 检索到的作者历史表达\n\n{source_lines}\n\n## 人工记录\n\n- 题意理解：\n- 观点与作者是否一致：\n- 语言与语气：\n- 论证方式：\n- AI 感：\n- 备注：\n\n## 运行信息\n\n- writer：{record['trace']['writer']['variant']}\n- 总耗时：{record['trace']['timing']['total_ms']} ms\n- 目标原回答 parent：{record['parent_id']}（已由检索过滤器排除）\n"""
    path.write_text(content, encoding="utf-8", newline="\n")


def write_summary(config: EvalRunConfig, records: list[dict[str, Any]], path: Path) -> None:
    average_total_ms = round(sum(item["trace"]["timing"]["total_ms"] for item in records) / len(records))
    content = f"""# {config.run_name}\n\n- split：{config.split}\n- 题目数：{len(records)}\n- query mode：{config.query_mode}\n- writer：{config.writer_prompt}\n- 平均总耗时：{average_total_ms} ms\n\n本轮不包含 LLM Judge。请在 `items/` 中阅读原回答、生成回答和检索材料，并填写人工记录。\n"""
    path.write_text(content, encoding="utf-8", newline="\n")


def config_to_dict(config: EvalRunConfig) -> dict[str, Any]:
    value = asdict(config)
    value["dataset_path"] = str(config.dataset_path)
    value["out_dir"] = str(config.out_dir)
    value["persona_pack_path"] = (
        str(config.persona_pack_path) if config.persona_pack_path is not None else None
    )
    value["narrative_schema_path"] = (
        str(config.narrative_schema_path) if config.narrative_schema_path is not None else None
    )
    return value


def prompt_version_for(writer_prompt: str) -> str:
    if writer_prompt == "rag_magic_if":
        return RAG_MAGIC_IF_PROMPT_VERSION
    if writer_prompt == "rag_magic_if_v2":
        return RAG_MAGIC_IF_V2_PROMPT_VERSION
    if writer_prompt == "strong_style_v1":
        from personaforge.persona.writer import STRONG_STYLE_PROMPT_VERSION

        return STRONG_STYLE_PROMPT_VERSION
    if writer_prompt == "strong_style_2pass_v1":
        from personaforge.persona.writer import STRONG_STYLE_2PASS_PROMPT_VERSION

        return STRONG_STYLE_2PASS_PROMPT_VERSION
    if writer_prompt == "pure_role_rag10_v1":
        return PURE_ROLE_RAG10_PROMPT_VERSION
    return writer_prompt


def prompt_sha256_for(writer_prompt: str) -> str:
    return hashlib.sha256(writer_system_prompt(writer_prompt).encode("utf-8")).hexdigest()


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(value: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def error_payload(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)[:1000]}


def git_revision() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": None}
