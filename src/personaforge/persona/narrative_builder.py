"""Build evidence-backed NarrativeSchema files from a bounded author sample.

The builder deliberately reuses the existing parent store, BGE-M3 encoder and
title clustering implementation.  It never writes the selected full articles
to the resulting schema; full text is sent only in the transient LLM request,
while the saved artifact keeps short, auditable excerpts and document IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from personaforge.ingest.embeddings import TextEncoder
from personaforge.llm import JsonChatClient
from personaforge.persona.narrative import (
    NARRATIVE_SCHEMA_FILENAME,
    NarrativeSchema,
    load_narrative_schema,
)
from personaforge.persona.routing_profile import (
    TitleEvidence,
    clean_title_evidence,
    cluster_title_embeddings,
    compute_corpus_version,
    l2_normalize,
)


DEFAULT_SCHEMA_CANDIDATES_PER_CLUSTER = 5
DEFAULT_SCHEMA_MAX_DOCUMENTS = 72
DEFAULT_SCHEMA_MAX_PAYLOAD_CHARS = 2_000_000
DEFAULT_SCHEMA_OUTPUT_TOKENS = 8_192


class NarrativeSchemaBuildError(RuntimeError):
    """Raised when a generated schema cannot be made evidence-backed."""


@dataclass(frozen=True, slots=True)
class NarrativeSchemaSelectionConfig:
    candidates_per_cluster: int = DEFAULT_SCHEMA_CANDIDATES_PER_CLUSTER
    max_documents: int = DEFAULT_SCHEMA_MAX_DOCUMENTS
    max_payload_chars: int = DEFAULT_SCHEMA_MAX_PAYLOAD_CHARS
    output_tokens: int = DEFAULT_SCHEMA_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        if self.candidates_per_cluster < 1:
            raise ValueError("candidates_per_cluster must be positive")
        if self.max_documents < 1:
            raise ValueError("max_documents must be positive")
        if self.max_payload_chars < 1000:
            raise ValueError("max_payload_chars must be at least 1000")
        if self.output_tokens < 512:
            raise ValueError("output_tokens must be at least 512")

    def as_dict(self) -> dict[str, int]:
        return {
            "candidates_per_cluster": self.candidates_per_cluster,
            "max_documents": self.max_documents,
            "max_payload_chars": self.max_payload_chars,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True, slots=True)
class SelectedNarrativeDocument:
    cluster_id: str
    rank: int
    selection_score: float
    evidence: TitleEvidence
    row: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NarrativeSchemaBuildResult:
    status: str
    schema: NarrativeSchema
    schema_path: str
    selected_document_count: int
    cluster_count: int
    llm_calls: int


class NarrativeSchemaBuilder:
    """Generate one author-level schema from representative parent documents."""

    def __init__(
        self,
        *,
        data_dir: Path,
        author_id: str,
        encoder: TextEncoder,
        llm: JsonChatClient,
        model_name: str = "BAAI/bge-m3",
        embedding_batch_size: int = 12,
        distance_threshold: float = 0.32,
        min_cluster_size: int = 3,
        selection: NarrativeSchemaSelectionConfig | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.author_id = author_id
        self.encoder = encoder
        self.llm = llm
        self.model_name = model_name
        self.embedding_batch_size = embedding_batch_size
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.selection = selection or NarrativeSchemaSelectionConfig()

    @property
    def author_dir(self) -> Path:
        return self.data_dir / "authors" / "zhihu" / self.author_id

    @property
    def index_dir(self) -> Path:
        return self.author_dir / "index"

    @property
    def parent_path(self) -> Path:
        return self.index_dir / "parents.jsonl"

    @property
    def schema_path(self) -> Path:
        return self.author_dir / NARRATIVE_SCHEMA_FILENAME

    @property
    def config_hash(self) -> str:
        config = {
            "builder_version": 2,
            "model_name": self.model_name,
            "distance_threshold": self.distance_threshold,
            "min_cluster_size": self.min_cluster_size,
            "embedding_batch_size": self.embedding_batch_size,
            "selection": self.selection.as_dict(),
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:20]

    def build(self, *, force: bool = False) -> NarrativeSchemaBuildResult:
        rows = self._load_parent_rows()
        eligible_rows = [row for row in rows if str(row.get("kind") or "").lower() in {"answer", "article"}]
        corpus_version = compute_corpus_version(eligible_rows)
        if not force:
            cached = self._load_reusable_schema(corpus_version)
            if cached is not None:
                selected_count = int(cached.corpus_snapshot.get("selected_document_count") or 0)
                cluster_count = int(cached.corpus_snapshot.get("cluster_count") or 0)
                return NarrativeSchemaBuildResult(
                    status="reused",
                    schema=cached,
                    schema_path=str(self.schema_path),
                    selected_document_count=selected_count,
                    cluster_count=cluster_count,
                    llm_calls=0,
                )

        title_evidence = clean_title_evidence(eligible_rows)
        if not title_evidence:
            raise NarrativeSchemaBuildError(f"No answer/article titles found for author {self.author_id!r}.")
        title_embeddings = self.encoder.encode_texts(
            [item.title for item in title_evidence],
            batch_size=self.embedding_batch_size,
        )
        dense_vectors = [l2_normalize(item.dense) for item in title_embeddings]
        clusters = cluster_title_embeddings(
            title_evidence,
            dense_vectors,
            distance_threshold=self.distance_threshold,
            min_cluster_size=self.min_cluster_size,
        )
        selected = select_representative_documents(
            clusters,
            eligible_rows,
            candidates_per_cluster=self.selection.candidates_per_cluster,
            max_documents=self.selection.max_documents,
        )
        if not selected:
            raise NarrativeSchemaBuildError("Representative selection produced no documents.")

        payload = [_document_payload(item) for item in selected]
        result, llm_calls = self._generate_schema_payload(payload)
        schema_payload = self._assemble_schema(
            result,
            corpus_version=corpus_version,
            rows=eligible_rows,
            clusters=clusters,
            selected=selected,
        )
        schema = self._write_and_validate(schema_payload)
        return NarrativeSchemaBuildResult(
            status="rebuilt",
            schema=schema,
            schema_path=str(self.schema_path),
            selected_document_count=len(selected),
            cluster_count=len(clusters),
            llm_calls=llm_calls,
        )

    def _load_parent_rows(self) -> list[dict[str, Any]]:
        if not self.parent_path.exists():
            raise FileNotFoundError(self.parent_path)
        rows: list[dict[str, Any]] = []
        for line in self.parent_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def _load_reusable_schema(self, corpus_version: str) -> NarrativeSchema | None:
        if not self.schema_path.exists():
            return None
        try:
            schema = load_narrative_schema(
                self.schema_path,
                parent_store_path=self.parent_path,
                verify_evidence=True,
            )
        except Exception:
            return None
        snapshot = schema.corpus_snapshot
        source = schema.source
        if snapshot.get("corpus_version") != corpus_version:
            return None
        if source.get("builder_config_hash") != self.config_hash:
            return None
        return schema

    def _generate_schema_payload(self, documents: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
        serialized = json.dumps(documents, ensure_ascii=False)
        if len(serialized) <= self.selection.max_payload_chars:
            return self._call_full_schema(documents), 1

        # Large authors are handled in evidence-preserving batches.  Each
        # batch produces candidate facets; the final call merges only those
        # compact, cited candidates rather than silently truncating articles.
        batches = _partition_documents(documents, self.selection.max_payload_chars)
        partials: list[dict[str, Any]] = []
        calls = 0
        for batch in batches:
            partials.append(self._call_partial_facets(batch))
            calls += 1
        merged = self._call_merge_schema(partials)
        return merged, calls + 1

    def _call_full_schema(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": _schema_system_prompt(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "author_id": self.author_id,
                        "documents": documents,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            result = self.llm.complete_json(
                messages,
                temperature=0.0,
                max_tokens=self.selection.output_tokens,
            )
        except Exception as exc:
            raise NarrativeSchemaBuildError("NarrativeSchema LLM generation failed.") from exc
        if not isinstance(result, dict):
            raise NarrativeSchemaBuildError("NarrativeSchema LLM response must be a JSON object.")
        return result

    def _call_partial_facets(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": _partial_schema_system_prompt(),
            },
            {
                "role": "user",
                "content": json.dumps({"documents": documents}, ensure_ascii=False),
            },
        ]
        try:
            result = self.llm.complete_json(
                messages,
                temperature=0.0,
                max_tokens=self.selection.output_tokens,
            )
        except Exception as exc:
            raise NarrativeSchemaBuildError("NarrativeSchema batch extraction failed.") from exc
        if not isinstance(result, dict):
            raise NarrativeSchemaBuildError("NarrativeSchema batch response must be a JSON object.")
        return result

    def _call_merge_schema(self, partials: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": _schema_merge_system_prompt(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"author_id": self.author_id, "partial_evidence_summaries": partials},
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            result = self.llm.complete_json(
                messages,
                temperature=0.0,
                max_tokens=self.selection.output_tokens,
            )
        except Exception as exc:
            raise NarrativeSchemaBuildError("NarrativeSchema merge failed.") from exc
        if not isinstance(result, dict):
            raise NarrativeSchemaBuildError("NarrativeSchema merge response must be a JSON object.")
        return result

    def _assemble_schema(
        self,
        result: dict[str, Any],
        *,
        corpus_version: str,
        rows: list[dict[str, Any]],
        clusters: list[dict[str, Any]],
        selected: list[SelectedNarrativeDocument],
    ) -> dict[str, Any]:
        allowed = {item.evidence.doc_id: item for item in selected}
        identity = _object_or_default(result.get("identity"), {"public_identity": self.author_id})
        global_summary = _required_result_text(result, "global_summary")
        core_traits = _required_result_list(result, "core_traits")
        facets = _normalize_facets(result.get("scene_facets"), allowed)
        if not facets:
            raise NarrativeSchemaBuildError("NarrativeSchema contains no evidence-backed scene_facets.")
        generation_policy = _object_or_default(result.get("generation_policy"), {})
        generation_policy.setdefault(
            "selection",
            "完整 Schema 可见，但只把与当前问题相关的视角作为软参考；本轮 RAG 原文优先。",
        )
        generation_policy.setdefault(
            "anchoring",
            "只约束观察姿态和表达边界，不替代当前原文的具体事实和立场。",
        )
        generation_policy.setdefault(
            "bounding",
            [
                "不得把 Schema 当作事实来源。",
                "不得复制代表文章的原句或固定模板。",
                "当前 RAG 没有支持时忽略相关视角。",
            ],
        )
        generation_policy.setdefault(
            "enacting",
            "自然参考相关视角，不解释 Schema，不输出证据 ID。",
        )
        selected_doc_ids = [item.evidence.doc_id for item in selected]
        return {
            "schema_version": 1,
            "schema_id": f"{self.author_id}.narrative.{corpus_version}",
            "author_id": self.author_id,
            "display_name": str(result.get("display_name") or self.author_id).strip(),
            "source": {
                "method": "从聚类后代表 parent 全文生成证据约束的场景化叙事记忆",
                "construction": "evidence_backed_representative_v1",
                "training_document_count": len(rows),
                "selected_document_count": len(selected),
                "cluster_count": len(clusters),
                "temporal_cutoff": _latest_timestamp(rows),
                "holdout_used": False,
                "llm_api_used": True,
                "builder_config_hash": self.config_hash,
            },
            "corpus_snapshot": {
                "source_asset": "index/parents.jsonl",
                "source_asset_role": "representative_parent_input",
                "parent_store": "index/parents.jsonl",
                "corpus_version": corpus_version,
                "cluster_count": len(clusters),
                "selected_document_count": len(selected),
                "selected_doc_ids": selected_doc_ids,
                "selection_config": self.selection.as_dict(),
            },
            "identity": identity,
            "global_summary": global_summary,
            "core_traits": core_traits,
            "scene_facets": facets,
            "generation_policy": generation_policy,
        }

    def _write_and_validate(self, payload: dict[str, Any]) -> NarrativeSchema:
        self.schema_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.schema_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            schema = load_narrative_schema(
                temporary,
                parent_store_path=self.parent_path,
                verify_evidence=True,
            )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        temporary.replace(self.schema_path)
        return schema


def select_representative_documents(
    clusters: list[dict[str, Any]],
    parent_rows: Iterable[dict[str, Any]],
    *,
    candidates_per_cluster: int = DEFAULT_SCHEMA_CANDIDATES_PER_CLUSTER,
    max_documents: int = DEFAULT_SCHEMA_MAX_DOCUMENTS,
) -> list[SelectedNarrativeDocument]:
    """Choose central, well-supported and engagement-aware parent documents."""

    if candidates_per_cluster < 1 or max_documents < 1:
        raise ValueError("Representative selection limits must be positive.")
    rows_by_id = {
        str(row.get("doc_id") or "").strip(): row
        for row in parent_rows
        if str(row.get("doc_id") or "").strip()
    }
    candidates: dict[str, list[SelectedNarrativeDocument]] = {}
    for cluster in clusters:
        evidence = list(cluster.get("evidence") or [])
        raw_vectors = cluster.get("vectors")
        vectors = np.asarray(raw_vectors if raw_vectors is not None else [], dtype=np.float32)
        if not evidence or len(vectors) != len(evidence):
            continue
        center = np.asarray(_normalized_centroid(vectors), dtype=np.float32)
        centrality = vectors @ center
        engagement = np.asarray([_row_engagement(rows_by_id.get(item.doc_id, {})) for item in evidence], dtype=np.float32)
        if len(engagement) and float(engagement.max()) > 0:
            engagement = engagement / float(engagement.max())
        completeness = np.asarray(
            [min(1.0, math.log1p(len(str(rows_by_id.get(item.doc_id, {}).get("text") or ""))) / math.log1p(12000)) for item in evidence],
            dtype=np.float32,
        )
        score = 0.65 * ((centrality + 1.0) / 2.0) + 0.25 * engagement + 0.10 * completeness
        order = sorted(
            range(len(evidence)),
            # First take the nearest titles to the cluster centre.  The
            # engagement-aware score is used later when a global cap must
            # choose among already-central candidates.
            key=lambda index: (
                -float(centrality[index]),
                -float(engagement[index]),
                -float(completeness[index]),
                evidence[index].doc_id,
            ),
        )
        selected: list[SelectedNarrativeDocument] = []
        for rank, index in enumerate(order[:candidates_per_cluster], start=1):
            row = rows_by_id.get(evidence[index].doc_id)
            if row is None:
                continue
            selected.append(
                SelectedNarrativeDocument(
                    cluster_id=str(cluster.get("cluster_id") or "unknown"),
                    rank=rank,
                    selection_score=float(score[index]),
                    evidence=evidence[index],
                    row=row,
                )
            )
        if selected:
            candidates[str(cluster.get("cluster_id") or "unknown")] = selected

    if not candidates:
        return []
    all_candidates = [item for values in candidates.values() for item in values]
    if len(all_candidates) <= max_documents:
        return all_candidates

    cluster_ids = list(candidates)
    minimum = 2 if len(cluster_ids) * 2 <= max_documents else 1
    selected: list[SelectedNarrativeDocument] = []
    for cluster_id in cluster_ids:
        selected.extend(candidates[cluster_id][:minimum])
    remaining = max_documents - len(selected)
    extras = [
        item
        for cluster_id in cluster_ids
        for item in candidates[cluster_id][minimum:]
    ]
    extras.sort(key=lambda item: (-item.selection_score, item.cluster_id, item.evidence.doc_id))
    selected.extend(extras[: max(0, remaining)])
    return selected


def narrative_schema_path(data_dir: Path, author_id: str) -> Path:
    return Path(data_dir) / "authors" / "zhihu" / author_id / NARRATIVE_SCHEMA_FILENAME


def _document_payload(item: SelectedNarrativeDocument) -> dict[str, Any]:
    row = item.row
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "cluster_id": item.cluster_id,
        "selection_rank": item.rank,
        "selection_score": round(item.selection_score, 6),
        "doc_id": item.evidence.doc_id,
        "title": item.evidence.title,
        "kind": item.evidence.kind,
        "created_at": row.get("created_at"),
        "updated_at": item.evidence.updated_at,
        "like_count": _count_value(row, metadata, "like_count"),
        "comment_count": _count_value(row, metadata, "comment_count"),
        "reaction_count": _count_value(row, metadata, "reaction_count"),
        "repin_count": _count_value(row, metadata, "repin_count"),
        "text": str(row.get("text") or row.get("markdown") or ""),
    }


def _partition_documents(documents: list[dict[str, Any]], max_chars: int) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 2
    for document in documents:
        size = len(json.dumps(document, ensure_ascii=False)) + 1
        if current and current_chars + size > max_chars:
            batches.append(current)
            current = []
            current_chars = 2
        current.append(document)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def _normalize_facets(value: Any, allowed: dict[str, SelectedNarrativeDocument]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    facets: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            continue
        title = _text(raw.get("title"))
        situation = _text(raw.get("situation"))
        thinking = _text(raw.get("thinking_pattern"))
        behavior = _text(raw.get("behavior_pattern"))
        if not all((title, situation, thinking, behavior)):
            continue
        cue_keys = _list_text(raw.get("cue_keys")) or [title]
        expression_signals = _list_text(raw.get("expression_signals")) or [behavior]
        boundary_anchors = _list_text(raw.get("boundary_anchors")) or [
            "只在当前 RAG 证据支持时使用该视角"
        ]
        facet_id = _slug(_text(raw.get("facet_id")) or title) or f"facet_{index + 1}"
        base_id = facet_id
        suffix = 2
        while facet_id in used_ids:
            facet_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(facet_id)
        evidence: list[dict[str, str]] = []
        raw_evidence = raw.get("source_evidence")
        if isinstance(raw_evidence, list):
            for evidence_index, item in enumerate(raw_evidence):
                if not isinstance(item, dict):
                    continue
                doc_id = _text(item.get("doc_id"))
                excerpt = _text(item.get("excerpt"))
                selected = allowed.get(doc_id)
                if selected is None or not excerpt or excerpt not in str(selected.row.get("text") or selected.row.get("markdown") or ""):
                    continue
                evidence.append(
                    {
                        "claim_id": _text(item.get("claim_id")) or f"{facet_id}:{evidence_index + 1}",
                        "doc_id": doc_id,
                        "excerpt": excerpt[:1000],
                    }
                )
        if not evidence:
            continue
        facets.append(
            {
                "facet_id": facet_id,
                "title": title,
                "cue_keys": cue_keys,
                "situation": situation,
                "thinking_pattern": thinking,
                "behavior_pattern": behavior,
                "expression_signals": expression_signals,
                "boundary_anchors": boundary_anchors,
                "source_evidence": evidence,
            }
        )
    return facets


def _schema_system_prompt() -> str:
    return """你是证据约束的作者叙事 Schema 构建器。

输入是同一位作者在多个标题聚类中选出的代表 parent 全文。请从跨文章重复出现的观察姿态、推理方式、表达动作和边界中，生成一个可供 Chat 参考的 NarrativeSchema。

硬性要求：
1. 当前 RAG 原文永远比 Schema 优先；Schema 只提供视角和思路，不是事实库、答案模板或固定口癖。
2. 不得把作者写过的领域直接当成人格结论。
3. 不得推断私人经历、实时事实、因果关系或材料没有支持的价值判断。
4. 每个 scene_facet 至少引用一个输入 doc_id，并给出输入全文中的逐字 excerpt；不能改写 excerpt。
5. 尽量让稳定 facet 的证据跨越多个领域；不稳定或只属于一个领域的结论要保守表述。
6. 输出必须是 JSON 对象，字段为：display_name、identity、global_summary、core_traits、scene_facets、generation_policy。
7. scene_facets 的字段为：facet_id、title、cue_keys、situation、thinking_pattern、behavior_pattern、expression_signals、boundary_anchors、source_evidence。
8. generation_policy 必须明确：相关 RAG 原文优先、Schema 不得固定复制过去表达、当前证据不支持时忽略视角。
"""


def _partial_schema_system_prompt() -> str:
    return """请从给出的作者代表 parent 全文中抽取可审计的作者视角候选。只输出 JSON：{"scene_facets":[...],"core_traits":[...]}。
每个 facet 必须包含 title、cue_keys、situation、thinking_pattern、behavior_pattern、expression_signals、boundary_anchors、source_evidence；source_evidence 的 doc_id 必须来自输入，excerpt 必须逐字来自输入全文。不要把领域名称直接当人格，不要补充输入没有支持的经历或事实。"""


def _schema_merge_system_prompt() -> str:
    return """你是证据约束的作者叙事 Schema 合并器。输入是同一作者不同代表文批次中已经带 doc_id 和逐字 excerpt 的候选 facet。请合并重复视角，保留跨批次稳定的思考方式，生成完整 JSON：display_name、identity、global_summary、core_traits、scene_facets、generation_policy。不得新增没有候选证据支持的结论；每个保留 facet 必须继续引用已有 doc_id 和逐字 excerpt。Schema 只提供视角参考，当前 RAG 原文优先，不是事实库或答案模板。"""


def _latest_timestamp(rows: Iterable[dict[str, Any]]) -> str | None:
    values = [str(row.get("updated_at") or row.get("created_at") or "").strip() for row in rows]
    values = [value for value in values if value]
    return max(values) if values else None


def _row_engagement(row: dict[str, Any]) -> float:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return math.log1p(
        _count_value(row, metadata, "like_count")
        + 0.5 * _count_value(row, metadata, "comment_count")
        + 0.5 * _count_value(row, metadata, "reaction_count")
        + 0.25 * _count_value(row, metadata, "repin_count")
    )


def _count_value(row: dict[str, Any], metadata: dict[str, Any], key: str) -> int:
    value = row.get(key, metadata.get(key))
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _object_or_default(value: Any, default: dict[str, Any]) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else dict(default)


def _required_result_text(value: dict[str, Any], key: str) -> str:
    result = _text(value.get(key))
    if not result:
        raise NarrativeSchemaBuildError(f"NarrativeSchema response is missing {key!r}.")
    return result


def _required_result_list(value: dict[str, Any], key: str) -> list[str]:
    result = _list_text(value.get(key))
    if not result:
        raise NarrativeSchemaBuildError(f"NarrativeSchema response is missing non-empty {key!r}.")
    return result


def _list_text(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _text(item))]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value[:80]


def _normalized_centroid(vectors: np.ndarray) -> list[float]:
    center = np.asarray(vectors, dtype=np.float32).mean(axis=0)
    return l2_normalize(center)
