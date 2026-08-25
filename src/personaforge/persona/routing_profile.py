"""Author routing profiles for external CreatorOS integrations.

This module deliberately reuses the existing parent-document store and BGE-M3
encoder.  It does not participate in answer retrieval or generation: the
vectors here describe domains and evidence-backed perspectives that a future
CreatorOS service can query independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
from uuid import UUID, uuid5

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from personaforge.ingest.embeddings import TextEncoder
from personaforge.ingest.loader import load_parent_documents
from personaforge.llm import JsonChatClient
from personaforge.persona.narrative import NarrativeSchema, load_narrative_schema_for_index
from personaforge.persona.pack import PersonaPack, load_persona_pack_for_index


ROUTING_PROFILE_FILENAME = "routing_profile.json"
ROUTING_COLLECTION = "creator_routing_profiles"
ROUTING_PROFILE_SCHEMA_VERSION = 1
DEFAULT_DISTANCE_THRESHOLD = 0.32
DEFAULT_MIN_CLUSTER_SIZE = 3
_POINT_NAMESPACE = UUID("b0f56c3a-17ba-45d1-9f26-4d84698e4c8b")


class RepresentativeEvidence(BaseModel):
    """Small, auditable evidence index; never stores the source article body."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str = ""
    kind: str = ""
    updated_at: str | None = None
    source_method: str = "title"
    field: str | None = None
    claim_id: str | None = None
    excerpt: str | None = None


class VectorRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_name: str = ROUTING_COLLECTION
    point_id: str
    vector_name: str = "dense"
    embedding_model: str
    dimension: int = Field(gt=0)
    normalized: bool = True
    corpus_version: str


class DomainPrototype(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prototype_id: str
    label: str
    description: str
    retrieval_text: str
    document_count: int = Field(ge=1)
    representative_evidence: list[RepresentativeEvidence]
    confidence: float = Field(ge=0, le=1)
    vector_ref: VectorRef
    status: Literal["stable", "provisional", "long_tail", "fallback"] = "stable"


class PerspectivePrototype(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prototype_id: str
    label: str
    summary: str
    values: list[str]
    trigger_cues: list[str]
    reasoning_pattern: str
    boundaries: list[str]
    retrieval_text: str
    representative_evidence: list[RepresentativeEvidence]
    confidence: float = Field(ge=0, le=1)
    source_method: Literal["narrative_schema", "persona_pack", "llm_sample"]
    vector_ref: VectorRef


class AuthorRoutingProfile(BaseModel):
    """Stable public contract returned to CreatorOS through the Web API."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = ROUTING_PROFILE_SCHEMA_VERSION
    author_id: str
    display_name: str = ""
    source: str = "zhihu"
    generated_at: str
    corpus_version: str
    config_hash: str
    status: Literal["ready", "domain_ready", "perspective_pending"]
    embedding_model: str
    embedding_dimension: int = Field(gt=0)
    qdrant_collection: str = ROUTING_COLLECTION
    domain_prototypes: list[DomainPrototype]
    perspective_prototypes: list[PerspectivePrototype]


class RoutingProfileResult(BaseModel):
    """Response envelope used by both the CLI and API."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["reused", "rebuilt"]
    profile: AuthorRoutingProfile
    profile_path: str


class TitleEvidence(BaseModel):
    """A normalized title and the parent-document fields needed for audit."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    normalized_title: str
    kind: str
    updated_at: str | None = None
    source: str = "zhihu"


def normalize_title(value: str | None) -> str:
    """Normalize title text without throwing away meaningful words."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[`*_~]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_title_evidence(rows: Iterable[dict[str, Any] | Any]) -> list[TitleEvidence]:
    """Filter empty titles and exact duplicates deterministically."""

    selected: dict[str, TitleEvidence] = {}
    for row in rows:
        if isinstance(row, dict):
            get = row.get
        else:
            get = lambda key, default=None: getattr(row, key, default)
        kind = str(get("kind", "") or "").strip().lower()
        if kind not in {"answer", "article"}:
            continue
        title = normalize_title(get("title", ""))
        if not title:
            continue
        key = title.casefold()
        candidate = TitleEvidence(
            doc_id=str(get("doc_id", "") or "").strip(),
            title=title,
            normalized_title=key,
            kind=kind,
            updated_at=_optional_text(get("updated_at")),
            source=str(get("source", "zhihu") or "zhihu"),
        )
        if not candidate.doc_id:
            continue
        previous = selected.get(key)
        if previous is None or (candidate.doc_id, candidate.updated_at or "") < (
            previous.doc_id,
            previous.updated_at or "",
        ):
            selected[key] = candidate
    return sorted(selected.values(), key=lambda item: (item.doc_id, item.updated_at or ""))


def compute_corpus_version(rows: Iterable[dict[str, Any] | TitleEvidence | Any]) -> str:
    """Hash sorted ``doc_id + updated_at`` pairs as the corpus snapshot key."""

    pairs: list[tuple[str, str]] = []
    for row in rows:
        if isinstance(row, dict):
            doc_id = str(row.get("doc_id") or "").strip()
            updated_at = str(row.get("updated_at") or "")
        else:
            doc_id = str(getattr(row, "doc_id", "") or "").strip()
            updated_at = str(getattr(row, "updated_at", "") or "")
        if doc_id:
            pairs.append((doc_id, updated_at))
    payload = "\n".join(f"{doc_id}\t{updated_at}" for doc_id, updated_at in sorted(set(pairs)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def l2_normalize(vector: Sequence[float]) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding must have a finite non-zero norm.")
    return (array / norm).astype(np.float32).tolist()


def _normalized_centroid(vectors: np.ndarray) -> list[float]:
    center = np.asarray(vectors, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(center))
    if not math.isfinite(norm) or norm <= 1e-12:
        return l2_normalize(np.asarray(vectors, dtype=np.float32)[0])
    return (center / norm).astype(np.float32).tolist()


def _representatives(
    evidence: list[TitleEvidence], vectors: np.ndarray, *, limit: int = 5
) -> list[TitleEvidence]:
    center = np.asarray(_normalized_centroid(vectors), dtype=np.float32)
    scores = np.asarray(vectors, dtype=np.float32) @ center
    order = sorted(range(len(evidence)), key=lambda index: (-float(scores[index]), evidence[index].doc_id))
    return [evidence[index] for index in order[: max(3, min(limit, len(order)))]]


def cluster_title_embeddings(
    evidence: list[TitleEvidence],
    embeddings: Sequence[Sequence[float]],
    *,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> list[dict[str, Any]]:
    """Cluster normalized title vectors and retain a deterministic long tail."""

    if not evidence:
        return []
    if len(evidence) != len(embeddings):
        raise ValueError("Title evidence and embedding counts must match.")
    matrix = np.asarray([l2_normalize(item) for item in embeddings], dtype=np.float32)
    if len(evidence) < 3:
        return [
            {
                "cluster_id": "fallback",
                "evidence": evidence,
                "vectors": matrix,
                "representatives": _representatives(evidence, matrix),
                "status": "fallback",
            }
        ]
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:  # pragma: no cover - exercised in installation checks.
        raise RuntimeError(
            "Title clustering requires scikit-learn; install the index extra."
        ) from exc
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(matrix)
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(index)
    ordered = sorted(grouped.values(), key=lambda indexes: min(evidence[i].doc_id for i in indexes))
    valid: list[dict[str, Any]] = []
    long_tail_indexes: list[int] = []
    for indexes in ordered:
        if len(indexes) < min_cluster_size:
            long_tail_indexes.extend(indexes)
            continue
        cluster_evidence = [evidence[i] for i in indexes]
        cluster_vectors = matrix[indexes]
        valid.append(
            {
                "cluster_id": "cluster-" + hashlib.sha1(
                    "|".join(item.doc_id for item in cluster_evidence).encode("utf-8")
                ).hexdigest()[:10],
                "evidence": cluster_evidence,
                "vectors": cluster_vectors,
                "representatives": _representatives(cluster_evidence, cluster_vectors),
                "status": "stable",
            }
        )
    if long_tail_indexes:
        tail_evidence = [evidence[i] for i in long_tail_indexes]
        tail_vectors = matrix[long_tail_indexes]
        valid.append(
            {
                "cluster_id": "long-tail",
                "evidence": tail_evidence,
                "vectors": tail_vectors,
                "representatives": _representatives(tail_evidence, tail_vectors),
                "status": "long_tail",
            }
        )
    return valid


class RoutingProfileBuilder:
    """Build and persist one author's routing profile."""

    def __init__(
        self,
        *,
        data_dir: Path,
        author_id: str,
        encoder: TextEncoder,
        llm: JsonChatClient | None = None,
        model_name: str = "BAAI/bge-m3",
        embedding_batch_size: int = 12,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        qdrant_path: Path | None = None,
        qdrant_client: Any | None = None,
        persist_qdrant: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.author_id = author_id
        self.encoder = encoder
        self.llm = llm
        self.model_name = model_name
        self.embedding_batch_size = embedding_batch_size
        if not 0 < distance_threshold < 2:
            raise ValueError("distance_threshold must be between 0 and 2.")
        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2.")
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive.")
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.qdrant_path = Path(qdrant_path or self.data_dir / "system" / "routing_qdrant")
        self.qdrant_client = qdrant_client
        self.persist_qdrant = persist_qdrant

    @property
    def author_dir(self) -> Path:
        return self.data_dir / "authors" / "zhihu" / self.author_id

    @property
    def index_dir(self) -> Path:
        return self.author_dir / "index"

    @property
    def profile_path(self) -> Path:
        return self.author_dir / ROUTING_PROFILE_FILENAME

    @property
    def config_hash(self) -> str:
        config = {
            "schema_version": ROUTING_PROFILE_SCHEMA_VERSION,
            "model_name": self.model_name,
            "distance_threshold": self.distance_threshold,
            "min_cluster_size": self.min_cluster_size,
            "embedding_batch_size": self.embedding_batch_size,
            "llm_enabled": self.llm is not None,
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:20]

    def build(self, *, force: bool = False) -> RoutingProfileResult:
        parents = self._load_parent_rows()
        eligible_rows = [row for row in parents if str(row.get("kind") or "").lower() in {"answer", "article"}]
        corpus_version = compute_corpus_version(eligible_rows)
        if not force and self.profile_path.exists():
            try:
                cached = AuthorRoutingProfile.model_validate_json(self.profile_path.read_text(encoding="utf-8"))
                if cached.corpus_version == corpus_version and cached.config_hash == self.config_hash:
                    return RoutingProfileResult(status="reused", profile=cached, profile_path=str(self.profile_path))
            except Exception:
                pass

        title_evidence = clean_title_evidence(eligible_rows)
        if not title_evidence:
            raise ValueError(f"No answer/article titles found for author {self.author_id!r}.")
        title_embeddings = self.encoder.encode_texts(
            [item.title for item in title_evidence], batch_size=self.embedding_batch_size
        )
        dense_vectors = [l2_normalize(item.dense) for item in title_embeddings]
        clusters = cluster_title_embeddings(
            title_evidence,
            dense_vectors,
            distance_threshold=self.distance_threshold,
            min_cluster_size=self.min_cluster_size,
        )
        narrative = self._load_narrative()
        pack = self._load_pack() if narrative is None else None
        domain_specs = self._domain_specs(clusters)
        perspective_specs, perspective_status = self._perspective_specs(
            narrative=narrative,
            pack=pack,
            clusters=clusters,
        )
        # Domain retrieval must use the normalized cluster center, not a fresh
        # embedding of the generated label. Perspective vectors are encoded
        # from their structured retrieval text separately.
        domain_vectors = [list(spec.pop("_center_vector")) for spec in domain_specs]
        perspective_embeddings = self.encoder.encode_texts(
            [spec["retrieval_text"] for spec in perspective_specs],
            batch_size=self.embedding_batch_size,
        ) if perspective_specs else []
        perspective_vectors = [l2_normalize(item.dense) for item in perspective_embeddings]
        dimension = len(domain_vectors[0] if domain_vectors else perspective_vectors[0])
        if any(len(vector) != dimension for vector in (*domain_vectors, *perspective_vectors)):
            raise ValueError("Domain and perspective embeddings must have the same dimension.")
        domain_prototypes: list[DomainPrototype] = []
        perspective_prototypes: list[PerspectivePrototype] = []
        vector_entries: list[tuple[str, str, Any, list[float], float, list[str]]] = []
        for spec, vector in zip(domain_specs, domain_vectors):
            point_id = str(uuid5(_POINT_NAMESPACE, f"{self.author_id}:{corpus_version}:{spec['prototype_id']}"))
            vector_ref = VectorRef(
                point_id=point_id,
                embedding_model=self.model_name,
                dimension=dimension,
                corpus_version=corpus_version,
            )
            prototype = DomainPrototype(
                **{key: value for key, value in spec.items() if not key.startswith("_")},
                vector_ref=vector_ref,
            )
            domain_prototypes.append(prototype)
            vector_entries.append(
                (prototype.prototype_id, "domain", prototype, vector, prototype.confidence, [item.doc_id for item in prototype.representative_evidence])
            )
        for spec, vector in zip(perspective_specs, perspective_vectors):
            point_id = str(uuid5(_POINT_NAMESPACE, f"{self.author_id}:{corpus_version}:{spec['prototype_id']}"))
            vector_ref = VectorRef(
                point_id=point_id,
                embedding_model=self.model_name,
                dimension=dimension,
                corpus_version=corpus_version,
            )
            prototype = PerspectivePrototype(**spec, vector_ref=vector_ref)
            perspective_prototypes.append(prototype)
            vector_entries.append(
                (prototype.prototype_id, "perspective", prototype, vector, prototype.confidence, [item.doc_id for item in prototype.representative_evidence])
            )

        qdrant_collection = ROUTING_COLLECTION
        if self.persist_qdrant:
            self._persist_vectors(
                vector_entries,
                dimension=dimension,
                author_id=self.author_id,
                corpus_version=corpus_version,
            )
        profile = AuthorRoutingProfile(
            author_id=self.author_id,
            display_name=self._display_name(narrative, pack),
            generated_at=datetime.now(timezone.utc).isoformat(),
            corpus_version=corpus_version,
            config_hash=self.config_hash,
            status="ready" if perspective_prototypes else perspective_status,
            embedding_model=self.model_name,
            embedding_dimension=dimension,
            qdrant_collection=qdrant_collection,
            domain_prototypes=domain_prototypes,
            perspective_prototypes=perspective_prototypes,
        )
        self._atomic_write(profile)
        if self.persist_qdrant:
            self._cleanup_old_points(corpus_version)
        return RoutingProfileResult(status="rebuilt", profile=profile, profile_path=str(self.profile_path))

    def _load_parent_rows(self) -> list[dict[str, Any]]:
        parent_path = self.index_dir / "parents.jsonl"
        if parent_path.exists():
            rows: list[dict[str, Any]] = []
            for line in parent_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
            return rows
        raw_dir = self.author_dir / "raw"
        return [item.to_dict() for item in load_parent_documents(raw_dir)]

    def _load_narrative(self) -> NarrativeSchema | None:
        try:
            return load_narrative_schema_for_index(self.index_dir, required=False)
        except Exception:
            return None

    def _load_pack(self) -> PersonaPack | None:
        try:
            return load_persona_pack_for_index(self.index_dir, required=False)
        except Exception:
            return None

    def _domain_specs(self, clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not clusters:
            return []
        llm_labels = self._llm_domain_labels(clusters)
        specs: list[dict[str, Any]] = []
        for index, cluster in enumerate(clusters):
            representatives = cluster["representatives"]
            titles = [item.title for item in representatives]
            label, description, confidence = llm_labels.get(index, self._provisional_domain(titles, cluster["status"]))
            evidence = [self._title_to_evidence(item) for item in representatives]
            retrieval_text = f"{label}。{description}。代表标题：{'；'.join(titles)}"
            specs.append(
                {
                    "prototype_id": f"domain:{cluster['cluster_id']}",
                    "label": label,
                    "description": description,
                    "retrieval_text": retrieval_text,
                    "document_count": len(cluster["evidence"]),
                    "representative_evidence": evidence,
                    "confidence": confidence,
                    "status": cluster["status"],
                    "_center_vector": _normalized_centroid(cluster["vectors"]),
                }
            )
        return specs

    def _provisional_domain(self, titles: list[str], status: str) -> tuple[str, str, float]:
        lead = titles[0] if titles else "未命名领域"
        label = f"provisional: {lead}"
        description = "围绕代表标题形成的暂定领域：" + "；".join(titles)
        return label, description, 0.35 if status in {"fallback", "long_tail"} else 0.45

    def _llm_domain_labels(self, clusters: list[dict[str, Any]]) -> dict[int, tuple[str, str, float]]:
        if self.llm is None:
            return {}
        payload = [
            {"cluster_index": index, "representative_titles": [item.title for item in cluster["representatives"]]}
            for index, cluster in enumerate(clusters)
        ]
        messages = [
            {
                "role": "system",
                "content": "你只根据每个聚类给出的代表标题生成领域标签和一句描述，不得引入标题之外的信息。输出 JSON: {clusters:[{cluster_index,label,description,confidence}]}。",
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            result = self.llm.complete_json(messages, temperature=0.0, max_tokens=2048)
        except Exception:
            return {}
        values = result.get("clusters") if isinstance(result, dict) else None
        if not isinstance(values, list):
            return {}
        parsed: dict[int, tuple[str, str, float]] = {}
        for item in values:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("cluster_index"))
                label = normalize_title(str(item.get("label") or ""))
                description = normalize_title(str(item.get("description") or ""))
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.65))))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(clusters) and label and description:
                parsed[index] = (label, description, confidence)
        return parsed

    def _perspective_specs(
        self,
        *,
        narrative: NarrativeSchema | None,
        pack: PersonaPack | None,
        clusters: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], Literal["domain_ready", "perspective_pending"]]:
        title_by_doc = {item.doc_id: item for cluster in clusters for item in cluster["evidence"]}
        if narrative is not None:
            specs = []
            for facet in narrative.scene_facets:
                evidence = [
                    RepresentativeEvidence(
                        doc_id=item.doc_id,
                        title=title_by_doc.get(item.doc_id, TitleEvidence(doc_id=item.doc_id, title="", normalized_title="", kind="")).title,
                        kind=title_by_doc.get(item.doc_id, TitleEvidence(doc_id=item.doc_id, title="", normalized_title="", kind="")).kind,
                        updated_at=title_by_doc.get(item.doc_id, TitleEvidence(doc_id=item.doc_id, title="", normalized_title="", kind="")).updated_at,
                        source_method="narrative_schema",
                        field="scene_facets",
                        claim_id=item.claim_id,
                        excerpt=item.excerpt[:500],
                    )
                    for item in facet.source_evidence
                ]
                values = list(narrative.core_traits) + list(facet.expression_signals)
                retrieval_text = "；".join(
                    [facet.title, narrative.global_summary, facet.situation, facet.thinking_pattern, facet.behavior_pattern, *values, *facet.cue_keys, *facet.boundary_anchors]
                )
                specs.append(
                    {
                        "prototype_id": f"perspective:narrative:{facet.facet_id}",
                        "label": facet.title,
                        "summary": f"{narrative.global_summary}；{facet.situation}",
                        "values": values,
                        "trigger_cues": list(facet.cue_keys),
                        "reasoning_pattern": facet.thinking_pattern,
                        "boundaries": list(facet.boundary_anchors),
                        "retrieval_text": retrieval_text,
                        "representative_evidence": evidence,
                        "confidence": 0.85 if evidence else 0.65,
                        "source_method": "narrative_schema",
                    }
                )
            return specs, "domain_ready" if not specs else "domain_ready"
        if pack is not None:
            specs = []
            for section in ("worldview", "reasoning", "response_strategy"):
                for claim in pack.claims_for(section):
                    source_evidence = (*claim.evidence, *claim.counterevidence)
                    evidence = [
                        RepresentativeEvidence(
                            doc_id=item.doc_id,
                            title=title_by_doc.get(item.doc_id, TitleEvidence(doc_id=item.doc_id, title="", normalized_title="", kind="")).title,
                            kind=title_by_doc.get(item.doc_id, TitleEvidence(doc_id=item.doc_id, title="", normalized_title="", kind="")).kind,
                            updated_at=title_by_doc.get(item.doc_id, TitleEvidence(doc_id=item.doc_id, title="", normalized_title="", kind="")).updated_at,
                            source_method="persona_pack",
                            field=section,
                            claim_id=claim.claim_id,
                            excerpt=item.excerpt[:500],
                        )
                        for item in source_evidence
                    ]
                    boundaries = [item for item in (claim.activation_condition, claim.avoid_overapplication) if item]
                    values = [claim.claim] if section == "worldview" else []
                    reasoning = claim.claim if section == "reasoning" else str(pack.generation_policy.get("selection_rule") or "")
                    retrieval_text = "；".join([claim.claim, *claim.scopes, *boundaries, reasoning])
                    specs.append(
                        {
                            "prototype_id": f"perspective:persona_pack:{claim.claim_id}",
                            "label": claim.claim,
                            "summary": claim.claim,
                            "values": values,
                            "trigger_cues": list(claim.scopes),
                            "reasoning_pattern": reasoning,
                            "boundaries": boundaries,
                            "retrieval_text": retrieval_text,
                            "representative_evidence": evidence,
                            "confidence": max(0.0, min(1.0, claim.confidence)),
                            "source_method": "persona_pack",
                        }
                    )
            return specs, "domain_ready"
        specs = self._llm_perspective_specs(clusters)
        return specs, "domain_ready" if specs else "perspective_pending"

    def _llm_perspective_specs(self, clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.llm is None:
            return []
        sampled: list[TitleEvidence] = []
        for cluster in clusters:
            sampled.extend(cluster["representatives"][:2])
        sampled = sampled[:24]
        evidence_payload = [{"doc_id": item.doc_id, "title": item.title, "kind": item.kind} for item in sampled]
        messages = [
            {
                "role": "system",
                "content": "从给出的代表标题中抽取可审计的作者视角原型。不得把领域当人格；每个结论必须引用一个给定 doc_id。输出 JSON: {prototypes:[{label,summary,values,trigger_cues,reasoning_pattern,boundaries,evidence_doc_ids,confidence}]}。",
            },
            {"role": "user", "content": json.dumps(evidence_payload, ensure_ascii=False)},
        ]
        try:
            result = self.llm.complete_json(messages, temperature=0.0, max_tokens=2048)
        except Exception:
            return []
        values = result.get("prototypes") if isinstance(result, dict) else None
        if not isinstance(values, list):
            return []
        allowed = {item.doc_id: item for item in sampled}
        specs: list[dict[str, Any]] = []
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            doc_ids = [str(value) for value in item.get("evidence_doc_ids", []) if str(value) in allowed]
            if not doc_ids:
                continue
            label = normalize_title(str(item.get("label") or ""))
            summary = normalize_title(str(item.get("summary") or ""))
            if not label or not summary:
                continue
            evidence = [self._title_to_evidence(allowed[doc_id], source_method="llm_sample") for doc_id in doc_ids]
            values_list = _text_list(item.get("values"))
            cues = _text_list(item.get("trigger_cues"))
            boundaries = _text_list(item.get("boundaries"))
            reasoning = normalize_title(str(item.get("reasoning_pattern") or ""))
            try:
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.45))))
            except (TypeError, ValueError):
                confidence = 0.45
            specs.append(
                {
                    "prototype_id": f"perspective:llm_sample:{index}",
                    "label": label,
                    "summary": summary,
                    "values": values_list,
                    "trigger_cues": cues,
                    "reasoning_pattern": reasoning,
                    "boundaries": boundaries,
                    "retrieval_text": "；".join([label, summary, *values_list, *cues, reasoning, *boundaries]),
                    "representative_evidence": evidence,
                    "confidence": confidence,
                    "source_method": "llm_sample",
                }
            )
        return specs

    def _persist_vectors(
        self,
        entries: list[tuple[str, str, Any, list[float], float, list[str]]],
        *,
        dimension: int,
        author_id: str,
        corpus_version: str,
    ) -> None:
        client = self.qdrant_client
        if client is None:
            try:
                from personaforge.ingest.qdrant_index import create_local_client

                client = create_local_client(self.qdrant_path)
            except ImportError as exc:  # pragma: no cover - optional dependency.
                raise RuntimeError("Qdrant persistence requires the index extra.") from exc
            self.qdrant_client = client
        from qdrant_client import models

        exists = bool(client.collection_exists(ROUTING_COLLECTION))
        if not exists:
            client.create_collection(
                collection_name=ROUTING_COLLECTION,
                vectors_config={"dense": models.VectorParams(size=dimension, distance=models.Distance.COSINE)},
            )
        points = []
        for prototype_id, prototype_type, prototype, vector, confidence, doc_ids in entries:
            point_id = prototype.vector_ref.point_id
            payload = {
                "author_id": author_id,
                "prototype_id": prototype_id,
                "prototype_type": prototype_type,
                "label": prototype.label,
                "description": prototype.description if isinstance(prototype, DomainPrototype) else prototype.summary,
                "retrieval_text": prototype.retrieval_text,
                "corpus_version": corpus_version,
                "confidence": confidence,
                "representative_doc_ids": doc_ids,
            }
            points.append(models.PointStruct(id=point_id, vector={"dense": vector}, payload=payload))
        if points:
            client.upsert(collection_name=ROUTING_COLLECTION, points=points, wait=True)

    def _cleanup_old_points(self, corpus_version: str) -> None:
        if self.qdrant_client is None:
            return
        from qdrant_client import models

        selector = models.Filter(
            must=[models.FieldCondition(key="author_id", match=models.MatchValue(value=self.author_id))],
            must_not=[models.FieldCondition(key="corpus_version", match=models.MatchValue(value=corpus_version))],
        )
        self.qdrant_client.delete(collection_name=ROUTING_COLLECTION, points_selector=selector, wait=True)

    def _atomic_write(self, profile: AuthorRoutingProfile) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.profile_path.with_suffix(".json.tmp")
        temporary.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(self.profile_path)

    def _display_name(self, narrative: NarrativeSchema | None, pack: PersonaPack | None) -> str:
        return (narrative.display_name if narrative else pack.display_name if pack else self.author_id) or self.author_id

    def _title_to_evidence(self, item: TitleEvidence, *, source_method: str = "title") -> RepresentativeEvidence:
        return RepresentativeEvidence(
            doc_id=item.doc_id,
            title=item.title,
            kind=item.kind,
            updated_at=item.updated_at,
            source_method=source_method,
        )


def routing_profile_path(data_dir: Path, author_id: str) -> Path:
    return Path(data_dir) / "authors" / "zhihu" / author_id / ROUTING_PROFILE_FILENAME


def load_routing_profile(data_dir: Path, author_id: str) -> AuthorRoutingProfile:
    path = routing_profile_path(data_dir, author_id)
    if not path.exists():
        raise FileNotFoundError(path)
    return AuthorRoutingProfile.model_validate_json(path.read_text(encoding="utf-8"))


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [normalize_title(str(item)) for item in value if normalize_title(str(item))]
