from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from personaforge.ingest.embeddings import SparseEmbedding, TextEmbedding
from personaforge.persona.routing_profile import (
    AuthorRoutingProfile,
    RoutingProfileBuilder,
    clean_title_evidence,
    cluster_title_embeddings,
    compute_corpus_version,
    _representatives,
)


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode_texts(self, texts: list[str], *, batch_size: int = 12) -> list[TextEmbedding]:
        self.calls += 1
        return [
            TextEmbedding(
                dense=[float(index + 1), float((index + 1) * 2), 1.0],
                sparse=SparseEmbedding(indices=[], values=[]),
            )
            for index, _ in enumerate(texts)
        ]


class FakeQdrant:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def scroll(self, **_kwargs):
        class Point:
            def __init__(self, point_id: str) -> None:
                self.id = point_id

        return [Point("keep"), Point("stale-same-corpus"), Point("stale-old-corpus")], None

    def delete(self, *, points_selector, **_kwargs) -> None:
        self.deleted.extend(points_selector)


def test_title_cleaning_filters_empty_and_exact_duplicates() -> None:
    rows = [
        {"doc_id": "b", "kind": "answer", "title": "  A\u3000title  ", "updated_at": "2"},
        {"doc_id": "a", "kind": "answer", "title": "A title", "updated_at": "1"},
        {"doc_id": "c", "kind": "pin", "title": "ignored", "updated_at": "1"},
        {"doc_id": "d", "kind": "article", "title": "", "updated_at": "1"},
    ]
    cleaned = clean_title_evidence(rows)
    assert [item.doc_id for item in cleaned] == ["a"]
    assert cleaned[0].normalized_title == "a title"


def test_corpus_version_uses_sorted_doc_and_updated_at() -> None:
    left = [{"doc_id": "b", "updated_at": "2"}, {"doc_id": "a", "updated_at": "1"}]
    right = [{"doc_id": "a", "updated_at": "1"}, {"doc_id": "b", "updated_at": "2"}]
    assert compute_corpus_version(left) == compute_corpus_version(right)
    assert compute_corpus_version(left) != compute_corpus_version(
        [{"doc_id": "a", "updated_at": "1"}, {"doc_id": "b", "updated_at": "3"}]
    )


def test_small_corpus_creates_fallback_and_representatives() -> None:
    evidence = clean_title_evidence(
        [
            {"doc_id": "a", "kind": "answer", "title": "Alpha"},
            {"doc_id": "b", "kind": "article", "title": "Beta"},
        ]
    )
    clusters = cluster_title_embeddings(evidence, [[1, 0], [0, 1]])
    assert clusters[0]["status"] == "fallback"
    assert [item.doc_id for item in clusters[0]["representatives"]] == ["a", "b"]


def test_representatives_are_nearest_to_normalized_center() -> None:
    evidence = clean_title_evidence(
        [{"doc_id": str(index), "kind": "answer", "title": f"Title {index}"} for index in range(5)]
    )
    representatives = _representatives(
        evidence,
        np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [-1, 0]], dtype=float),
        limit=3,
    )
    assert [item.doc_id for item in representatives] == ["2", "3", "1"]


def test_profile_serialization_and_reuse(tmp_path: Path) -> None:
    index_dir = tmp_path / "authors" / "zhihu" / "demo" / "index"
    index_dir.mkdir(parents=True)
    (index_dir / "parents.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "kind": kind,
                    "title": title,
                    "updated_at": "2024-01-01",
                    "source": "zhihu",
                },
                ensure_ascii=False,
            )
            for doc_id, kind, title in (("a", "answer", "Alpha"), ("b", "article", "Beta"))
        ),
        encoding="utf-8",
    )
    encoder = FakeEncoder()
    builder = RoutingProfileBuilder(
        data_dir=tmp_path,
        author_id="demo",
        encoder=encoder,
        persist_qdrant=False,
    )
    first = builder.build()
    assert first.status == "rebuilt"
    assert first.profile.status == "perspective_pending"
    assert first.profile.domain_prototypes[0].vector_ref.point_id
    raw_profile = Path(first.profile_path).read_text(encoding="utf-8")
    assert '"dense": [' not in raw_profile
    saved = AuthorRoutingProfile.model_validate_json(raw_profile)
    assert saved.model_dump() == first.profile.model_dump()
    calls = encoder.calls
    second = builder.build()
    assert second.status == "reused"
    assert encoder.calls == calls


def test_qdrant_cleanup_keeps_only_active_point_ids(tmp_path: Path) -> None:
    qdrant = FakeQdrant()
    builder = RoutingProfileBuilder(
        data_dir=tmp_path,
        author_id="demo",
        encoder=FakeEncoder(),
        qdrant_client=qdrant,
        persist_qdrant=True,
    )
    builder._cleanup_old_points(active_point_ids={"keep"})
    assert qdrant.deleted == ["stale-same-corpus", "stale-old-corpus"]


def test_agglomerative_configuration_is_used_when_dependency_available() -> None:
    pytest.importorskip("sklearn")
    evidence = clean_title_evidence(
        [{"doc_id": str(index), "kind": "answer", "title": f"Title {index}"} for index in range(4)]
    )
    clusters = cluster_title_embeddings(
        evidence,
        [[1, 0], [1, 0.01], [0, 1], [0, 1.01]],
        distance_threshold=0.32,
        min_cluster_size=2,
    )
    assert clusters
    assert all(len(cluster["representatives"]) >= 2 for cluster in clusters)
