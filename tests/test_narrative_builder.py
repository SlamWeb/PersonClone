from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from personaforge.ingest.embeddings import SparseEmbedding, TextEmbedding
from personaforge.persona.narrative_builder import (
    NarrativeSchemaBuilder,
    NarrativeSchemaSelectionConfig,
    select_representative_documents,
)
from personaforge.persona.routing_profile import clean_title_evidence, cluster_title_embeddings


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode_texts(self, texts: list[str], *, batch_size: int = 12) -> list[TextEmbedding]:
        self.calls += 1
        values = ([1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99], [0.7, 0.7])
        return [
            TextEmbedding(dense=values[index % len(values)], sparse=SparseEmbedding(indices=[], values=[]))
            for index, _text in enumerate(texts)
        ]


class FakeSchemaLlm:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def complete_json(self, messages: list[dict[str, str]], *, temperature: float = 0.0, max_tokens: int = 1024) -> dict[str, object]:
        self.calls.append(messages)
        payload = json.loads(messages[-1]["content"])
        documents = payload.get("documents") or []
        first = documents[0]
        excerpt = first["text"][:18]
        return {
            "display_name": "测试作者",
            "identity": {"public_identity": "公开写作者"},
            "global_summary": "先处理问题结构，再在证据边界内表达。",
            "core_traits": ["先看前提", "证据不足时说明边界"],
            "scene_facets": [
                {
                    "facet_id": "premise-and-boundary",
                    "title": "先拆前提再判断",
                    "cue_keys": ["为什么", "是否"],
                    "situation": "问题带有未经验证的前提时",
                    "thinking_pattern": "先检查前提，再区分事实和推测",
                    "behavior_pattern": "先指出关键前提，再给出有边界的判断",
                    "expression_signals": ["直接指出关键处"],
                    "boundary_anchors": ["不补写材料没有支持的事实"],
                    "source_evidence": [{"claim_id": "facet-1", "doc_id": first["doc_id"], "excerpt": excerpt}],
                }
            ],
            "generation_policy": {"selection": "RAG 原文优先"},
        }


def _write_parents(tmp_path: Path) -> None:
    index_dir = tmp_path / "authors" / "zhihu" / "demo" / "index"
    index_dir.mkdir(parents=True)
    rows = []
    for index, (kind, title, vector_group) in enumerate(
        [
            ("answer", "问题前提一", "a"),
            ("answer", "问题前提二", "a"),
            ("article", "表达边界一", "b"),
            ("answer", "表达边界二", "b"),
            ("answer", "边缘观察", "c"),
        ]
    ):
        rows.append(
            {
                "doc_id": f"doc-{index}",
                "kind": kind,
                "title": title,
                "updated_at": f"2024-01-0{index + 1}",
                "text": f"这是第 {index} 篇代表全文，包含可审计的证据。TAIL-{index}",
                "metadata": {"like_count": index * 100, "comment_count": index},
            }
        )
    (index_dir / "parents.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_representative_selection_keeps_clusters_and_uses_engagement_as_tiebreaker() -> None:
    evidence = clean_title_evidence(
        [
            {"doc_id": "a", "kind": "answer", "title": "A", "text": "short", "metadata": {"like_count": 1}},
            {"doc_id": "b", "kind": "answer", "title": "B", "text": "long " * 20, "metadata": {"like_count": 1000}},
        ]
    )
    clusters = [
        {
            "cluster_id": "cluster-a",
            "evidence": evidence,
            "vectors": np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        }
    ]
    selected = select_representative_documents(
        clusters,
        [
            {"doc_id": "a", "text": "short", "metadata": {"like_count": 1}},
            {"doc_id": "b", "text": "long " * 20, "metadata": {"like_count": 1000}},
        ],
        candidates_per_cluster=1,
        max_documents=1,
    )
    assert [item.evidence.doc_id for item in selected] == ["b"]


def test_schema_builder_sends_full_selected_documents_and_reuses_cache(tmp_path: Path) -> None:
    _write_parents(tmp_path)
    encoder = FakeEncoder()
    llm = FakeSchemaLlm()
    builder = NarrativeSchemaBuilder(
        data_dir=tmp_path,
        author_id="demo",
        encoder=encoder,
        llm=llm,
        distance_threshold=0.32,
        min_cluster_size=2,
        selection=NarrativeSchemaSelectionConfig(
            candidates_per_cluster=3,
            max_documents=4,
            max_payload_chars=100_000,
        ),
    )

    first = builder.build()
    assert first.status == "rebuilt"
    assert first.schema.facet_count == 1
    assert first.selected_document_count <= 4
    assert first.schema.source["llm_api_used"] is True
    assert first.schema_path.endswith("narrative_schema.json")
    request_payload = json.loads(llm.calls[0][-1]["content"])
    assert request_payload["documents"]
    assert all("text" in item and item["text"] for item in request_payload["documents"])
    raw_schema = Path(first.schema_path).read_text(encoding="utf-8")
    assert '"documents"' not in raw_schema
    assert "TAIL-" not in raw_schema

    encoder_calls = encoder.calls
    llm_calls = len(llm.calls)
    second = builder.build()
    assert second.status == "reused"
    assert encoder.calls == encoder_calls
    assert len(llm.calls) == llm_calls

    parent_path = tmp_path / "authors" / "zhihu" / "demo" / "index" / "parents.jsonl"
    rows = [json.loads(line) for line in parent_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["updated_at"] = "2025-01-01"
    parent_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    third = builder.build()
    assert third.status == "rebuilt"
    assert len(llm.calls) > llm_calls
