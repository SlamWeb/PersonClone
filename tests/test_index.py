from __future__ import annotations

import json
import uuid

from personaforge.ingest.embeddings import SparseEmbedding, TextEmbedding
from personaforge.ingest.index import index_corpus
from personaforge.ingest.qdrant_index import collection_name_for_author, point_id_for_node


class FakeEncoder:
    def encode_texts(self, texts: list[str], *, batch_size: int = 12) -> list[TextEmbedding]:
        return [
            TextEmbedding(
                dense=[float(index), float(index + 1), float(index + 2)],
                sparse=SparseEmbedding(indices=[index + 10], values=[0.5]),
            )
            for index, _ in enumerate(texts)
        ]


def write_nodes(index_dir) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "node_id": "zhihu:answer:1:title:0",
            "parent_id": "zhihu:answer:1",
            "node_type": "title",
            "text": "标题一",
            "source": "zhihu",
            "kind": "answer",
            "source_id": "1",
            "title": "标题一",
            "path": "answer/answer-1.md",
            "author_token": "alice",
            "index": 0,
            "metadata": {},
        },
        {
            "node_id": "zhihu:answer:1:passage:0",
            "parent_id": "zhihu:answer:1",
            "node_type": "passage",
            "text": "正文一",
            "source": "zhihu",
            "kind": "answer",
            "source_id": "1",
            "title": "标题一",
            "path": "answer/answer-1.md",
            "author_token": "alice",
            "index": 0,
            "metadata": {},
        },
    ]
    (index_dir / "nodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def test_collection_name_is_author_scoped() -> None:
    assert collection_name_for_author("zhihu", "wu-ren-jun-28") == "personaforge__zhihu__wu-ren-jun-28"


def test_point_id_for_node_is_stable_uuid() -> None:
    point_id = point_id_for_node("zhihu:answer:1:passage:0")

    assert point_id == point_id_for_node("zhihu:answer:1:passage:0")
    uuid.UUID(point_id)


def test_index_corpus_orchestrates_embedding_and_upload(tmp_path) -> None:
    index_dir = tmp_path / "index"
    write_nodes(index_dir)
    calls = {"collection": None, "dense_size": None, "points": []}

    def recreate_collection(client, collection_name: str, dense_size: int) -> None:
        calls["collection"] = collection_name
        calls["dense_size"] = dense_size

    def make_point(node, embedding):
        return {
            "id": node["node_id"],
            "parent_id": node["parent_id"],
            "dense": embedding.dense,
            "sparse": embedding.sparse.indices,
        }

    def upload_points(client, collection_name: str, points):
        calls["points"].extend(points)
        return len(points)

    result = index_corpus(
        index_dir,
        author="alice",
        qdrant_path=tmp_path / "qdrant",
        encoder=FakeEncoder(),
        client=object(),
        batch_size=2,
        recreate_collection_fn=recreate_collection,
        make_point_fn=make_point,
        upload_points_fn=upload_points,
    )

    assert result.collection_name == "personaforge__zhihu__alice"
    assert result.node_count == 2
    assert result.dense_size == 3
    assert calls["collection"] == "personaforge__zhihu__alice"
    assert calls["dense_size"] == 3
    assert len(calls["points"]) == 2
    manifest = json.loads((index_dir / "qdrant_manifest.json").read_text(encoding="utf-8"))
    assert manifest["collection_policy"] == "one collection per author"


def test_index_manifest_uses_portable_path_for_qdrant_inside_index(tmp_path) -> None:
    index_dir = tmp_path / "staging" / "index"
    write_nodes(index_dir)

    result = index_corpus(
        index_dir,
        author="alice",
        encoder=FakeEncoder(),
        client=object(),
        recreate_collection_fn=lambda *_args: None,
        make_point_fn=lambda node, _embedding: node,
        upload_points_fn=lambda _client, _collection, points: len(points),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["qdrant_path"] == "qdrant"
