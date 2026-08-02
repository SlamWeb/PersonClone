"""Build a Qdrant child-node index from ingest artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from personaforge.crawler.models import utc_now_iso
from personaforge.ingest.embeddings import BgeM3Encoder, TextEncoder
from personaforge.ingest.qdrant_index import (
    collection_name_for_author,
    create_local_client,
    make_point,
    recreate_collection,
    upload_points,
)


@dataclass(slots=True)
class IndexResult:
    index_dir: Path
    qdrant_path: Path
    collection_name: str
    node_count: int
    dense_size: int
    manifest_path: Path


def index_corpus(
    index_dir: Path,
    *,
    author: str,
    qdrant_path: Path | None = None,
    encoder: TextEncoder | None = None,
    client: Any | None = None,
    batch_size: int = 12,
    recreate: bool = True,
    recreate_collection_fn: Callable[[Any, str, int], None] | None = None,
    make_point_fn: Callable[[dict[str, Any], Any], Any] | None = None,
    upload_points_fn: Callable[[Any, str, list[Any]], int] | None = None,
) -> IndexResult:
    nodes = load_nodes(index_dir / "nodes.jsonl")
    if not nodes:
        raise ValueError(f"No nodes found in {index_dir / 'nodes.jsonl'}")

    source = str(nodes[0].get("source") or "unknown")
    collection_name = collection_name_for_author(source, author)
    qdrant_path = qdrant_path or index_dir / "qdrant"
    encoder = encoder or BgeM3Encoder()
    client = client or create_local_client(qdrant_path)
    recreate_collection_fn = recreate_collection_fn or _recreate_collection_adapter
    make_point_fn = make_point_fn or make_point
    upload_points_fn = upload_points_fn or _upload_points_adapter

    dense_size: int | None = None
    indexed_count = 0
    for batch_nodes in _batched(nodes, batch_size):
        embeddings = encoder.encode_texts([str(node["text"]) for node in batch_nodes], batch_size=batch_size)
        if dense_size is None:
            dense_size = len(embeddings[0].dense)
            if recreate:
                recreate_collection_fn(client, collection_name, dense_size)
        points = [make_point_fn(node, embedding) for node, embedding in zip(batch_nodes, embeddings, strict=True)]
        indexed_count += upload_points_fn(client, collection_name, points)

    if dense_size is None:
        raise ValueError("Unable to infer dense vector size from embeddings.")

    manifest_path = index_dir / "qdrant_manifest.json"
    manifest = {
        "indexed_at": utc_now_iso(),
        "collection_name": collection_name,
        "qdrant_path": _manifest_qdrant_path(index_dir, qdrant_path),
        "node_count": indexed_count,
        "dense_size": dense_size,
        "vectors": {
            "dense": "BGE-M3 dense embedding",
            "sparse": "BGE-M3 lexical weights",
        },
        "collection_policy": "one collection per author",
        "source": source,
        "author_token": author,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return IndexResult(
        index_dir=index_dir,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        node_count=indexed_count,
        dense_size=dense_size,
        manifest_path=manifest_path,
    )


def load_nodes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"nodes.jsonl does not exist: {path}")
    nodes: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            nodes.append(json.loads(line))
    return nodes


def index_result_to_dict(result: IndexResult) -> dict[str, object]:
    value = asdict(result)
    for key in ("index_dir", "qdrant_path", "manifest_path"):
        value[key] = str(value[key])
    return value


def _manifest_qdrant_path(index_dir: Path, qdrant_path: Path) -> str:
    """Keep the default local Qdrant path valid when an index directory moves."""
    try:
        return str(qdrant_path.resolve().relative_to(index_dir.resolve()))
    except ValueError:
        return str(qdrant_path)


def _batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _recreate_collection_adapter(client: Any, collection_name: str, dense_size: int) -> None:
    recreate_collection(client, collection_name, dense_size=dense_size)


def _upload_points_adapter(client: Any, collection_name: str, points: list[Any]) -> int:
    return upload_points(client, collection_name, points)
