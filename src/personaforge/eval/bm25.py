"""Chinese BM25 retrieval over the existing child-node artifact."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from personaforge.ingest.retrieve import ChildHit


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_+#.\-]+|[\u4e00-\u9fff]+")


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize Chinese search text while retaining useful ASCII terms."""
    try:
        import jieba
    except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
        raise RuntimeError('BM25 requires the index extras: pip install -e ".[index]"') from exc

    jieba.setLogLevel(logging.WARNING)
    tokens: list[str] = []
    for raw_token in jieba.cut_for_search(text):
        token = raw_token.strip().lower()
        if token and TOKEN_PATTERN.fullmatch(token):
            tokens.append(token)
    return tokens


@dataclass(slots=True)
class Bm25ChildIndex:
    """An in-memory BM25 index built from the same nodes used by Qdrant."""

    nodes: list[dict[str, Any]]
    tokenized_corpus: list[list[str]]
    model: Any
    k1: float
    b: float

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        *,
        exclude_parent_ids: set[str] | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> "Bm25ChildIndex":
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
            raise RuntimeError('BM25 requires the index extras: pip install -e ".[index]"') from exc

        excluded = exclude_parent_ids or set()
        nodes = [
            row
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
            if str(row.get("parent_id") or "") not in excluded
        ]
        tokenized = [tokenize_for_bm25(str(row.get("text") or "")) for row in nodes]
        return cls(
            nodes=nodes,
            tokenized_corpus=tokenized,
            model=BM25Okapi(tokenized, k1=k1, b=b),
            k1=k1,
            b=b,
        )

    def search(self, query: str, *, child_top_k: int = 100) -> list[ChildHit]:
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens or not self.nodes:
            return []
        scores = self.model.get_scores(query_tokens)
        ranked_indexes = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
        hits: list[ChildHit] = []
        for node_index in ranked_indexes:
            score = float(scores[node_index])
            if abs(score) < 1e-12:
                continue
            row = self.nodes[node_index]
            hits.append(
                ChildHit(
                    rank=len(hits) + 1,
                    score=score,
                    node_id=str(row.get("node_id") or ""),
                    parent_id=str(row.get("parent_id") or ""),
                    node_type=str(row.get("node_type") or ""),
                    title=str(row.get("title") or ""),
                    path=str(row.get("path") or ""),
                    route="raw_bm25",
                )
            )
            if len(hits) >= child_top_k:
                break
        return hits
