from __future__ import annotations

from personaforge.eval.replay import rebuild_parent_hits


def test_rebuild_parent_hits_preserves_rank_and_parent_content() -> None:
    serialized = [
        {
            "rank": 1,
            "parent_id": "zhihu:answer:1",
            "score": 0.4,
            "title": "第一篇",
            "path": "answer-1.md",
            "first_hits": [
                {
                    "rank": 3,
                    "score": 0.8,
                    "node_id": "zhihu:answer:1:passage:0",
                    "parent_id": "zhihu:answer:1",
                    "node_type": "passage",
                    "title": "第一篇",
                    "path": "answer-1.md",
                    "route": "literal_question:dense",
                }
            ],
        },
        {
            "rank": 2,
            "parent_id": "zhihu:answer:2",
            "score": 0.3,
            "title": "第二篇",
            "path": "answer-2.md",
            "first_hits": [],
        },
    ]
    parents = {
        "zhihu:answer:1": {
            "doc_id": "zhihu:answer:1",
            "title": "第一篇",
            "text": "第一篇全文",
        },
        "zhihu:answer:2": {
            "doc_id": "zhihu:answer:2",
            "title": "第二篇",
            "text": "第二篇全文",
        },
    }

    hits = rebuild_parent_hits(serialized, parents_by_id=parents)

    assert [hit.parent_id for hit in hits] == [
        "zhihu:answer:1",
        "zhihu:answer:2",
    ]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].parent == parents["zhihu:answer:1"]
    assert hits[0].first_hits[0].route == "literal_question:dense"
