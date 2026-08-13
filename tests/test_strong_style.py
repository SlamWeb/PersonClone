from __future__ import annotations

from personaforge.eval.strong_style import build_content_plan, select_expression_hits
from personaforge.ingest.retrieve import ParentHit


class FakeJsonClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def complete_json(self, messages, *, temperature=0.0, max_tokens=1024):
        return self.payloads.pop(0)


def _hit(parent_id: str, rank: int) -> ParentHit:
    return ParentHit(
        rank=rank,
        parent_id=parent_id,
        score=1.0 / rank,
        title=f"标题-{parent_id}",
        path=f"{parent_id}.md",
        parent={"title": f"标题-{parent_id}", "text": f"作者表达 {parent_id}。"},
    )


def test_expression_selection_validates_ids_and_falls_back() -> None:
    candidates = [_hit("p1", 6), _hit("p2", 7), _hit("p3", 8)]
    client = FakeJsonClient([{"selected_parent_ids": ["unknown", "p2"]}])

    selected, trace = select_expression_hits(
        query="问题",
        objective_background="",
        candidates=candidates,
        llm=client,
        top_k=3,
    )

    assert [hit.parent_id for hit in selected] == ["p2", "p1", "p3"]
    assert trace["fallback"] is True


def test_content_plan_is_compact_and_traceable() -> None:
    client = FakeJsonClient([{
        "core_claim": "核心判断",
        "entry_angle": "切入角度",
        "supporting_points": "两个依据",
        "avoid": "不要跑题",
    }])

    plan, trace = build_content_plan(
        query="问题",
        objective_background="背景",
        content_hits=[_hit("p1", 1)],
        llm=client,
    )

    assert plan["core_claim"] == "核心判断"
    assert trace["prompt_version"] == "strong-style-content-plan-v1"
