from __future__ import annotations

from personaforge.ingest.query_understanding import (
    SearchResult,
    build_background_and_retrieval_queries,
    build_grounded_query_plan,
    plan_web_search,
)


class FakeJsonClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    def complete_json(self, messages, *, temperature=0.0, max_tokens=1024):
        self.messages.append(messages)
        return self.replies.pop(0)


class FakeSearchClient:
    def __init__(self):
        self.queries = []

    def search_many(self, queries, *, max_results=5):
        self.queries = list(queries)
        return [
            SearchResult(
                query=queries[0],
                title="事件标题",
                url="https://example.local/news",
                content="客观事件内容。",
            )
        ]


def test_plan_web_search_keeps_router_schema_small() -> None:
    llm = FakeJsonClient([{"needs_web": True, "search_queries": ["武亮 生活费", "武亮 大一 电脑"]}])

    plan = plan_web_search("如何评价武亮直播言论？", llm=llm)

    assert plan.needs_web is True
    assert plan.search_queries == ["武亮 生活费", "武亮 大一 电脑"]


def test_search_planner_prompt_covers_old_word_with_new_online_meaning() -> None:
    llm = FakeJsonClient([{"needs_web": True, "search_queries": ['"力工" 网络梗 是什么意思']}])

    plan = plan_web_search("为什么大多数人在嘲笑力工？", llm=llm)
    prompt = "\n".join(message["content"] for message in llm.messages[0])

    assert plan.needs_web is True
    assert "不能按字典第一义猜测" in prompt
    assert "为什么大多数人在嘲笑力工" in prompt
    assert '"力工梭哈" 彩礼 婚恋' in prompt


def test_background_transform_returns_fixed_four_routes() -> None:
    llm = FakeJsonClient(
        [
            {
                "objective_background": "",
                "retrieval_queries": [
                    {"route": "literal_question", "query": "女生 配得感"},
                    {"route": "event_background", "query": "女生 配得感"},
                    {"route": "mechanism_scene", "query": "女生 觉得自己值得更好关系"},
                    {"route": "colloquial_surface", "query": "女的 觉得自己值得更好的"},
                ],
            }
        ]
    )

    result = build_background_and_retrieval_queries("如何看待女生常说的配得感", search_results=[], llm=llm)

    assert result.objective_background == ""
    assert [item.route for item in result.retrieval_queries] == [
        "literal_question",
        "event_background",
        "mechanism_scene",
        "colloquial_surface",
    ]


def test_background_transform_prompt_requires_resolved_meaning_in_retrieval() -> None:
    llm = FakeJsonClient(
        [
            {
                "objective_background": "此题中的“力工”不是体力劳动者，而是婚恋语境中的网络标签。",
                "retrieval_queries": [
                    {"route": "literal_question", "query": "嘲笑力工"},
                    {"route": "event_background", "query": "力工梭哈 彩礼 婚恋"},
                    {"route": "mechanism_scene", "query": "男人攒钱 彩礼买房 结婚梭哈"},
                    {"route": "colloquial_surface", "query": "辛苦赚钱全给彩礼 被嘲笑"},
                ],
            }
        ]
    )
    search_results = [
        SearchResult(
            query='"力工" 网络梗',
            title="力工梭哈",
            url="https://example.local/ligong",
            content="力工梭哈用于描述把积蓄集中投入彩礼和婚姻的男性。",
        )
    ]

    result = build_background_and_retrieval_queries(
        "为什么大多数人在嘲笑力工？",
        search_results=search_results,
        llm=llm,
    )
    prompt = "\n".join(message["content"] for message in llm.messages[0])

    assert "不是体力劳动者" in result.objective_background
    assert result.retrieval_queries[2].query == "男人攒钱 彩礼买房 结婚梭哈"
    assert "不得继续围绕被排除的字面义检索" in prompt


def test_grounded_query_plan_runs_search_between_two_llm_calls() -> None:
    llm = FakeJsonClient(
        [
            {"needs_web": True, "search_queries": ["武亮 生活费 1500 2000"]},
            {
                "objective_background": "武亮直播言论引发关于大学生活费的讨论。",
                "retrieval_queries": [
                    {"route": "literal_question", "query": "武亮 大学生活费"},
                    {"route": "event_background", "query": "大一 电脑 男生生活费 女生生活费"},
                    {"route": "mechanism_scene", "query": "父母 管控 男大学生 资源"},
                    {"route": "colloquial_surface", "query": "老登 管男小登 花钱"},
                ],
            },
        ]
    )
    search = FakeSearchClient()

    plan = build_grounded_query_plan("如何评价武亮直播言论？", llm=llm, search_client=search)

    assert search.queries == ["武亮 生活费 1500 2000"]
    assert plan.search_results[0].title == "事件标题"
    assert len(plan.transform.retrieval_queries) == 4
