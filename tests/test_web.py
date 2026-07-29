from __future__ import annotations

import json
from time import perf_counter

from personaforge.ingest.query_understanding import RetrievalQuery
from personaforge.ingest.retrieve import ChildHit, ParentHit, RetrieveResult
from personaforge.persona.suggestions import validate_suggestions
from personaforge.web.app import _chat_stream_events, _frontend_dist_dir, create_app
from personaforge.web.service import (
    ChatProgress,
    PersonaChatService,
    PreparedChat,
    WebConfig,
    list_local_personas,
    sources_from_parent_hits,
)
from personaforge.web.schemas import ChatStreamRequest
from personaforge.web.streaming import sse_event


def test_list_local_personas_finds_indexed_authors(tmp_path) -> None:
    index_dir = tmp_path / "authors" / "zhihu" / "alice" / "index"
    (index_dir / "qdrant").mkdir(parents=True)
    (index_dir / "parents.jsonl").write_text("", encoding="utf-8")

    personas = list_local_personas(tmp_path)

    assert [item.author for item in personas] == ["alice"]
    assert personas[0].source == "zhihu"
    assert personas[0].display_name == "alice"
    assert personas[0].persona_pack_available is False


def test_list_local_personas_reads_profile_metadata(tmp_path) -> None:
    author_dir = tmp_path / "authors" / "zhihu" / "alice"
    index_dir = author_dir / "index"
    (index_dir / "qdrant").mkdir(parents=True)
    (index_dir / "parents.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (author_dir / "profile.json").write_text(
        json.dumps({"nickname": "Alice", "avatar_url": "https://example.com/a.jpg"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (author_dir / "persona_pack.json").write_text("{}", encoding="utf-8")

    persona = list_local_personas(tmp_path)[0]

    assert persona.display_name == "Alice"
    assert persona.avatar_url == "https://example.com/a.jpg"
    assert persona.content_count == 2
    assert persona.persona_pack_available is True


def test_save_turn_creates_author_scoped_session(tmp_path) -> None:
    service = PersonaChatService(WebConfig(data_dir=tmp_path))
    prepared = PreparedChat(
        session_id="s1",
        author="alice",
        query="问题？",
        query_mode="raw",
        writer_prompt="strong_identity",
        objective_background="",
        query_trace=None,
        retrieve_result=RetrieveResult(
            query="问题？",
            collection_name="zhihu__alice",
            child_top_k=100,
            parent_top_k=20,
            routes={},
            parents=[],
            retrieval_queries=[],
        ),
        messages=[],
    )

    service.save_turn(prepared, "回答", [{"rank": 1, "title": "来源"}])

    session = service.get_session("alice", "s1")
    assert session["title"] == "问题？"
    assert [message["role"] for message in session["messages"]] == ["user", "assistant"]
    assert service.list_sessions("alice")[0]["message_count"] == 2


def test_trace_records_retrieval_without_copying_parent_full_text(tmp_path) -> None:
    child = ChildHit(
        rank=1,
        score=0.2,
        node_id="node-1",
        parent_id="zhihu:answer:1",
        node_type="passage",
        title="标题",
        path="answer/1.md",
        route="literal_question:dense",
    )
    parent = ParentHit(
        rank=1,
        parent_id="zhihu:answer:1",
        score=0.1,
        title="标题",
        path="answer/1.md",
        first_hits=[child],
        parent={"text": "完整正文不应该进入 trace"},
    )
    result = RetrieveResult(
        query="问题？",
        collection_name="zhihu__alice",
        child_top_k=100,
        parent_top_k=20,
        routes={"literal_question:dense": [child]},
        parents=[parent],
        retrieval_queries=[RetrievalQuery(route="literal_question", query="问题？")],
    )
    prepared = PreparedChat(
        session_id="s1",
        author="alice",
        query="问题？",
        query_mode="grounded",
        writer_prompt="strong_identity",
        objective_background="客观背景",
        query_trace={"search_plan": {"needs_web": False, "search_queries": []}},
        retrieve_result=result,
        messages=[{"role": "system", "content": "系统提示"}, {"role": "user", "content": "用户问题"}],
        trace_id="trace-test-1",
        trace_created_at="2026-01-01T00:00:00+00:00",
        trace_started_at=perf_counter(),
        query_understanding_duration_ms=12,
        retrieval_duration_ms=34,
        writer_build_duration_ms=5,
    )
    service = PersonaChatService(WebConfig(data_dir=tmp_path))

    service.record_prepared_trace(prepared)
    service.complete_trace(prepared, "回答")
    trace = service.get_trace("alice", "trace-test-1")

    assert trace["status"] == "completed"
    assert trace["input"]["query"] == "问题？"
    assert trace["retrieval"]["parents"][0]["first_hits"][0]["route"] == "literal_question:dense"
    assert trace["generation"]["answer_characters"] == 2
    assert "完整正文" not in json.dumps(trace, ensure_ascii=False)
    assert trace["schema_version"] == "personaforge.web.trace.v1"
    assert trace["capture"]["mode"] == "summary"


def test_full_trace_capture_keeps_local_prompt_and_parent_context(tmp_path) -> None:
    parent = ParentHit(
        rank=1,
        parent_id="zhihu:answer:1",
        score=0.1,
        title="标题",
        path="answer/1.md",
        parent={"text": "完整正文"},
    )
    prepared = PreparedChat(
        session_id="s1",
        author="alice",
        query="问题？",
        query_mode="raw",
        writer_prompt="strong_identity",
        objective_background="",
        query_trace=None,
        retrieve_result=RetrieveResult(
            query="问题？",
            collection_name="zhihu__alice",
            child_top_k=100,
            parent_top_k=20,
            routes={},
            parents=[parent],
        ),
        messages=[{"role": "system", "content": "完整 prompt"}],
        trace_capture="full",
        trace_id="trace-full-1",
    )
    service = PersonaChatService(WebConfig(data_dir=tmp_path))

    service.complete_trace(prepared, "回答")
    trace = service.get_trace("alice", "trace-full-1")

    assert trace["capture"]["mode"] == "full"
    assert trace["writer"]["full_messages"][0]["content"] == "完整 prompt"
    assert trace["retrieval"]["full_parent_context"][0]["parent"]["text"] == "完整正文"


def test_list_suggestions_reads_profile_suggestions(tmp_path) -> None:
    path = tmp_path / "authors" / "zhihu" / "alice" / "profile_suggestions.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"suggestions": ["一个新问题？"]}, ensure_ascii=False), encoding="utf-8")

    service = PersonaChatService(WebConfig(data_dir=tmp_path))

    assert service.list_suggestions("alice") == ["一个新问题？"]


def test_validate_suggestions_filters_near_duplicate_source_title() -> None:
    suggestions = validate_suggestions(
        [
            "为什么孩子越大越不愿意跟我们说话？",
            "为什么现在的年轻人越来越不想结婚？",
            "为什么很多父母越关心孩子，孩子越想逃？",
        ],
        source_titles=["为什么孩子越大越不愿意跟我们说话？", "为什么男人都不想结婚了？"],
        count=4,
    )

    assert suggestions == ["为什么很多父母越关心孩子，孩子越想逃？"]


def test_sources_from_parent_hits_hides_parent_full_text() -> None:
    hit = ParentHit(
        rank=1,
        parent_id="zhihu:answer:1",
        score=0.1,
        title="标题",
        path="answer/1.md",
        first_hits=[
            ChildHit(
                rank=3,
                score=0.2,
                node_id="node-1",
                parent_id="zhihu:answer:1",
                node_type="passage",
                title="标题",
                path="answer/1.md",
                route="literal_question:dense",
            )
        ],
        parent={
            "kind": "answer",
            "source_id": "1",
            "metadata": {"question_id": "99"},
            "text": "完整正文不应该进 sources",
        },
    )

    sources = sources_from_parent_hits([hit])

    assert sources[0]["title"] == "标题"
    assert sources[0]["url"] == "https://www.zhihu.com/question/99/answer/1"
    assert "完整正文" not in json.dumps(sources, ensure_ascii=False)
    assert sources[0]["first_hits"][0]["node_type"] == "passage"


def test_old_trace_gets_public_source_url_when_read(tmp_path) -> None:
    index_dir = tmp_path / "authors" / "zhihu" / "alice" / "index"
    index_dir.mkdir(parents=True)
    (index_dir / "parents.jsonl").write_text(
        json.dumps(
            {
                "doc_id": "zhihu:answer:1",
                "kind": "answer",
                "source_id": "1",
                "metadata": {"question_id": "99"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    trace_dir = tmp_path / "authors" / "zhihu" / "alice" / "traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "old.json").write_text(
        json.dumps(
            {
                "trace_id": "old",
                "retrieval": {"parents": [{"parent_id": "zhihu:answer:1"}]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    trace = PersonaChatService(WebConfig(data_dir=tmp_path)).get_trace("alice", "old")

    assert trace["retrieval"]["parents"][0]["url"] == "https://www.zhihu.com/question/99/answer/1"


def test_sse_event_serializes_utf8_json() -> None:
    event = sse_event("token", {"text": "你好"})

    assert event.startswith("event: token\n")
    assert 'data: {"text": "你好"}' in event
    assert event.endswith("\n\n")


def test_create_app_registers_trace_endpoint(tmp_path) -> None:
    app = create_app(WebConfig(data_dir=tmp_path))

    paths = {route.path for route in app.routes}

    assert "/api/personas/{author}/traces/{trace_id}" in paths
    assert "/api/personas/preview" in paths
    assert "/api/author-jobs" in paths
    assert "/api/author-jobs/{job_id}/cancel" in paths


def test_frontend_dist_dir_accepts_deployment_override(tmp_path, monkeypatch) -> None:
    dist_dir = tmp_path / "web-dist"
    monkeypatch.setenv("PERSONAFORGE_WEB_DIST", str(dist_dir))

    assert _frontend_dist_dir() == dist_dir.resolve()


def test_chat_stream_emits_status_before_first_token() -> None:
    prepared = PreparedChat(
        session_id="s1",
        author="alice",
        query="问题？",
        query_mode="raw",
        writer_prompt="strong_identity",
        objective_background="",
        query_trace=None,
        retrieve_result=RetrieveResult(
            query="问题？",
            collection_name="zhihu__alice",
            child_top_k=100,
            parent_top_k=20,
            routes={},
            parents=[],
            retrieval_queries=[],
        ),
        messages=[],
        trace_id="trace-1",
    )

    class FakeService:
        def iter_prepare_chat(self, **_kwargs):
            yield ChatProgress(stage="retrieval", label="正在检索历史表达")
            yield ChatProgress(stage="writer", label="正在准备回答")
            yield prepared

        def stream_answer(self, _prepared):
            yield "回答"

        def save_turn(self, *_args):
            return {}

        def complete_trace(self, *_args):
            return None

        def fail_trace(self, *_args):
            return None

    events = list(
        _chat_stream_events(
            FakeService(),  # type: ignore[arg-type]
            ChatStreamRequest(
                author="alice",
                query="问题？",
                query_mode="raw",
                writer_prompt="strong_identity",
                parent_top_k=20,
            ),
        )
    )

    assert [event.split("\n", 1)[0] for event in events] == [
        "event: status",
        "event: status",
        "event: meta",
        "event: status",
        "event: token",
        "event: done",
    ]
