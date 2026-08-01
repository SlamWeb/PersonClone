from personaforge.ingest.retrieve import RetrieveResult
from personaforge.web.app import _persistent_chat_stream_events
from personaforge.web.chat_tasks import ChatTaskManager
from personaforge.web.conversations import ConversationStore
from personaforge.web.service import ChatProgress, PreparedChat


class NoopJsonClient:
    def complete_json(self, _messages, **_kwargs):
        return {}


class FakeChatService:
    def __init__(self, store):
        self.conversations = store
        self.memory_updates = []
        self.fail_memory_trace = False

    def iter_prepare_chat(self, **kwargs):
        yield ChatProgress(stage="conversation_context", label="正在理解对话")
        yield PreparedChat(
            session_id=kwargs["session_id"],
            author=kwargs["author"],
            query=kwargs["query"],
            query_mode=kwargs["query_mode"],
            writer_prompt=kwargs["writer_prompt"],
            objective_background="",
            query_trace={"search_plan": {"needs_web": False, "search_queries": []}},
            retrieve_result=RetrieveResult(
                query=kwargs["query"],
                collection_name="zhihu__alice",
                child_top_k=100,
                parent_top_k=0,
                routes={},
                parents=[],
            ),
            messages=[{"role": "user", "content": kwargs["query"]}],
            trace_id="trace-1",
            turn_id=kwargs["turn_id"],
        )

    def stream_answer(self, _prepared):
        yield "第一"
        yield "回答"

    def save_turn(self, prepared, answer, sources):
        self.conversations.complete_turn(
            prepared.turn_id,
            answer=answer,
            sources=sources,
            trace_id=prepared.trace_id,
        )

    def complete_trace(self, _prepared, _answer):
        return None

    def fail_trace(self, _prepared, _error):
        return None

    def update_memory_trace(self, **kwargs):
        if self.fail_memory_trace:
            raise OSError("trace write failed")
        self.memory_updates.append(kwargs)

    def llm_client(self):
        return NoopJsonClient()


def test_chat_task_runs_after_request_creation_and_persists_events(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    service = FakeChatService(store)
    manager = ChatTaskManager(
        service,  # type: ignore[arg-type]
        store=store,
        worker_count=1,
        token_flush_characters=1,
    )
    turn = manager.create_turn(
        author="alice",
        conversation_id=None,
        query="问题",
        query_mode="raw",
        writer_prompt="strong_identity",
        parent_top_k=20,
        trace_capture="summary",
    )

    assert manager.run_once() is True

    completed = store.get_turn(turn.id)
    events = store.list_events(turn.id)
    session = store.get_conversation("alice", turn.conversation_id)

    assert completed.status == "completed"
    assert session["messages"][1]["text"] == "第一回答"
    assert [event["event"] for event in events][-1] == "done"
    assert any(event["event"] == "status" for event in events)
    assert next(event for event in events if event["event"] == "status")["payload"]["label"]
    assert [
        event["payload"]["stage"]
        for event in events
        if event["event"] == "status"
    ] == ["queued", "conversation_context", "generation"]
    assert any(event["event"] == "token" for event in events)
    assert service.memory_updates[0]["memory_update"]["status"] == "skipped"

    stream = list(_persistent_chat_stream_events(manager, turn.id, initial_turn=turn))
    assert stream[0].startswith("event: accepted\n")
    assert f'"session_id": "{turn.conversation_id}"' in stream[0]
    assert any(chunk.startswith("event: status\n") for chunk in stream)


def test_chat_task_retry_reuses_original_user_message(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    turn = store.create_turn(
        author="alice",
        conversation_id=None,
        query="问题",
        query_mode="raw",
        writer_prompt="strong_identity",
        parent_top_k=20,
        trace_capture="summary",
    )
    store.claim_turn(turn.id)
    store.fail_turn(turn.id, {"message": "失败"})
    before = store.get_conversation("alice", turn.conversation_id)

    retried = store.retry_turn(turn.id)
    after = store.get_conversation("alice", turn.conversation_id)

    assert retried.status == "queued"
    assert len(before["messages"]) == len(after["messages"]) == 2
    assert after["messages"][0]["text"] == "问题"
    assert after["messages"][1]["text"] == ""


def test_memory_trace_failure_does_not_reopen_completed_turn(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    service = FakeChatService(store)
    service.fail_memory_trace = True
    manager = ChatTaskManager(service, store=store, worker_count=1)  # type: ignore[arg-type]
    turn = manager.create_turn(
        author="alice",
        conversation_id=None,
        query="问题",
        query_mode="raw",
        writer_prompt="strong_identity",
        parent_top_k=20,
        trace_capture="summary",
    )

    assert manager.run_once() is True

    assert store.get_turn(turn.id).status == "completed"
    assert [event["event"] for event in store.list_events(turn.id)][-1] == "done"
