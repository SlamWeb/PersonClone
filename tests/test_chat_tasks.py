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


def test_memory_extraction_failure_keeps_answer_completed(tmp_path, monkeypatch):
    from personaforge.web.user_memory import UserMemoryStore
    import personaforge.web.chat_tasks as tasks
    store = ConversationStore(tmp_path)
    service = FakeChatService(store)
    service.user_memories = UserMemoryStore(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError('private provider input must not appear in trace')
    monkeypatch.setattr(tasks, 'update_user_memories', fail)
    manager = ChatTaskManager(service, store=store, worker_count=1)
    turn = manager.create_turn(author='alice',conversation_id=None,query='问题',query_mode='raw',
        writer_prompt='strong_identity',parent_top_k=20,trace_capture='summary')
    manager.run_once()
    assert store.get_turn(turn.id).status == 'completed'
    assert store.list_events(turn.id)[-1]['event'] == 'done'
    assert service.memory_updates[0]['memory_update']['user_memory']['status'] == 'failed'
    assert 'private provider input' not in str(service.memory_updates)


def test_idle_recovery_uses_durable_conversation_without_prepared_chat(tmp_path):
    from personaforge.web.user_memory import UserMemoryStore
    from test_user_memory import FakeLlm
    store = ConversationStore(tmp_path)
    service = FakeChatService(store)
    service.user_memories = UserMemoryStore(tmp_path)
    llm = FakeLlm([{'candidates':[]}])
    service.llm_client = lambda: llm
    turn = store.save_completed_turn(conversation_id='c',author='alice',query='Redis AOF 是什么？',
        answer='日志。',sources=[],trace_id=None)
    manager = ChatTaskManager(service,store=store,worker_count=1)
    assert manager.recover_idle_memories(idle_seconds=0) == 1
    assert service.user_memories.window_checkpoint('local-user','c') == store.get_completed_turns('c')[0].sequence
    assert manager.recover_idle_memories(idle_seconds=0) == 0


def test_memory_checkpoint_waits_for_retryable_earlier_turn(tmp_path):
    store = ConversationStore(tmp_path)
    failed = store.create_turn(author='alice',conversation_id='c',query='先前的问题',query_mode='raw',
        writer_prompt='strong_identity',parent_top_k=20,trace_capture='summary')
    store.claim_turn(failed.id)
    store.fail_turn(failed.id,{'message':'failed'})
    store.save_completed_turn(conversation_id='c',author='alice',query='后来的问题',
        answer='回答',sources=[],trace_id=None)
    assert len(store.get_completed_turns('c')) == 1
    assert store.get_memory_eligible_turns('c') == []
    store.retry_turn(failed.id)
    store.claim_turn(failed.id)
    store.complete_turn(failed.id,answer='重试回答',sources=[],trace_id=None)
    assert len(store.get_memory_eligible_turns('c')) == 2
