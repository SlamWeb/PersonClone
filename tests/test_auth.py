from __future__ import annotations

from fastapi.testclient import TestClient

from personaforge.web.app import create_app
from personaforge.web.auth import AuthStore
from personaforge.web.conversations import ConversationStore
from personaforge.web.service import WebConfig
from personaforge.web.user_memory import UserMemoryStore


def test_bootstrap_claims_existing_local_conversations(tmp_path) -> None:
    conversations = ConversationStore(tmp_path)
    memories = UserMemoryStore(tmp_path)
    turn = conversations.save_completed_turn(
        conversation_id="legacy-conversation",
        author="alice",
        query="旧问题",
        answer="旧回答",
        sources=[],
        trace_id=None,
    )
    memories.advance_window_checkpoint("local-user", turn.conversation_id, 4)
    auth = AuthStore(tmp_path)

    user = auth.bootstrap_admin(username="owner", password="password-123")

    assert conversations.get_conversation(
        "alice",
        turn.conversation_id,
        owner_id=user.id,
    )["title"] == "旧问题"
    assert conversations.list_conversations("alice", owner_id="local-user") == []
    assert memories.window_checkpoint(user.id, turn.conversation_id) == 4
    assert memories.window_checkpoint("local-user", turn.conversation_id) == 0


def test_browser_session_bootstrap_login_and_logout(tmp_path) -> None:
    app = create_app(WebConfig(data_dir=tmp_path, auth_required=True))

    with TestClient(app) as client:
        state = client.get("/api/auth/state")
        assert state.json() == {"configured": False, "authenticated": False, "user": None}
        assert client.get("/api/personas").status_code == 401

        bootstrapped = client.post(
            "/api/auth/bootstrap",
            json={"username": "owner", "password": "password-123", "display_name": "Owner"},
        )
        assert bootstrapped.status_code == 200
        assert bootstrapped.json()["user"]["role"] == "admin"
        assert client.get("/api/personas").status_code == 200

        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/personas").status_code == 401

        logged_in = client.post(
            "/api/auth/login",
            json={"username": "owner", "password": "password-123"},
        )
        assert logged_in.status_code == 200
        assert logged_in.json()["authenticated"] is True


def test_users_cannot_read_each_others_conversations_or_turns(tmp_path) -> None:
    auth = AuthStore(tmp_path)
    alice = auth.create_user(username="alice", password="password-123")
    bob = auth.create_user(username="bob", password="password-456")
    conversations = ConversationStore(tmp_path)
    turn = conversations.save_completed_turn(
        conversation_id="alice-chat",
        author="creator",
        query="问题",
        answer="回答",
        sources=[],
        trace_id=None,
        owner_id=alice.id,
    )

    assert conversations.get_conversation(
        "creator",
        "alice-chat",
        owner_id=alice.id,
    )["messages"]
    assert conversations.list_conversations("creator", owner_id=bob.id) == []

    try:
        conversations.get_conversation("creator", "alice-chat", owner_id=bob.id)
    except FileNotFoundError:
        pass
    else:  # pragma: no cover - explicit access-control assertion.
        raise AssertionError("Bob read Alice's conversation")

    try:
        conversations.get_turn_for_owner(turn.id, bob.id)
    except KeyError:
        pass
    else:  # pragma: no cover - explicit access-control assertion.
        raise AssertionError("Bob read Alice's turn")


def test_existing_conversation_cannot_be_appended_by_another_owner(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    first = store.create_turn(
        author="creator",
        conversation_id=None,
        query="Alice's question",
        query_mode="grounded",
        writer_prompt="strong_identity",
        parent_top_k=20,
        trace_capture="summary",
        owner_id="alice",
    )

    try:
        store.create_turn(
            author="creator",
            conversation_id=first.conversation_id,
            query="Bob's question",
            query_mode="grounded",
            writer_prompt="strong_identity",
            parent_top_k=20,
            trace_capture="summary",
            owner_id="bob",
        )
    except FileNotFoundError:
        pass
    else:  # pragma: no cover - explicit access-control assertion.
        raise AssertionError("Bob appended to Alice's conversation")


def test_api_filters_sessions_and_turns_by_logged_in_user(tmp_path) -> None:
    auth = AuthStore(tmp_path)
    alice = auth.create_user(username="alice", password="password-123")
    auth.create_user(username="bob", password="password-456")
    store = ConversationStore(tmp_path)
    turn = store.save_completed_turn(
        conversation_id="alice-chat",
        author="creator",
        query="问题",
        answer="回答",
        sources=[],
        trace_id=None,
        owner_id=alice.id,
    )
    app = create_app(WebConfig(data_dir=tmp_path, auth_required=True))

    with TestClient(app) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "password-123"},
        ).status_code == 200
        assert len(client.get("/api/personas/creator/sessions").json()["sessions"]) == 1
        assert client.get(f"/api/chat/turns/{turn.id}").status_code == 200

        client.post("/api/auth/logout")
        assert client.post(
            "/api/auth/login",
            json={"username": "bob", "password": "password-456"},
        ).status_code == 200
        assert client.get("/api/personas/creator/sessions").json()["sessions"] == []
        assert client.get("/api/personas/creator/sessions/alice-chat").status_code == 404
        assert client.get(f"/api/chat/turns/{turn.id}").status_code == 404
