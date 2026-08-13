from __future__ import annotations

import pytest

from personaforge.web.deployment_guard import (
    DeploymentGuard,
    DeploymentGuardConfig,
    DeploymentGuardError,
)


class FakeConversationStore:
    def __init__(self, *, active_global: int = 0, active_by_user: dict[str, int] | None = None) -> None:
        self.active_global = active_global
        self.active_by_user = active_by_user or {}

    def count_active_turns(self, *, owner_id: str | None = None) -> int:
        if owner_id is None:
            return self.active_global
        return self.active_by_user.get(owner_id, 0)


def test_chat_guard_limits_user_and_global_activity() -> None:
    store = FakeConversationStore()
    guard = DeploymentGuard(
        DeploymentGuardConfig(chat_active_per_user=1, chat_active_global=2)
    )

    guard.admit_chat("alice", store)
    store.active_by_user["alice"] = 1
    with pytest.raises(DeploymentGuardError, match="正在生成"):
        guard.admit_chat("alice", store)

    store.active_by_user["alice"] = 0
    store.active_global = 2
    with pytest.raises(DeploymentGuardError, match="任务较多"):
        guard.admit_chat("bob", store)


def test_chat_guard_limits_request_window() -> None:
    store = FakeConversationStore()
    guard = DeploymentGuard(
        DeploymentGuardConfig(chat_requests_per_window=2, chat_window_seconds=600)
    )

    guard.admit_chat("alice", store)
    guard.admit_chat("alice", store)
    with pytest.raises(DeploymentGuardError, match="过于频繁"):
        guard.admit_chat("alice", store)


def test_login_guard_counts_failures_and_clears_on_success() -> None:
    guard = DeploymentGuard(
        DeploymentGuardConfig(login_failures_per_window=2, login_window_seconds=300)
    )

    assert guard.allow_login_attempt("127.0.0.1") is True
    guard.record_login_failure("127.0.0.1")
    guard.record_login_failure("127.0.0.1")
    assert guard.allow_login_attempt("127.0.0.1") is False
    guard.clear_login_failures("127.0.0.1")
    assert guard.allow_login_attempt("127.0.0.1") is True


def test_guard_can_be_disabled_for_controlled_local_tests() -> None:
    guard = DeploymentGuard(DeploymentGuardConfig(enabled=False))
    store = FakeConversationStore(active_global=100, active_by_user={"alice": 100})

    for _ in range(20):
        guard.admit_chat("alice", store)
    for _ in range(20):
        guard.record_login_failure("127.0.0.1")
    assert guard.allow_login_attempt("127.0.0.1") is True
