"""Small process-local guards for a privately shared Web deployment.

The guard intentionally protects only interactive Chat admission. Offline
evaluation and judge workers have their own queues, budgets, and concurrency
controls and must not be coupled to this limiter.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic


class DeploymentGuardError(RuntimeError):
    """A request is temporarily rejected by a deployment guard."""

    def __init__(self, message: str, *, retry_after: int) -> None:
        super().__init__(message)
        self.retry_after = max(1, int(retry_after))


@dataclass(frozen=True, slots=True)
class DeploymentGuardConfig:
    enabled: bool = True
    chat_requests_per_window: int = 10
    chat_window_seconds: int = 600
    chat_active_per_user: int = 1
    chat_active_global: int = 2
    login_failures_per_window: int = 8
    login_window_seconds: int = 300


class DeploymentGuard:
    """Protect an intentionally small, single-process deployment.

    This is not a distributed rate limiter. It is appropriate for the current
    one-process Uvicorn deployment and fails closed for Chat admission while
    leaving background evaluation managers untouched.
    """

    def __init__(self, config: DeploymentGuardConfig | None = None) -> None:
        self.config = config or DeploymentGuardConfig()
        self._lock = Lock()
        self._chat_requests: dict[str, deque[float]] = defaultdict(deque)
        self._login_failures: dict[str, deque[float]] = defaultdict(deque)

    def admit_chat(self, owner_id: str, store: object) -> None:
        """Reserve one Chat task admission or raise a retryable error."""

        if not self.config.enabled:
            return
        now = monotonic()
        with self._lock:
            requests = self._prune(
                self._chat_requests[owner_id],
                now=now,
                window=self.config.chat_window_seconds,
            )
            if len(requests) >= self.config.chat_requests_per_window:
                retry_after = self._retry_after(requests, now, self.config.chat_window_seconds)
                raise DeploymentGuardError(
                    "Chat 请求过于频繁，请稍后再试。",
                    retry_after=retry_after,
                )

            active_user = _count_active_turns(store, owner_id=owner_id)
            if active_user >= self.config.chat_active_per_user:
                raise DeploymentGuardError(
                    "你已有一个回答正在生成，请等待完成后再发送。",
                    retry_after=2,
                )
            active_global = _count_active_turns(store)
            if active_global >= self.config.chat_active_global:
                raise DeploymentGuardError(
                    "当前生成任务较多，请稍后再试。",
                    retry_after=5,
                )

            requests.append(now)

    def allow_login_attempt(self, client_key: str) -> bool:
        """Return whether another login/bootstrap attempt may be made."""

        if not self.config.enabled:
            return True
        now = monotonic()
        with self._lock:
            failures = self._prune(
                self._login_failures[client_key],
                now=now,
                window=self.config.login_window_seconds,
            )
            return len(failures) < self.config.login_failures_per_window

    def record_login_failure(self, client_key: str) -> None:
        if not self.config.enabled:
            return
        now = monotonic()
        with self._lock:
            failures = self._prune(
                self._login_failures[client_key],
                now=now,
                window=self.config.login_window_seconds,
            )
            failures.append(now)

    def clear_login_failures(self, client_key: str) -> None:
        with self._lock:
            self._login_failures.pop(client_key, None)

    @staticmethod
    def _prune(events: deque[float], *, now: float, window: int) -> deque[float]:
        cutoff = now - max(1, window)
        while events and events[0] <= cutoff:
            events.popleft()
        return events

    @staticmethod
    def _retry_after(events: deque[float], now: float, window: int) -> int:
        if not events:
            return 1
        return max(1, int(events[0] + max(1, window) - now + 0.999))


def _count_active_turns(store: object, *, owner_id: str | None = None) -> int:
    counter = getattr(store, "count_active_turns", None)
    if counter is None:
        raise RuntimeError("Conversation store does not support deployment admission checks.")
    return int(counter(owner_id=owner_id))
