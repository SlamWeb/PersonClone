"""Minimal chat-completion provider abstraction."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol

from personaforge.env import first_env_value, load_env_file

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


class JsonChatClient(Protocol):
    def complete_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Return plain text from chat messages."""

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, object]:
        """Return a JSON object from chat messages."""

    def stream_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """Yield text chunks from chat messages."""


@dataclass(frozen=True, slots=True)
class LlmUsage:
    """Provider-reported usage for one completion request."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cache_hit_tokens": self.prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": self.prompt_cache_miss_tokens,
        }


@dataclass(slots=True)
class DeepSeekJsonClient:
    api_key: str
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    model: str = DEFAULT_DEEPSEEK_MODEL
    timeout_seconds: float = 90.0
    thinking: str | None = "disabled"
    last_usage: LlmUsage | None = None

    @classmethod
    def from_env(cls, env_file: Path = Path(".env")) -> "DeepSeekJsonClient":
        load_env_file(env_file)
        api_key = first_env_value("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Missing DeepSeek API key: set DEEPSEEK_API_KEY in .env or environment.")
        return cls(
            api_key=api_key,
            base_url=first_env_value("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL,
            model=first_env_value("PERSONAFORGE_QUERY_MODEL", "DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL,
        )

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, object]:
        payload, _usage = self.complete_json_with_usage(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return payload

    def complete_json_with_usage(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[dict[str, object], LlmUsage | None]:
        """Return JSON and this request's usage without relying on shared mutable state."""

        body: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.thinking:
            body["thinking"] = {"type": self.thinking}
        payload = _post_json(
            _chat_endpoint(self.base_url),
            body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout_seconds=self.timeout_seconds,
        )
        usage = _usage_from_payload(payload)
        self.last_usage = usage
        try:
            text = str(payload["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected chat completion payload: {payload!r}") from exc
        return parse_json_object(text), usage

    def complete_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, object] | None = None,
    ) -> str:
        body: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if self.thinking:
            body["thinking"] = {"type": self.thinking}
        payload = _post_json(
            _chat_endpoint(self.base_url),
            body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout_seconds=self.timeout_seconds,
        )
        self.last_usage = _usage_from_payload(payload)
        try:
            text = str(payload["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected chat completion payload: {payload!r}") from exc
        return text

    def stream_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        yield from self.stream_text_with_usage(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def stream_text_with_usage(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        on_usage: Callable[[LlmUsage], None] | None = None,
    ) -> Iterator[str]:
        body: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.thinking:
            body["thinking"] = {"type": self.thinking}
        def receive_usage(usage: LlmUsage) -> None:
            self.last_usage = usage
            if on_usage is not None:
                on_usage(usage)

        yield from _post_json_stream(
            _chat_endpoint(self.base_url),
            body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout_seconds=self.timeout_seconds,
            on_usage=receive_usage,
        )


def parse_json_object(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from LLM.")
    return value


def _post_json(
    url: str,
    body: dict[str, object],
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 90.0,
) -> dict[str, object]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        **(headers or {}),
    }
    request = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    return json.loads(payload)


def _post_json_stream(
    url: str,
    body: dict[str, object],
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 90.0,
    on_usage: Callable[[LlmUsage], None] | None = None,
) -> Iterator[str]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        **(headers or {}),
    }
    request = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_part = line.removeprefix("data:").strip()
                if data_part == "[DONE]":
                    break
                try:
                    payload = json.loads(data_part)
                except json.JSONDecodeError:
                    continue
                usage = _usage_from_payload(payload)
                if usage is not None and on_usage is not None:
                    on_usage(usage)
                text = _stream_delta_text(payload)
                if text:
                    yield text
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _stream_delta_text(payload: dict[str, object]) -> str:
    try:
        choices = payload["choices"]
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta") or {}
        if not isinstance(delta, dict):
            return ""
        return str(delta.get("content") or "")
    except (KeyError, TypeError):
        return ""


def _usage_from_payload(payload: dict[str, object]) -> LlmUsage | None:
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, dict):
        return None

    def as_int(key: str) -> int | None:
        value = raw_usage.get(key)
        return int(value) if isinstance(value, int | float) else None

    return LlmUsage(
        prompt_tokens=as_int("prompt_tokens"),
        completion_tokens=as_int("completion_tokens"),
        total_tokens=as_int("total_tokens"),
        prompt_cache_hit_tokens=as_int("prompt_cache_hit_tokens"),
        prompt_cache_miss_tokens=as_int("prompt_cache_miss_tokens"),
    )


def _chat_endpoint(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"
