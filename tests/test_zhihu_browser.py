from __future__ import annotations

import json

from personaforge.crawler.zhihu_browser import (
    content_id_from_url,
    has_zhihu_login_cookie,
    link_matches_kind,
    launch_chromium,
    load_storage_state,
    normalize_zhihu_link,
    question_id_from_answer_url,
    title_selectors_for_kind,
    content_selectors_for_kind,
)


def test_normalize_zhihu_link_accepts_absolute_and_relative_links() -> None:
    assert (
        normalize_zhihu_link("https://www.zhihu.com/question/1/answer/2?utm=abc#x")
        == "https://www.zhihu.com/question/1/answer/2"
    )
    assert normalize_zhihu_link("/question/1/answer/2") == "https://www.zhihu.com/question/1/answer/2"
    assert normalize_zhihu_link("https://example.com/question/1/answer/2") is None


def test_link_matching_and_id_extraction() -> None:
    answer = "https://www.zhihu.com/question/123/answer/456"
    article = "https://zhuanlan.zhihu.com/p/789"
    pin = "https://www.zhihu.com/pin/111"

    assert link_matches_kind(answer, "answer")
    assert link_matches_kind(article, "article")
    assert link_matches_kind(pin, "pin")
    assert content_id_from_url(answer, "answer") == "456"
    assert content_id_from_url(article, "article") == "789"
    assert content_id_from_url(pin, "pin") == "111"
    assert question_id_from_answer_url(answer) == "123"


def test_storage_state_helpers_accept_playwright_state_and_cookie_list(tmp_path) -> None:
    playwright_state = {"cookies": [{"name": "z_c0", "value": "token", "domain": ".zhihu.com"}], "origins": []}
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(playwright_state), encoding="utf-8")
    assert load_storage_state(state_path) == playwright_state

    cookie_path = tmp_path / "cookies.json"
    cookie_path.write_text(json.dumps(playwright_state["cookies"]), encoding="utf-8")
    assert load_storage_state(cookie_path)["cookies"] == playwright_state["cookies"]
    assert has_zhihu_login_cookie(playwright_state["cookies"])


def test_browser_selectors_cover_answer_article_and_pin() -> None:
    assert ".QuestionHeader-title" in title_selectors_for_kind("answer")
    assert ".Post-Title" in title_selectors_for_kind("article")
    assert ".PinItem-title" in title_selectors_for_kind("pin")
    assert ".RichContent-inner" in content_selectors_for_kind("answer")
    assert ".Post-RichText" in content_selectors_for_kind("article")
    assert ".PinItem-content" in content_selectors_for_kind("pin")


def test_launch_chromium_uses_installed_chrome_when_bundled_browser_is_missing() -> None:
    launched: list[dict[str, object]] = []
    expected = object()

    class Chromium:
        def launch(self, **kwargs):
            launched.append(kwargs)
            if "channel" not in kwargs:
                raise RuntimeError("Executable doesn't exist")
            if kwargs["channel"] == "chrome":
                return expected
            raise AssertionError("Edge should not be needed")

    class Playwright:
        chromium = Chromium()

    assert launch_chromium(Playwright(), headless=True) is expected
    assert launched == [{"headless": True}, {"channel": "chrome", "headless": True}]
