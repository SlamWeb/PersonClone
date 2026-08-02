from __future__ import annotations

import json

from personaforge import cli
from personaforge.crawler.models import ContentItem, CreatorProfile


class FakePublicCrawler:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def crawl_profile(self, user: str) -> CreatorProfile:
        return CreatorProfile(
            source="zhihu",
            author_token=user,
            nickname="Fake Author",
            profile_url=f"https://www.zhihu.com/people/{user}",
        )

    def crawl_user(self, user: str, *, kinds, max_items):
        return [
            ContentItem(
                source="zhihu",
                kind="answer",
                id="1",
                title="Fake question",
                url="https://www.zhihu.com/question/1/answer/1",
                author_token=user,
                content_html="<p>Fake answer.</p>",
                content_text="Fake answer.",
                fetched_at="2026-01-01T00:00:00+00:00",
            )
        ]


class PartialPublicCrawler(FakePublicCrawler):
    def crawl_user(self, user: str, *, kinds, max_items):
        kind = tuple(kinds)[0]
        if kind != "pin":
            return []
        return [
            ContentItem(
                source="zhihu",
                kind="pin",
                id="pin-1",
                title="Public pin",
                url="https://www.zhihu.com/pin/pin-1",
                author_token=user,
                content_html="<p>Pin.</p>",
                content_text="Pin.",
                fetched_at="2026-01-01T00:00:00+00:00",
            )
        ]


class MissingKindBrowserCrawler:
    requested_kinds: list[str] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def crawl_profile(self, user: str) -> CreatorProfile:
        raise AssertionError("Public profile should be reused")

    def crawl_user(self, user: str, *, kinds, max_items):
        kind = tuple(kinds)[0]
        self.requested_kinds.append(kind)
        return [
            ContentItem(
                source="zhihu",
                kind=kind,
                id=f"{kind}-1",
                title=f"Browser {kind}",
                url=f"https://example.com/{kind}/1",
                author_token=user,
                content_html=f"<p>{kind}.</p>",
                content_text=f"{kind}.",
                fetched_at="2026-01-01T00:00:00+00:00",
            )
        ]


def test_cli_crawl_writes_raw_markdown(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "ZhihuPublicCrawler", FakePublicCrawler)

    out_dir = tmp_path / "raw"
    code = cli.main(["crawl", "zhihu", "alice", "--out-dir", str(out_dir), "--no-browser", "--quiet"])

    assert code == 0
    assert json.loads((out_dir / "profile.json").read_text(encoding="utf-8"))["nickname"] == "Fake Author"
    assert (out_dir / "manifest.jsonl").exists()
    manifest = json.loads((out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert manifest["path"] == "answer/answer-1-Fake-question.md"
    markdown_files = sorted(path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*.md"))
    assert markdown_files == ["answer/answer-1-Fake-question.md"]


def test_cli_crawl_default_output_is_author_scoped(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "ZhihuPublicCrawler", FakePublicCrawler)
    monkeypatch.chdir(tmp_path)

    code = cli.main(["crawl", "zhihu", "alice", "--no-browser", "--quiet"])

    assert code == 0
    raw_dir = tmp_path / "data" / "authors" / "zhihu" / "alice" / "raw"
    assert (raw_dir / "profile.json").exists()
    assert (raw_dir / "manifest.jsonl").exists()
    assert (raw_dir / "answer" / "answer-1-Fake-question.md").exists()


def test_cli_crawl_falls_back_per_missing_content_kind(monkeypatch, tmp_path) -> None:
    MissingKindBrowserCrawler.requested_kinds = []
    monkeypatch.setattr(cli, "ZhihuPublicCrawler", PartialPublicCrawler)
    monkeypatch.setattr(cli, "ZhihuBrowserCrawler", MissingKindBrowserCrawler)

    out_dir = tmp_path / "raw"
    code = cli.main(["crawl", "zhihu", "alice", "--out-dir", str(out_dir), "--all", "--quiet"])

    assert code == 0
    assert MissingKindBrowserCrawler.requested_kinds == ["answer", "article"]
    rows = [json.loads(line) for line in (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["kind"] for row in rows} == {"answer", "article", "pin"}
