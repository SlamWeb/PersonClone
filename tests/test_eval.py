from __future__ import annotations

import json

from personaforge.eval.dataset import prepare_temporal_dataset
from personaforge.eval.runner import EvalRunConfig, run_temporal_eval
from personaforge.ingest.query_understanding import RetrievalQuery
from personaforge.ingest.retrieve import ChildHit, ParentHit, RetrieveResult


class FakeLlm:
    model = "fake-model"

    def complete_text(self, messages, *, temperature=0.7, max_tokens=2048):
        return "生成回答"


def test_prepare_temporal_dataset_excludes_every_future_parent(tmp_path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    parents = [
        parent("zhihu:answer:1", "2025-01-01T00:00:00+08:00", "训练问题一"),
        parent("zhihu:pin:2", "2025-02-01T00:00:00+08:00", "训练想法", kind="pin"),
        parent("zhihu:answer:3", "2025-03-01T00:00:00+08:00", "开发问题"),
        parent("zhihu:article:4", "2025-03-02T00:00:00+08:00", "未来文章", kind="article"),
        parent("zhihu:answer:5", "2025-04-01T00:00:00+08:00", "测试问题一"),
        parent("zhihu:answer:6", "2025-05-01T00:00:00+08:00", "测试问题二"),
    ]
    write_parents(index_dir, parents)

    result = prepare_temporal_dataset(
        author="alice",
        index_dir=index_dir,
        out_dir=tmp_path / "eval",
        dev_size=1,
        test_size=2,
        min_answer_characters=10,
    )

    records = [json.loads(line) for line in result.dataset_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert [record["item_id"] for record in records] == ["dev-01", "test-01", "test-02"]
    assert [record["parent_id"] for record in records] == ["zhihu:answer:3", "zhihu:answer:5", "zhihu:answer:6"]
    assert manifest["selection"]["temporal_cutoff"] == "2025-03-01T00:00:00+08:00"
    assert manifest["excluded_parent_ids"] == [
        "zhihu:answer:3",
        "zhihu:answer:5",
        "zhihu:answer:6",
        "zhihu:article:4",
    ]


def test_prepare_sparse_author_test_only_uses_all_answers_including_short_ones(tmp_path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    parents = [
        parent("zhihu:article:old", "2024-01-01T00:00:00+08:00", "旧文章", kind="article"),
        parent("zhihu:answer:1", "2024-02-01T00:00:00+08:00", "短回答一", text="很短"),
        parent("zhihu:answer:2", "2024-03-01T00:00:00+08:00", "长回答二"),
        parent("zhihu:answer:3", "2024-04-01T00:00:00+08:00", "短回答三", text="也很短"),
    ]
    write_parents(index_dir, parents)

    result = prepare_temporal_dataset(
        author="sparse-author",
        index_dir=index_dir,
        out_dir=tmp_path / "eval",
        test_size=3,
        test_only=True,
    )

    records = [json.loads(line) for line in result.dataset_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.dev_count == 0
    assert result.test_count == 3
    assert [record["split"] for record in records] == ["test", "test", "test"]
    assert [record["parent_id"] for record in records] == [
        "zhihu:answer:1",
        "zhihu:answer:2",
        "zhihu:answer:3",
    ]
    assert manifest["selection"]["protocol"] == "sparse_author_test_only"
    assert manifest["selection"]["test_only"] is True
    assert manifest["selection"]["temporal_cutoff"] is None
    assert manifest["selection"]["corpus_policy"] == "exclude_eval_answers_only"
    assert manifest["excluded_parent_ids"] == [
        "zhihu:answer:1",
        "zhihu:answer:2",
        "zhihu:answer:3",
    ]


def test_eval_runner_writes_machine_and_human_artifacts(monkeypatch, tmp_path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    parents = [
        parent("zhihu:answer:1", "2025-01-01T00:00:00+08:00", "训练问题一"),
        parent("zhihu:answer:2", "2025-02-01T00:00:00+08:00", "开发问题"),
        parent("zhihu:answer:3", "2025-03-01T00:00:00+08:00", "测试问题一"),
        parent("zhihu:answer:4", "2025-04-01T00:00:00+08:00", "测试问题二"),
    ]
    write_parents(index_dir, parents)
    dataset = prepare_temporal_dataset(
        author="alice",
        index_dir=index_dir,
        out_dir=tmp_path / "eval",
        dev_size=1,
        test_size=2,
        min_answer_characters=10,
    )

    monkeypatch.setattr("personaforge.eval.runner.retrieve_parents", fake_retrieve)
    config = EvalRunConfig(
        author="alice",
        dataset_path=dataset.dataset_path,
        split="dev",
        run_name="baseline",
        out_dir=dataset.dataset_path.parent,
        query_mode="raw",
    )

    result = run_temporal_eval(
        config,
        index_dir=index_dir,
        qdrant_path=tmp_path / "qdrant",
        encoder=object(),
        llm=FakeLlm(),
    )

    run = json.loads(result.runs_path.read_text(encoding="utf-8").splitlines()[0])
    assert result.item_count == 1
    assert run["answer"] == "生成回答"
    assert "人工记录" in (result.run_dir / "items" / "dev-01.md").read_text(encoding="utf-8")
    assert json.loads(result.manifest_path.read_text(encoding="utf-8"))["status"] == "completed"


def parent(
    doc_id: str,
    created_at: str,
    title: str,
    *,
    kind: str = "answer",
    text: str | None = None,
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "kind": kind,
        "title": title,
        "text": text or "这是一段足够长的原作者回答，用于构造临时评测数据。",
        "created_at": created_at,
        "path": f"{doc_id}.md",
    }


def write_parents(index_dir, rows) -> None:
    (index_dir / "parents.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def fake_retrieve(query, **kwargs) -> RetrieveResult:
    child = ChildHit(
        rank=1,
        score=0.9,
        node_id="zhihu:answer:1:title:0",
        parent_id="zhihu:answer:1",
        node_type="title",
        title="训练问题一",
        path="answer-1.md",
        route="dense",
    )
    parent_hit = ParentHit(
        rank=1,
        score=0.1,
        parent_id="zhihu:answer:1",
        title="训练问题一",
        path="answer-1.md",
        first_hits=[child],
        parent={"text": "训练正文"},
    )
    return RetrieveResult(
        query=query,
        collection_name="personaforge__zhihu__alice",
        child_top_k=100,
        parent_top_k=20,
        routes={"dense": [child]},
        parents=[parent_hit],
        retrieval_queries=[RetrievalQuery(route="original_semantics", query=query)],
    )
