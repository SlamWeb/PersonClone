from __future__ import annotations

import json
import hashlib
from pathlib import Path

from personaforge.eval.retrieval_judge import label_pool, materialize_codex_labels
from personaforge.eval.retrieval_gold_qrels import (
    _judge_batch_resilient,
    compare_v1_v2,
    export_codex_gold_handoff,
    extract_gold_units,
    label_gold_aware_pool,
    materialize_codex_gold_labels,
)
from personaforge.eval.dataset import sha256_json
from personaforge.eval.retrieval_metrics import compute_retrieval_metrics, compute_split_metrics, sort_candidates_by_relevance
from personaforge.web.retrieval_evaluation import RetrievalEvaluationStore
from personaforge.llm import LlmUsage


class FakeJudge:
    model = "fake-retrieval-judge"

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, messages, *, temperature=0.0, max_tokens=900):
        self.calls += 1
        return {"score": 2, "confidence": "high", "evidence": "直接命中", "reason": "覆盖问题核心对象"}


class FakeGoldAwareJudge:
    model = "fake-gold-aware-judge"

    def complete_json(self, messages, *, temperature=0.0, max_tokens=6500):
        system = messages[0]["content"]
        if "拆成少量、原子的评估单元" in system:
            return {
                "stance": ["作者支持结论 A"],
                "reasoning": ["因为机制 B"],
                "example": [],
                "expression": ["先下判断再解释"],
            }
        payload = json.loads(messages[1]["content"].split("\n", 1)[1])
        labels = []
        for candidate in payload["candidates"]:
            useful = "支持" in candidate["material"]
            labels.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "content_support": 2 if useful else 0,
                    "persona_expression_support": 1 if useful else 0,
                    "confidence": "high",
                    "content_candidate_evidence": "直接支持" if useful else "",
                    "content_gold_unit_ids": ["stance-1"] if useful else [],
                    "persona_candidate_evidence": "先判断" if useful else "",
                    "persona_gold_unit_ids": ["expression-1"] if useful else [],
                    "reason": "可复用" if useful else "没有支撑",
                }
            )
        return {"labels": labels}


class ScriptedGoldAwareJudge:
    model = "scripted-gold-aware-judge"

    def __init__(self, *, fail_on_calls: set[int] | None = None) -> None:
        self.calls = 0
        self.fail_on_calls = fail_on_calls or set()

    def complete_json(self, messages, *, temperature=0.0, max_tokens=6500):
        self.calls += 1
        if self.calls in self.fail_on_calls:
            raise RuntimeError("simulated provider interruption")
        payload = json.loads(messages[1]["content"].split("\n", 1)[1])
        return {
            "labels": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "content_support": 1,
                    "persona_expression_support": 1,
                    "confidence": "high",
                    "content_candidate_evidence": "局部支持",
                    "content_gold_unit_ids": ["stance-1"],
                    "persona_candidate_evidence": "表达可迁移",
                    "persona_gold_unit_ids": ["expression-1"],
                    "reason": "需要复评的边界样本",
                }
                for candidate in payload["candidates"]
            ]
        }


class SplitRecoveryGoldAwareJudge:
    model = "split-recovery-gold-aware-judge"

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, messages, *, temperature=0.0, max_tokens=6500):
        self.calls += 1
        payload = json.loads(messages[1]["content"].split("\n", 1)[1])
        candidates = payload["candidates"]
        if len(candidates) > 1:
            candidates = candidates[:-1]
        return {
            "labels": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "content_support": 0,
                    "persona_expression_support": 0,
                    "confidence": "high",
                    "content_candidate_evidence": "",
                    "content_gold_unit_ids": [],
                    "persona_candidate_evidence": "",
                    "persona_gold_unit_ids": [],
                    "reason": "无支撑",
                }
                for candidate in candidates
            ]
        }


class UsageAwareGoldJudge(FakeGoldAwareJudge):
    def __init__(self) -> None:
        self.payload_key_orders: list[list[str]] = []

    def complete_json_with_usage(self, messages, *, temperature=0.0, max_tokens=6500):
        payload = json.loads(messages[1]["content"].split("\n", 1)[1])
        self.payload_key_orders.append(list(payload))
        result = super().complete_json(messages, temperature=temperature, max_tokens=max_tokens)
        return result, LlmUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_cache_hit_tokens=80,
            prompt_cache_miss_tokens=20,
        )


def write_pool(data_dir: Path) -> Path:
    directory = data_dir / "eval" / "demo"
    directory.mkdir(parents=True)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "personaforge.eval.retrieval_pool.v0",
                "pool_id": "demo.pool",
                "dataset_id": "demo",
                "author": "author-a",
                "split": "dev",
                "pool_file": "pool.jsonl",
            }
        ),
        encoding="utf-8",
    )
    record = {
        "item_id": "dev-01",
        "query": "测试问题",
        "candidates": [
            {
                "parent_id": "parent-a",
                "title": "材料 A",
                "text": "完整材料 A",
                "url": "https://example.com/a",
                "kind": "answer",
                "route_ranks": {"raw_dense": {"rank": 1, "score": 0.9}},
            },
            {
                "parent_id": "parent-b",
                "title": "材料 B",
                "text": "完整材料 B",
                "url": "https://example.com/b",
                "kind": "article",
                "route_ranks": {"raw_dense": {"rank": 2, "score": 0.8}},
            },
        ],
    }
    (directory / "pool.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def test_metrics_use_labels_without_route_duplicates() -> None:
    records = [
        {
            "item_id": "q1",
            "candidates": [
                {"parent_id": "a", "route_ranks": {"raw_dense": {"rank": 1, "score": 0.9}}},
                {"parent_id": "b", "route_ranks": {"raw_dense": {"rank": 2, "score": 0.8}}},
            ],
        }
    ]
    metrics = compute_retrieval_metrics(records, {("q1", "a"): 2, ("q1", "b"): 0})
    route = metrics["routes"]["raw_dense"]
    assert route["coverage"] == 1.0
    assert route["hit_at_k"] == 1.0
    assert route["mrr_at_k"] == 1.0
    assert route["precision_at_k"] == 0.5
    assert route["map_at_k"] == 1.0
    assert route["no_relevant_query_count"] == 0


def test_multi_k_metrics_include_recall_and_split_reports() -> None:
    records = [
        {
            "item_id": "dev-01",
            "split": "dev",
            "candidates": [
                {"parent_id": "a", "route_ranks": {"raw_dense": {"rank": 1, "score": 0.9}}},
                {"parent_id": "b", "route_ranks": {"raw_dense": {"rank": 2, "score": 0.8}}},
            ],
        },
        {
            "item_id": "test-01",
            "split": "test",
            "candidates": [
                {"parent_id": "c", "route_ranks": {"raw_dense": {"rank": 1, "score": 0.9}}},
                {"parent_id": "d", "route_ranks": {"raw_dense": {"rank": 2, "score": 0.8}}},
            ],
        },
    ]
    labels = {("dev-01", "a"): 2, ("dev-01", "b"): 1, ("test-01", "c"): 0, ("test-01", "d"): 2}
    metrics = compute_split_metrics(records, labels, cutoff=1, cutoffs=(1, 2))

    assert metrics["cutoffs"] == [1, 2]
    assert set(metrics["splits"]) == {"dev", "test"}
    assert metrics["routes"]["raw_dense"]["by_cutoff"]["1"]["recall_at_k"] == 0.25
    assert metrics["routes"]["raw_dense"]["by_cutoff"]["2"]["recall_at_k"] == 1.0


def test_label_pool_is_resumable_and_web_can_read_ranked_report(tmp_path: Path) -> None:
    manifest_path = write_pool(tmp_path)
    judge = FakeJudge()
    first = label_pool(manifest_path, client=judge)
    assert first["manifest"]["completed"] == 2
    assert judge.calls == 2

    second = label_pool(manifest_path, client=judge)
    assert second["manifest"]["completed"] == 2
    assert judge.calls == 2

    store = RetrievalEvaluationStore(tmp_path)
    report = store.llm_query("demo.pool", "llm_relevance_v1", "dev-01")
    assert report["labeled_count"] == 2
    assert [row["relevance_order"] for row in report["candidates"]] == [1, 2]
    assert report["candidates"][0]["score"] == 2


def test_relevance_sort_keeps_route_ranks_for_audit() -> None:
    candidates = [
        {"parent_id": "a", "route_ranks": {"raw_dense": {"rank": 1, "score": 0.9}}},
        {"parent_id": "b", "route_ranks": {"raw_sparse": {"rank": 2, "score": 0.8}}},
    ]
    rows = sort_candidates_by_relevance(
        candidates,
        {("q1", "a"): {"score": 0}, ("q1", "b"): {"score": 2}},
        item_id="q1",
    )
    assert rows[0]["parent_id"] == "b"
    assert rows[0]["route_ranks"]["raw_sparse"]["rank"] == 2


def test_label_pool_compacts_duplicate_rows_before_resume(tmp_path: Path) -> None:
    manifest_path = write_pool(tmp_path)
    output_dir = manifest_path.parent / "llm_labels" / "llm_relevance_v1"
    output_dir.mkdir(parents=True)
    duplicate = {
        "item_id": "dev-01",
        "parent_id": "parent-a",
        "score": 2,
        "status": "completed",
        "confidence": "high",
    }
    (output_dir / "labels.jsonl").write_text(
        json.dumps(duplicate) + "\n" + json.dumps({**duplicate, "score": 1}) + "\n",
        encoding="utf-8",
    )
    judge = FakeJudge()
    result = label_pool(manifest_path, client=judge)
    rows = [
        json.loads(line)
        for line in (output_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert len({(row["item_id"], row["parent_id"]) for row in rows}) == 2
    assert result["manifest"]["completed"] == 2


def test_codex_review_materializes_explicit_zero_labels_only_when_complete(tmp_path: Path) -> None:
    manifest_path = write_pool(tmp_path)
    pool_path = manifest_path.parent / "pool.jsonl"
    pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    review_path = manifest_path.parent / "codex_review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": "personaforge.eval.retrieval_codex_review.v1",
                "pool_id": "demo.pool",
                "pool_sha256": pool_sha,
                "reviewer": "codex-test",
                "items": [
                    {
                        "item_id": "dev-01",
                        "review_complete": True,
                        "nonzero_labels": [
                            {"parent_id": "parent-a", "score": 2, "reason": "直接支持", "evidence": "材料 A"}
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = materialize_codex_labels(manifest_path, review_path, label_set="codex_test")
    rows = [json.loads(line) for line in result["labels_path"].read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {(row["parent_id"], row["score"]) for row in rows} == {("parent-a", 2), ("parent-b", 0)}
    assert result["metrics"]["cutoffs"] == [1, 3, 5, 10, 20, 30]


def test_gold_aware_qrels_are_dual_axis_resumable_and_comparable(tmp_path: Path) -> None:
    dataset = [
        {
            "item_id": "dev-01",
            "split": "dev",
            "query": "测试问题",
            "parent_id": "gold-parent",
            "gold_answer": "作者支持结论 A，因为机制 B。",
        }
    ]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps(dataset[0], ensure_ascii=False) + "\n", encoding="utf-8")
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    pool_record = {
        "item_id": "dev-01",
        "split": "dev",
        "query": "测试问题",
        "target_parent_id": "gold-parent",
        "candidates": [
            {
                "parent_id": "helpful",
                "title": "有用材料",
                "text": "这篇材料直接支持结论。",
                "route_ranks": {"raw_dense": {"rank": 1, "score": 1.0}},
            },
            {
                "parent_id": "useless",
                "title": "无用材料",
                "text": "完全无关。",
                "route_ranks": {"raw_dense": {"rank": 2, "score": 0.5}},
            },
        ],
    }
    (pool_dir / "pool.jsonl").write_text(json.dumps(pool_record, ensure_ascii=False) + "\n", encoding="utf-8")
    pool_manifest = {
        "schema_version": "personaforge.eval.retrieval_pool.v1",
        "pool_id": "gold-aware-demo",
        "dataset_id": "demo",
        "dataset_sha256": sha256_json(dataset),
        "dataset_path": str(dataset_path),
        "pool_file": "pool.jsonl",
        "split": "all",
    }
    pool_manifest_path = pool_dir / "manifest.json"
    pool_manifest_path.write_text(json.dumps(pool_manifest), encoding="utf-8")

    judge = FakeGoldAwareJudge()
    units = extract_gold_units(dataset_path, client=judge)
    progress_events: list[tuple[str, int, int]] = []
    result = label_gold_aware_pool(
        pool_manifest_path,
        dataset_path=dataset_path,
        gold_units_path=units["path"],
        client=judge,
        batch_size=2,
        max_concurrency=1,
        stability_sample_rate=0.0,
        progress=lambda phase, current, total: progress_events.append((phase, current, total)),
    )
    labels = [json.loads(line) for line in result["labels_path"].read_text(encoding="utf-8").splitlines()]
    assert result["manifest"]["completed"] == 2
    assert result["metrics"]["no_combined_score"] is True
    assert result["metrics"]["axes"]["content_support"]["recall_scope"] == "six_route_candidate_union"
    assert {(row["parent_id"], row["content_support"], row["persona_expression_support"]) for row in labels} == {
        ("helpful", 2, 1),
        ("useless", 0, 0),
    }
    assert ("pass-1", 2, 2) in progress_events

    v1_dir = pool_dir / "llm_labels" / "v1"
    v1_dir.mkdir()
    (v1_dir / "labels.jsonl").write_text(
        "".join(
            json.dumps({"item_id": "dev-01", "parent_id": parent_id, "score": score, "status": "completed"}) + "\n"
            for parent_id, score in [("helpful", 0), ("useless", 0)]
        ),
        encoding="utf-8",
    )
    (v1_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "personaforge.eval.retrieval_llm_labels.v1", "label_set": "v1", "pool_id": "gold-aware-demo", "labels_file": "labels.jsonl"}),
        encoding="utf-8",
    )
    comparison = compare_v1_v2(
        pool_manifest_path,
        v1_label_manifest=v1_dir / "manifest.json",
        v2_label_manifest=result["manifest_path"],
    )
    assert comparison["report"]["all"]["v1_zero_to_v2_positive"] == 1


def test_gold_aware_prompt_is_candidate_first_and_usage_is_request_local() -> None:
    judge = UsageAwareGoldJudge()
    usage_rows = []
    rows = _judge_batch_resilient(
        judge,
        item_id="dev-01",
        question="测试问题",
        gold_answer="作者真实回答",
        gold_units={
            "stance": [{"id": "stance-1", "text": "判断"}],
            "reasoning": [{"id": "reasoning-1", "text": "机制"}],
            "example": [],
            "expression": [{"id": "expression-1", "text": "表达"}],
        },
        candidates=[{"parent_id": "candidate-a", "title": "A", "text": "支持材料"}],
        pass_index=1,
        max_tokens=6500,
        max_attempts=1,
        usage_callback=usage_rows.append,
    )
    assert rows[0]["parent_id"] == "candidate-a"
    assert judge.payload_key_orders == [["candidates", "question", "gold_answer", "gold_units"]]
    assert usage_rows[0]["prompt_cache_hit_tokens"] == 80
    assert usage_rows[0]["prompt_cache_miss_tokens"] == 20


def test_dual_axis_codex_handoff_round_trip(tmp_path: Path) -> None:
    dataset = [{"item_id": "dev-01", "split": "dev", "query": "测试问题", "parent_id": "gold", "gold_answer": "作者支持结论。"}]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps(dataset[0], ensure_ascii=False) + "\n", encoding="utf-8")
    units_path = tmp_path / "gold_units.jsonl"
    units_path.write_text(
        json.dumps(
            {
                "item_id": "dev-01",
                "gold_answer_sha256": hashlib.sha256(dataset[0]["gold_answer"].encode("utf-8")).hexdigest(),
                "units": {
                    "stance": [{"id": "stance-1", "text": "作者支持结论"}],
                    "reasoning": [{"id": "reasoning-1", "text": "理由"}],
                    "example": [],
                    "expression": [{"id": "expression-1", "text": "先判断"}],
                },
            },
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    pool_record = {
        "item_id": "dev-01",
        "split": "dev",
        "query": "测试问题",
        "target_parent_id": "gold",
        "candidates": [{"parent_id": "candidate-a", "title": "A", "text": "支持结论", "route_ranks": {"raw_dense": {"rank": 1}}}],
    }
    (pool_dir / "pool.jsonl").write_text(json.dumps(pool_record, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = pool_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "personaforge.eval.retrieval_pool.v1", "pool_id": "codex-dual", "dataset_sha256": sha256_json(dataset), "pool_file": "pool.jsonl"}),
        encoding="utf-8",
    )

    handoff = export_codex_gold_handoff(
        manifest_path,
        dataset_path=dataset_path,
        gold_units_path=units_path,
    )
    review = json.loads(handoff["template_path"].read_text(encoding="utf-8"))
    review["items"][0]["review_complete"] = True
    review["items"][0]["labels"] = [{
        "candidate_id": "candidate-a",
        "content_support": 2,
        "persona_expression_support": 1,
        "confidence": "high",
        "content_candidate_evidence": "支持结论",
        "content_gold_unit_ids": ["stance-1"],
        "persona_candidate_evidence": "先判断",
        "persona_gold_unit_ids": ["expression-1"],
        "reason": "同时提供内容和表达支撑",
    }]
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
    result = materialize_codex_gold_labels(
        manifest_path,
        review_path,
        dataset_path=dataset_path,
        gold_units_path=units_path,
    )
    assert result["manifest"]["status"] == "completed"
    assert result["manifest"]["labeler"] == "codex_handoff"
    row = json.loads(result["labels_path"].read_text(encoding="utf-8"))
    assert (row["content_support"], row["persona_expression_support"]) == (2, 1)


def test_gold_units_can_be_frozen_for_one_split_only(tmp_path: Path) -> None:
    rows = [
        {"item_id": "dev-01", "split": "dev", "query": "开发题", "gold_answer": "开发答案"},
        {"item_id": "test-01", "split": "test", "query": "测试题", "gold_answer": "测试答案"},
    ]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    judge = FakeGoldAwareJudge()
    result = extract_gold_units(
        dataset_path,
        client=judge,
        out_path=tmp_path / "gold_units_dev.jsonl",
        splits=["dev"],
    )
    written = [json.loads(line) for line in result["path"].read_text(encoding="utf-8").splitlines()]
    assert [row["item_id"] for row in written] == ["dev-01"]
    assert result["manifest"]["splits"] == ["dev"]


def test_gold_aware_resume_finishes_pending_stability_passes(tmp_path: Path) -> None:
    dataset = [
        {
            "item_id": "dev-01",
            "split": "dev",
            "query": "测试问题",
            "parent_id": "gold-parent",
            "gold_answer": "作者支持结论 A，因为机制 B。",
        }
    ]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps(dataset[0], ensure_ascii=False) + "\n", encoding="utf-8")
    gold_units_path = tmp_path / "gold_units.jsonl"
    gold_units_path.write_text(
        json.dumps(
            {
                "item_id": "dev-01",
                "gold_answer_sha256": hashlib.sha256(dataset[0]["gold_answer"].encode("utf-8")).hexdigest(),
                "units": {
                    "stance": [{"id": "stance-1", "text": "作者支持结论 A"}],
                    "reasoning": [{"id": "reasoning-1", "text": "因为机制 B"}],
                    "example": [],
                    "expression": [{"id": "expression-1", "text": "先判断再解释"}],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    pool_record = {
        "item_id": "dev-01",
        "split": "dev",
        "query": "测试问题",
        "target_parent_id": "gold-parent",
        "candidates": [
            {
                "parent_id": "candidate-1",
                "title": "边界材料",
                "text": "这篇材料提供局部支持。",
                "route_ranks": {"raw_dense": {"rank": 1, "score": 1.0}},
            }
        ],
    }
    (pool_dir / "pool.jsonl").write_text(json.dumps(pool_record, ensure_ascii=False) + "\n", encoding="utf-8")
    pool_manifest_path = pool_dir / "manifest.json"
    pool_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "personaforge.eval.retrieval_pool.v1",
                "pool_id": "resume-demo",
                "dataset_id": "demo",
                "dataset_sha256": sha256_json(dataset),
                "dataset_path": str(dataset_path),
                "pool_file": "pool.jsonl",
                "split": "all",
            }
        ),
        encoding="utf-8",
    )

    interrupted = ScriptedGoldAwareJudge(fail_on_calls={2})
    partial = label_gold_aware_pool(
        pool_manifest_path,
        dataset_path=dataset_path,
        gold_units_path=gold_units_path,
        client=interrupted,
        label_set="resume-v2",
        batch_size=1,
        max_concurrency=1,
        max_attempts=1,
        stability_sample_rate=0.0,
    )
    assert interrupted.calls == 2
    assert partial["manifest"]["status"] == "partial"
    assert partial["manifest"]["completed"] == 0
    assert partial["manifest"]["progress"] == {
        "pass1_completed": 1,
        "judge_pass1_completed": 1,
        "missing_pass1": 0,
        "pass2_required": 1,
        "pass2_completed": 0,
        "pending_pass2": 1,
        "pass3_required": 0,
        "pass3_completed": 0,
        "pending_pass3": 0,
        "stability_completed": 0,
    }
    assert partial["labels_path"].read_text(encoding="utf-8") == ""

    resumed_judge = ScriptedGoldAwareJudge()
    resumed = label_gold_aware_pool(
        pool_manifest_path,
        dataset_path=dataset_path,
        gold_units_path=gold_units_path,
        client=resumed_judge,
        label_set="resume-v2",
        batch_size=1,
        max_concurrency=1,
        max_attempts=1,
        stability_sample_rate=0.0,
    )
    assert resumed_judge.calls == 1
    assert resumed["manifest"]["status"] == "completed"
    assert resumed["manifest"]["completed"] == 1
    label = json.loads(resumed["labels_path"].read_text(encoding="utf-8"))
    assert label["repeat_count"] == 2
    assert label["required_passes"] == [1, 2]
    assert label["completed_passes"] == [1, 2]
    assert label["stability_complete"] is True

    seeded_judge = ScriptedGoldAwareJudge(fail_on_calls={1})
    seeded = label_gold_aware_pool(
        pool_manifest_path,
        dataset_path=dataset_path,
        gold_units_path=gold_units_path,
        client=seeded_judge,
        label_set="seed-copy-v2",
        seed_label_manifest=resumed["manifest_path"],
        batch_size=1,
        max_concurrency=1,
        max_attempts=1,
        stability_sample_rate=0.0,
    )
    assert seeded_judge.calls == 0
    assert seeded["manifest"]["status"] == "completed"
    assert seeded["manifest"]["seeded"] == 1


def test_gold_aware_batch_falls_back_to_smaller_groups() -> None:
    judge = SplitRecoveryGoldAwareJudge()
    rows = _judge_batch_resilient(
        judge,
        item_id="dev-01",
        question="测试问题",
        gold_answer="作者真实回答",
        gold_units={
            "stance": [{"id": "stance-1", "text": "判断"}],
            "reasoning": [{"id": "reasoning-1", "text": "机制"}],
            "example": [],
            "expression": [{"id": "expression-1", "text": "表达"}],
        },
        candidates=[
            {"parent_id": "candidate-a", "title": "A", "text": "材料 A"},
            {"parent_id": "candidate-b", "title": "B", "text": "材料 B"},
        ],
        pass_index=1,
        max_tokens=6500,
        max_attempts=1,
    )
    assert judge.calls == 3
    assert [row["parent_id"] for row in rows] == ["candidate-a", "candidate-b"]
