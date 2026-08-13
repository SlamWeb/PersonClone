from __future__ import annotations

from personaforge.studies.study1_materials import (
    build_codex_prompt,
    clone_bank_for_v2,
    prepare_bank,
    stable_choice,
    text_stats,
    write_json,
    write_jsonl,
)


def test_clone_bank_for_v2_preserves_stimuli_and_changes_identity(tmp_path) -> None:
    source = tmp_path / "legacy.json"
    payload = {
        "schema_version": "personaforge.study1.material-bank.v1",
        "study_id": "legacy-study",
        "items": [{"item_id": str(index), "responses": {}} for index in range(5)],
    }
    write_json(source, payload)
    bank = clone_bank_for_v2(
        source=source,
        out_dir=tmp_path / "v2",
        study_id="fresh-study-v2",
    )
    assert bank["study_id"] == "fresh-study-v2"
    assert bank["protocol_version"] == "study1-v2"
    assert bank["items"] == payload["items"]
    assert bank["cloned_from"]["study_id"] == "legacy-study"


def test_stable_choice_ignores_candidate_input_order() -> None:
    candidates = [
        {"answer_id": "3"},
        {"answer_id": "1"},
        {"answer_id": "2"},
    ]
    first = stable_choice(candidates, study_id="study", question_id="question")
    second = stable_choice(list(reversed(candidates)), study_id="study", question_id="question")
    assert first == second


def test_text_stats_preserve_natural_length_metadata() -> None:
    stats = text_stats("第一句。\n\n第二段！还有一句？")
    assert stats == {"chars": 15, "paragraphs": 2, "sentences": 3}


def test_codex_prompt_contains_only_supplied_reference_text() -> None:
    prompt = build_codex_prompt(
        question="测试问题",
        persona_pack={"claim": "稳定画像"},
        reference_parents=[{"title": "历史标题", "text": "历史正文"}],
    )
    assert "测试问题" in prompt
    assert "历史正文" in prompt
    assert "禁止联网" in prompt
    assert "未提供的 Gold 原文" not in prompt


def test_prepare_bank_supports_a_second_author_without_code_changes(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    parent_store = tmp_path / "authors" / "second-author" / "parents.jsonl"
    out_dir = tmp_path / "studies" / "second-author-study"
    persona_pack = tmp_path / "authors" / "second-author" / "persona_pack.json"
    write_jsonl(
        dataset_dir / "dataset.jsonl",
        [
            {
                "item_id": "dev-01",
                "split": "dev",
                "query": "第二位作者会如何回答？",
                "parent_id": "zhihu:answer:101",
                "gold_answer": "目标作者的冻结回答。",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
    )
    write_jsonl(
        parent_store,
        [
            {
                "doc_id": "zhihu:answer:101",
                "title": "测试标题",
                "text": "目标作者的冻结回答。",
                "metadata": {"question_id": "9001"},
            },
            {
                "doc_id": "zhihu:answer:99",
                "title": "历史表达",
                "text": "时间切分点之前的历史表达。",
                "metadata": {"question_id": "8001"},
            },
        ],
    )
    for run_name, answer in (("rag-v1", "RAG 回答"), ("persona-v1", "画像回答")):
        write_jsonl(
            dataset_dir / "runs" / run_name / "runs.jsonl",
            [
                {
                    "item_id": "dev-01",
                    "answer": answer,
                    "trace": {
                        "retrieval": {
                            "parents": [{"parent_id": "zhihu:answer:99"}]
                        }
                    },
                }
            ],
        )
        write_json(
            dataset_dir / "runs" / run_name / "manifest.json",
            {"writer_model": "test-model", "config": {}},
        )
    write_json(persona_pack, {"identity": "second-author"})

    bank = prepare_bank(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        parent_store_path=parent_store,
        rag_run_name="rag-v1",
        persona_run_name="persona-v1",
        persona_pack_path=persona_pack,
        author="second-author",
        author_label="第二位作者",
        study_id="second-author-study-v1",
    )

    assert bank["study_id"] == "second-author-study-v1"
    assert bank["author"] == {
        "platform": "zhihu",
        "token": "second-author",
        "label": "第二位作者",
    }
    assert bank["inputs"]["persona_pack_path"] == str(persona_pack)
    assert bank["items"][0]["reference_parent_ids"] == ["zhihu:answer:99"]
