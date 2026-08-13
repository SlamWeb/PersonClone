from __future__ import annotations

import csv
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from personaforge.studies.study1_service import (
    AI_SOURCES,
    PROTOCOL_VERSION,
    Study1Store,
    StudyExposureRequest,
    StudyHighlight,
    StudyPairwiseRequest,
    StudyPointwiseRequest,
    StudyProfileRequest,
    StudyTransitionRequest,
    build_assignment,
)
from personaforge.studies.study1_analysis import analyze_bundle, render_markdown
from personaforge.web.app import create_app
from personaforge.web.service import WebConfig


def _bank(
    data_dir: Path,
    *,
    folder: str = "demo",
    study_id: str = "study-demo-v2",
    author: str = "demo-author",
    protocol_version: str = PROTOCOL_VERSION,
) -> dict:
    target = data_dir / "studies" / folder
    target.mkdir(parents=True, exist_ok=True)
    sources = ("gold", "rag_identity", "persona_pack", "codex", "other_human")
    items = []
    for index in range(10):
        text = f"第{index}题的回答正文。"
        items.append(
            {
                "item_id": f"dev-{index:02d}",
                "question_id": f"question-{index}",
                "question": f"测试问题 {index}？",
                "responses": {
                    source: {"text": f"{text}{source} 提供自己的解释。"}
                    for source in sources
                },
            }
        )
    bank = {
        "schema_version": "personaforge.study1.material-bank.v2",
            "protocol_version": protocol_version,
        "study_id": study_id,
        "author": {"platform": "zhihu", "token": author},
        "items": items,
    }
    (target / "material_bank.json").write_text(
        json.dumps(bank, ensure_ascii=False), encoding="utf-8"
    )
    return bank


def _profile(code: str) -> StudyProfileRequest:
    return StudyProfileRequest(
        participant_code=code,
        follow_duration="1_to_3_years",
        reading_frequency="weekly",
        familiarity="familiar",
        ai_frequency="almost_daily",
        consent=True,
    )


def _highlight(text: str, *, start: int = 0, impact: int = 2) -> StudyHighlight:
    return StudyHighlight(
        annotation_id=f"annotation-{uuid4().hex}",
        start=start,
        end=start + 1,
        selected_text=text[start : start + 1],
        impact=impact,
        reason="这处文字影响了判断",
    )


def _submit_pointwise(store: Study1Store, state: dict, score: int = 1) -> dict:
    trial = state["trial"]
    return store.save_pointwise(
        state["session_id"],
        trial["trial_id"],
        StudyPointwiseRequest(
            overall_score=score,
            highlights=[_highlight(trial["answer"], impact=2 if score >= 0 else -2)],
            primary_reason="整篇最关键的判断理由",
            elapsed_ms=1000,
            submit=True,
        ),
    )


def _submit_pairwise(store: Study1Store, state: dict) -> dict:
    trial = state["trial"]
    return store.save_pairwise(
        state["session_id"],
        trial["trial_id"],
        StudyPairwiseRequest(
            choice="left",
            confidence="fairly_sure",
            selected_reason="选择它的关键原因",
            rejected_reason="没有选择另一篇的关键原因",
            elapsed_ms=1000,
            submit=True,
        ),
    )


def _complete_formal_trials(store: Study1Store, state: dict) -> dict:
    state = _submit_pointwise(store, state)
    state = _submit_pointwise(store, state, score=-1)
    assert state["phase"] == "transition"
    state = store.acknowledge_transition(
        state["session_id"], StudyTransitionRequest(acknowledge=True)
    )
    state = _submit_pairwise(store, state)
    return _submit_pairwise(store, state)


def test_study1_v2_complete_flow_has_transition_and_four_unique_questions(tmp_path: Path) -> None:
    _bank(tmp_path)
    store = Study1Store(tmp_path)
    code = store.create_codes(1)[0]
    first = store.start(_profile(code))
    resumed = store.start(_profile(code.lower()))
    assert first["session_id"] == resumed["session_id"]
    assert first["protocol_version"] == PROTOCOL_VERSION

    questions = {first["trial"]["question"]}
    state = _submit_pointwise(store, first)
    questions.add(state["trial"]["question"])
    state = _submit_pointwise(store, state, score=-1)
    assert state["phase"] == "transition"
    assert state["progress"] == {"completed": 2, "total": 4}

    state = store.acknowledge_transition(
        state["session_id"], StudyTransitionRequest(acknowledge=True)
    )
    questions.add(state["trial"]["question"])
    state = _submit_pairwise(store, state)
    questions.add(state["trial"]["question"])
    state = _submit_pairwise(store, state)
    assert state["phase"] == "exposure"
    assert len(questions) == 4

    state = store.save_exposure(state["session_id"], StudyExposureRequest(value="no"))
    assert state["phase"] == "completed"
    assert state["progress"] == {"completed": 4, "total": 4}
    detail = store.detail(first["session_id"])
    assert len(detail["pointwise"]) == 2
    assert len(detail["pairwise"]) == 2
    assert detail["phase2_started_at"]
    assert {event["event_type"] for event in detail["events"]} >= {
        "phase2_entered",
        "pair_choice_selected",
        "pair_confidence_selected",
        "trial_submitted",
        "study_submitted",
    }


def test_legacy_material_is_replay_only_and_catalog_prefers_v2(tmp_path: Path) -> None:
    _bank(tmp_path, folder="legacy", study_id="study-demo-v1", protocol_version="study1-v1")
    _bank(tmp_path, folder="current", study_id="study-demo-v2")
    store = Study1Store(tmp_path)

    catalog = store.study_catalog()
    legacy = next(item for item in catalog if item["study_id"] == "study-demo-v1")
    current = next(item for item in catalog if item["study_id"] == "study-demo-v2")
    assert legacy["available"] is True
    assert legacy["recruitable"] is False
    assert current["recruitable"] is True
    with pytest.raises(ValueError, match="旧协议"):
        store.create_codes(1, "study-demo-v1")


def test_pointwise_validates_score_spans_reasons_and_allows_mixed_direction(tmp_path: Path) -> None:
    _bank(tmp_path)
    store = Study1Store(tmp_path)
    state = store.start(_profile(store.create_codes(1)[0]))
    trial = state["trial"]
    text = trial["answer"]

    with pytest.raises(ValueError, match="作者相似度"):
        store.save_pointwise(
            state["session_id"],
            trial["trial_id"],
            StudyPointwiseRequest(
                highlights=[_highlight(text)],
                primary_reason="理由",
                submit=True,
            ),
        )
    with pytest.raises(ValueError, match="至少标注一处"):
        store.save_pointwise(
            state["session_id"],
            trial["trial_id"],
            StudyPointwiseRequest(
                overall_score=1, primary_reason="理由", submit=True
            ),
        )

    mixed = [_highlight(text, start=0, impact=2), _highlight(text, start=2, impact=-2)]
    saved = store.save_pointwise(
        state["session_id"],
        trial["trial_id"],
        StudyPointwiseRequest(
            overall_score=-2,
            highlights=mixed,
            primary_reason="整体仍然不像，但存在一处正向证据",
            submit=False,
        ),
    )
    assert [item["impact"] for item in saved["draft"]["highlights"]] == [2, -2]

    overlapping = [_highlight(text, start=0), StudyHighlight(
        annotation_id="overlap-annotation", start=0, end=2, selected_text=text[:2], impact=-1, reason="重叠"
    )]
    with pytest.raises(ValueError, match="不能重叠"):
        store.save_pointwise(
            state["session_id"], trial["trial_id"],
            StudyPointwiseRequest(overall_score=0, highlights=overlapping, primary_reason="理由", submit=False),
        )


def test_transition_cannot_be_bypassed_and_previous_keeps_draft(tmp_path: Path) -> None:
    _bank(tmp_path)
    store = Study1Store(tmp_path)
    state = store.start(_profile(store.create_codes(1)[0]))
    first_trial = state["trial"]
    state = _submit_pointwise(store, state)
    state = _submit_pointwise(store, state)
    assert state["phase"] == "transition"

    assignment = store.detail(state["session_id"])["pairwise"][0]["trial"]
    with pytest.raises(ValueError, match="只能保存当前题目"):
        store.save_pairwise(
            state["session_id"], assignment["trial_id"],
            StudyPairwiseRequest(choice="left", confidence="close", selected_reason="选", rejected_reason="不选", submit=True),
        )

    previous = store.navigate_previous(state["session_id"])
    assert previous["phase"] == "pointwise"
    assert previous["draft"]["overall_score"] == 1
    store.navigate_previous(previous["session_id"])
    first_again = store.state(previous["session_id"])
    assert first_again["trial"]["trial_id"] == first_trial["trial_id"]


def test_pairwise_requires_choice_confidence_and_both_reasons(tmp_path: Path) -> None:
    _bank(tmp_path)
    store = Study1Store(tmp_path)
    state = store.start(_profile(store.create_codes(1)[0]))
    state = _submit_pointwise(store, state)
    state = _submit_pointwise(store, state)
    state = store.acknowledge_transition(
        state["session_id"], StudyTransitionRequest(acknowledge=True)
    )
    trial = state["trial"]
    with pytest.raises(ValueError, match="选择理由和不选择理由"):
        store.save_pairwise(
            state["session_id"], trial["trial_id"],
            StudyPairwiseRequest(choice="right", confidence="very_sure", submit=True),
        )


def test_assignment_uses_all_three_ai_systems_and_never_repeats_questions(tmp_path: Path) -> None:
    bank = _bank(tmp_path)
    pointwise_counts = {source: 0 for source in ("gold", "rag_identity", "persona_pack", "codex", "other_human")}
    ai_gold_counts = {source: 0 for source in AI_SOURCES}
    side_counts = {source: {"left": 0, "right": 0} for source in AI_SOURCES}
    for index in range(1500):
        assignment = build_assignment(bank, f"PF-SIM-{index:05d}")
        questions = [trial["question_id"] for trial in assignment["pointwise"] + assignment["pairwise"]]
        assert len(questions) == len(set(questions)) == 4
        assert {trial["hidden_pair_type"] for trial in assignment["pairwise"]} == {"gold_vs_ai", "ai_vs_ai"}
        seen_ai: set[str] = set()
        for trial in assignment["pointwise"]:
            pointwise_counts[trial["hidden_source"]] += 1
        for trial in assignment["pairwise"]:
            for side in ("left", "right"):
                source = trial[side]["hidden_source"]
                if source in AI_SOURCES:
                    seen_ai.add(source)
                    side_counts[source][side] += 1
            if trial["hidden_pair_type"] == "gold_vs_ai":
                source = next(trial[side]["hidden_source"] for side in ("left", "right") if trial[side]["hidden_source"] != "gold")
                ai_gold_counts[source] += 1
        assert seen_ai == set(AI_SOURCES)

    assert max(pointwise_counts.values()) - min(pointwise_counts.values()) < 100
    assert max(ai_gold_counts.values()) - min(ai_gold_counts.values()) < 100
    for counts in side_counts.values():
        assert abs(counts["left"] - counts["right"]) < 100


def test_analysis_bundle_is_relational_and_auditable(tmp_path: Path) -> None:
    _bank(tmp_path)
    store = Study1Store(tmp_path)
    state = store.start(_profile(store.create_codes(1)[0]))
    session_id = state["session_id"]
    state = _complete_formal_trials(store, state)
    store.save_exposure(session_id, StudyExposureRequest(value="unsure"))

    payload, filename = store.analysis_bundle("study-demo-v2")
    assert filename == "study1-study-demo-v2-analysis.zip"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "sessions.csv", "trials.csv", "span_annotations.csv",
            "feature_coding_template.csv", "events.jsonl", "raw.jsonl",
            "data_dictionary.md",
        }
        sessions = list(csv.DictReader(io.StringIO(archive.read("sessions.csv").decode())))
        trials = list(csv.DictReader(io.StringIO(archive.read("trials.csv").decode())))
        spans = list(csv.DictReader(io.StringIO(archive.read("span_annotations.csv").decode())))
        assert len(sessions) == 1
        assert len(trials) == 4
        assert len(spans) == 2
        assert {row["session_id"] for row in trials + spans} == {session_id}
        assert all(row["material_sha256"] == sessions[0]["material_sha256"] for row in trials + spans)
        assert "不得相乘" in archive.read("data_dictionary.md").decode("utf-8")

    bundle_path = tmp_path / filename
    bundle_path.write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        coding_rows = list(csv.DictReader(io.StringIO(archive.read("feature_coding_template.csv").decode())))
    span_coding = next(row for row in coding_rows if row["evidence_role"] == "pointwise_span")
    span_coding["coder_id"] = "coder-1"
    span_coding["feature_dimension"] = "立场与世界观"
    span_coding["feature_realization"] = "核心判断一致"
    coding_path = tmp_path / "coded.csv"
    with coding_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(span_coding))
        writer.writeheader()
        writer.writerow(span_coding)
    report = analyze_bundle(bundle_path, coding_path=coding_path, bootstrap_iterations=100)
    assert report["integrity"]["completed_sessions"] == 1
    assert report["integrity"]["duplicate_annotation_ids"] == 0
    assert report["features"]["立场与世界观"]["participant_trial_units"] == 1
    assert "不因果" not in render_markdown(report)
    assert "not causal" in report["analysis_boundary"]


def test_double_submit_advances_only_once(tmp_path: Path) -> None:
    _bank(tmp_path)
    store = Study1Store(tmp_path)
    state = store.start(_profile(store.create_codes(1)[0]))
    trial = state["trial"]
    payload = StudyPointwiseRequest(
        overall_score=1,
        highlights=[_highlight(trial["answer"])],
        primary_reason="关键理由",
        submit=True,
    )

    def submit_once() -> str:
        try:
            store.save_pointwise(state["session_id"], trial["trial_id"], payload)
            return "saved"
        except ValueError:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: submit_once(), range(2)))
    assert outcomes.count("saved") == 1
    assert outcomes.count("stale") == 1
    assert store.state(state["session_id"])["progress"]["completed"] == 1


def test_multiple_authors_and_participants_are_isolated(tmp_path: Path) -> None:
    _bank(tmp_path, folder="alpha", study_id="study-alpha-v2", author="author-alpha")
    _bank(tmp_path, folder="beta", study_id="study-beta-v2", author="author-beta")
    store = Study1Store(tmp_path)
    codes = store.create_codes(8, "study-alpha-v2")
    with ThreadPoolExecutor(max_workers=8) as executor:
        states = list(executor.map(lambda code: store.start(_profile(code), "study-alpha-v2"), codes))
    assert len({state["session_id"] for state in states}) == 8
    beta_code = store.create_codes(1, "study-beta-v2")[0]
    beta = store.start(_profile(beta_code), "study-beta-v2")
    assert beta["author"] == "author-beta"
    with pytest.raises(ValueError, match="参与码不存在"):
        store.start(_profile(codes[0]), "study-beta-v2")


def test_material_freeze_and_legacy_protocol_are_not_mixed(tmp_path: Path) -> None:
    _bank(tmp_path)
    store = Study1Store(tmp_path)
    store.create_codes(1)
    material_path = tmp_path / "studies" / "demo" / "material_bank.json"
    material = json.loads(material_path.read_text(encoding="utf-8"))
    material["items"][0]["responses"]["gold"]["text"] += " 改动。"
    material_path.write_text(json.dumps(material, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="材料已冻结"):
        store.create_codes(1)


def test_public_session_api_requires_resume_token_and_hides_sources(tmp_path: Path) -> None:
    _bank(tmp_path)
    store = Study1Store(tmp_path)
    code = store.create_codes(1)[0]
    app = create_app(WebConfig(data_dir=tmp_path))
    with TestClient(app) as client:
        started = client.post("/api/studies/study1/sessions", json=_profile(code).model_dump()).json()
        assert "hidden_source" not in json.dumps(started)
        assert client.get(f"/api/studies/study1/sessions/{started['session_id']}").status_code == 403
        resumed = client.get(
            f"/api/studies/study1/sessions/{started['session_id']}",
            headers={"X-Study-Session-Token": started["resume_token"]},
        )
        assert resumed.status_code == 200
        assert "hidden_source" not in resumed.text


def test_public_meta_contains_author_avatar(tmp_path: Path) -> None:
    _bank(tmp_path, author="avatar-author")
    author_dir = tmp_path / "authors" / "zhihu" / "avatar-author"
    author_dir.mkdir(parents=True)
    (author_dir / "profile.json").write_text(
        json.dumps({"nickname": "头像作者", "avatar_url": "https://example.com/avatar.jpg"}, ensure_ascii=False),
        encoding="utf-8",
    )
    meta = Study1Store(tmp_path).public_meta("study-demo-v2")
    assert meta["author_label"] == "头像作者"
    assert meta["avatar_url"] == "https://example.com/avatar.jpg"
    assert meta["protocol_version"] == PROTOCOL_VERSION
    assert meta["total"] == 4
