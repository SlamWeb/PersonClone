"""Reproducible descriptive analysis for a Study 1 V2 analysis bundle."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


def _read_text(source: Path, name: str) -> str:
    if source.is_dir():
        return (source / name).read_text(encoding="utf-8-sig")
    with zipfile.ZipFile(source) as archive:
        return archive.read(name).decode("utf-8-sig")


def _read_csv(source: Path, name: str) -> list[dict[str, str]]:
    text = _read_text(source, name)
    return list(csv.DictReader(io.StringIO(text))) if text.strip() else []


def _number(value: str | None) -> float | None:
    try:
        return float(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return statistics.fmean(rows) if rows else None


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    mean_x, mean_y = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry)
    )
    return numerator / denominator if denominator else None


def participant_cluster_bootstrap(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
    *,
    iterations: int = 2000,
    seed: int = 20260806,
) -> dict[str, float | int | None]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["session_id"])].append(row)
    cluster_ids = sorted(clusters)
    estimate = statistic(rows)
    if not cluster_ids or estimate is None:
        return {"estimate": estimate, "lower_95": None, "upper_95": None, "clusters": len(cluster_ids)}
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled: list[dict[str, Any]] = []
        for _ in cluster_ids:
            sampled.extend(clusters[rng.choice(cluster_ids)])
        value = statistic(sampled)
        if value is not None:
            samples.append(value)
    samples.sort()
    if not samples:
        return {"estimate": estimate, "lower_95": None, "upper_95": None, "clusters": len(cluster_ids)}
    lower = samples[max(0, round(0.025 * (len(samples) - 1)))]
    upper = samples[min(len(samples) - 1, round(0.975 * (len(samples) - 1)))]
    return {"estimate": estimate, "lower_95": lower, "upper_95": upper, "clusters": len(cluster_ids)}


def analyze_bundle(
    source: Path,
    *,
    coding_path: Path | None = None,
    bootstrap_iterations: int = 2000,
    seed: int = 20260806,
) -> dict[str, Any]:
    sessions = _read_csv(source, "sessions.csv")
    trials = _read_csv(source, "trials.csv")
    spans = _read_csv(source, "span_annotations.csv")
    coding = (
        list(csv.DictReader(io.StringIO(coding_path.read_text(encoding="utf-8-sig"))))
        if coding_path
        else _read_csv(source, "feature_coding_template.csv")
    )
    session_ids = {row["session_id"] for row in sessions}
    trial_keys = [(row["session_id"], row["trial_id"]) for row in trials]
    span_ids = [row["annotation_id"] for row in spans]
    completed = {row["session_id"] for row in sessions if row.get("completed_at")}
    submitted_trials = [row for row in trials if row.get("status") == "submitted"]
    integrity = {
        "sessions": len(sessions),
        "completed_sessions": len(completed),
        "trials": len(trials),
        "submitted_trials": len(submitted_trials),
        "spans": len(spans),
        "duplicate_trial_keys": len(trial_keys) - len(set(trial_keys)),
        "duplicate_annotation_ids": len(span_ids) - len(set(span_ids)),
        "orphan_trial_sessions": sorted({row["session_id"] for row in trials} - session_ids),
        "orphan_span_trials": sum(
            (row["session_id"], row["trial_id"]) not in set(trial_keys) for row in spans
        ),
        "completed_sessions_without_four_submitted_trials": sorted(
            session_id
            for session_id in completed
            if sum(row["session_id"] == session_id for row in submitted_trials) != 4
        ),
        "material_hashes": sorted({row.get("material_sha256", "") for row in sessions if row.get("material_sha256")}),
        "protocol_versions": dict(Counter(row.get("protocol_version", "") for row in sessions)),
    }

    pointwise = [row for row in submitted_trials if row.get("kind") == "pointwise"]
    pairwise = [row for row in submitted_trials if row.get("kind") == "pairwise"]
    balance = {
        "pointwise_source": dict(Counter(row.get("hidden_source", "") for row in pointwise)),
        "pair_type": dict(Counter(row.get("pair_type", "") for row in pairwise)),
        "left_source": dict(Counter(row.get("left_hidden_source", "") for row in pairwise)),
        "right_source": dict(Counter(row.get("right_hidden_source", "") for row in pairwise)),
        "confidence": dict(Counter(row.get("confidence", "") for row in pairwise)),
    }

    pointwise_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pointwise:
        score = _number(row.get("overall_score"))
        if score is not None:
            pointwise_by_source[row["hidden_source"]].append({**row, "value": score})
    source_scores = {
        source_name: participant_cluster_bootstrap(
            rows,
            lambda sample: _mean(float(row["value"]) for row in sample),
            iterations=bootstrap_iterations,
            seed=seed,
        )
        for source_name, rows in sorted(pointwise_by_source.items())
    }

    ai_vs_gold = [row for row in pairwise if row.get("pair_type") == "gold_vs_ai"]
    ai_selected_rows = []
    for row in ai_vs_gold:
        left = row.get("left_hidden_source")
        chosen = row.get("chosen_source")
        ai_source = row.get("right_hidden_source") if left == "gold" else left
        ai_selected_rows.append({**row, "ai_source": ai_source, "value": 1.0 if chosen == ai_source else 0.0})
    gold_confusion = participant_cluster_bootstrap(
        ai_selected_rows,
        lambda sample: _mean(float(row["value"]) for row in sample),
        iterations=bootstrap_iterations,
        seed=seed + 1,
    )
    gold_confusion_close_excluded = participant_cluster_bootstrap(
        [row for row in ai_selected_rows if row.get("confidence") != "close"],
        lambda sample: _mean(float(row["value"]) for row in sample),
        iterations=bootstrap_iterations,
        seed=seed + 2,
    )

    pair_wins: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pairwise:
        pair_wins[row.get("pair_type", "")][row.get("chosen_source", "")] += 1

    coding_by_evidence: dict[str, set[str]] = defaultdict(set)
    for row in coding:
        dimension = (row.get("feature_dimension") or "").strip()
        if dimension:
            coding_by_evidence[row["evidence_id"]].add(dimension)
    span_by_id = {row["annotation_id"]: row for row in spans}
    trial_by_key = {(row["session_id"], row["trial_id"]): row for row in pointwise}
    feature_units: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for evidence_id, dimensions in coding_by_evidence.items():
        span = span_by_id.get(evidence_id)
        if not span:
            continue
        impact = _number(span.get("impact"))
        if impact is None:
            continue
        for dimension in dimensions:
            feature_units[(span["session_id"], span["trial_id"], dimension)].append(impact)
    feature_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (session_id, trial_id, dimension), impacts in feature_units.items():
        trial = trial_by_key.get((session_id, trial_id))
        score = _number(trial.get("overall_score")) if trial else None
        if score is None:
            continue
        feature_rows[dimension].append(
            {
                "session_id": session_id,
                "trial_id": trial_id,
                "impact": statistics.fmean(impacts),
                "absolute_impact": statistics.fmean(abs(value) for value in impacts),
                "overall_score": score,
            }
        )
    features = {}
    for dimension, rows in sorted(feature_rows.items()):
        features[dimension] = {
            "participant_trial_units": len(rows),
            "mean_impact": _mean(row["impact"] for row in rows),
            "mean_absolute_impact": _mean(row["absolute_impact"] for row in rows),
            "spearman_impact_vs_overall": spearman(
                [row["impact"] for row in rows], [row["overall_score"] for row in rows]
            ),
        }

    return {
        "analysis_boundary": "Study 1 reports descriptive associations and discrimination, not causal feature effects.",
        "integrity": integrity,
        "balance": balance,
        "pointwise": {"overall_score_by_source": source_scores},
        "pairwise": {
            "ai_selected_over_gold": gold_confusion,
            "ai_selected_over_gold_excluding_close": gold_confusion_close_excluded,
            "chosen_source_counts_by_pair_type": {key: dict(value) for key, value in pair_wins.items()},
        },
        "features": features,
    }


def render_markdown(report: dict[str, Any]) -> str:
    integrity = report["integrity"]
    lines = [
        "# Study 1 V2 分析报告",
        "",
        f"> {report['analysis_boundary']}",
        "",
        "## 数据完整性",
        "",
        f"- 会话：{integrity['sessions']}，完成：{integrity['completed_sessions']}",
        f"- Trial：{integrity['trials']}，已提交：{integrity['submitted_trials']}",
        f"- 划线证据：{integrity['spans']}",
        f"- 重复 trial key：{integrity['duplicate_trial_keys']}；重复 annotation ID：{integrity['duplicate_annotation_ids']}",
        "",
        "## 来源平衡",
        "",
        "```json",
        json.dumps(report["balance"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 单篇与配对",
        "",
        "```json",
        json.dumps({"pointwise": report["pointwise"], "pairwise": report["pairwise"]}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Feature 编码结果",
        "",
    ]
    if report["features"]:
        lines.extend(["```json", json.dumps(report["features"], ensure_ascii=False, indent=2), "```"])
    else:
        lines.append("尚未在 `feature_coding_template.csv` 中填写 feature_dimension，因此暂不计算 feature 关联。")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分析 Study 1 V2 导出的 ZIP。")
    parser.add_argument("bundle", type=Path, help="管理员下载的 analysis ZIP，或已解压目录。")
    parser.add_argument("--coding", type=Path, help="已完成人工编码的 feature_coding_template.csv。")
    parser.add_argument("--out-dir", type=Path, default=Path("data/studies/analysis"))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args(argv)
    report = analyze_bundle(
        args.bundle,
        coding_path=args.coding,
        bootstrap_iterations=max(100, args.bootstrap_iterations),
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "study1_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "study1_analysis.md").write_text(render_markdown(report), encoding="utf-8")
    print(args.out_dir / "study1_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
