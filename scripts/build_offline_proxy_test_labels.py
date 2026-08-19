"""Build a no-API provisional Test label set from completed Dev labels.

This is deliberately a proxy, not a replacement for human or hosted-LLM
judgment.  It learns a small deterministic classifier from the same author's
completed Dev dual-axis labels and applies it to the frozen Test pool.  The
output uses the normal Gold-aware label schema so the existing report UI can
inspect it, while the manifest marks the result as provisional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import string
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from personaforge.eval.retrieval_gold_qrels import (
    GOLD_LABEL_SCHEMA_VERSION,
    compute_dual_axis_metrics,
)
from personaforge.eval.retrieval_judge import load_label_set


AXES = ("content_support", "persona_expression_support")
PROMPT_VERSION = "offline-proxy-char-tfidf-v1"
PUNCTUATION = set(string.punctuation) | set("，。！？；：、（）【】《》“”‘’—…·「」『』")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dev-label-manifest", type=Path, required=True)
    parser.add_argument("--label-set", default="codex_offline_proxy_test_v1")
    args = parser.parse_args()

    pool_manifest_path = args.pool_manifest.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    dev_label_manifest = args.dev_label_manifest.expanduser().resolve()
    pool_manifest = _read_json(pool_manifest_path)
    pool_path = pool_manifest_path.parent / str(pool_manifest.get("pool_file") or "pool.jsonl")
    records = _read_jsonl(pool_path)
    dataset = {str(row.get("item_id") or ""): row for row in _read_jsonl(dataset_path)}
    dev_manifest, dev_labels = load_label_set(dev_label_manifest)

    test_records = [row for row in records if str(row.get("split") or "") == "test"]
    dev_records = [row for row in records if str(row.get("split") or "") == "dev"]
    if not test_records or not dev_records:
        raise SystemExit("The pool must contain both Dev and Test records.")
    if dev_manifest.get("status") != "completed":
        raise SystemExit("The calibration label set must be completed.")
    if str(dev_manifest.get("pool_id") or "") != str(pool_manifest.get("pool_id") or ""):
        raise SystemExit("Dev labels do not belong to the supplied pool.")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )
    feature_rows = _make_feature_rows(records, dataset, vectorizer)
    train_rows = [row for row in feature_rows if row["split"] == "dev"]
    test_rows = [row for row in feature_rows if row["split"] == "test"]
    train_keyed = {
        (str(key[0]), str(key[1])): value
        for key, value in dev_labels.items()
        if value.get("status") == "completed"
    }

    usable_train = [
        row
        for row in train_rows
        if (row["item_id"], row["parent_id"]) in train_keyed
        and all(train_keyed[(row["item_id"], row["parent_id"])].get(axis) in {0, 1, 2} for axis in AXES)
    ]
    if len(usable_train) < 30:
        raise SystemExit(f"Too few completed Dev labels for calibration: {len(usable_train)}")

    x_train = [row["features"] for row in usable_train]
    x_test = [row["features"] for row in test_rows]
    classifiers: dict[str, LogisticRegression] = {}
    for axis in AXES:
        y_train = [int(train_keyed[(row["item_id"], row["parent_id"])][axis]) for row in usable_train]
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=0,
            ),
        )
        classifier.fit(x_train, y_train)
        classifiers[axis] = classifier

    now = _utc_now()
    label_rows: list[dict[str, Any]] = []
    counts = {axis: Counter() for axis in AXES}
    for row in test_rows:
        scores: dict[str, int] = {}
        probabilities: dict[str, list[float]] = {}
        for axis, classifier in classifiers.items():
            probabilities[axis] = [float(value) for value in classifier.predict_proba([row["features"]])[0]]
            scores[axis] = int(classifier.predict([row["features"]])[0])
            counts[axis][scores[axis]] += 1

        evidence = str(row["text"]).strip().replace("\n", " ")[:120]
        positive = any(value > 0 for value in scores.values())
        label_rows.append(
            {
                "axis_ranges": {axis: 0 for axis in AXES},
                "candidate_sha256": _sha256_text(row["text"]),
                "completed_passes": [1],
                "confidence": "low",
                "content_candidate_evidence": evidence if scores["content_support"] > 0 else "",
                "content_gold_unit_ids": [],
                "content_support": scores["content_support"],
                "exact_agreement": True,
                "gold_answer_sha256": _sha256_text(str(dataset[row["item_id"]].get("gold_answer") or "")),
                "item_id": row["item_id"],
                "label_source": "codex_offline_proxy",
                "model": "offline_dev_calibrated_classifier",
                "parent_id": row["parent_id"],
                "persona_candidate_evidence": evidence if scores["persona_expression_support"] > 0 else "",
                "persona_expression_support": scores["persona_expression_support"],
                "persona_gold_unit_ids": [],
                "pool_id": str(pool_manifest.get("pool_id") or ""),
                "prompt_version": PROMPT_VERSION,
                "reason": (
                    "离线代理：依据同作者 Dev 双轴标签校准的文本相似度与表达特征分类。"
                    if positive
                    else "离线代理：未达到同作者 Dev 标签校准出的正向支撑阈值。"
                ),
                "repeat_count": 1,
                "required_passes": [1],
                "split": "test",
                "status": "completed",
                "stability_complete": True,
                "updated_at": now,
                "proxy_probabilities": probabilities,
            }
        )

    output_dir = pool_manifest_path.parent / "llm_labels" / args.label_set
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.jsonl"
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"
    _write_jsonl(labels_path, label_rows)
    metrics = compute_dual_axis_metrics(
        test_records,
        label_rows,
        recall_scope=str(pool_manifest.get("recall_scope") or "eligible_author_corpus_before_cutoff"),
    )
    _write_json(metrics_path, metrics)
    manifest = {
        "schema_version": GOLD_LABEL_SCHEMA_VERSION,
        "status": "completed",
        "label_set": args.label_set,
        "labeler": "codex_offline_proxy",
        "reviewer": "codex",
        "model": "offline_dev_calibrated_classifier",
        "label_source": "codex_offline_proxy",
        "provisional": True,
        "not_gold": True,
        "pool_id": str(pool_manifest.get("pool_id") or ""),
        "pool_manifest": str(pool_manifest_path),
        "pool_manifest_sha256": _sha256_file(pool_manifest_path),
        "dataset_path": str(dataset_path),
        "dataset_file_sha256": _sha256_file(dataset_path),
        "calibration_label_manifest": str(dev_label_manifest),
        "calibration_label_manifest_sha256": _sha256_file(dev_label_manifest),
        "prompt_version": PROMPT_VERSION,
        "axes": {
            "content_support": {"label": "内容支撑", "values": [0, 1, 2]},
            "persona_expression_support": {"label": "作者表达支撑", "values": [0, 1, 2]},
        },
        "default_axis": "content_support",
        "selected_splits": ["test"],
        "splits": ["test"],
        "no_combined_score": True,
        "labels_file": labels_path.name,
        "metrics_file": metrics_path.name,
        "total": len(label_rows),
        "completed": len(label_rows),
        "stability_completed": 0,
        "updated_at": now,
        "provenance": {
            "external_api_called": False,
            "method": "character TF-IDF pair features plus a deterministic logistic classifier",
            "calibration_rows": len(usable_train),
            "calibration_label_set": str(dev_manifest.get("label_set") or dev_label_manifest.parent.name),
            "warning": "仅用于无 API 时的页面与工程链路验证，不作为人工或正式 Gold 标注。",
        },
        "score_counts": {axis: {str(key): value for key, value in sorted(counter.items())} for axis, counter in counts.items()},
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "label_set": args.label_set,
        "total": len(label_rows),
        "output_dir": str(output_dir),
        "score_counts": manifest["score_counts"],
    }, ensure_ascii=False, indent=2))
    return 0


def _make_feature_rows(
    records: list[dict[str, Any]],
    dataset: dict[str, dict[str, Any]],
    vectorizer: TfidfVectorizer,
) -> list[dict[str, Any]]:
    raw_texts: list[str] = []
    triples: list[tuple[dict[str, Any], dict[str, Any], str, str, str]] = []
    for record in records:
        item_id = str(record.get("item_id") or "")
        gold = str(dataset.get(item_id, {}).get("gold_answer") or "")
        query = str(record.get("query") or dataset.get(item_id, {}).get("query") or "")
        for candidate in record.get("candidates") or []:
            text = str(candidate.get("text") or "")
            triples.append((record, candidate, gold, query, text))
            raw_texts.extend((gold, query, text))
    matrix = vectorizer.fit_transform(raw_texts)
    rows: list[dict[str, Any]] = []
    index = 0
    for record, candidate, gold, query, text in triples:
        gold_vec = matrix[index]
        query_vec = matrix[index + 1]
        text_vec = matrix[index + 2]
        index += 3
        features = [
            _cosine(gold_vec, text_vec),
            _cosine(query_vec, text_vec),
            _character_overlap(gold, text),
            _character_overlap(query, text),
            _length_ratio(gold, text),
            _length_ratio(query, text),
            _punctuation_similarity(gold, text),
            _paragraph_similarity(gold, text),
        ]
        rows.append(
            {
                "item_id": str(record.get("item_id") or ""),
                "parent_id": str(candidate.get("parent_id") or ""),
                "split": str(record.get("split") or ""),
                "text": text,
                "features": features,
            }
        )
    return rows


def _cosine(left: Any, right: Any) -> float:
    value = float(left.multiply(right).sum())
    left_norm = math.sqrt(float(left.multiply(left).sum()))
    right_norm = math.sqrt(float(right.multiply(right).sum()))
    return value / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _character_overlap(left: str, right: str) -> float:
    left_chars = {value for value in left if not value.isspace()}
    right_chars = {value for value in right if not value.isspace()}
    if not left_chars or not right_chars:
        return 0.0
    return len(left_chars & right_chars) / math.sqrt(len(left_chars) * len(right_chars))


def _length_ratio(left: str, right: str) -> float:
    left_len = max(1, len(left.strip()))
    right_len = max(1, len(right.strip()))
    return min(left_len, right_len) / max(left_len, right_len)


def _punctuation_similarity(left: str, right: str) -> float:
    left_counts = Counter(value for value in left if value in PUNCTUATION)
    right_counts = Counter(value for value in right if value in PUNCTUATION)
    keys = set(left_counts) | set(right_counts)
    if not keys:
        return 1.0
    numerator = sum(left_counts[key] * right_counts[key] for key in keys)
    left_norm = math.sqrt(sum(left_counts[key] ** 2 for key in keys))
    right_norm = math.sqrt(sum(right_counts[key] ** 2 for key in keys))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _paragraph_similarity(left: str, right: str) -> float:
    left_count = max(1, len([part for part in left.split("\n") if part.strip()]))
    right_count = max(1, len([part for part in right.split("\n") if part.strip()]))
    return min(left_count, right_count) / max(left_count, right_count)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
