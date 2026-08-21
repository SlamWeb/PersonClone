from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean


BASELINE_ROUTE = "transformed_rrf"
RERANK_ROUTE = "transformed_rrf_reranked"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dcg(scores: list[int], *, exponential: bool) -> float:
    gains = [(2**score - 1) if exponential else score for score in scores]
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def ndcg(scores: list[int], ideal_scores: list[int], *, exponential: bool) -> float:
    ideal = dcg(ideal_scores, exponential=exponential)
    return dcg(scores, exponential=exponential) / ideal if ideal else 0.0


def route_scores(route: list[dict], labels: dict[str, int], k: int) -> list[int]:
    return [labels[item["parent_id"]] for item in route[:k]]


def summarize(
    baseline_rows: dict[str, dict],
    rerank_rows: dict[str, dict],
    labels_by_item: dict[str, dict[str, int]],
    *,
    k: int,
) -> list[dict]:
    results = []
    for item_id, labels in labels_by_item.items():
        baseline = route_scores(baseline_rows[item_id]["routes"][BASELINE_ROUTE], labels, k)
        reranked = route_scores(rerank_rows[item_id]["routes"][RERANK_ROUTE], labels, k)
        ideal = sorted(labels.values(), reverse=True)[:k]
        row = {
            "item_id": item_id,
            "baseline_scores": baseline,
            "reranked_scores": reranked,
            "baseline_ndcg_exp": ndcg(baseline, ideal, exponential=True),
            "reranked_ndcg_exp": ndcg(reranked, ideal, exponential=True),
            "baseline_ndcg_linear": ndcg(baseline, ideal, exponential=False),
            "reranked_ndcg_linear": ndcg(reranked, ideal, exponential=False),
            "baseline_useful_precision": mean(score >= 1 for score in baseline),
            "reranked_useful_precision": mean(score >= 1 for score in reranked),
            "baseline_strong_precision": mean(score == 2 for score in baseline),
            "reranked_strong_precision": mean(score == 2 for score in reranked),
        }
        row["delta_ndcg_exp"] = row["reranked_ndcg_exp"] - row["baseline_ndcg_exp"]
        row["delta_ndcg_linear"] = row["reranked_ndcg_linear"] - row["baseline_ndcg_linear"]
        results.append(row)
    return results


def aggregate(rows: list[dict]) -> dict[str, float]:
    keys = [
        "baseline_ndcg_exp",
        "reranked_ndcg_exp",
        "delta_ndcg_exp",
        "baseline_ndcg_linear",
        "reranked_ndcg_linear",
        "delta_ndcg_linear",
        "baseline_useful_precision",
        "reranked_useful_precision",
        "baseline_strong_precision",
        "reranked_strong_precision",
    ]
    return {key: mean(row[key] for row in rows) for key in keys}


def clipped(text: str, limit: int) -> str:
    normalized = text.strip()
    return normalized if len(normalized) <= limit else normalized[:limit].rstrip() + "……"


def build_review_packet(
    *,
    output: Path,
    answer_key_output: Path,
    audit_labels: dict[str, dict[str, int]],
    old_labels: dict[tuple[str, str], int],
    dataset_rows: dict[str, dict],
    pool_rows: dict[str, dict],
    limit: int,
) -> None:
    disagreements = []
    for item_id, labels in audit_labels.items():
        for parent_id, audit_score in labels.items():
            old_score = old_labels[(item_id, parent_id)]
            if old_score != audit_score:
                disagreements.append(
                    {
                        "item_id": item_id,
                        "parent_id": parent_id,
                        "old_score": old_score,
                        "audit_score": audit_score,
                        "difference": abs(audit_score - old_score),
                    }
                )
    disagreements.sort(key=lambda row: (-row["difference"], row["item_id"], row["parent_id"]))
    selected = disagreements[:limit]

    lines = [
        "# Reranker 人工复核 20 条（盲审）",
        "",
        "只填写 `你的评分`：`0` 无用，`1` 有间接帮助，`2` 直接支撑关键判断。",
        "不要猜原 Judge 或审计者给了多少分。材料为完整 Parent 的截断预览，可点击来源阅读全文。",
        "",
    ]
    for index, row in enumerate(selected, start=1):
        dataset = dataset_rows[row["item_id"]]
        candidates = {candidate["parent_id"]: candidate for candidate in pool_rows[row["item_id"]]["candidates"]}
        candidate = candidates[row["parent_id"]]
        lines.extend(
            [
                f"## {index}. 待评估材料（编号 {row['item_id']}）",
                "",
                f"### 待评估问题：{dataset['query']}",
                "",
                "**作者对这个问题的原回答（预览）：**",
                "",
                clipped(dataset["gold_answer"], 1000),
                "",
                f"### 候选历史材料：[{candidate['title'] or candidate['kind']}]({candidate['url']})",
                "",
                clipped(candidate["text"], 1600),
                "",
                "**请判断：这篇候选历史材料能在多大程度上帮助该作者回答上面的待评估问题？**",
                "",
                "- `0`：不能提供帮助",
                "- `1`：提供相关背景、局部机制或例子，但仍需明显推断",
                "- `2`：直接支撑关键判断、因果机制或核心论据",
                "",
                "**你的评分：** `__`",
                "",
                "---",
                "",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    answer_key_output.write_text(
        json.dumps({"items": selected}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and reranked Top-K under audit labels.")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--old-labels", type=Path, required=True)
    parser.add_argument("--baseline-rankings", type=Path, required=True)
    parser.add_argument("--rerank-rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--review-key-output", type=Path)
    parser.add_argument("--review-limit", type=int, default=20)
    parser.add_argument("--completed-user-review", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    audit_labels = {
        item_id: {parent_id: int(score) for parent_id, score in labels.items()}
        for item_id, labels in audit["labels"].items()
    }
    audited_pairs = {(item_id, parent_id) for item_id, labels in audit_labels.items() for parent_id in labels}

    old_labels = {
        (row["item_id"], row["parent_id"]): int(row["content_support"])
        for row in load_jsonl(args.old_labels)
        if (row["item_id"], row["parent_id"]) in audited_pairs
    }
    old_labels_by_item = {
        item_id: {parent_id: old_labels[(item_id, parent_id)] for parent_id in labels}
        for item_id, labels in audit_labels.items()
    }
    baseline_rows = {row["item_id"]: row for row in load_jsonl(args.baseline_rankings)}
    rerank_rows = {row["item_id"]: row for row in load_jsonl(args.rerank_rankings)}

    audit_rows = summarize(baseline_rows, rerank_rows, audit_labels, k=args.k)
    old_rows = summarize(baseline_rows, rerank_rows, old_labels_by_item, k=args.k)
    old_by_item = {row["item_id"]: row for row in old_rows}
    comparisons = []
    for row in audit_rows:
        old = old_by_item[row["item_id"]]
        comparisons.append(
            {
                **row,
                "old_delta_ndcg_exp": old["delta_ndcg_exp"],
                "old_delta_ndcg_linear": old["delta_ndcg_linear"],
            }
        )

    payload = {
        "audit_id": audit["audit_id"],
        "k": args.k,
        "routes": {"baseline": BASELINE_ROUTE, "reranked": RERANK_ROUTE},
        "old_qrels": {"aggregate": aggregate(old_rows), "per_item": old_rows},
        "audit_qrels": {"aggregate": aggregate(audit_rows), "per_item": audit_rows},
        "comparison": comparisons,
    }

    if args.completed_user_review:
        if not args.review_key_output:
            parser.error("--completed-user-review requires --review-key-output")
        review_lines = args.completed_user_review.read_text(encoding="utf-8").splitlines()
        user_scores = []
        for line in review_lines:
            if not line.startswith("**你的评分：**"):
                continue
            match = re.search(r"_([012])_", line)
            if match:
                user_scores.append(int(match.group(1)))
        review_items = json.loads(args.review_key_output.read_text(encoding="utf-8"))["items"]
        if len(user_scores) != len(review_items):
            parser.error(
                f"completed review has {len(user_scores)} scores, expected {len(review_items)}"
            )
        adjudicated_labels = {item_id: dict(labels) for item_id, labels in audit_labels.items()}
        adjudication_rows = []
        for item, user_score in zip(review_items, user_scores, strict=True):
            adjudicated_labels[item["item_id"]][item["parent_id"]] = user_score
            adjudication_rows.append({**item, "user_score": user_score})
        adjudicated_metrics = summarize(
            baseline_rows,
            rerank_rows,
            adjudicated_labels,
            k=args.k,
        )
        payload["user_adjudication"] = {
            "reviewed_count": len(adjudication_rows),
            "user_vs_audit_exact": mean(
                row["user_score"] == row["audit_score"] for row in adjudication_rows
            ),
            "user_vs_old_exact": mean(
                row["user_score"] == row["old_score"] for row in adjudication_rows
            ),
            "user_vs_audit_within_one": mean(
                abs(row["user_score"] - row["audit_score"]) <= 1 for row in adjudication_rows
            ),
            "user_vs_old_within_one": mean(
                abs(row["user_score"] - row["old_score"]) <= 1 for row in adjudication_rows
            ),
            "score_distribution": {
                str(score): sum(row["user_score"] == score for row in adjudication_rows)
                for score in range(3)
            },
            "aggregate": aggregate(adjudicated_metrics),
            "per_item": adjudicated_metrics,
            "items": adjudication_rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    review_generation_args = [args.dataset, args.pool, args.review_output]
    if any(review_generation_args) and (not all(review_generation_args) or not args.review_key_output):
        parser.error("--dataset, --pool, --review-output and --review-key-output must be supplied together")
    if all(review_generation_args):
        dataset_rows = {row["item_id"]: row for row in load_jsonl(args.dataset)}
        pool_rows = {row["item_id"]: row for row in load_jsonl(args.pool)}
        build_review_packet(
            output=args.review_output,
            answer_key_output=args.review_key_output,
            audit_labels=audit_labels,
            old_labels=old_labels,
            dataset_rows=dataset_rows,
            pool_rows=pool_rows,
            limit=args.review_limit,
        )

    print("item       old_delta   audit_delta  baseline->reranked (audit exp nDCG)")
    for row in comparisons:
        print(
            f"{row['item_id']:<10} {row['old_delta_ndcg_exp']:+.3f}      "
            f"{row['delta_ndcg_exp']:+.3f}       "
            f"{row['baseline_ndcg_exp']:.3f}->{row['reranked_ndcg_exp']:.3f}"
        )
    print("\naggregate")
    print(json.dumps(payload["audit_qrels"]["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
