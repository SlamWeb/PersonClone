from __future__ import annotations

from personaforge.eval.retrieval_metrics import compute_split_metrics_from_rankings


def test_independent_rankings_keep_qrels_denominator_and_report_unknown_pool_items() -> None:
    records = [
        {
            "item_id": "dev-01",
            "split": "dev",
            "candidates": [
                {"parent_id": "p-1"},
                {"parent_id": "p-2"},
                {"parent_id": "p-3"},
            ],
        },
        {
            "item_id": "test-01",
            "split": "test",
            "candidates": [
                {"parent_id": "p-4"},
                {"parent_id": "p-5"},
            ],
        },
    ]
    labels = {
        ("dev-01", "p-1"): 2,
        ("dev-01", "p-2"): 1,
        ("dev-01", "p-3"): 0,
        ("dev-01", "not-in-qrels"): 2,
        ("test-01", "p-4"): 2,
        ("test-01", "p-5"): 0,
    }
    rankings = [
        {
            "item_id": "dev-01",
            "split": "dev",
            "routes": {
                "raw_dense": [
                    {"rank": 1, "parent_id": "outside", "score": 0.9},
                    {"rank": 2, "parent_id": "p-2", "score": 0.8},
                    {"rank": 3, "parent_id": "p-1", "score": 0.7},
                ]
            },
        },
        {
            "item_id": "test-01",
            "split": "test",
            "routes": {
                "raw_dense": [
                    {"rank": 1, "parent_id": "p-5", "score": 0.9},
                    {"rank": 2, "parent_id": "p-4", "score": 0.8},
                ]
            },
        },
    ]

    report = compute_split_metrics_from_rankings(
        records,
        rankings,
        labels,
        ranking_id="six_route_parent_top100_v1",
        ndcg_cutoffs=(1, 3),
        precision_cutoffs=(1, 3),
        recall_cutoffs=(1, 3),
    )
    route = report["routes"]["raw_dense"]

    assert report["ranking_id"] == "six_route_parent_top100_v1"
    assert report["candidate_count"] == 5
    assert report["judged_candidate_count"] == 5
    assert report["relevant_candidate_count"] == 3
    assert report["coverage"] == 1.0
    assert route["by_cutoff"]["1"]["pool_outside_count"] == 1
    assert route["by_cutoff"]["1"]["unjudged_top_count"] == 1
    assert report["splits"]["dev"]["routes"]["raw_dense"]["by_cutoff"]["3"]["useful_recall_at_k"] is None
    assert report["splits"]["dev"]["routes"]["raw_dense"]["by_cutoff"]["3"]["strong_recall_at_k"] is None
    assert set(report["splits"]) == {"dev", "test"}
    assert report["splits"]["dev"]["routes"]["raw_dense"]["by_cutoff"]["3"]["pool_outside_count"] == 1


def test_ranking_report_does_not_count_labels_outside_qrels() -> None:
    records = [{
        "item_id": "dev-01",
        "split": "dev",
        "candidates": [{"parent_id": "p-1"}],
    }]
    rankings = [{
        "item_id": "dev-01",
        "split": "dev",
        "routes": {"raw_dense": [{"rank": 1, "parent_id": "p-1", "score": 1.0}]},
    }]

    report = compute_split_metrics_from_rankings(
        records,
        rankings,
        {("dev-01", "p-1"): 1, ("dev-01", "stale-parent"): 2},
        ndcg_cutoffs=(1,),
        precision_cutoffs=(1,),
        recall_cutoffs=(1,),
    )

    assert report["candidate_count"] == 1
    assert report["judged_candidate_count"] == 1
    assert report["relevant_candidate_count"] == 1
    assert report["coverage"] == 1.0


def test_ranking_report_exposes_linear_ndcg_alongside_exponential_ndcg() -> None:
    records = [{
        "item_id": "dev-01",
        "split": "dev",
        "candidates": [
            {"parent_id": "p-1"},
            {"parent_id": "p-2"},
            {"parent_id": "p-3"},
        ],
    }]
    rankings = [{
        "item_id": "dev-01",
        "split": "dev",
        "routes": {
            "raw_dense": [
                {"rank": 1, "parent_id": "p-1", "score": 0.9},
                {"rank": 2, "parent_id": "p-2", "score": 0.8},
                {"rank": 3, "parent_id": "p-3", "score": 0.7},
            ]
        },
    }]

    report = compute_split_metrics_from_rankings(
        records,
        rankings,
        {("dev-01", "p-1"): 1, ("dev-01", "p-2"): 2, ("dev-01", "p-3"): 0},
        ndcg_cutoffs=(3,),
        precision_cutoffs=(3,),
        recall_cutoffs=(3,),
    )
    metric = report["routes"]["raw_dense"]["by_cutoff"]["3"]

    assert metric["linear_ndcg_at_k"] > metric["ndcg_at_k"]
    assert metric["linear_ndcg_at_k"] < 1.0


def test_query_diagnostics_exposes_qrels_distribution_and_route_metrics() -> None:
    from personaforge.eval.retrieval_metrics import compute_query_diagnostics

    record = {
        "item_id": "q1",
        "candidates": [
            {"parent_id": "a", "route_ranks": {"raw_dense": {"rank": 1, "score": 0.9}}},
            {"parent_id": "b", "route_ranks": {"raw_dense": {"rank": 2, "score": 0.8}}},
            {"parent_id": "c", "route_ranks": {"raw_dense": {"rank": 3, "score": 0.7}}},
            {"parent_id": "d", "route_ranks": {"raw_dense": {"rank": 4, "score": 0.6}}},
        ],
    }
    diagnostics = compute_query_diagnostics(
        record,
        {("q1", "a"): 2, ("q1", "b"): 1, ("q1", "c"): 0},
        cutoffs=(1, 3),
    )

    assert diagnostics["qrels"] == {
        "candidate_count": 4,
        "labeled_count": 3,
        "unlabelled_count": 1,
        "zero_count": 1,
        "useful_count": 2,
        "strong_count": 1,
        "useful_recall_denominator": 2,
        "strong_recall_denominator": 1,
        "coverage": 0.75,
    }
    metric = diagnostics["routes"]["raw_dense"]["by_cutoff"]["1"]
    assert metric["useful_recall_at_k"] is None
    assert metric["strong_recall_at_k"] is None
    assert metric["unjudged_top_count"] == 0


def test_ranking_report_exposes_only_declared_cutoff_groups() -> None:
    records = [{
        "item_id": "dev-01",
        "split": "dev",
        "candidates": [{"parent_id": "p-1"}],
    }]
    rankings = [{
        "item_id": "dev-01",
        "split": "dev",
        "routes": {"raw_dense": [{"rank": 1, "parent_id": "p-1", "score": 1.0}]},
    }]
    report = compute_split_metrics_from_rankings(
        records,
        rankings,
        {("dev-01", "p-1"): 2},
        ndcg_cutoffs=(1, 3, 10),
        precision_cutoffs=(1, 5),
        recall_cutoffs=(10, 50, 100),
    )

    assert report["cutoff_groups"] == {
        "ndcg": [1, 3, 10],
        "precision": [1, 5],
        "recall": [10, 50, 100],
    }
    assert report["route_depths"]["raw_dense"] == 1
    assert report["max_supported_cutoff"] == 1


def test_query_diagnostics_exposes_recall_distribution_for_a_fully_judged_query() -> None:
    from personaforge.eval.retrieval_metrics import compute_query_diagnostics

    record = {
        "item_id": "q1",
        "candidates": [
            {"parent_id": "a", "route_ranks": {"raw_dense": {"rank": 1, "score": 0.9}}},
            {"parent_id": "b", "route_ranks": {"raw_dense": {"rank": 2, "score": 0.8}}},
        ],
    }
    diagnostics = compute_query_diagnostics(
        record,
        {("q1", "a"): 2, ("q1", "b"): 0},
        cutoffs=(1, 2),
    )

    metric = diagnostics["routes"]["raw_dense"]["by_cutoff"]["1"]
    assert metric["useful_recall_min_at_k"] == 1.0
    assert metric["useful_recall_median_at_k"] == 1.0
    assert metric["useful_recall_max_at_k"] == 1.0
    assert metric["useful_recall_zero_query_count"] == 0
    assert metric["useful_recall_one_query_count"] == 1
