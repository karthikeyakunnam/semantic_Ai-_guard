import csv
import sys
from pathlib import Path

import numpy as np
import pytest

EVALUATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALUATION_DIR))
import benchmark  # noqa: E402
from metrics import (  # noqa: E402
    confusion_counts,
    ndcg_at_k,
    precision_at_k,
    precision_recall_f1,
    reciprocal_rank,
    recall_at_k,
    summarize_rankings,
)


def test_confusion_matrix_and_precision_recall_f1():
    counts = confusion_counts([1, 0, 1, 0], [1, 1, 0, 0])
    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert precision_recall_f1(counts) == pytest.approx((0.5, 0.5, 0.5))


def test_top_k_precision_recall_mrr_and_ndcg():
    labels = {"a": 0, "b": 1, "c": 1}
    ranked = ["a", "b", "c"]
    assert precision_at_k(labels, ranked, 2) == pytest.approx(0.5)
    assert recall_at_k(labels, ranked, 2) == pytest.approx(0.5)
    assert reciprocal_rank(labels, ranked) == pytest.approx(0.5)
    assert ndcg_at_k(labels, ranked, 2) == pytest.approx(1 / np.log2(3) / (1 + 1 / np.log2(3)))
    assert reciprocal_rank({"a": 0}, ["a"]) == 0.0
    assert ndcg_at_k({"a": 0}, ["a"], 1) == 0.0


def test_faiss_candidate_recall_is_separate_from_reranker_recall():
    metrics = summarize_rankings(
        {"q": {"positive-1": 1, "positive-2": 1, "negative": 0}},
        {"q": ["negative"]},
        {"q": ["positive-1", "negative"]},
        candidate_k=2,
        output_k=1,
    )
    assert metrics["faiss_candidate_recall_at_candidate_k"] == pytest.approx(0.5)
    assert metrics["recall_at_k_given_faiss_candidates"] == pytest.approx(0.0)
    assert metrics["fn"] == 2
    assert metrics["fp"] == 1


def test_micro_precision_recall_and_f1_at_k():
    result = summarize_rankings(
        {"positive_query": {"hit": 1, "miss": 0}, "negative_query": {"n1": 0, "n2": 0}},
        {"positive_query": ["hit"], "negative_query": ["n1"]},
        {"positive_query": ["hit", "miss"], "negative_query": ["n1", "n2"]},
        candidate_k=2,
        output_k=1,
    )
    assert result["precision_at_k"] == pytest.approx(0.5)
    assert result["recall_at_k"] == pytest.approx(1.0)
    assert result["f1_at_k"] == pytest.approx(2 / 3)


def test_synthetic_dataset_is_balanced_by_scenario_and_labels_exact_copy():
    dataset = EVALUATION_DIR / "datasets" / "synthetic_pairs.csv"
    queries, candidates, labels = benchmark.load_dataset(dataset)
    assert len(queries) == 10
    assert len(candidates) == 10
    assert sum(sum(items.values()) for items in labels.values()) == 6
    assert {scenario for scenario, _ in queries.values()} == {
        "exact_copy", "paraphrasing", "partial_copy", "common_topic_independent", "unrelated"
    }
    exact_query = queries["exact-01"][1]
    assert exact_query == candidates["exact-target-01"]
    assert labels["exact-01"]["exact-target-01"] == 1
    assert labels["common-01"]["common-topic-target-01"] == 0


def test_load_dataset_rejects_incomplete_candidate_pools(tmp_path):
    path = tmp_path / "incomplete.csv"
    path.write_text(
        "scenario,query_id,query_passage,candidate_id,candidate_passage,relevant\n"
        "exact_copy,q1,query,c1,candidate,1\n",
        encoding="utf-8",
    )
    # One query with one candidate is a complete pool and is valid.
    assert len(benchmark.load_dataset(path)[2]["q1"]) == 1
    path.write_text(
        "scenario,query_id,query_passage,candidate_id,candidate_passage,relevant\n"
        "exact_copy,q1,query,c1,candidate-1,1\n"
        "exact_copy,q1,query,c2,candidate-2,0\n"
        "exact_copy,q2,query-2,c1,candidate-1,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complete candidate pool"):
        benchmark.load_dataset(path)


def test_benchmark_writes_pairwise_aggregate_and_markdown_outputs(tmp_path, monkeypatch):
    dataset = tmp_path / "pairs.csv"
    rows = [
        {"scenario": "exact_copy", "query_id": "q1", "query_passage": "q1", "candidate_id": "c1", "candidate_passage": "c1", "relevant": 1},
        {"scenario": "exact_copy", "query_id": "q1", "query_passage": "q1", "candidate_id": "c2", "candidate_passage": "c2", "relevant": 0},
        {"scenario": "unrelated", "query_id": "q2", "query_passage": "q2", "candidate_id": "c1", "candidate_passage": "c1", "relevant": 0},
        {"scenario": "unrelated", "query_id": "q2", "query_passage": "q2", "candidate_id": "c2", "candidate_passage": "c2", "relevant": 1},
    ]
    with dataset.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    vectors = {"q1": [1.0, 0.0], "q2": [0.0, 1.0], "c1": [1.0, 0.0], "c2": [0.0, 1.0]}
    monkeypatch.setattr(benchmark, "get_model", lambda: object())
    monkeypatch.setattr(benchmark, "get_reranker", lambda: object())
    monkeypatch.setattr(benchmark, "encode", lambda texts: np.asarray([vectors[text] for text in texts], dtype=np.float32))

    def fake_search(queries, corpus, top_k):
        scores = queries @ corpus.T
        ids = np.argsort(-scores, axis=1)[:, :top_k]
        return np.take_along_axis(scores, ids, axis=1), ids

    monkeypatch.setattr(benchmark, "search", fake_search)
    monkeypatch.setattr(
        benchmark,
        "rerank",
        lambda pairs: [0.9 if query == "q2" and candidate == "c2" or query == "q1" and candidate == "c2" else 0.1 for query, candidate in pairs],
    )

    output = tmp_path / "results"
    benchmark.run_benchmark(dataset, output, candidate_k=2, output_k=1, repeats=1)
    pairwise = list(csv.DictReader((output / "pairwise_results.csv").open(encoding="utf-8")))
    aggregates = list(csv.DictReader((output / "aggregate_metrics.csv").open(encoding="utf-8")))
    report = (output / "benchmark_report.md").read_text(encoding="utf-8")
    assert len(pairwise) == 4
    assert len(aggregates) == 6  # Overall plus two scenarios, for both pipelines.
    assert all(bool(row["faiss_candidate"]) == bool(row["cross_encoder_score"]) for row in pairwise)
    assert "Synthetic controlled evaluation only" in report
    assert "Confusion matrices" in report
    assert "Cross-Encoder minus FAISS-only" in report
    assert "embedding_model_load_seconds" in aggregates[0]
    assert "cross_encoder_model_load_seconds" in aggregates[0]
    assert "dataset_sha256" in aggregates[0]
