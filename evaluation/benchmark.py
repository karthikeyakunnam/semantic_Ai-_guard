"""Run deterministic synthetic retrieval and reranking evaluation, entirely locally."""
import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.embedding import encode, get_model  # noqa: E402
from app.services.reranking import get_reranker, rerank  # noqa: E402
from app.services.vector_search import search  # noqa: E402

def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


REQUIRED_COLUMNS = {"scenario", "query_id", "query_passage", "candidate_id", "candidate_passage", "relevant"}


def load_dataset(path: Path):
    queries, candidates, labels = {}, {}, defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError("CSV is missing columns: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            query_id, candidate_id = row["query_id"].strip(), row["candidate_id"].strip()
            query_text, candidate_text = row["query_passage"].strip(), row["candidate_passage"].strip()
            scenario = row["scenario"].strip()
            try:
                label = int(row["relevant"])
            except (TypeError, ValueError):
                raise ValueError(f"Invalid relevant label on CSV line {line}") from None
            if not all((query_id, candidate_id, query_text, candidate_text, scenario)):
                raise ValueError(f"Empty required field on CSV line {line}")
            if label not in (0, 1):
                raise ValueError(f"Relevant label must be 0 or 1 on CSV line {line}")
            current_query = (scenario, query_text)
            if query_id in queries and queries[query_id] != current_query:
                raise ValueError(f"query_id {query_id!r} maps to inconsistent scenario or text")
            if candidate_id in candidates and candidates[candidate_id] != candidate_text:
                raise ValueError(f"candidate_id {candidate_id!r} maps to inconsistent text")
            if candidate_id in labels[query_id]:
                raise ValueError(f"Duplicate pair for query {query_id!r} and candidate {candidate_id!r}")
            queries[query_id] = current_query
            candidates[candidate_id] = candidate_text
            labels[query_id][candidate_id] = label
    if not queries or not candidates:
        raise ValueError("Dataset must contain at least one query and candidate")
    expected = set(candidates)
    for query_id, query_labels in labels.items():
        if set(query_labels) != expected:
            raise ValueError(f"Query {query_id!r} does not label the complete candidate pool")
    return queries, candidates, labels


def _metrics_rows(pipeline_name, scope, query_ids, labels, predicted, retrieved, candidate_k, output_k):
    from metrics import summarize_rankings

    result = summarize_rankings(
        {query_id: labels[query_id] for query_id in query_ids},
        {query_id: predicted[query_id] for query_id in query_ids},
        {query_id: retrieved[query_id] for query_id in query_ids},
        candidate_k,
        output_k,
    )
    return {
        "pipeline": pipeline_name,
        "scope": scope,
        "query_count": len(query_ids),
        "candidate_k": candidate_k,
        "output_k": output_k,
        **result,
    }


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(path, dataset_path, metadata, metrics_rows, timing):
    repo_root = Path(__file__).resolve().parents[1]
    try:
        dataset_display = Path(dataset_path).resolve().relative_to(repo_root)
    except ValueError:
        dataset_display = Path(dataset_path).resolve()
    overall = {row["pipeline"]: row for row in metrics_rows if row["scope"] == "overall"}
    faiss_row, rerank_row = overall["faiss_only"], overall["faiss_cross_encoder"]
    scenario_rows = [row for row in metrics_rows if row["scope"] != "overall"]
    lines = [
        "# SemanticGuard AI — Synthetic Retrieval Evaluation",
        "",
        "> **Synthetic controlled evaluation only.** This dataset is authored for reproducible pipeline checks. It is not real student work, not independently human-reviewed, and must not be presented as real-world plagiarism detection accuracy.",
        "",
        "## Run configuration",
        "",
        f"- Dataset: `{dataset_display}` (SHA-256 `{metadata['dataset_sha256']}`)",
        f"- Generated at (UTC): {metadata['run_timestamp_utc']}",
        f"- Queries / candidate passages / labeled pairs: {metadata['query_count']} / {metadata['candidate_count']} / {metadata['pair_count']}",
        f"- Relevant / non-relevant pairs: {metadata['relevant_count']} / {metadata['pair_count'] - metadata['relevant_count']}",
        f"- Scenarios: {', '.join(metadata['scenarios'])}",
        f"- Models: `{metadata['embedding_model']}` and `{metadata['cross_encoder_model']}`",
        f"- Platform: {metadata['platform']} {metadata['machine']}; Python {metadata['python_version']}; NumPy {metadata['numpy_version']}",
        f"- Package versions: Sentence Transformers {metadata['sentence_transformers_version']}; Torch {metadata['torch_version']}; FAISS {metadata['faiss_version']}",
        f"- FAISS candidate pool K={metadata['candidate_k']}; evaluated output K={metadata['output_k']}; timing repeats={metadata['repeats']}",
        "",
        "The two evaluated rankers use **identical FAISS top-K candidates** for every query. FAISS candidate recall is reported separately because a reranker cannot promote a passage that retrieval did not return.",
        "",
        "## Aggregate ranking metrics",
        "",
        "Precision@K / Recall@K / F1@K are micro retrieval metrics over the returned top-K results and the full labeled candidate pool. MRR and nDCG@K are macro-averaged by query; a query with no relevant candidates contributes zero. Confusion matrices treat returned top-K pairs as predicted relevant and every other pair in the labeled pool as predicted non-relevant.",
        "",
        "| Pipeline | Precision@K | Recall@K | F1@K | MRR | nDCG@K | FAISS candidate recall@candidate-K | Recall@K among retrieved positives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (faiss_row, rerank_row):
        lines.append(
            f"| {row['pipeline']} | {row['precision_at_k']:.3f} | {row['recall_at_k']:.3f} | {row['f1_at_k']:.3f} | {row['mrr']:.3f} | {row['ndcg_at_k']:.3f} | {row['faiss_candidate_recall_at_candidate_k']:.3f} | {row['recall_at_k_given_faiss_candidates']:.3f} |"
        )
    lines.extend([
        "",
        f"Cross-Encoder minus FAISS-only: Precision@K {rerank_row['precision_at_k'] - faiss_row['precision_at_k']:+.3f}; Recall@K {rerank_row['recall_at_k'] - faiss_row['recall_at_k']:+.3f}; F1@K {rerank_row['f1_at_k'] - faiss_row['f1_at_k']:+.3f}; MRR {rerank_row['mrr'] - faiss_row['mrr']:+.3f}; nDCG@K {rerank_row['ndcg_at_k'] - faiss_row['ndcg_at_k']:+.3f}.",
        "",
        "## Confusion matrices",
        "",
        "Counts are micro-aggregated over every labeled query/candidate pair (rows: actual non-relevant/relevant; columns: predicted non-relevant/relevant).",
        "",
        "| Pipeline | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in (faiss_row, rerank_row):
        lines.append(f"| {row['pipeline']} | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} |")
    lines.extend([
        "",
        "## Scenario breakdown",
        "",
        "| Scenario | Pipeline | Queries | Precision@K | Recall@K | F1@K | MRR | nDCG@K | Candidate recall@candidate-K | TN | FP | FN | TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in scenario_rows:
        lines.append(
            f"| {row['scope']} | {row['pipeline']} | {row['query_count']} | {row['precision_at_k']:.3f} | {row['recall_at_k']:.3f} | {row['f1_at_k']:.3f} | {row['mrr']:.3f} | {row['ndcg_at_k']:.3f} | {row['faiss_candidate_recall_at_candidate_k']:.3f} | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} |"
        )
    lines.extend([
        "",
        "## Timing",
        "",
        "Model loading is measured separately from inference. Inference times are mean wall-clock durations over the configured repeats; the first warm-up pass is excluded. FAISS search timing includes its worker-process startup on macOS. Model load timing includes any first-use download if weights are not cached.",
        "",
        "| Stage | Mean seconds |",
        "|---|---:|",
        f"| Embedding model load | {timing['embedding_model_load_seconds']:.4f} |",
        f"| Cross-Encoder model load | {timing['cross_encoder_model_load_seconds']:.4f} |",
        f"| Embedding inference | {timing['embedding_inference_seconds']:.4f} |",
        f"| FAISS search | {timing['faiss_search_seconds']:.4f} |",
        f"| Cross-Encoder inference on FAISS candidates | {timing['cross_encoder_inference_seconds']:.4f} |",
        f"| FAISS-only inference total | {timing['embedding_inference_seconds'] + timing['faiss_search_seconds']:.4f} |",
        f"| FAISS + Cross-Encoder inference total | {timing['embedding_inference_seconds'] + timing['faiss_search_seconds'] + timing['cross_encoder_inference_seconds']:.4f} |",
        "",
        "## Interpretation and limitations",
        "",
        "- Exact-copy, paraphrase, and partial-copy labels are known by construction; common-topic independent and unrelated pairs are labeled non-matches by construction.",
        "- This small synthetic set measures these examples and this fixed candidate corpus only. It does not estimate performance on real assignments, disciplines, languages, or adversarial paraphrases.",
        "- A relevant pair means the passages were constructed to overlap or express the same specific content. It is not a finding of student misconduct.",
        "- Embedding cosine and Cross-Encoder scores are ranking signals, not probabilities of plagiarism.",
        "- A reranker can only reorder FAISS candidates. Candidate recall is the upper bound on end-to-end recall for that retrieved pool.",
        "- Results can change with model revisions, hardware, library versions, and K settings. No general accuracy claim is warranted from this dataset.",
        "",
        "The reproducible per-pair scores and aggregate metrics are saved beside this report as `pairwise_results.csv` and `aggregate_metrics.csv`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(dataset_path: Path, results_dir: Path, candidate_k=5, output_k=1, repeats=3):
    if candidate_k < output_k or output_k < 1 or repeats < 1:
        raise ValueError("Require candidate_k >= output_k >= 1 and repeats >= 1")
    queries, candidates, labels = load_dataset(Path(dataset_path))
    query_ids, candidate_ids = list(queries), list(candidates)
    query_texts = [queries[query_id][1] for query_id in query_ids]
    candidate_texts = [candidates[candidate_id] for candidate_id in candidate_ids]

    started = time.perf_counter()
    get_model()
    embedding_load = time.perf_counter() - started
    started = time.perf_counter()
    get_reranker()
    reranker_load = time.perf_counter() - started

    all_texts = list(dict.fromkeys(query_texts + candidate_texts))
    text_index = {text: index for index, text in enumerate(all_texts)}
    encode([all_texts[0]])  # Warm-up excluded from inference timings.
    search(np.zeros((1, 384), dtype=np.float32), np.zeros((1, 384), dtype=np.float32), 1)
    rerank([(query_texts[0], candidate_texts[0])])

    embedding_times, faiss_times, cross_times = [], [], []
    cosine_scores = faiss_indices = cross_scores_by_query = None
    for _ in range(repeats):
        started = time.perf_counter()
        vectors = encode(all_texts)
        embedding_times.append(time.perf_counter() - started)
        query_vectors = np.asarray([vectors[text_index[text]] for text in query_texts], dtype=np.float32)
        corpus_vectors = np.asarray([vectors[text_index[text]] for text in candidate_texts], dtype=np.float32)

        started = time.perf_counter()
        cosine_scores, faiss_indices = search(query_vectors, corpus_vectors, len(candidate_ids))
        faiss_times.append(time.perf_counter() - started)

        pairs, refs = [], []
        for query_index, query_id in enumerate(query_ids):
            for candidate_index in faiss_indices[query_index, :candidate_k]:
                candidate_id = candidate_ids[int(candidate_index)]
                pairs.append((queries[query_id][1], candidates[candidate_id]))
                refs.append((query_id, candidate_id))
        started = time.perf_counter()
        pair_scores = rerank(pairs)
        cross_times.append(time.perf_counter() - started)
        cross_scores_by_query = {query_id: {} for query_id in query_ids}
        for (query_id, candidate_id), score in zip(refs, pair_scores):
            cross_scores_by_query[query_id][candidate_id] = float(score)

    labels_by_query = {query_id: dict(labels[query_id]) for query_id in query_ids}
    faiss_ranked, retrieved, reranked_ranked = {}, {}, {}
    cosine_by_pair, faiss_rank_by_pair = {}, {}
    for query_index, query_id in enumerate(query_ids):
        ranking_indices = [int(item) for item in faiss_indices[query_index]]
        ranked_ids = [candidate_ids[item] for item in ranking_indices]
        faiss_ranked[query_id] = ranked_ids
        retrieved[query_id] = ranked_ids[:candidate_k]
        for rank, (candidate_index, score) in enumerate(zip(ranking_indices, cosine_scores[query_index]), start=1):
            candidate_id = candidate_ids[candidate_index]
            cosine_by_pair[(query_id, candidate_id)] = float(score)
            faiss_rank_by_pair[(query_id, candidate_id)] = rank
        reranked_ranked[query_id] = sorted(
            retrieved[query_id],
            key=lambda candidate_id: cross_scores_by_query[query_id][candidate_id],
            reverse=True,
        )

    scenarios = defaultdict(list)
    for query_id, (scenario, _) in queries.items():
        scenarios[scenario].append(query_id)
    metrics_rows = []
    for pipeline, predicted in (("faiss_only", faiss_ranked), ("faiss_cross_encoder", reranked_ranked)):
        metrics_rows.append(_metrics_rows(pipeline, "overall", query_ids, labels_by_query, predicted, retrieved, candidate_k, output_k))
        for scenario, scenario_query_ids in scenarios.items():
            metrics_rows.append(_metrics_rows(pipeline, scenario, scenario_query_ids, labels_by_query, predicted, retrieved, candidate_k, output_k))

    average = lambda values: float(sum(values) / len(values)) if values else 0.0
    timing = {
        "embedding_model_load_seconds": embedding_load,
        "cross_encoder_model_load_seconds": reranker_load,
        "embedding_inference_seconds": average(embedding_times),
        "faiss_search_seconds": average(faiss_times),
        "cross_encoder_inference_seconds": average(cross_times),
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "run_timestamp_utc": timestamp,
        "query_count": len(query_ids),
        "candidate_count": len(candidate_ids),
        "pair_count": sum(len(row) for row in labels_by_query.values()),
        "relevant_count": sum(sum(row.values()) for row in labels_by_query.values()),
        "scenarios": sorted(scenarios),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sentence_transformers_version": _package_version("sentence-transformers"),
        "torch_version": _package_version("torch"),
        "faiss_version": _package_version("faiss-cpu"),
        "candidate_k": candidate_k,
        "output_k": output_k,
        "repeats": repeats,
        "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metric_fields = ["pipeline", "scope", "query_count", "candidate_k", "output_k", "precision_at_k", "recall_at_k", "f1_at_k", "mrr", "ndcg_at_k", "faiss_candidate_recall_at_candidate_k", "recall_at_k_given_faiss_candidates", "tn", "fp", "fn", "tp", "micro_precision", "micro_recall", "micro_f1", "pair_count", "relevant_count", "predicted_count", "run_timestamp_utc", "dataset_sha256", "embedding_model", "cross_encoder_model", "sentence_transformers_version", "torch_version", "faiss_version", "embedding_model_load_seconds", "cross_encoder_model_load_seconds", "embedding_inference_seconds", "faiss_search_seconds", "cross_encoder_inference_seconds", "faiss_only_inference_total_seconds", "reranked_inference_total_seconds"]
    metric_output = [{field: row.get(field, timing.get(field, metadata.get(field, ""))) for field in metric_fields} for row in metrics_rows]
    for row in metric_output:
        row["faiss_only_inference_total_seconds"] = timing["embedding_inference_seconds"] + timing["faiss_search_seconds"]
        row["reranked_inference_total_seconds"] = timing["embedding_inference_seconds"] + timing["faiss_search_seconds"] + timing["cross_encoder_inference_seconds"]
    _write_csv(results_dir / "aggregate_metrics.csv", metric_output)

    pair_rows = []
    for query_id in query_ids:
        scenario, query_text = queries[query_id]
        faiss_candidates = retrieved[query_id]
        reranked_candidates = reranked_ranked[query_id]
        for candidate_id in candidate_ids:
            pair_rows.append({
                "scenario": scenario,
                "query_id": query_id,
                "candidate_id": candidate_id,
                "relevant": labels_by_query[query_id][candidate_id],
                "cosine_score": round(cosine_by_pair[(query_id, candidate_id)], 8),
                "faiss_rank": faiss_rank_by_pair[(query_id, candidate_id)],
                "faiss_candidate": int(candidate_id in faiss_candidates),
                "cross_encoder_score": round(cross_scores_by_query[query_id][candidate_id], 8) if candidate_id in cross_scores_by_query[query_id] else "",
                "reranked_rank": reranked_candidates.index(candidate_id) + 1 if candidate_id in reranked_candidates else "",
                "faiss_predicted_relevant": int(candidate_id in faiss_ranked[query_id][:output_k]),
                "reranked_predicted_relevant": int(candidate_id in reranked_candidates[:output_k]),
                "benchmark_timestamp_utc": timestamp,
                "dataset_sha256": metadata["dataset_sha256"],
                "embedding_model": metadata["embedding_model"],
                "cross_encoder_model": metadata["cross_encoder_model"],
                "sentence_transformers_version": metadata["sentence_transformers_version"],
                "torch_version": metadata["torch_version"],
                "faiss_version": metadata["faiss_version"],
            })
    _write_csv(results_dir / "pairwise_results.csv", pair_rows)
    _markdown_report(results_dir / "benchmark_report.md", Path(dataset_path), metadata, metrics_rows, timing)
    return {"metadata": metadata, "metrics": metrics_rows, "timing": timing, "results_dir": results_dir}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "datasets" / "synthetic_pairs.csv")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--candidate-k", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=1, help="number of results evaluated per query")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    result = run_benchmark(args.dataset, args.results_dir, args.candidate_k, args.top_k, args.repeats)
    overall = {row["pipeline"]: row for row in result["metrics"] if row["scope"] == "overall"}
    print(f"Synthetic benchmark completed for {result['metadata']['pair_count']} labeled pairs.")
    for pipeline, row in overall.items():
        print(f"{pipeline}: P@K={row['precision_at_k']:.3f} R@K={row['recall_at_k']:.3f} F1@K={row['f1_at_k']:.3f} MRR={row['mrr']:.3f} nDCG@K={row['ndcg_at_k']:.3f} FAISS candidate recall={row['faiss_candidate_recall_at_candidate_k']:.3f}")
    print(f"Results: {result['results_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
