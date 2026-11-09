"""Ranking and retrieval metrics for binary passage relevance labels."""
from collections.abc import Sequence


def confusion_counts(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, int]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal lengths")
    if any(label not in (0, 1) for label in labels) or any(pred not in (0, 1) for pred in predictions):
        raise ValueError("labels and predictions must be binary")
    tp = sum(label == 1 and pred == 1 for label, pred in zip(labels, predictions))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predictions))
    tn = sum(label == 0 and pred == 0 for label, pred in zip(labels, predictions))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predictions))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_f1(counts: dict[str, int]) -> tuple[float, float, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def precision_at_k(labels_by_id: dict[str, int], ranked_ids: Sequence[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    chosen = ranked_ids[:k]
    if not chosen:
        return 0.0
    return sum(labels_by_id[item] for item in chosen) / len(chosen)


def recall_at_k(labels_by_id: dict[str, int], ranked_ids: Sequence[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    relevant_count = sum(labels_by_id.values())
    if not relevant_count:
        return 0.0
    return sum(labels_by_id[item] for item in ranked_ids[:k]) / relevant_count


def reciprocal_rank(labels_by_id: dict[str, int], ranked_ids: Sequence[str]) -> float:
    for rank, candidate_id in enumerate(ranked_ids, start=1):
        if labels_by_id[candidate_id]:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels_by_id: dict[str, int], ranked_ids: Sequence[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    import math

    dcg = sum(labels_by_id[item] / math.log2(rank + 1) for rank, item in enumerate(ranked_ids[:k], start=1))
    ideal_relevant = min(sum(labels_by_id.values()), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def summarize_rankings(
    labels_by_query: dict[str, dict[str, int]],
    predicted_by_query: dict[str, Sequence[str]],
    candidate_retrieval_by_query: dict[str, Sequence[str]],
    candidate_k: int,
    output_k: int,
) -> dict[str, float | int]:
    labels_all, predictions_all = [], []
    total_tp_at_k = total_predictions_at_k = total_relevant = 0
    reciprocal_ranks, ndcgs = [], []
    candidate_tp = retrieved_relevant = rerank_tp = 0
    for query_id, labels in labels_by_query.items():
        predictions = list(predicted_by_query.get(query_id, ()))[:output_k]
        retrieved = list(candidate_retrieval_by_query.get(query_id, ()))[:candidate_k]
        predicted_set = set(predictions)
        labels_all.extend(labels.values())
        predictions_all.extend(int(candidate_id in predicted_set) for candidate_id in labels)
        hits = sum(labels[candidate_id] for candidate_id in predictions)
        query_relevant = sum(labels.values())
        retrieved_hits = sum(labels[candidate_id] for candidate_id in retrieved)
        total_tp_at_k += hits
        total_predictions_at_k += len(predictions)
        total_relevant += query_relevant
        candidate_tp += retrieved_hits
        retrieved_relevant += retrieved_hits
        rerank_tp += hits
        reciprocal_ranks.append(reciprocal_rank(labels, predictions))
        ndcgs.append(ndcg_at_k(labels, predictions, output_k))

    counts = confusion_counts(labels_all, predictions_all)
    micro_precision, micro_recall, micro_f1 = precision_recall_f1(counts)
    precision = total_tp_at_k / total_predictions_at_k if total_predictions_at_k else 0.0
    recall = total_tp_at_k / total_relevant if total_relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **counts,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "ndcg_at_k": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "faiss_candidate_recall_at_candidate_k": candidate_tp / total_relevant if total_relevant else 0.0,
        "recall_at_k_given_faiss_candidates": rerank_tp / retrieved_relevant if retrieved_relevant else 0.0,
        "predicted_count": total_predictions_at_k,
        "relevant_count": total_relevant,
        "pair_count": len(labels_all),
    }
