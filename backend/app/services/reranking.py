from functools import lru_cache

import numpy as np

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker():
    # Import lazily so the Cross-Encoder is loaded only when a comparison needs
    # reranking, after FAISS has produced the bounded candidate set.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def rerank(pairs: list[tuple[str, str]]) -> list[float]:
    """Return sigmoid-normalized MS MARCO passage-relevance scores."""
    if not pairs:
        return []
    logits = np.asarray(get_reranker().predict(pairs, show_progress_bar=False)).reshape(-1)
    # Stable sigmoid avoids an eager torch import and preserves the raw ranking.
    scores = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    scores[~positive] = exp_logits / (1.0 + exp_logits)
    return [float(score) for score in scores]
