import pickle
import subprocess
import sys
from pathlib import Path

import faiss
import numpy as np


def search(query_vectors, corpus_vectors, top_k=5):
    corpus = np.ascontiguousarray(corpus_vectors, dtype="float32")
    queries = np.ascontiguousarray(query_vectors, dtype="float32")
    if corpus.ndim != 2 or queries.ndim != 2 or corpus.shape[0] == 0 or queries.shape[0] == 0:
        raise ValueError("FAISS search requires non-empty two-dimensional vector arrays")
    if corpus.shape[1] != queries.shape[1]:
        raise ValueError("Query and corpus vector dimensions must match")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if sys.platform == "darwin":
        # The macOS PyTorch and pip FAISS wheels may bundle distinct OpenMP
        # runtimes. Keep FAISS in a short-lived process so their runtimes never
        # coexist. Only numerical vectors cross this boundary, not document text.
        payload = pickle.dumps((queries, corpus, top_k), protocol=pickle.HIGHEST_PROTOCOL)
        result = subprocess.run(
            [sys.executable, "-m", "app.services.faiss_worker"],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )
        return pickle.loads(result.stdout)

    index = faiss.IndexFlatIP(corpus.shape[1])
    index.add(corpus)
    return index.search(queries, min(top_k, len(corpus)))
