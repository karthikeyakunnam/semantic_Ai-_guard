"""Isolated FAISS process used on macOS to avoid duplicate OpenMP runtimes."""
import pickle
import sys

import faiss
import numpy as np


def main():
    queries, corpus, top_k = pickle.load(sys.stdin.buffer)
    index = faiss.IndexFlatIP(corpus.shape[1])
    index.add(corpus)
    scores, ids = index.search(queries, min(top_k, len(corpus)))
    pickle.dump((scores, ids), sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
