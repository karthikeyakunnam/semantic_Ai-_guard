import numpy as np
def cosine_matrix(a, b): return np.asarray(a) @ np.asarray(b).T
def document_score(matrix, threshold=0.45):
    if matrix.size == 0: return 0.0
    best_a = matrix.max(axis=1); best_b = matrix.max(axis=0)
    # Symmetric coverage: reduces bias when documents differ greatly in length.
    return float(np.clip((best_a.mean() + best_b.mean()) / 2, 0, 1))
