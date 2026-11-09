# SemanticGuard Phase 4 — Synthetic False-Positive Evaluation

**Controlled synthetic examples only; these results are not real-world plagiarism accuracy. Scores are not plagiarism probabilities.**

Validation SHA-256: `83b967d13f361b173d2243b5206c5ed14f898e0d0ba80c39d80082b88d9ab5c3`
Held-out test SHA-256: `8576ab736787a79e1d5dc7492a823e1396b90356e24c6fed4ed015470560c800`

Validation and test contain disjoint topic and template identifiers. Weights and threshold were calibrated on validation, written to `hybrid_calibration.json`, then frozen before test scoring.

Frozen weights: `{'semantic': 0.2, 'lexical': 0.4, 'phrase': 0.4}`; threshold: `0.20`.

## Held-out test metrics

| Pipeline | Precision | Recall | F1 | Specificity | False-positive rate | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| faiss_baseline | 0.750 | 1.000 | 0.857 | 0.667 | 0.333 | 2 | 1 | 0 | 3 |
| faiss_cross_encoder | 0.750 | 1.000 | 0.857 | 0.667 | 0.333 | 2 | 1 | 0 | 3 |
| hybrid | 0.667 | 0.667 | 0.667 | 0.667 | 0.333 | 2 | 1 | 1 | 2 |

## Scenario breakdown

| Scenario | Pipeline | TN | FP | FN | TP | FPR | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Common prompt | faiss_baseline | 0 | 1 | 0 | 0 | 1.000 | 0.000 |
| Common prompt | faiss_cross_encoder | 0 | 1 | 0 | 0 | 1.000 | 0.000 |
| Common prompt | hybrid | 0 | 1 | 0 | 0 | 1.000 | 0.000 |
| Common topic independent | faiss_baseline | 1 | 0 | 0 | 0 | 0.000 | 0.000 |
| Common topic independent | faiss_cross_encoder | 1 | 0 | 0 | 0 | 0.000 | 0.000 |
| Common topic independent | hybrid | 1 | 0 | 0 | 0 | 0.000 | 0.000 |
| Exact copying | faiss_baseline | 0 | 0 | 0 | 1 | 0.000 | 1.000 |
| Exact copying | faiss_cross_encoder | 0 | 0 | 0 | 1 | 0.000 | 1.000 |
| Exact copying | hybrid | 0 | 0 | 0 | 1 | 0.000 | 1.000 |
| Paraphrase | faiss_baseline | 0 | 0 | 0 | 1 | 0.000 | 1.000 |
| Paraphrase | faiss_cross_encoder | 0 | 0 | 0 | 1 | 0.000 | 1.000 |
| Paraphrase | hybrid | 0 | 0 | 1 | 0 | 0.000 | 0.000 |
| Partial copy | faiss_baseline | 0 | 0 | 0 | 1 | 0.000 | 1.000 |
| Partial copy | faiss_cross_encoder | 0 | 0 | 0 | 1 | 0.000 | 1.000 |
| Partial copy | hybrid | 0 | 0 | 0 | 1 | 0.000 | 1.000 |
| Unrelated | faiss_baseline | 1 | 0 | 0 | 0 | 0.000 | 0.000 |
| Unrelated | faiss_cross_encoder | 1 | 0 | 0 | 0 | 0.000 | 0.000 |
| Unrelated | hybrid | 1 | 0 | 0 | 0 | 0.000 | 0.000 |

## Timing and memory

Embedding model load: 1.6995s
Cross-Encoder model load: 2.4761s
Cross-Encoder validation inference: 0.0244s
Cross-Encoder test inference: 0.0133s
Process peak RSS increase over the full benchmark run: 29.17 MiB (includes model loading and inference).

The pair dataset tests threshold behavior, not FAISS retrieval recall: each row is a labeled candidate pair. Phase 3 separately reports retrieval/ranking metrics. The baseline is the current cosine cutoff (0.45); Cross-Encoder reranking has identical candidate membership and thus identical binary decisions in this one-candidate-per-pair design. Hybrid predictions use validation-frozen weights/threshold.

Results are authored examples with one observation per scenario. Per-scenario false-positive rates are therefore highly unstable and descriptive only. A negative result does not support general claims about academic integrity or student misconduct.
