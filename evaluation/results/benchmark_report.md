# SemanticGuard AI — Synthetic Retrieval Evaluation

> **Synthetic controlled evaluation only.** This dataset is authored for reproducible pipeline checks. It is not real student work, not independently human-reviewed, and must not be presented as real-world plagiarism detection accuracy.

## Run configuration

- Dataset: `evaluation/datasets/synthetic_pairs.csv` (SHA-256 `660d1497796ebcb37dfc483d3c14bc989705271fe199e1b8b382c3b8a5c58d65`)
- Generated at (UTC): 2026-11-06T11:19:13.138755+00:00
- Queries / candidate passages / labeled pairs: 10 / 10 / 100
- Relevant / non-relevant pairs: 6 / 94
- Scenarios: common_topic_independent, exact_copy, paraphrasing, partial_copy, unrelated
- Models: `sentence-transformers/all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Platform: Darwin arm64; Python 3.12.13; NumPy 2.5.4
- Package versions: Sentence Transformers 6.2.0; Torch 2.15.0; FAISS 1.15.1
- FAISS candidate pool K=5; evaluated output K=1; timing repeats=3

The two evaluated rankers use **identical FAISS top-K candidates** for every query. FAISS candidate recall is reported separately because a reranker cannot promote a passage that retrieval did not return.

## Aggregate ranking metrics

Precision@K / Recall@K / F1@K are micro retrieval metrics over the returned top-K results and the full labeled candidate pool. MRR and nDCG@K are macro-averaged by query; a query with no relevant candidates contributes zero. Confusion matrices treat returned top-K pairs as predicted relevant and every other pair in the labeled pool as predicted non-relevant.

| Pipeline | Precision@K | Recall@K | F1@K | MRR | nDCG@K | FAISS candidate recall@candidate-K | Recall@K among retrieved positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| faiss_only | 0.600 | 1.000 | 0.750 | 0.600 | 0.600 | 1.000 | 1.000 |
| faiss_cross_encoder | 0.600 | 1.000 | 0.750 | 0.600 | 0.600 | 1.000 | 1.000 |

Cross-Encoder minus FAISS-only: Precision@K +0.000; Recall@K +0.000; F1@K +0.000; MRR +0.000; nDCG@K +0.000.

## Confusion matrices

Counts are micro-aggregated over every labeled query/candidate pair (rows: actual non-relevant/relevant; columns: predicted non-relevant/relevant).

| Pipeline | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| faiss_only | 90 | 4 | 0 | 6 |
| faiss_cross_encoder | 90 | 4 | 0 | 6 |

## Scenario breakdown

| Scenario | Pipeline | Queries | Precision@K | Recall@K | F1@K | MRR | nDCG@K | Candidate recall@candidate-K | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact_copy | faiss_only | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18 | 0 | 0 | 2 |
| paraphrasing | faiss_only | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18 | 0 | 0 | 2 |
| partial_copy | faiss_only | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18 | 0 | 0 | 2 |
| common_topic_independent | faiss_only | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 18 | 2 | 0 | 0 |
| unrelated | faiss_only | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 18 | 2 | 0 | 0 |
| exact_copy | faiss_cross_encoder | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18 | 0 | 0 | 2 |
| paraphrasing | faiss_cross_encoder | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18 | 0 | 0 | 2 |
| partial_copy | faiss_cross_encoder | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18 | 0 | 0 | 2 |
| common_topic_independent | faiss_cross_encoder | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 18 | 2 | 0 | 0 |
| unrelated | faiss_cross_encoder | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 18 | 2 | 0 | 0 |

## Timing

Model loading is measured separately from inference. Inference times are mean wall-clock durations over the configured repeats; the first warm-up pass is excluded. FAISS search timing includes its worker-process startup on macOS. Model load timing includes any first-use download if weights are not cached.

| Stage | Mean seconds |
|---|---:|
| Embedding model load | 1.9155 |
| Cross-Encoder model load | 2.3656 |
| Embedding inference | 0.0458 |
| FAISS search | 0.2433 |
| Cross-Encoder inference on FAISS candidates | 0.0907 |
| FAISS-only inference total | 0.2891 |
| FAISS + Cross-Encoder inference total | 0.3797 |

## Interpretation and limitations

- Exact-copy, paraphrase, and partial-copy labels are known by construction; common-topic independent and unrelated pairs are labeled non-matches by construction.
- This small synthetic set measures these examples and this fixed candidate corpus only. It does not estimate performance on real assignments, disciplines, languages, or adversarial paraphrases.
- A relevant pair means the passages were constructed to overlap or express the same specific content. It is not a finding of student misconduct.
- Embedding cosine and Cross-Encoder scores are ranking signals, not probabilities of plagiarism.
- A reranker can only reorder FAISS candidates. Candidate recall is the upper bound on end-to-end recall for that retrieved pool.
- Results can change with model revisions, hardware, library versions, and K settings. No general accuracy claim is warranted from this dataset.

The reproducible per-pair scores and aggregate metrics are saved beside this report as `pairwise_results.csv` and `aggregate_metrics.csv`.
