import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.database.database import get_db
from app.database.models import Analysis, Document
from app.services.chunking import chunk_passages
from app.services.embedding import encode
from app.services.reranking import RERANKER_MODEL, rerank
from app.services.similarity import cosine_matrix, document_score
from app.services.vector_search import search
from app.services.hybrid_similarity import classify_passages, overlap_signals, hybrid_score

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analysis", tags=["analysis"])


class CompareRequest(BaseModel):
    document_a_id: int
    document_b_id: int


@router.post("/compare")
def compare(req: CompareRequest, db: Session = Depends(get_db)):
    if req.document_a_id == req.document_b_id:
        raise HTTPException(400, "Choose two different documents")
    a, b = db.get(Document, req.document_a_id), db.get(Document, req.document_b_id)
    if not a or not b:
        raise HTTPException(404, "Document not found")

    all_a = classify_passages(chunk_passages(a.text), settings.boilerplate_phrases)
    all_b = classify_passages(chunk_passages(b.text), settings.boilerplate_phrases)
    passages_a = [p for p in all_a if not p.excluded]
    passages_b = [p for p in all_b if not p.excluded]
    if not passages_a or not passages_b:
        raise HTTPException(422, "Not enough text to compare")

    texts_a, texts_b = [p.text for p in passages_a], [p.text for p in passages_b]
    vectors_a, vectors_b = encode(texts_a), encode(texts_b)
    matrix = cosine_matrix(vectors_a, vectors_b)
    faiss_scores, faiss_ids = search(vectors_a, vectors_b, settings.top_k)

    # Rerank only the FAISS candidate pairs; do not run a quadratic cross-encoder
    # over every passage pair. A model/cache/download failure degrades gracefully
    # to the original FAISS ordering, and logs no assignment text.
    candidate_pairs = []
    candidate_refs = []
    for i in range(len(passages_a)):
        for score, candidate_id in zip(faiss_scores[i], faiss_ids[i]):
            j = int(candidate_id)
            if j >= 0:
                candidate_refs.append((i, j, float(score)))
                candidate_pairs.append((texts_a[i], texts_b[j]))

    reranker_available = True
    try:
        cross_scores = rerank(candidate_pairs)
        if len(cross_scores) != len(candidate_refs):
            raise ValueError("Cross-encoder returned an unexpected number of scores")
    except Exception as exc:
        logger.warning(
            "Cross-encoder unavailable; retaining FAISS ranking (%s)",
            type(exc).__name__,
        )
        cross_scores = [None] * len(candidate_refs)
        reranker_available = False

    by_source: dict[int, list[dict]] = {i: [] for i in range(len(passages_a))}
    for (i, j, faiss_score), cross_score in zip(candidate_refs, cross_scores):
        by_source[i].append(
            {
                "passage": texts_b[j],
                "similarity": round(faiss_score, 4),
                "rerank_score": round(cross_score, 4) if cross_score is not None else None,
                "_faiss_score": faiss_score,
                "_rerank_score": cross_score,
            }
        )

    matches = []
    for i, candidates in by_source.items():
        if not candidates or max(item["_faiss_score"] for item in candidates) < settings.min_similarity:
            continue
        candidates.sort(
            key=lambda item: (
                item["_rerank_score"] if item["_rerank_score"] is not None else item["_faiss_score"]
            ),
            reverse=True,
        )
        best = candidates[0]
        matches.append(
            {
                "passage_a": texts_a[i],
                "passage_b": best["passage"],
                # Keep the existing similarity field as the original cosine score.
                "similarity": best["similarity"],
                "rerank_score": best["rerank_score"],
                "candidates": [
                    {key: value for key, value in item.items() if not key.startswith("_")}
                    for item in candidates
                ],
            }
        )
    matches.sort(
        key=lambda item: (
            item["rerank_score"] if item["rerank_score"] is not None else item["similarity"]
        ),
        reverse=True,
    )

    calibration_path = Path(__file__).resolve().parents[3] / "evaluation" / "results" / "hybrid_calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        weights = calibration["weights"]
        hybrid_threshold = float(calibration["threshold"])
        calibration_status = "validation_calibrated"
    else:
        weights = {"semantic": 0.50, "lexical": 0.25, "phrase": 0.25}
        hybrid_threshold = 0.50
        calibration_status = "uncalibrated_default"
    hybrid_matches = []
    for position, (i, j, faiss_score) in enumerate(candidate_refs):
        signals = overlap_signals(texts_a[i], texts_b[j])
        cross_score = cross_scores[position]
        hybrid_matches.append({
            "passage_a": texts_a[i], "passage_b": texts_b[j],
            "source_a": {"start_char": passages_a[i].start_char, "end_char": passages_a[i].end_char},
            "source_b": {"start_char": passages_b[j].start_char, "end_char": passages_b[j].end_char},
            "semantic_similarity": round(float(faiss_score), 4),
            **{key: round(value, 4) for key, value in signals.items()},
            "hybrid_score": round(hybrid_score(float(faiss_score), signals, weights), 4),
            "rerank_score": round(cross_score, 4) if cross_score is not None else None,
        })
    hybrid_matches.sort(key=lambda item: item["hybrid_score"], reverse=True)
    for item in hybrid_matches:
        item["passes_review_threshold"] = item["hybrid_score"] >= hybrid_threshold
    hybrid_matches = [item for item in hybrid_matches if item["passes_review_threshold"]]
    score = document_score(matrix)  # Stable bi-encoder aggregate for API compatibility.
    row = Analysis(
        document_a_id=a.id,
        document_b_id=b.id,
        similarity=score,
        matches_json=json.dumps(matches),
        audit_json=json.dumps({"hybrid_matches": hybrid_matches, "excluded_passages": {"document_a": [p.as_dict() for p in all_a if p.excluded], "document_b": [p.as_dict() for p in all_b if p.excluded]}, "weights": weights, "threshold": hybrid_threshold, "calibration_status": calibration_status}),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    response = {
        "id": row.id,
        "document_a": {"id": a.id, "filename": a.filename},
        "document_b": {"id": b.id, "filename": b.filename},
        "similarity": round(score, 4),
        "match_count": len(matches),
        "matches": matches,
        "hybrid_matches": hybrid_matches,
        "hybrid_calibration_status": calibration_status,
        "excluded_passages": {
            "document_a": [p.as_dict() for p in all_a if p.excluded],
            "document_b": [p.as_dict() for p in all_b if p.excluded],
        },
        "notice": "Similarity indicates passages for human review; it is not proof of plagiarism.",
        "reranker_used": reranker_available,
        "reranker_model": RERANKER_MODEL if reranker_available else None,
    }
    if not reranker_available:
        response["reranker_notice"] = (
            "Cross-encoder unavailable; candidates are ordered by embedding similarity. "
            "Download/cache the reranker model to enable reranking."
        )
    return response
