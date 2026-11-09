import io, json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database.models import Analysis, Document
router = APIRouter(prefix="/api/reports", tags=["reports"])
@router.get("")
def list_reports(db: Session = Depends(get_db)):
    rows = db.query(Analysis).order_by(Analysis.created_at.desc()).all()
    return [{"id": r.id, "document_a": r.document_a.filename, "document_b": r.document_b.filename, "similarity": r.similarity, "match_count": len(json.loads(r.matches_json)), "created_at": r.created_at.isoformat()} for r in rows]
@router.get("/{report_id}")
def report(report_id: int, db: Session = Depends(get_db)):
    row = db.get(Analysis, report_id)
    if not row: raise HTTPException(404, "Report not found")
    audit = json.loads(row.audit_json) if row.audit_json else {}
    return {"id": row.id, "document_a": {"id": row.document_a.id, "filename": row.document_a.filename}, "document_b": {"id": row.document_b.id, "filename": row.document_b.filename}, "similarity": row.similarity, "matches": json.loads(row.matches_json), **audit, "created_at": row.created_at.isoformat(), "notice": "Similarity indicates passages for human review; it is not proof of plagiarism."}
@router.get("/{report_id}/download")
def download(report_id: int, db: Session = Depends(get_db)):
    row = db.get(Analysis, report_id)
    if not row: raise HTTPException(404, "Report not found")
    buf = io.BytesIO(); pdf = canvas.Canvas(buf, pagesize=letter); width, height = letter; y = height - 50
    def line(s, indent=0):
        nonlocal y
        for part in [s[i:i+95] for i in range(0, len(s), 95)] or [""]:
            if y < 55: pdf.showPage(); y = height - 50
            pdf.drawString(48 + indent, y, part.encode("latin-1", "replace").decode("latin-1")); y -= 15
    line("SemanticGuard AI - Similarity Review Report"); y -= 8
    line(f"Documents: {row.document_a.filename} vs {row.document_b.filename}"); line(f"Overall similarity: {row.similarity:.1%}"); line("Similarity is a review signal, not proof of plagiarism."); y -= 10
    for i, m in enumerate(json.loads(row.matches_json), 1):
        line(f"Match {i} - {m['similarity']:.1%}"); line("Document A: " + m["passage_a"], 12); line("Document B: " + m["passage_b"], 12); y -= 8
    audit = json.loads(row.audit_json) if row.audit_json else {}
    for i, m in enumerate(audit.get("hybrid_matches", []), 1):
        line(f"Hybrid signal {i}: {m.get('hybrid_score', 0):.1%}; semantic {m.get('semantic_similarity', 0):.1%}; lexical {m.get('lexical_similarity', 0):.1%}; phrase {m.get('phrase_overlap', 0):.1%}; partial-copy signal {m.get('partial_copy_signal', 0):.1%}")
        line(f"Document A extracted-text range: [{m['source_a']['start_char']}, {m['source_a']['end_char']})")
        line(f"Document B extracted-text range: [{m['source_b']['start_char']}, {m['source_b']['end_char']})")
    excluded = audit.get("excluded_passages", {})
    for document_key, title in (("document_a", row.document_a.filename), ("document_b", row.document_b.filename)):
        for passage in excluded.get(document_key, []):
            line(f"Excluded audit passage from {title}: {passage['exclusion_reason']} [{passage['start_char']}, {passage['end_char']})")
            line(passage["text"], 12)
    pdf.save(); buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=semanticguard-report-{report_id}.pdf"})
