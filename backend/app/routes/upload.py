import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.config import settings
from app.database.database import get_db
from app.database.models import Document
from app.services.pdf_extractor import extract_pdf
from app.services.docx_extractor import extract_docx
router = APIRouter(prefix="/api/documents", tags=["documents"])
@router.post("")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".pdf", ".docx"}: raise HTTPException(415, "Upload a PDF or DOCX file")
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if not data: raise HTTPException(400, "The uploaded file is empty")
    if len(data) > settings.max_upload_mb * 1024 * 1024: raise HTTPException(413, f"Maximum file size is {settings.max_upload_mb} MB")
    stored = f"{uuid.uuid4().hex}{ext}"; path = settings.upload_dir / stored
    path.write_bytes(data)
    try: text = extract_pdf(str(path)) if ext == ".pdf" else extract_docx(str(path))
    except Exception as exc:
        path.unlink(missing_ok=True); raise HTTPException(422, f"Could not read document: {exc}")
    if len(text) < 50:
        path.unlink(missing_ok=True); raise HTTPException(422, "No usable text found in the uploaded document. Scanned PDFs may need OCR, which is not enabled.")
    doc = Document(filename=Path(file.filename or "document").name, stored_name=stored, file_type=ext[1:], text=text)
    db.add(doc); db.commit(); db.refresh(doc)
    return {"id": doc.id, "filename": doc.filename, "file_type": doc.file_type, "characters": len(text), "uploaded_at": doc.uploaded_at.isoformat()}
@router.get("")
def list_documents(db: Session = Depends(get_db)):
    return [{"id": d.id, "filename": d.filename, "file_type": d.file_type, "characters": len(d.text), "uploaded_at": d.uploaded_at.isoformat()} for d in db.query(Document).order_by(Document.uploaded_at.desc()).all()]
@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc: raise HTTPException(404, "Document not found")
    (settings.upload_dir / doc.stored_name).unlink(missing_ok=True); db.delete(doc); db.commit()
