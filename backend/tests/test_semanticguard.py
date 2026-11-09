import io
import numpy as np
import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database.database import Base, get_db
from app.database.models import Document
from app.main import app
from app.services.embedding import encode
from app.services.similarity import cosine_matrix, document_score
from app.services.reranking import rerank as rerank_pairs
from app.services.vector_search import search


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def make_docx(text: str) -> bytes:
    doc = DocxDocument()
    for paragraph in text.split("\n"):
        doc.add_paragraph(paragraph)
    output = io.BytesIO()
    doc.save(output)
    return output.getvalue()


def make_pdf(text: str) -> bytes:
    output = io.BytesIO()
    pdf = canvas.Canvas(output)
    pdf.drawString(72, 720, text)
    pdf.save()
    return output.getvalue()


def upload(client, filename: str, payload: bytes):
    return client.post(
        "/api/documents",
        files={"file": (filename, payload, "application/octet-stream")},
    )


def test_health_and_missing_upload(client):
    assert client.get("/api/health").json()["status"] == "ok"
    response = client.post("/api/documents")
    assert response.status_code == 422


def test_pdf_and_docx_upload_extract_and_list(client):
    pdf = upload(client, "assignment.pdf", make_pdf("A sufficiently long PDF sentence for reliable text extraction from the generated sample document."))
    docx = upload(client, "assignment.docx", make_docx("A sufficiently long Word sentence for reliable extraction from this generated assignment document."))
    assert pdf.status_code == 200, pdf.text
    assert docx.status_code == 200, docx.text
    assert pdf.json()["file_type"] == "pdf"
    assert docx.json()["file_type"] == "docx"
    assert len(client.get("/api/documents").json()) == 2


def test_pdf_upload_to_docx_comparison_offsets_and_downloadable_report(client, monkeypatch):
    sentence = "Climate researchers compare long term temperature records to understand how regional patterns change over many decades."
    pdf_response = upload(client, "climate.pdf", make_pdf(sentence))
    docx_response = upload(client, "climate.docx", make_docx(sentence))
    assert pdf_response.status_code == 200, pdf_response.text
    assert docx_response.status_code == 200, docx_response.text
    monkeypatch.setattr("app.routes.analysis.encode", lambda texts: np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)))
    monkeypatch.setattr("app.routes.analysis.rerank", lambda pairs: [0.9] * len(pairs))

    compared = client.post("/api/analysis/compare", json={"document_a_id": pdf_response.json()["id"], "document_b_id": docx_response.json()["id"]})
    assert compared.status_code == 200, compared.text
    payload = compared.json()
    assert payload["matches"] and payload["hybrid_matches"]
    match = payload["hybrid_matches"][0]
    assert match["source_a"]["end_char"] > match["source_a"]["start_char"]
    assert match["source_b"]["end_char"] > match["source_b"]["start_char"]
    history = client.get(f"/api/reports/{payload['id']}").json()
    assert history["hybrid_matches"][0]["source_a"] == match["source_a"]
    report_pdf = client.get(f"/api/reports/{payload['id']}/download")
    assert report_pdf.status_code == 200
    assert report_pdf.content.startswith(b"%PDF")


def test_invalid_pdf_docx_and_empty_extracted_document(client):
    bad_pdf = upload(client, "bad.pdf", b"not a PDF")
    bad_docx = upload(client, "bad.docx", b"not a DOCX zip")
    empty_docx = upload(client, "empty.docx", make_docx(""))
    assert bad_pdf.status_code == 422
    assert bad_docx.status_code == 422
    assert empty_docx.status_code == 422


def test_rejects_unsupported_file_type(client):
    response = upload(client, "notes.txt", b"some plain text that is long enough for extraction")
    assert response.status_code == 415


def test_similarity_score_boundaries():
    assert document_score(np.array([[0.25, 0.5], [0.3, 0.4]])) == pytest.approx(0.425)
    assert document_score(np.array([[1.0]])) == pytest.approx(1.0)
    assert document_score(np.array([[0.0]])) == pytest.approx(0.0)
    assert document_score(np.empty((0, 0))) == 0.0


def test_faiss_retrieves_top_cosine_passages():
    corpus = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    queries = np.array([[0.99, 0.01]], dtype=np.float32)
    scores, ids = search(queries, corpus, top_k=2)
    assert ids[0, 0] == 0
    assert scores[0, 0] > scores[0, 1]


def test_model_generates_normalized_sentence_embeddings():
    vectors = encode(["Semantic search compares meaning, not only matching words."])
    assert vectors.shape == (1, 384)
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-4)


def test_identical_paraphrased_and_unrelated_assignment_scores():
    original = "A dog is running through a grassy field on a sunny day."
    paraphrase = "A canine sprints across the green lawn beneath the sunshine."
    unrelated = "The chemical composition of water is two hydrogen atoms and one oxygen atom."
    vectors = encode([original, paraphrase, unrelated])
    matrix = cosine_matrix(vectors, vectors)
    identical = float(matrix[0, 0])
    paraphrased = float(matrix[0, 1])
    unrelated_score = float(matrix[0, 2])
    assert identical == pytest.approx(1.0, abs=1e-4)
    assert paraphrased > unrelated_score
    assert paraphrased <= 1.0 and unrelated_score >= -1.0


def test_compare_endpoint_persists_analysis_history_and_pdf_download(client, monkeypatch):
    first = upload(client, "first.docx", make_docx("This paragraph is sufficiently long to be indexed in the semantic comparison."))
    second = upload(client, "second.docx", make_docx("This paragraph is sufficiently long to be indexed in the semantic comparison."))
    assert first.status_code == 200 and second.status_code == 200

    # Isolate endpoint behavior from model/network variation; model-backed semantic behavior
    # is covered separately above.
    text_to_vector = {
        "This paragraph is sufficiently long to be indexed in the semantic comparison.": np.array([1.0, 0.0], dtype=np.float32)
    }
    monkeypatch.setattr("app.routes.analysis.encode", lambda texts: np.stack([text_to_vector[t] for t in texts]))
    monkeypatch.setattr("app.routes.analysis.rerank", lambda pairs: [0.91] * len(pairs))
    response = client.post("/api/analysis/compare", json={"document_a_id": first.json()["id"], "document_b_id": second.json()["id"]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["similarity"] == pytest.approx(1.0)
    assert result["match_count"] >= 1
    assert result["reranker_used"] is True
    assert result["matches"][0]["rerank_score"] == pytest.approx(0.91)
    assert result["matches"][0]["similarity"] == pytest.approx(1.0)

    history = client.get("/api/reports")
    assert history.status_code == 200
    assert len(history.json()) == 1
    detail = client.get(f"/api/reports/{result['id']}")
    assert detail.status_code == 200
    assert "hybrid_matches" in detail.json()
    assert "excluded_passages" in detail.json()
    hybrid_match = detail.json()["hybrid_matches"][0]
    assert hybrid_match["source_a"]["start_char"] < hybrid_match["source_a"]["end_char"]
    download = client.get(f"/api/reports/{result['id']}/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/pdf"
    assert download.content.startswith(b"%PDF")


def test_analysis_api_error_handling(client):
    response = client.post("/api/analysis/compare", json={"document_a_id": 404, "document_b_id": 405})
    assert response.status_code == 404
    same = client.post("/api/analysis/compare", json={"document_a_id": 1, "document_b_id": 1})
    assert same.status_code == 400
    missing_report = client.get("/api/reports/999")
    assert missing_report.status_code == 404


def test_empty_document_analysis_returns_client_error(client):
    # Insert empty extracted text directly to exercise the comparison guard.
    db = next(app.dependency_overrides[get_db]())
    a = Document(filename="empty-a.docx", stored_name="empty-a", file_type="docx", text="")
    b = Document(filename="empty-b.docx", stored_name="empty-b", file_type="docx", text="")
    db.add_all([a, b])
    db.commit()
    ids = (a.id, b.id)
    db.close()
    response = client.post("/api/analysis/compare", json={"document_a_id": ids[0], "document_b_id": ids[1]})
    assert response.status_code == 422


def test_uploaded_content_is_not_logged(client, caplog):
    marker = "PRIVATE_STUDENT_CONTENT_8472"
    response = upload(client, "private.docx", make_docx(marker + " plus enough extra words to pass the text length check."))
    assert response.status_code == 200
    assert marker not in caplog.text


def test_rerank_service_returns_one_score_per_candidate(monkeypatch):
    class FakeCrossEncoder:
        def predict(self, pairs, show_progress_bar=False):
            assert len(pairs) == 2
            return np.array([-2.0, 2.0])

    monkeypatch.setattr("app.services.reranking.get_reranker", lambda: FakeCrossEncoder())
    assert rerank_pairs([("query", "candidate one"), ("query", "candidate two")]) == pytest.approx([1 / (1 + np.exp(2)), 1 / (1 + np.exp(-2))])


def test_compare_returns_candidates_in_cross_encoder_order(client, monkeypatch):
    db = next(app.dependency_overrides[get_db]())
    source = "alpha " * 140 + "\n" + "beta " * 140
    target = "gamma " * 140 + "\n" + "delta " * 140
    a = Document(filename="source.docx", stored_name="source", file_type="docx", text=source)
    b = Document(filename="target.docx", stored_name="target", file_type="docx", text=target)
    db.add_all([a, b])
    db.commit()
    ids = (a.id, b.id)
    db.close()

    monkeypatch.setattr(settings, "top_k", 2)
    monkeypatch.setattr(settings, "min_similarity", 0.45)
    def fake_encode(texts):
        vectors = []
        for text in texts:
            if text.startswith("alpha") or text.startswith("gamma"):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.8, 0.6] if text.startswith("beta") else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)
    monkeypatch.setattr("app.routes.analysis.encode", fake_encode)
    monkeypatch.setattr(
        "app.routes.analysis.rerank",
        lambda pairs: [0.95 if pair[1].startswith("delta") else 0.05 for pair in pairs],
    )

    response = client.post("/api/analysis/compare", json={"document_a_id": ids[0], "document_b_id": ids[1]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["reranker_used"] is True
    match = next(item for item in result["matches"] if item["passage_a"].startswith("alpha"))
    assert match["passage_b"].startswith("delta")
    assert match["similarity"] == pytest.approx(0.0)
    assert match["rerank_score"] == pytest.approx(0.95)
    assert match["candidates"][0]["passage"].startswith("delta")


def test_compare_falls_back_to_faiss_when_reranker_is_unavailable(client, monkeypatch):
    db = next(app.dependency_overrides[get_db]())
    a = Document(filename="fallback-a.docx", stored_name="fallback-a", file_type="docx", text="first passage with enough words for testing semantic fallback ordering.")
    b = Document(filename="fallback-b.docx", stored_name="fallback-b", file_type="docx", text="second passage with enough words for testing semantic fallback ordering.")
    db.add_all([a, b])
    db.commit()
    ids = (a.id, b.id)
    db.close()
    monkeypatch.setattr("app.routes.analysis.encode", lambda texts: np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)))
    def unavailable(_pairs):
        raise OSError("model weights are not cached")
    monkeypatch.setattr("app.routes.analysis.rerank", unavailable)

    response = client.post("/api/analysis/compare", json={"document_a_id": ids[0], "document_b_id": ids[1]})
    assert response.status_code == 200
    result = response.json()
    assert result["reranker_used"] is False
    assert result["reranker_model"] is None
    assert "embedding similarity" in result["reranker_notice"]
    assert all(match["rerank_score"] is None for match in result["matches"])
