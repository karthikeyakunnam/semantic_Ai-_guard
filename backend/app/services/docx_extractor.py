from docx import Document
def extract_docx(path: str) -> str:
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        parts.extend(cell.text for row in table.rows for cell in row.cells if cell.text.strip())
    return "\n".join(parts)
