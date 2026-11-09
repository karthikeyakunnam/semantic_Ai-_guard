import pymupdf as fitz
def extract_pdf(path: str) -> str:
    with fitz.open(path) as pdf:
        return "\n".join(page.get_text("text") for page in pdf)
