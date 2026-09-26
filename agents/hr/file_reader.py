"""
Utility: Extract plain text from uploaded PDF, DOCX, or TXT files.
"""

import io
from pypdf import PdfReader


def extract_text_from_upload(uploaded_file) -> str:
    """
    Accepts a Streamlit UploadedFile.
    Returns plain text.
    Supports: .pdf, .docx, .txt
    """
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()

    try:
        if name.endswith(".pdf"):
            return _read_pdf(uploaded_file)
        elif name.endswith(".docx"):
            return _read_docx(uploaded_file)
        elif name.endswith(".txt"):
            return _read_txt(uploaded_file)
        else:
            return f"(Unsupported file type: {name})"
    except Exception as e:
        return f"(Error reading {name}: {e})"


def _read_pdf(file) -> str:
    data = file.read()
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages).strip()


def _read_docx(file) -> str:
    from docx import Document
    data = file.read()
    doc = Document(io.BytesIO(data))
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(lines).strip()


def _read_txt(file) -> str:
    data = file.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")