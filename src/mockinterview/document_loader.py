from __future__ import annotations
from pathlib import Path


def load_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".txt", ".md"):
        return Path(path).read_text(encoding="utf-8")
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        import docx
        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs)
    raise ValueError(f"Unsupported file type: {ext}")
