"""
utils/pdf_parser.py — Extract plain text from PDF or text resume uploads.
"""

from __future__ import annotations

import io
from pathlib import Path


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF byte stream using pdfplumber."""
    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n\n".join(text_parts).strip()


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Route to the correct extractor based on file extension.

    Supported: .pdf, .txt, .md
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)

    if ext in (".txt", ".md", ""):
        # Try UTF-8, fall back to latin-1
        try:
            return file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1").strip()

    raise ValueError(f"Unsupported file type: {ext!r}. Please upload a PDF or TXT file.")
