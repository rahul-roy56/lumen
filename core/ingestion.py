"""Document loading, text extraction, chunking, and metadata management."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from core.config import CHUNK_SIZE, CHUNK_OVERLAP, CHARS_PER_TOKEN

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata attached to every text chunk."""

    filename: str
    page_number: int | None  # None for non-paged formats
    chunk_index: int
    total_chunks: int = 0


@dataclass
class DocumentChunk:
    """A single text chunk with its metadata."""

    text: str
    metadata: ChunkMetadata


@dataclass
class IngestionResult:
    """Result of ingesting a single document."""

    filename: str
    num_chunks: int
    chunks: list[DocumentChunk] = field(default_factory=list)
    error: str | None = None


# ── Text Extraction ────────────────────────────────────────

def _extract_pdf(file_bytes: bytes, filename: str) -> list[tuple[str, int]]:
    """Extract text from PDF, returning list of (text, page_number) tuples."""
    pages: list[tuple[str, int]] = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages.append((text.strip(), page_num + 1))
            else:
                # OCR fallback for scanned pages
                try:
                    from PIL import Image
                    import pytesseract
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        pages.append((ocr_text.strip(), page_num + 1))
                except ImportError:
                    logger.warning("pytesseract not available — skipping OCR for %s page %d", filename, page_num + 1)
                except Exception as e:
                    logger.warning("OCR failed for %s page %d: %s", filename, page_num + 1, e)
        doc.close()
    except Exception as e:
        raise RuntimeError(f"Failed to extract PDF '{filename}': {e}") from e
    return pages


def _extract_docx(file_bytes: bytes, filename: str) -> list[tuple[str, int]]:
    """Extract text from DOCX, returning list of (text, page_number=None) tuples."""
    try:
        doc = DocxDocument(io.BytesIO(file_bytes))
        full_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        if full_text.strip():
            return [(full_text.strip(), None)]
        return []
    except Exception as e:
        raise RuntimeError(f"Failed to extract DOCX '{filename}': {e}") from e


def _extract_txt(file_bytes: bytes, filename: str) -> list[tuple[str, int]]:
    """Extract text from TXT file."""
    try:
        text = file_bytes.decode("utf-8", errors="replace")
        if text.strip():
            return [(text.strip(), None)]
        return []
    except Exception as e:
        raise RuntimeError(f"Failed to read TXT '{filename}': {e}") from e


EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
}


# ── Chunking ───────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preserving paragraph and sentence boundaries."""
    char_chunk = chunk_size * CHARS_PER_TOKEN
    char_overlap = overlap * CHARS_PER_TOKEN

    if len(text) <= char_chunk:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + char_chunk
        chunk = text[start:end]

        # Try to break on boundaries (best to worst): paragraph > sentence > word
        if end < len(text):
            # 1. Try double newline (paragraph boundary)
            para_break = chunk.rfind("\n\n")
            # 2. Try single newline
            line_break = chunk.rfind("\n")
            # 3. Try sentence boundary
            sent_break = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("! "))

            # Pick the best break point that's past 40% of the chunk
            min_break = int(char_chunk * 0.4)
            if para_break > min_break:
                break_point = para_break
            elif sent_break > min_break:
                break_point = sent_break + 1  # include the period
            elif line_break > min_break:
                break_point = line_break
            else:
                break_point = -1

            if break_point > 0:
                chunk = text[start : start + break_point + 1]
                end = start + break_point + 1

        stripped = chunk.strip()
        if stripped:
            chunks.append(stripped)
        start = end - char_overlap
        if start >= len(text):
            break

    return [c for c in chunks if c]


# ── Public API ─────────────────────────────────────────────

def ingest_file(uploaded_file: BinaryIO, filename: str) -> IngestionResult:
    """Ingest a single uploaded file: extract text, chunk, and attach metadata."""
    suffix = Path(filename).suffix.lower()
    extractor = EXTRACTORS.get(suffix)

    if extractor is None:
        return IngestionResult(filename=filename, num_chunks=0, error=f"Unsupported file type: {suffix}")

    try:
        file_bytes = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
        pages = extractor(file_bytes, filename)
    except RuntimeError as e:
        return IngestionResult(filename=filename, num_chunks=0, error=str(e))

    all_chunks: list[DocumentChunk] = []
    global_idx = 0

    for text, page_num in pages:
        text_chunks = _chunk_text(text)
        for chunk_text in text_chunks:
            meta = ChunkMetadata(
                filename=filename,
                page_number=page_num,
                chunk_index=global_idx,
            )
            all_chunks.append(DocumentChunk(text=chunk_text, metadata=meta))
            global_idx += 1

    # Back-fill total_chunks
    for chunk in all_chunks:
        chunk.metadata.total_chunks = len(all_chunks)

    return IngestionResult(filename=filename, num_chunks=len(all_chunks), chunks=all_chunks)


def ingest_multiple(files: list[tuple[BinaryIO, str]]) -> list[IngestionResult]:
    """Ingest multiple files and return all results."""
    results: list[IngestionResult] = []
    for file_obj, filename in files:
        result = ingest_file(file_obj, filename)
        results.append(result)
    return results
