from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "indexes"

CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.json"
INDEX_PATH = INDEX_DIR / "ncap.index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 75


@dataclass
class Chunk:
    chunk_id: int
    source_file: str
    page_number: int
    text: str


def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)  # join hyphenated line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text page by page from a PDF."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = clean_text(raw_text)

        if text:
            pages.append((page_idx, text))

    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_chunks() -> list[Chunk]:
    """Read all PDFs from data/raw and convert them into chunks."""
    pdf_paths = sorted(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {RAW_DATA_DIR}. "
            "Add some PDF files to data/raw/ first."
        )

    chunks: list[Chunk] = []
    chunk_id = 0

    for pdf_path in tqdm(pdf_paths, desc="Reading PDFs"):
        pages = extract_pdf_pages(pdf_path)

        for page_number, page_text in pages:
            page_chunks = chunk_text(
                page_text,
                chunk_size=CHUNK_SIZE_WORDS,
                overlap=CHUNK_OVERLAP_WORDS,
            )

            for text in page_chunks:
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source_file=pdf_path.name,
                        page_number=page_number,
                        text=text,
                    )
                )
                chunk_id += 1

    return chunks


def save_chunks(chunks: list[Chunk]) -> None:
    """Save chunk metadata and text to JSON."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump([asdict(chunk) for chunk in chunks], f, indent=2, ensure_ascii=False)


def build_faiss_index(chunks: list[Chunk]) -> None:
    """Embed chunks locally and save a FAISS index."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk.text for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    # Inner product over normalized embeddings is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))


def main() -> None:
    print("Building chunks...")
    chunks = build_chunks()
    print(f"Created {len(chunks)} chunks.")

    print(f"Saving chunks to {CHUNKS_PATH}...")
    save_chunks(chunks)

    print("Building FAISS index...")
    build_faiss_index(chunks)

    print(f"Saved FAISS index to {INDEX_PATH}.")
    print("Done.")


if __name__ == "__main__":
    main()