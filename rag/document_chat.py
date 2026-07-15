"""Document RAG — chat over uploaded PDFs.

PDFs are chunked, embedded with a sentence-transformer, indexed in FAISS, and
retrieved to ground Gemini's answers. Both heavy dependencies (sentence-
transformers, faiss) are imported lazily so the rest of the app runs when they
are absent; the UI checks :func:`is_available` and explains what to install.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.config import settings
from utils import llm
from utils.helpers import AutoBusinessError
from utils.logger import get_logger

logger = get_logger(__name__)

_INDEX_FILE = settings.paths.vector_store / "docs.faiss"
_META_FILE = settings.paths.vector_store / "docs_meta.pkl"

_embedder: Any = None


def is_available() -> bool:
    """True when both embedder and FAISS can be imported."""
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class Chunk:
    text: str
    source: str
    page: int


@dataclass
class SearchHit:
    chunk: Chunk
    score: float

    @property
    def citation(self) -> str:
        return f"{self.chunk.source} (p.{self.chunk.page})"


@dataclass
class ChatTurn:
    question: str
    answer: str
    hits: list[SearchHit] = field(default_factory=list)
    grounded: bool = True

    @property
    def citations(self) -> list[str]:
        return sorted({h.citation for h in self.hits})


def _get_embedder() -> Any:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(settings.rag.embedding_model)
        logger.info("Loaded embedder %s", settings.rag.embedding_model)
    return _embedder


def _extract_pdf(file: Any, name: str) -> list[Chunk]:
    from pypdf import PdfReader

    reader = PdfReader(file)
    chunks: list[Chunk] = []
    size, overlap = settings.rag.chunk_size, settings.rag.chunk_overlap
    for pageno, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            piece = text[start:start + size]
            if piece.strip():
                chunks.append(Chunk(piece.strip(), name, pageno))
            start += size - overlap
    return chunks


def index_documents(files: list[Any], replace: bool = False) -> dict[str, Any]:
    """Ingest PDFs into the FAISS index."""
    if not is_available():
        raise AutoBusinessError(
            "Document chat needs sentence-transformers and faiss-cpu. "
            "Install them: pip install sentence-transformers faiss-cpu"
        )
    import faiss

    all_chunks: list[Chunk] = []
    failures: list[str] = []
    for f in files:
        name = getattr(f, "name", "document.pdf")
        try:
            all_chunks.extend(_extract_pdf(f, name))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")

    if not all_chunks:
        raise AutoBusinessError("No extractable text found in the uploaded PDFs.")

    embedder = _get_embedder()
    vectors = embedder.encode([c.text for c in all_chunks], show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)

    existing_meta: list[Chunk] = []
    if not replace and _INDEX_FILE.exists() and _META_FILE.exists():
        index = faiss.read_index(str(_INDEX_FILE))
        with open(_META_FILE, "rb") as fh:
            existing_meta = pickle.load(fh)
    else:
        index = faiss.IndexFlatIP(vectors.shape[1])

    index.add(vectors)
    meta = existing_meta + all_chunks

    faiss.write_index(index, str(_INDEX_FILE))
    with open(_META_FILE, "wb") as fh:
        pickle.dump(meta, fh)

    logger.info("Indexed %d chunks (%d total)", len(all_chunks), len(meta))
    return {"documents": len(files), "chunks": len(all_chunks),
            "total_chunks": len(meta), "failures": failures}


def _search(question: str, k: int) -> list[SearchHit]:
    import faiss

    if not (_INDEX_FILE.exists() and _META_FILE.exists()):
        return []
    index = faiss.read_index(str(_INDEX_FILE))
    with open(_META_FILE, "rb") as fh:
        meta: list[Chunk] = pickle.load(fh)

    embedder = _get_embedder()
    q = np.asarray(embedder.encode([question]), dtype="float32")
    faiss.normalize_L2(q)
    scores, idx = index.search(q, min(k, len(meta)))

    hits: list[SearchHit] = []
    for score, i in zip(scores[0], idx[0]):
        if 0 <= i < len(meta):
            hits.append(SearchHit(meta[i], float(score)))
    return hits


def ask(question: str, history: list[ChatTurn] | None = None, k: int | None = None) -> ChatTurn:
    """Answer a question grounded in the indexed documents."""
    if not is_available():
        raise AutoBusinessError("Document chat is unavailable — embedder not installed.")
    if not llm.is_available():
        raise AutoBusinessError("Document chat needs a Gemini API key (add it in Settings).")

    k = k or settings.rag.top_k
    hits = _search(question, k)
    if not hits:
        return ChatTurn(question, "No documents are indexed yet. Upload PDFs first.", [], grounded=False)

    context = "\n\n".join(f"[{h.citation}]\n{h.chunk.text}" for h in hits)
    prompt = (
        f"Answer the question using ONLY the context below. If the answer isn't "
        f"there, say so.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    answer = llm.generate(prompt, temperature=0.2)
    return ChatTurn(question, answer, hits, grounded=True)


__all__ = ["is_available", "index_documents", "ask", "ChatTurn", "SearchHit", "Chunk"]
