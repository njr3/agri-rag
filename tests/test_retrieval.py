import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agri_rag.chunking import Chunk, chunk_document, split_sentences
from agri_rag.retrieval import BM25Retriever, HybridRetriever, LSARetriever

TOY_DOCS = [
    {"id": "a", "lang": "en", "title": "Irrigation timing", "text": "Water tomatoes early in the morning. Avoid wetting the leaves to reduce fungal disease."},
    {"id": "b", "lang": "en", "title": "Pest control", "text": "Aphids can be washed off with a strong jet of water. Ladybird beetles are a natural predator of aphids."},
    {"id": "c", "lang": "en", "title": "Soil pH", "text": "Most vegetables grow best in slightly acidic soil, around pH 6 to 6.5. Test soil before adding lime."},
]


def test_split_sentences_basic():
    sentences = split_sentences("First one. Second one! Third one?")
    assert sentences == ["First one.", "Second one!", "Third one?"]


def test_chunk_document_respects_window():
    doc = {"id": "x", "text": "S1. S2. S3. S4. S5.", "lang": "en", "title": "t"}
    chunks = chunk_document(doc, max_sentences=2, overlap_sentences=0)
    assert len(chunks) == 3
    assert chunks[0].text == "S1. S2."
    assert chunks[0].doc_id == "x"


def test_bm25_retrieves_lexically_matching_doc():
    chunks = []
    for d in TOY_DOCS:
        chunks.extend(chunk_document(d, max_sentences=5))
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("how do I control aphids", top_k=1)
    assert results[0].chunk.doc_id == "b"


def test_hybrid_returns_requested_top_k():
    chunks = []
    for d in TOY_DOCS:
        chunks.extend(chunk_document(d, max_sentences=5))
    retriever = HybridRetriever(chunks)
    results = retriever.retrieve("soil acidity for vegetables", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.doc_id == "c"


def test_lsa_retriever_runs_on_tiny_corpus():
    # regression check: small corpora shouldn't crash TruncatedSVD by
    # requesting more components than there are chunks
    chunks = [Chunk("only", "only::0", "a single short chunk of text", "en", "t")]
    retriever = LSARetriever(chunks, n_components=50)
    results = retriever.retrieve("anything", top_k=1)
    assert results[0].chunk.doc_id == "only"
