import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agri_rag.chunking import Chunk
from agri_rag.evaluation import faithfulness_score, precision_at_k, recall_at_k, reciprocal_rank
from agri_rag.generation import ExtractiveGenerator, GeneratedAnswer
from agri_rag.retrieval import RetrievedChunk


def _chunk(doc_id, text):
    c = Chunk(doc_id, f"{doc_id}::0", text, "en", "t")
    return RetrievedChunk(c, 1.0, 1)


def test_extractive_generator_is_trivially_faithful():
    retrieved = [_chunk("a", "Mulch reduces evaporation and moderates soil temperature.")]
    answer = ExtractiveGenerator().generate("does mulch help", retrieved)
    result = faithfulness_score(answer, retrieved)
    assert result.score == 1.0


def test_faithfulness_flags_unsupported_sentence():
    retrieved = [_chunk("a", "Mulch reduces evaporation and moderates soil temperature.")]
    fabricated = GeneratedAnswer(
        text="Mulch reduces evaporation. Mulch also cures groundnut rosette virus completely.",
        source_chunk_ids=["a::0"],
        method="test",
    )
    result = faithfulness_score(fabricated, retrieved)
    assert result.total_sentences == 2
    assert result.supported_sentences == 1
    assert "groundnut rosette virus" in result.unsupported[0]


def test_faithfulness_zero_with_no_context():
    fabricated = GeneratedAnswer(text="Something not grounded in anything.", source_chunk_ids=[], method="test")
    result = faithfulness_score(fabricated, [])
    assert result.score == 0.0


def test_precision_recall_reciprocal_rank():
    retrieved_ids = ["docA::0", "docB::1", "docC::0"]
    relevant = {"docB"}
    assert precision_at_k(retrieved_ids, relevant, k=3) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved_ids, relevant, k=3) == 1.0
    assert reciprocal_rank(retrieved_ids, relevant) == 0.5


def test_reciprocal_rank_zero_when_not_found():
    assert reciprocal_rank(["a::0", "b::0"], {"z"}) == 0.0
