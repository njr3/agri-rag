"""
Two kinds of evaluation, kept separate on purpose:

Retrieval metrics (precision@k, recall@k, MRR) measure whether the right
chunk got surfaced at all. They need a gold relevance label and don't
care what the generator does with the chunk afterward.

Faithfulness measures something different: given what was retrieved,
does the generated answer actually stick to it? A generator can retrieve
perfectly and still hallucinate, or retrieve badly and still produce a
faithful — if unhelpful — answer that correctly says "I don't know."
Conflating the two into one score would hide which stage is actually
failing when something goes wrong.

The faithfulness scorer here works by decomposing the answer into
sentences and checking each one's TF-IDF cosine similarity against the
concatenated retrieved context, above a fixed threshold. This is a
coarse proxy for entailment, not real natural language inference — a
sentence can score as "supported" by lexical overlap while subtly
misstating what the source says. That's a genuine limitation, not a
detail I'm glossing over, and it's called out in the README rather than
presented as a solved problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import split_sentences
from .generation import GeneratedAnswer
from .retrieval import RetrievedChunk


def precision_at_k(retrieved_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if _doc_of(cid) in relevant_doc_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    top_k_docs = {_doc_of(cid) for cid in retrieved_ids[:k]}
    hits = len(top_k_docs & relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_doc_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if _doc_of(cid) in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def _doc_of(chunk_id: str) -> str:
    return chunk_id.split("::")[0]


@dataclass
class FaithfulnessResult:
    score: float
    supported_sentences: int
    total_sentences: int
    unsupported: list[str]


def faithfulness_score(
    answer: GeneratedAnswer, retrieved: list[RetrievedChunk], threshold: float = 0.15
) -> FaithfulnessResult:
    sentences = split_sentences(answer.text)
    if not sentences:
        return FaithfulnessResult(1.0, 0, 0, [])

    context_texts = [r.chunk.text for r in retrieved]
    if not context_texts:
        return FaithfulnessResult(0.0, 0, len(sentences), sentences)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    corpus = context_texts + sentences
    matrix = vectorizer.fit_transform(corpus)
    context_matrix = matrix[: len(context_texts)]
    sentence_matrix = matrix[len(context_texts) :]

    sims = cosine_similarity(sentence_matrix, context_matrix)
    max_sim_per_sentence = sims.max(axis=1)

    supported = max_sim_per_sentence >= threshold
    unsupported = [s for s, ok in zip(sentences, supported) if not ok]

    return FaithfulnessResult(
        score=float(supported.sum() / len(sentences)),
        supported_sentences=int(supported.sum()),
        total_sentences=len(sentences),
        unsupported=unsupported,
    )
