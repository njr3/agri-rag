"""
Three retrievers, compared head to head in eval/run_eval.py:

BM25Retriever
    Sparse lexical matching. Rank-BM25's Okapi variant. Strong when the
    query shares vocabulary with the source text, weak when a farmer
    phrases a question with none of the words the document uses.

LSARetriever
    A lightweight "dense" retriever: TF-IDF followed by truncated SVD
    (latent semantic analysis) to get a low-dimensional embedding, then
    cosine similarity. I used this instead of a transformer embedding
    model on purpose — see the README for why, but the short version is
    that off-the-shelf multilingual sentence embeddings are trained
    almost entirely on high-resource languages and their quality on
    something like Wolof is unverified at best, so treating them as a
    drop-in solution for a low-resource-language deployment would be
    the wrong takeaway from this project. LSA is a weaker semantic
    model, but it's a known quantity, works the same regardless of
    language, and doesn't require a multi-hundred-megabyte download to
    reproduce the eval.

HybridRetriever
    Reciprocal rank fusion of the two rankings above. Doesn't need score
    normalization between BM25 and cosine similarity, which are on
    different scales and not directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(self.chunks[i], float(scores[i]), rank)
            for rank, i in enumerate(order, start=1)
        ]


class LSARetriever:
    name = "lsa_dense"

    def __init__(self, chunks: list[Chunk], n_components: int = 50):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        n_components = min(n_components, max(2, len(texts) - 1))
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf = self._vectorizer.fit_transform(texts)
        self._svd = TruncatedSVD(n_components=n_components, random_state=0)
        self._doc_vectors = self._svd.fit_transform(tfidf)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        q_tfidf = self._vectorizer.transform([query])
        q_vec = self._svd.transform(q_tfidf)
        sims = cosine_similarity(q_vec, self._doc_vectors)[0]
        order = np.argsort(sims)[::-1][:top_k]
        return [
            RetrievedChunk(self.chunks[i], float(sims[i]), rank)
            for rank, i in enumerate(order, start=1)
        ]


class HybridRetriever:
    name = "hybrid_rrf"

    def __init__(self, chunks: list[Chunk], k_rrf: int = 60):
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.lsa = LSARetriever(chunks)
        self.k_rrf = k_rrf

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        # pull a wider candidate pool from each retriever before fusing,
        # so a chunk that's rank 8 in one list but rank 1 in the other
        # still gets a chance to surface.
        pool_k = max(top_k * 4, 10)
        bm25_results = self.bm25.retrieve(query, top_k=pool_k)
        lsa_results = self.lsa.retrieve(query, top_k=pool_k)

        rrf_scores: dict[str, float] = {}
        chunk_lookup = {}
        for results in (bm25_results, lsa_results):
            for r in results:
                rrf_scores[r.chunk.chunk_id] = rrf_scores.get(r.chunk.chunk_id, 0.0) + 1.0 / (
                    self.k_rrf + r.rank
                )
                chunk_lookup[r.chunk.chunk_id] = r.chunk

        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(chunk_lookup[cid], score, rank)
            for rank, (cid, score) in enumerate(ranked, start=1)
        ]


RETRIEVERS = {"bm25": BM25Retriever, "lsa_dense": LSARetriever, "hybrid_rrf": HybridRetriever}
