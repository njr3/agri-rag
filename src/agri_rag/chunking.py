"""
Chunking for the demo corpus.

The seed documents here are short enough (one topic, a few sentences)
that each document is really already one chunk. The function below still
does real sentence-aware splitting with an overlap window, because any
corpus built from actual extension manuals or FAO guides won't be this
tidy, and I'd rather the chunker be exercised now than rewritten later
against a bigger corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    lang: str
    title: str


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def chunk_document(
    doc: dict, max_sentences: int = 4, overlap_sentences: int = 1
) -> list[Chunk]:
    """Sliding window over sentences, not characters, so a chunk never cuts
    a sentence in half — that matters more for the faithfulness check
    later than it does for retrieval."""
    sentences = split_sentences(doc["text"])
    if not sentences:
        return []

    chunks = []
    start = 0
    idx = 0
    step = max(1, max_sentences - overlap_sentences)
    while start < len(sentences):
        window = sentences[start : start + max_sentences]
        chunks.append(
            Chunk(
                doc_id=doc["id"],
                chunk_id=f"{doc['id']}::{idx}",
                text=" ".join(window),
                lang=doc.get("lang", "en"),
                title=doc.get("title", ""),
            )
        )
        idx += 1
        start += step
    return chunks


def chunk_corpus(docs: list[dict], **kwargs) -> list[Chunk]:
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, **kwargs))
    return chunks
