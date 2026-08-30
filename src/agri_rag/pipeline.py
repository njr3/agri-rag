from __future__ import annotations

from dataclasses import dataclass

from .evaluation import FaithfulnessResult, faithfulness_score
from .generation import ExtractiveGenerator, GeneratedAnswer
from .retrieval import RetrievedChunk


@dataclass
class PipelineResult:
    query: str
    retrieved: list[RetrievedChunk]
    answer: GeneratedAnswer
    faithfulness: FaithfulnessResult


class RAGPipeline:
    def __init__(self, retriever, generator=None, top_k: int = 3):
        self.retriever = retriever
        self.generator = generator or ExtractiveGenerator()
        self.top_k = top_k

    def answer(self, query: str) -> PipelineResult:
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        generated = self.generator.generate(query, retrieved)
        faith = faithfulness_score(generated, retrieved)
        return PipelineResult(query, retrieved, generated, faith)
