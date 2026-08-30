"""
Turns retrieved chunks into an answer.

If ANTHROPIC_API_KEY is set, this calls Claude with the retrieved
passages as the only allowed source of information and an instruction
to answer only from them. If it isn't set — which is the default, and
what the eval harness actually runs against — it falls back to an
extractive generator that returns the single highest-scoring chunk
verbatim, with its source attached.

The extractive fallback is not a weaker version of the same thing for
demo convenience. It's a legitimate baseline: it has a faithfulness
score of 1.0 by construction, since it can't say anything the source
document didn't say, and comparing the LLM generator's faithfulness
against that baseline is exactly what makes the faithfulness metric in
eval/run_eval.py meaningful instead of just a number with no reference
point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .retrieval import RetrievedChunk


@dataclass
class GeneratedAnswer:
    text: str
    source_chunk_ids: list[str]
    method: str


class ExtractiveGenerator:
    name = "extractive_fallback"

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> GeneratedAnswer:
        if not retrieved:
            return GeneratedAnswer("No relevant information found in the corpus.", [], self.name)
        top = retrieved[0].chunk
        return GeneratedAnswer(top.text, [top.chunk_id], self.name)


class ClaudeGenerator:
    name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.client = None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=api_key)
            except Exception:
                self.client = None

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> GeneratedAnswer:
        if not self.client or not retrieved:
            return ExtractiveGenerator().generate(query, retrieved)

        context = "\n\n".join(f"[{r.chunk.chunk_id}] {r.chunk.text}" for r in retrieved)
        prompt = (
            "Answer the farmer's question using only the passages below. "
            "If the passages don't contain the answer, say so directly "
            "instead of guessing. Keep the answer to two or three sentences.\n\n"
            f"Passages:\n{context}\n\nQuestion: {query}"
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in response.content if block.type == "text")
            return GeneratedAnswer(text.strip(), [r.chunk.chunk_id for r in retrieved], self.name)
        except Exception:
            return ExtractiveGenerator().generate(query, retrieved)
