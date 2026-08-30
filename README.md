# agri-rag

A retrieval-augmented question answering system for agricultural extension
questions, built around a question I kept running into rather than a
technology I wanted to show off: farmers and extension workers ask
questions in their own words, and a lot of the advice they need already
exists in written form somewhere, but retrieving the right passage and
then not having the model make something up on top of it are two separate
problems that most RAG demos treat as solved by default.

This project treats them as two problems, evaluates them separately, and
is honest about where the second one — not hallucinating — is genuinely
hard to measure well with the tools available here.

## What's actually being tested

Two things, kept apart deliberately:

**Retrieval quality** — does the right passage get surfaced for a given
question. Measured with precision@k, recall@k, and mean reciprocal rank
against a small labeled eval set, comparing three retrieval strategies
head to head: BM25 (sparse/lexical), an LSA-based dense retriever
(TF-IDF followed by truncated SVD, then cosine similarity), and a hybrid
of the two combined with reciprocal rank fusion.

**Faithfulness** — given what was retrieved, does the generated answer
stick to it, or does it add things the source material doesn't support.
Measured by decomposing the answer into sentences and checking each
one's similarity against the retrieved context, with an extractive
baseline (return the top passage verbatim) included specifically because
it has a faithfulness score of 1.0 by construction and gives the metric
something to be measured against.

## Why LSA instead of a transformer embedding model

This was a deliberate choice, not a shortcut. The eventual goal for a
system like this is to work for farmers asking questions in Wolof or
other West African languages that most off-the-shelf multilingual
sentence embedding models were not meaningfully evaluated on during
training. Dropping in a model like LaBSE or multilingual-e5 and calling
the low-resource-language problem solved would be the wrong lesson to
take from this project — it would be borrowing credibility from a
benchmark (usually English, French, and a handful of well-resourced
languages) that doesn't cover the actual target case.

So the retrieval architecture here is built to be embedding-agnostic: the
`LSARetriever` class is one implementation of a small interface, TF-IDF
and SVD, that works identically regardless of language and doesn't
depend on what a pretrained model happened to see during training. It's
a genuinely weaker semantic model than a good transformer embedding
would be — it can't do true paraphrase matching the way a trained
embedding can — but it's a known, honest baseline rather than an
unverified one. Swapping in a stronger multilingual embedding model, and
specifically evaluating it on Wolof-language queries against this same
corpus and eval harness, is the natural next step and is called out
below rather than glossed over.

## Demo corpus and eval set

`corpus/docs.json` is fifteen short documents on staple-crop agronomy
for the region — millet spacing, groundnut rotation, Striga control,
post-harvest storage, that kind of thing — thirteen in English, two in
French, written for this project rather than pulled from any single
source. `eval/questions.json` is sixteen questions against that corpus,
each with the correct document(s) labeled, including two French
questions and one question that legitimately requires two documents to
answer well.

This is a seed corpus for exercising the architecture, not a claim that
fifteen documents constitute real extension coverage. Swapping in a
larger corpus pulled from FAO or CGIAR extension material is exactly
what `corpus/docs.json`'s schema (id, lang, title, text) is meant to
make easy.

## Running the evaluation

```
pip install -r requirements.txt
python eval/run_eval.py
python -m pytest tests/
```

Current result on the demo corpus:

| retriever | precision@3 | recall@3 | MRR | mean faithfulness |
|---|---|---|---|---|
| bm25 | 0.333 | 0.938 | 0.833 | 1.000 |
| lsa_dense | 0.354 | 0.875 | 0.812 | 1.000 |
| hybrid_rrf | 0.333 | 0.938 | 0.802 | 1.000 |

BM25 edges out the dense retriever here, and hybrid doesn't beat BM25
alone. I want to be direct about why, rather than let the table imply
something the corpus doesn't support: the eval questions were written in
close-to-natural farmer phrasing but still share real vocabulary with
the documents they target, and on a fifteen-document corpus with that
much lexical overlap, sparse retrieval doesn't have much room to lose to
a semantic method. The one case where the gap actually shows —
`eval/run_eval.py`'s per-question breakdown flags it — is the
intercropping question, where BM25 gets a perfect hit and the LSA
retriever misses it entirely. Dense retrieval's real advantage, matching
meaning across genuinely different vocabulary or across languages, is
a claim this corpus is too small and too lexically friendly to test
properly. That's a scaling question, not a settled one, and it's the
first thing on the roadmap.

Faithfulness is 1.0 across the board here because the eval runs against
the extractive fallback generator by default (no `ANTHROPIC_API_KEY`
needed to reproduce this table). Set the key and pass `ClaudeGenerator()`
into the pipeline to see how faithfulness looks for actual generated
answers instead of extracted ones — that's where the metric starts to
do real work, and where I'd expect scores below 1.0 to show up.

## Using it

```python
import json
from pathlib import Path
from agri_rag.chunking import chunk_corpus
from agri_rag.retrieval import HybridRetriever
from agri_rag.generation import ClaudeGenerator
from agri_rag.pipeline import RAGPipeline

docs = json.loads(Path("corpus/docs.json").read_text())
chunks = chunk_corpus(docs)
retriever = HybridRetriever(chunks)
pipeline = RAGPipeline(retriever, generator=ClaudeGenerator(), top_k=3)

result = pipeline.answer("How do I stop weevils from ruining my stored grain?")
print(result.answer.text)
print(result.faithfulness.score, result.faithfulness.unsupported)
```

## Layout

```
src/agri_rag/
  chunking.py     sentence-aware sliding-window chunking
  retrieval.py    BM25, LSA-dense, and hybrid RRF retrievers
  generation.py   extractive fallback + optional Claude-backed generator
  evaluation.py   precision@k / recall@k / MRR, faithfulness scorer
  pipeline.py     wires retrieval + generation + faithfulness together
corpus/
  docs.json       seed agronomic corpus (13 English, 2 French)
eval/
  questions.json  16 labeled eval questions
  run_eval.py     compares all three retrievers, writes results table
tests/
  test_retrieval.py    chunking + all three retrievers
  test_evaluation.py   faithfulness scorer + retrieval metrics
results/
  eval_results.md       output of the last eval run
```

## Limitations and what's next

- The faithfulness scorer is a lexical-overlap proxy, not real entailment.
  It will call a sentence "supported" if it shares enough n-grams with
  the context even when it subtly misstates what the source says, and it
  can flag a correct paraphrase as unsupported if the wording diverges
  too far from the source. A proper next step is a small NLI-based
  faithfulness check (even a lightweight cross-lingual NLI model) run
  alongside this one, so the two can be compared rather than trusting
  either alone.
- The corpus is too small and too lexically aligned with the eval
  questions to say anything definitive about when dense or hybrid
  retrieval earns its complexity over BM25. That comparison needs a
  larger corpus and, more importantly, eval questions written by someone
  who wasn't also the one writing the source documents.
- No actual Wolof (or other low-resource regional language) content or
  queries are in this repo yet. The architecture is built not to assume
  a specific embedding model so that gap can be closed later without a
  redesign, but closing it is real work: sourcing or translating
  agronomic content, writing eval questions with a native speaker, and
  evaluating whatever embedding model gets chosen specifically on that
  language rather than assuming its English/French benchmark numbers
  transfer.

## License

MIT — see LICENSE.
