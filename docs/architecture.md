# Architecture notes

## Why retrieval and faithfulness are scored separately

It would be easy to report a single "answer quality" number and call it
done, but that number wouldn't tell you which stage to fix when it's
low. A pipeline can retrieve the right passage and still generate a
faithful-but-unhelpful non-answer, or retrieve nothing useful and still
generate something that happens to sound right. Keeping precision/recall
and faithfulness as separate metrics, computed independently, means a
regression in one is diagnosable without having to guess whether it's
retrieval or generation that broke.

## Why the extractive generator is a first-class option, not a stub

`ExtractiveGenerator` isn't there to keep the demo running before the
"real" generator gets plugged in. It's a legitimate zero-hallucination
baseline: since it can only return text that was already retrieved, its
faithfulness score is 1.0 by definition. That gives `faithfulness_score`
something concrete to be validated against — if a generated answer from
`ClaudeGenerator` scores lower than 1.0, that's evidence the metric is
measuring something real, not just always returning 1.0 regardless of
input. Without the extractive baseline in the repo, there'd be no way to
tell those two situations apart.

## Why reciprocal rank fusion instead of a weighted score blend

BM25 scores and cosine similarity scores from the LSA retriever live on
different, not-directly-comparable scales — a BM25 score of 8.2 doesn't
mean anything relative to a cosine similarity of 0.3 without an
arbitrary normalization step that would need its own justification and
its own failure modes (min-max normalizing over a candidate pool that
changes per query is a common source of bugs in hybrid retrieval code).
Reciprocal rank fusion sidesteps the whole problem by only ever looking
at rank position, never at the raw score, which is why it's a standard
choice for combining heterogeneous retrievers.

## Where the faithfulness scorer is weakest

The scorer works by decomposing an answer into sentences and checking
each one against retrieved context via TF-IDF cosine similarity above a
threshold. Two concrete failure modes worth naming rather than leaving
implicit:

1. A sentence can share enough vocabulary with the context to score as
   "supported" while actually inverting or overstating what the source
   says — the check has no notion of negation or numeric magnitude, only
   lexical overlap.
2. A sentence that's a faithful paraphrase using different words than
   the source can score as "unsupported" if the vocabulary diverges
   enough, which would show up as a false alarm.

Both point toward the same fix: an NLI-based entailment check as a
second faithfulness signal, run alongside the current one rather than
replacing it, since disagreement between the two would itself be useful
information about which sentences need a closer look.

## Why chunking is sentence-aware with an overlap window

The demo corpus doesn't strictly need this — most documents are three or
four sentences and would round-trip through a naive character-based
splitter without visible damage. It's implemented properly anyway
because the chunker is the one piece of this pipeline that would
otherwise need a rewrite the moment a real, longer-form corpus (an
actual FAO extension manual, for instance) got substituted in, and at
that point silently truncating a sentence mid-word would corrupt exactly
the kind of specific, numeric agronomic advice (application rates,
spacing, timing) that this whole project exists to retrieve accurately.
