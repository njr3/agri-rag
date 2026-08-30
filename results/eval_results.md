# Retrieval strategy comparison

Corpus: 15 documents, 17 chunks. Eval set: 16 questions.

| retriever | precision@3 | recall@3 | MRR | mean faithfulness |
|---|---|---|---|---|
| bm25 | 0.333 | 0.938 | 0.833 | 1.000 |
| lsa_dense | 0.354 | 0.875 | 0.812 | 1.000 |
| hybrid_rrf | 0.333 | 0.938 | 0.802 | 1.000 |

## Per-question detail (where the retrievers disagree)

- **q8** (What are the benefits of planting millet and cowpea together in the same field?): bm25=1.00, lsa_dense=0.00, hybrid_rrf=1.00