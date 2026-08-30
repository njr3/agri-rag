"""
Runs all three retrieval strategies against the eval question set and
reports precision@3, recall@3, MRR, and mean faithfulness for each.

    python eval/run_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agri_rag.chunking import chunk_corpus
from agri_rag.evaluation import precision_at_k, recall_at_k, reciprocal_rank
from agri_rag.pipeline import RAGPipeline
from agri_rag.retrieval import RETRIEVERS

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(Path(path).read_text())


def evaluate_retriever(name, retriever_cls, chunks, questions, k=3):
    retriever = retriever_cls(chunks)
    pipeline = RAGPipeline(retriever, top_k=k)

    precisions, recalls, rrs, faithfulness_scores = [], [], [], []
    per_question = []

    for q in questions:
        result = pipeline.answer(q["question"])
        retrieved_ids = [r.chunk.chunk_id for r in result.retrieved]
        relevant = set(q["relevant_docs"])

        p = precision_at_k(retrieved_ids, relevant, k)
        r = recall_at_k(retrieved_ids, relevant, k)
        rr = reciprocal_rank(retrieved_ids, relevant)

        precisions.append(p)
        recalls.append(r)
        rrs.append(rr)
        faithfulness_scores.append(result.faithfulness.score)

        per_question.append(
            {
                "id": q["id"],
                "question": q["question"],
                "retrieved": retrieved_ids,
                "relevant": list(relevant),
                "precision_at_k": p,
                "recall_at_k": r,
                "reciprocal_rank": rr,
                "faithfulness": result.faithfulness.score,
            }
        )

    summary = {
        "retriever": name,
        "mean_precision_at_k": sum(precisions) / len(precisions),
        "mean_recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
        "mean_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
    }
    return summary, per_question


def main():
    docs = load_json(ROOT / "corpus" / "docs.json")
    questions = load_json(ROOT / "eval" / "questions.json")
    chunks = chunk_corpus(docs)

    lines = ["# Retrieval strategy comparison", "", f"Corpus: {len(docs)} documents, {len(chunks)} chunks. Eval set: {len(questions)} questions.", ""]
    lines.append("| retriever | precision@3 | recall@3 | MRR | mean faithfulness |")
    lines.append("|---|---|---|---|---|")

    all_summaries = []
    for name, cls in RETRIEVERS.items():
        summary, per_question = evaluate_retriever(name, cls, chunks, questions)
        all_summaries.append((summary, per_question))
        lines.append(
            f"| {name} | {summary['mean_precision_at_k']:.3f} | {summary['mean_recall_at_k']:.3f} "
            f"| {summary['mrr']:.3f} | {summary['mean_faithfulness']:.3f} |"
        )

    lines.append("")
    lines.append("## Per-question detail (where the retrievers disagree)")
    lines.append("")
    by_retriever = {s["retriever"]: {d["id"]: d for d in detail} for s, detail in all_summaries}
    question_ids = [q["id"] for q in questions]
    any_disagreement = False
    for qid in question_ids:
        recalls = {name: by_retriever[name][qid]["recall_at_k"] for name in by_retriever}
        if max(recalls.values()) - min(recalls.values()) > 0.01:
            any_disagreement = True
            q_text = by_retriever["bm25"][qid]["question"]
            recall_str = ", ".join(f"{name}={v:.2f}" for name, v in recalls.items())
            lines.append(f"- **{qid}** ({q_text}): {recall_str}")
    if not any_disagreement:
        lines.append("(no disagreements — all three retrievers scored every question identically at k=3)")

    text = "\n".join(lines)
    print(text)
    out_path = ROOT / "results" / "eval_results.md"
    out_path.write_text(text)
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
