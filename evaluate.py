"""
Evaluate retrieval strategies on the test set.

    python evaluate.py                 -> run all strategies
    python evaluate.py vector hybrid   -> run only these (results are saved per strategy)
    python evaluate.py summary         -> just print the tables from saved results
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

from rag.embeddings import get_embedding_model
from rag.evaluation.judge import Judge
from rag.evaluation.metrics import citation_hit, refusal_correct, timed
from rag.llm import get_llm
from rag.pipeline import RAGPipeline, RoutedRAGPipeline
from rag.retrievers.factory import get_retriever
from rag.router import QueryRouter
from rag.vector_store import load_vector_store

TEST_SET = Path("data/eval/test_set.json")
RESULTS_DIR = Path("results")
ALL_STRATEGIES = ["vector", "hybrid", "rerank", "hyde", "fusion", "auto"]
METRICS = ["context_relevance", "faithfulness", "answer_relevance", "completeness", "citation_hit", "refusal_ok"]
PAUSE_SECONDS = 2   # short pause between questions to stay under free-tier rate limits


def build_pipeline(strategy: str, vector_store, llm):
    """Same as chat.py: 'auto' uses the router, anything else is a fixed strategy."""
    if strategy == "auto":
        return RoutedRAGPipeline(vector_store, llm, QueryRouter(llm))
    return RAGPipeline(retriever=get_retriever(strategy, vector_store, llm=llm), llm=llm)


def evaluate_question(rag, judge: Judge, item: dict) -> dict:
    """Ask one test question and score the result."""
    question = item["question"]
    result, seconds = timed(rag.ask, question)

    answer = result["answer"]
    context = "\n\n".join(source["content"] for source in result["sources"])
    should_refuse = item["source_file"] is None

    row = {
        "question": question,
        "type": item["type"],
        "answer": answer,
        "seconds": seconds,
        "refusal_ok": refusal_correct(answer, should_refuse),
    }

    # The triad, completeness and citations only make sense when an answer exists in the handbook
    if not should_refuse:
        row["context_relevance"] = judge.context_relevance(question, context)
        row["faithfulness"] = judge.faithfulness(context, answer)
        row["answer_relevance"] = judge.answer_relevance(question, answer)
        row["completeness"] = judge.completeness(item["golden_answer"], answer)
        row["citation_hit"] = citation_hit(result["sources"], item["source_file"])

    if "route" in result:   # only for 'auto': which strategy did the router pick?
        row["route"] = result["route"]["strategy"]

    return row


def run_strategy(strategy: str, test_set: list[dict], vector_store, llm, judge: Judge) -> None:
    print(f"\n=== Evaluating: {strategy} ===")
    rag = build_pipeline(strategy, vector_store, llm)

    rows = []
    for number, item in enumerate(test_set, start=1):
        row = evaluate_question(rag, judge, item)
        row["strategy"] = strategy
        rows.append(row)
        print(f"  [{number:>2}/{len(test_set)}] {item['question'][:55]:<55} {row['seconds']}s")
        time.sleep(PAUSE_SECONDS)

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{strategy}.csv", index=False)
    print(f"  Saved {RESULTS_DIR / f'{strategy}.csv'}")


def print_summary() -> None:
    files = sorted(RESULTS_DIR.glob("*.csv"))
    if not files:
        print("No results yet. Run: python evaluate.py")
        return

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["quality"] = df[METRICS].mean(axis=1)   # one score per question (skips metrics that don't apply)

    # 1. Overall table
    summary = df.groupby("strategy")[METRICS + ["quality", "seconds"]].mean().round(2)
    summary = summary.sort_values("quality", ascending=False)
    print("\n=== Overall (higher is better, except seconds) ===")
    print(summary.to_string())

    # 2. Which strategy works best for which kind of question?
    by_type = df.pivot_table(index="type", columns="strategy", values="quality", aggfunc="mean").round(2)
    print("\n=== Quality by question type ===")
    print(by_type.to_string())

    print("\n=== Best strategy per question type ===")
    for question_type, strategy in by_type.idxmax(axis=1).items():
        print(f"  {question_type:<13} -> {strategy}")

    best = summary.index[0]
    print(f"\nOverall winner: {best} (quality {summary.loc[best, 'quality']}, {summary.loc[best, 'seconds']}s per question)")


def main():
    args = sys.argv[1:]
    strategies = [] if args == ["summary"] else (args or ALL_STRATEGIES)

    if strategies:
        test_set = json.loads(TEST_SET.read_text(encoding="utf-8"))
        vector_store = load_vector_store(get_embedding_model())
        llm = get_llm()
        judge = Judge()

        for strategy in strategies:
            run_strategy(strategy, test_set, vector_store, llm, judge)

    print_summary()


if __name__ == "__main__":
    main()