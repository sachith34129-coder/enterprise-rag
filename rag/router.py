"""
Query router
------------
Uses the LLM to classify a question, then picks the best retrieval
strategy for that type of question.
"""

import random
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from rag.prompts import ROUTER_PROMPT

# Which strategy handles which type of question
ROUTES = {
    "EXACT": "hybrid",     # keyword search catches exact terms
    "FACTUAL": "rerank",   # precise ordering for clear questions
    "VAGUE": "hyde",       # fake answer bridges casual wording
    "BROAD": "fusion",     # many rephrasings cover more ground
    "CHAT": "none",        # no documents needed
}
DEFAULT_CATEGORY = "FACTUAL"


class QueryRouter:
    def __init__(self, llm: BaseChatModel, ab_test_rate: float = 0.0):
        self.chain = ROUTER_PROMPT | llm | StrOutputParser()
        self.ab_test_rate = ab_test_rate   # e.g. 0.2 = send 20% of traffic to the experiment

    def classify(self, question: str) -> str:
        reply = self.chain.invoke({"question": question})
        words = re.findall(r"[A-Z]+", reply.upper())
        category = words[0] if words else ""
        return category if category in ROUTES else DEFAULT_CATEGORY

    def route(self, question: str) -> dict:
        category = self.classify(question)
        strategy = ROUTES[category]

        # A/B test: occasionally try the cheap "vector" strategy instead,
        # so we can later measure whether the expensive one is worth it
        is_experiment = False
        if strategy != "none" and random.random() < self.ab_test_rate:
            strategy = "vector"
            is_experiment = True

        return {"category": category, "strategy": strategy, "is_experiment": is_experiment}


if __name__ == "__main__":
    from rag.llm import get_llm

    router = QueryRouter(get_llm())

    tests = [
        ("What is EOS?", "EXACT"),
        ("When are Heartbeats due?", "EXACT"),
        ("How long is the sabbatical?", "FACTUAL"),
        ("Do they pay for gym stuff?", "VAGUE"),
        ("what if I quit", "VAGUE"),
        ("What benefits and perks do employees get?", "BROAD"),
        ("Hi there!", "CHAT"),
        ("Thanks, that helped", "CHAT"),
    ]

    correct = 0
    print(f"{'QUESTION':<45} {'EXPECTED':<9} {'GOT':<9} STRATEGY")
    print("-" * 80)
    for question, expected in tests:
        decision = router.route(question)
        mark = "✓" if decision["category"] == expected else "✗"
        correct += decision["category"] == expected
        print(f"{question:<45} {expected:<9} {decision['category']:<9} {decision['strategy']:<8} {mark}")

    print(f"\nRouter accuracy: {correct}/{len(tests)}")