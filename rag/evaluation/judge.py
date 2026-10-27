"""
LLM judge
---------
Scores RAG outputs with YES (1) / NO (0) on four checks:
  - context_relevance : did we retrieve useful chunks?
  - faithfulness      : is the answer supported by those chunks? (no hallucination)
  - answer_relevance  : does the answer address the question?
  - completeness      : does the answer contain all key facts from the golden answer?
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.llm import get_llm
from rag.prompts import (
    ANSWER_RELEVANCE_PROMPT,
    COMPLETENESS_PROMPT,
    CONTEXT_RELEVANCE_PROMPT,
    FAITHFULNESS_PROMPT,
)

# A smaller, different model than the one answering:
# cheaper, higher free-tier limits, and less likely to "go easy" on its own answers
JUDGE_MODEL = "openai/gpt-oss-20b"


class Judge:
    def __init__(self, llm: BaseChatModel | None = None):
        self.llm = llm or get_llm(model=JUDGE_MODEL, temperature=0, max_tokens=5)

    def _ask(self, prompt: ChatPromptTemplate, **inputs) -> int:
        """Run one judge prompt and turn the YES/NO reply into 1/0."""
        reply = (prompt | self.llm | StrOutputParser()).invoke(inputs)
        return 1 if reply.strip().upper().startswith("YES") else 0

    def context_relevance(self, question: str, context: str) -> int:
        return self._ask(CONTEXT_RELEVANCE_PROMPT, question=question, context=context)

    def faithfulness(self, context: str, answer: str) -> int:
        return self._ask(FAITHFULNESS_PROMPT, context=context, answer=answer)

    def answer_relevance(self, question: str, answer: str) -> int:
        return self._ask(ANSWER_RELEVANCE_PROMPT, question=question, answer=answer)

    def completeness(self, golden_answer: str, answer: str) -> int:
        return self._ask(COMPLETENESS_PROMPT, golden_answer=golden_answer, answer=answer)


if __name__ == "__main__":
    judge = Judge()

    pto_context = (
        "Employees get 15 days of PTO per year. To request time off, "
        "submit form HR-401 to your manager at least 5 days in advance."
    )

    # (check, inputs, expected score, what it tests)
    tests = [
        ("context_relevance", dict(question="How many PTO days do I get?", context=pto_context), 1,
         "relevant context"),
        ("context_relevance", dict(question="What is the dress code?", context=pto_context), 0,
         "irrelevant context"),
        ("faithfulness", dict(context=pto_context, answer="You get 15 days of PTO per year."), 1,
         "supported answer"),
        ("faithfulness", dict(context=pto_context, answer="You get 15 days of PTO, which can be extended to 20."), 0,
         "made-up extension"),
        ("answer_relevance", dict(question="How do I request time off?",
                                  answer="Submit form HR-401 to your manager at least 5 days in advance."), 1,
         "answers the question"),
        ("answer_relevance", dict(question="How do I request time off?",
                                  answer="Employees get 15 days of PTO per year."), 0,
         "true but doesn't answer"),
        ("completeness", dict(golden_answer="Every three years, employees may take a 6-week paid sabbatical.",
                              answer="You can take a paid sabbatical of 6 weeks once every three years."), 1,
         "all key facts"),
        ("completeness", dict(golden_answer="Every three years, employees may take a 6-week paid sabbatical.",
                              answer="You can take a paid sabbatical."), 0,
         "missing key facts"),
    ]

    correct = 0
    print(f"{'CHECK':<18} {'TESTS':<25} {'EXPECTED':<9} {'GOT':<4}")
    print("-" * 62)
    for check, inputs, expected, description in tests:
        score = getattr(judge, check)(**inputs)
        correct += score == expected
        mark = "✓" if score == expected else "✗"
        print(f"{check:<18} {description:<25} {expected:<9} {score:<4} {mark}")

    print(f"\nJudge agreed with expected answers: {correct}/{len(tests)}")