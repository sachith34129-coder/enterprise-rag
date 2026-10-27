"""
Simple metrics
--------------
Plain-Python checks that need no LLM: free, instant, and 100% consistent.
"""

import time

REFUSAL_PHRASES = ["couldn't find", "could not find"]


def citation_hit(sources: list[dict], expected_file: str) -> int:
    """1 if the correct source file is among the retrieved sources, else 0."""
    retrieved_files = {source["source"] for source in sources}
    return 1 if expected_file in retrieved_files else 0


def is_refusal(answer: str) -> bool:
    """True if the bot said it couldn't find the answer."""
    text = answer.lower().replace("’", "'")   # treat curly and straight apostrophes the same
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def refusal_correct(answer: str, should_refuse: bool) -> int:
    """
    1 if the bot refused exactly when it should have:
      - unanswerable question + refused  -> correct
      - answerable question  + answered  -> correct
    """
    return 1 if is_refusal(answer) == should_refuse else 0


def timed(function, *args):
    """Run function(*args) and return (result, seconds taken)."""
    start = time.perf_counter()
    result = function(*args)
    return result, round(time.perf_counter() - start, 2)


if __name__ == "__main__":
    sources = [{"source": "how-we-work.md"}, {"source": "our-rituals.md"}]

    print("--- citation_hit ---")
    print("Correct file retrieved :", citation_hit(sources, "how-we-work.md"))    # expect 1
    print("Correct file missing   :", citation_hit(sources, "severance.md"))      # expect 0

    print("\n--- refusal_correct ---")
    refusal = "I couldn't find that in the handbook."
    answer = "The cooldown period is two weeks."
    print("Unanswerable + refused :", refusal_correct(refusal, should_refuse=True))   # expect 1
    print("Unanswerable + answered:", refusal_correct(answer, should_refuse=True))    # expect 0 (made something up?)
    print("Answerable + answered  :", refusal_correct(answer, should_refuse=False))   # expect 1
    print("Answerable + refused   :", refusal_correct(refusal, should_refuse=False))  # expect 0 (retrieval missed it)

    print("\n--- timed ---")
    result, seconds = timed(sum, range(1_000_000))
    print(f"sum = {result}, took {seconds}s")