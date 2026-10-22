"""
Reciprocal Rank Fusion (RRF)
----------------------------
Merges several ranked lists into one. A chunk scores 1 / (60 + rank)
in each list it appears in; chunks that rank high in MANY lists win.
"""

from langchain_core.documents import Document


def reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = 60) -> list[Document]:
    scores = {}   # chunk text -> fused score
    chunks = {}   # chunk text -> Document

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            key = chunk.page_content
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            chunks[key] = chunk

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [chunks[key] for key in ranked_keys]


if __name__ == "__main__":
    A = Document(page_content="A")
    B = Document(page_content="B")
    C = Document(page_content="C")
    D = Document(page_content="D")

    vector_results = [A, B, C]
    keyword_results = [C, A, D]

    fused = reciprocal_rank_fusion([vector_results, keyword_results])
    print("Vector :", [d.page_content for d in vector_results])
    print("Keyword:", [d.page_content for d in keyword_results])
    print("Fused  :", [d.page_content for d in fused])