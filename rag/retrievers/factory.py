"""
Retriever factory
-----------------
Builds a retriever by name, so the rest of the app can switch strategies
with one word:  get_retriever("hybrid", vector_store)
"""

from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever

from rag.retrievers.hybrid import HybridRetriever
from rag.retrievers.keyword import KeywordRetriever
from rag.vector_store import get_all_chunks


def get_retriever(strategy: str, vector_store: Chroma, k: int = 4) -> BaseRetriever:
    if strategy == "vector":
        return vector_store.as_retriever(search_kwargs={"k": k})

    if strategy == "keyword":
        return KeywordRetriever(chunks=get_all_chunks(vector_store), k=k)

    if strategy == "hybrid":
        return HybridRetriever(
            vector_retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
            keyword_retriever=KeywordRetriever(chunks=get_all_chunks(vector_store), k=10),
            k=k,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    from rag.embeddings import get_embedding_model
    from rag.vector_store import load_vector_store

    vector_store = load_vector_store(get_embedding_model())

    questions = [
        "What is EOS?",                                              # exact acronym
        "How much time do we get to fix bugs between projects?",     # paraphrase
    ]

    for question in questions:
        print(f"\n{'=' * 70}\nQ: {question}")
        for strategy in ["vector", "keyword", "hybrid"]:
            retriever = get_retriever(strategy, vector_store)
            print(f"\n  [{strategy}]")
            for chunk in retriever.invoke(question):
                print(f"    {chunk.metadata['source']:<25} {chunk.page_content[:70]!r}")