"""
Keyword retriever (BM25)
------------------------
Finds chunks that share the most important WORDS with the question.
Great for exact terms (names, acronyms, product codes) that embeddings can miss.
"""

import re

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """'How long is the Cooldown?' -> ['how', 'long', 'is', 'the', 'cooldown']"""
    return re.findall(r"\w+", text.lower())


class KeywordRetriever(BaseRetriever):
    chunks: list[Document]
    k: int = 4
    bm25: BM25Okapi | None = None

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context) -> None:
        # Build the BM25 index once, when the retriever is created
        self.bm25 = BM25Okapi([tokenize(chunk.page_content) for chunk in self.chunks])

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        scores = self.bm25.get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        return [self.chunks[i] for i in best if scores[i] > 0]  # skip chunks with no matching words


if __name__ == "__main__":
    from rag.chunker import split_documents
    from rag.loader import load_documents

    chunks = split_documents(load_documents("data/handbook"))
    retriever = KeywordRetriever(chunks=chunks, k=3)

    for question in ["What is EOS?", "When are Heartbeats due?", "What is the capital of France?"]:
        print(f"\nQ: {question}")
        for chunk in retriever.invoke(question):
            print(f"  [{chunk.metadata['source']}] {chunk.page_content[:100]!r}")