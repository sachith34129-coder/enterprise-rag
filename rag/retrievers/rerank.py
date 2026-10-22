"""
Rerank retriever
----------------
Two-stage retrieval:
  1. A base retriever fetches many candidates (fast, rough)
  2. A cross-encoder reads each (question, chunk) pair and re-orders them (slow, accurate)
"""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder


class RerankRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    k: int = 4
    model: CrossEncoder | None = None

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context) -> None:
        # Load the cross-encoder once, when the retriever is created
        print(f"Loading reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        # Stage 1: get candidates
        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []

        # Stage 2: score every (question, chunk) pair, highest score = most relevant
        pairs = [(query, chunk.page_content) for chunk in candidates]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in ranked[: self.k]]


if __name__ == "__main__":
    from rag.embeddings import get_embedding_model
    from rag.vector_store import load_vector_store

    vector_store = load_vector_store(get_embedding_model())
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    retriever = RerankRetriever(base_retriever=base_retriever, k=4)

    question = "How much time do we get to fix bugs between projects?"
    print(f"\nQ: {question}")

    # BEFORE: the vector search order, with the reranker's score for each chunk
    candidates = base_retriever.invoke(question)
    scores = retriever.model.predict([(question, c.page_content) for c in candidates])
    print("\n--- BEFORE: vector search order ---")
    for rank, (chunk, score) in enumerate(zip(candidates, scores), start=1):
        print(f"#{rank:<2} rerank score={score:6.2f}  {chunk.metadata['source']:<25} {chunk.page_content[:60]!r}")

    # AFTER: re-ordered by the reranker, top 4 kept
    print("\n--- AFTER: reranked top 4 ---")
    for rank, chunk in enumerate(retriever.invoke(question), start=1):
        print(f"#{rank}  {chunk.metadata['source']:<25} {chunk.page_content[:60]!r}")