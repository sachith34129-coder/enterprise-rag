"""
RAG Fusion retriever
--------------------
1. The LLM rewrites the question in several different ways
2. We search with the original + every rewrite
3. Reciprocal Rank Fusion merges all the result lists
"""

import re

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from rag.prompts import FUSION_PROMPT
from rag.retrievers.rrf import reciprocal_rank_fusion


class FusionRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    llm: BaseChatModel
    num_queries: int = 3
    k: int = 4

    def generate_queries(self, question: str) -> list[str]:
        chain = FUSION_PROMPT | self.llm | StrOutputParser()
        reply = chain.invoke({"question": question, "num_queries": self.num_queries})

        queries = []
        for line in reply.splitlines():
            line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip()  # remove "1." or "-" if the LLM added them
            if line:
                queries.append(line)
        return queries[: self.num_queries]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        all_queries = [query] + self.generate_queries(query)
        result_lists = [self.base_retriever.invoke(q) for q in all_queries]
        fused = reciprocal_rank_fusion(result_lists)
        return fused[: self.k]


if __name__ == "__main__":
    from rag.embeddings import get_embedding_model
    from rag.llm import get_llm
    from rag.vector_store import load_vector_store

    vector_store = load_vector_store(get_embedding_model())
    fusion_retriever = FusionRetriever(
        base_retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
        llm=get_llm(),
    )

    question = "What is the WFH policy?"
    print(f"Q: {question}")

    # 1. What rewrites does the LLM come up with?
    print("\n--- Generated queries ---")
    for q in fusion_retriever.generate_queries(question):
        print(f"  - {q}")

    # 2. Compare plain vector search vs fusion
    print("\n--- Plain vector search ---")
    for chunk in vector_store.as_retriever(search_kwargs={"k": 4}).invoke(question):
        print(f"  {chunk.metadata['source']:<25} {chunk.page_content[:70]!r}")

    print("\n--- RAG Fusion ---")
    for chunk in fusion_retriever.invoke(question):
        print(f"  {chunk.metadata['source']:<25} {chunk.page_content[:70]!r}")