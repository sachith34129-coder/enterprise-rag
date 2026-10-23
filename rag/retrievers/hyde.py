"""
HyDE retriever (Hypothetical Document Embeddings)
-------------------------------------------------
1. The LLM writes a fake, handbook-style answer to the question
2. We search with (question + fake answer) instead of the question alone
The fake answer is never shown to the user - it only improves the search.
"""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from rag.prompts import HYDE_PROMPT


class HyDERetriever(BaseRetriever):
    base_retriever: BaseRetriever
    llm: BaseChatModel

    def generate_hypothetical_answer(self, question: str) -> str:
        chain = HYDE_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"question": question})

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        hypothetical_answer = self.generate_hypothetical_answer(query)
        search_text = f"{query}\n{hypothetical_answer}"
        return self.base_retriever.invoke(search_text)


if __name__ == "__main__":
    from rag.embeddings import get_embedding_model
    from rag.llm import get_llm
    from rag.vector_store import load_vector_store

    vector_store = load_vector_store(get_embedding_model())
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    hyde_retriever = HyDERetriever(base_retriever=vector_retriever, llm=get_llm())

    question = "Do they pay for gym stuff?"
    print(f"Q: {question}")

    # 1. What does the fake answer look like?
    print("\n--- Hypothetical answer (never shown to the user) ---")
    print(hyde_retriever.generate_hypothetical_answer(question))

    # 2. Compare: searching with the raw question vs with HyDE
    print("\n--- Plain vector search ---")
    for chunk in vector_retriever.invoke(question):
        print(f"  {chunk.metadata['source']:<25} {chunk.page_content[:70]!r}")

    print("\n--- HyDE search ---")
    for chunk in hyde_retriever.invoke(question):
        print(f"  {chunk.metadata['source']:<25} {chunk.page_content[:70]!r}")