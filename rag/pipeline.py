"""
Pipeline module
---------------
The full RAG flow:  question -> retrieve chunks -> build prompt -> LLM -> answer + sources
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from rag.prompts import RAG_PROMPT, format_context


class RAGPipeline:
    def __init__(self, retriever: BaseRetriever, llm: BaseChatModel):
        self.retriever = retriever
        self.chain = RAG_PROMPT | llm | StrOutputParser()

    def ask(self, question: str) -> dict:
        # 1. Retrieve: find the most relevant chunks
        chunks = self.retriever.invoke(question)

        # 2. Augment + Generate: put chunks in the prompt and call the LLM
        answer = self.chain.invoke({
            "context": format_context(chunks),
            "question": question,
        })

        # 3. Return the answer together with its sources (for citations)
        sources = [
            {"source": chunk.metadata["source"], "content": chunk.page_content}
            for chunk in chunks
        ]
        return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    from rag.embeddings import get_embedding_model
    from rag.llm import get_llm
    from rag.vector_store import load_vector_store

    vector_store = load_vector_store(get_embedding_model())
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    rag = RAGPipeline(retriever=retriever, llm=get_llm())

    questions = [
        "How long is the cooldown period?",
        "Can I work a second job?",
        "What is the capital of France?",   # not in the handbook
    ]

    for question in questions:
        result = rag.ask(question)
        files = sorted({s["source"] for s in result["sources"]})
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"Sources: {', '.join(files)}")