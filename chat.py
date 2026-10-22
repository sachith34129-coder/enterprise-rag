"""
Ask questions about the handbook in the terminal:   python chat.py
"""

from rag.embeddings import get_embedding_model
from rag.llm import get_llm
from rag.pipeline import RAGPipeline
from rag.vector_store import load_vector_store


def main():
    vector_store = load_vector_store(get_embedding_model())
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    rag = RAGPipeline(retriever=retriever, llm=get_llm())

    print("Ask about the handbook (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        result = rag.ask(question)
        files = sorted({s["source"] for s in result["sources"]})
        print(f"\nBot: {result['answer']}")
        print(f"Sources: {', '.join(files)}\n")


if __name__ == "__main__":
    main()