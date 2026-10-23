"""
Ask questions about the handbook in the terminal:   python chat.py
"""

from rag.embeddings import get_embedding_model
from rag.llm import get_llm
from rag.pipeline import RAGPipeline, RoutedRAGPipeline
from rag.retrievers.factory import get_retriever
from rag.router import QueryRouter
from rag.vector_store import load_vector_store

STRATEGY = "auto"   # "auto" = router decides | or fix one: "vector", "keyword", "hybrid", "rerank", "hyde", "fusion"


def main():
    vector_store = load_vector_store(get_embedding_model())
    llm = get_llm()

    if STRATEGY == "auto":
        rag = RoutedRAGPipeline(vector_store, llm, QueryRouter(llm))
    else:
        rag = RAGPipeline(retriever=get_retriever(STRATEGY, vector_store, llm=llm), llm=llm)

    print(f"Ask about the handbook - strategy: {STRATEGY} (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        result = rag.ask(question)

        if "route" in result:
            route = result["route"]
            print(f"\n[router: {route['category']} → {route['strategy']}]")
        print(f"Bot: {result['answer']}")
        if result["sources"]:
            files = sorted({s["source"] for s in result["sources"]})
            print(f"Sources: {', '.join(files)}")
        print()


if __name__ == "__main__":
    main()


"""
You: hi
[router: CHAT → none]
Bot: Hello! I can help with questions about company policies...

You: What is EOS?
[router: EXACT → hybrid]
Bot: EOS stands for Everyone on Support...
Sources: our-rituals.md

You: do they pay for gym stuff
[router: VAGUE → hyde]
Bot: ...
Sources: benefits-and-perks.md

You: What benefits do employees get?
[router: BROAD → fusion]
Bot: ...
"""