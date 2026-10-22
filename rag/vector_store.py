"""
Vector store module
-------------------
Embeds chunks and saves them in a Chroma database on disk, then lets us
search them by meaning.

  build_vector_store() -> run once (or whenever the documents change)
  load_vector_store()  -> run every time we want to search
"""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

PERSIST_DIR = "vector_db"
COLLECTION_NAME = "handbook"


def build_vector_store(chunks: list[Document], embedding_model: Embeddings) -> Chroma:
    """Embed all chunks and save them to disk. Deletes any old database first."""
    if Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)  # start fresh, otherwise re-running adds duplicates

    print(f"Embedding {len(chunks)} chunks... (this can take a minute)")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"Saved vector store to '{PERSIST_DIR}/'")
    return vector_store


def load_vector_store(embedding_model: Embeddings) -> Chroma:
    """Open the database we built earlier."""
    if not Path(PERSIST_DIR).exists():
        raise FileNotFoundError(f"No vector store at '{PERSIST_DIR}/'. Build it first.")

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIR,
    )

#optiming part skip for now

def get_all_chunks(vector_store: Chroma) -> list[Document]:
    """Read every stored chunk back out of Chroma (used by keyword search)."""
    data = vector_store.get()
    return [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(data["documents"], data["metadatas"])
    ]


if __name__ == "__main__":
    from rag.chunker import split_documents
    from rag.embeddings import get_embedding_model
    from rag.loader import load_documents

    # 1. Build: loader -> chunker -> embeddings -> Chroma
    documents = load_documents("data/handbook")
    chunks = split_documents(documents)
    embedding_model = get_embedding_model()
    build_vector_store(chunks, embedding_model)

    # 2. Load it back from disk (proves it was saved)
    vector_store = load_vector_store(embedding_model)
    print(f"Chunks in database: {len(vector_store.get()['ids'])}")

    # 3. Search by meaning
    question = "How long is the cooldown period?"
    results = vector_store.similarity_search_with_score(question, k=4)

    print(f"\nQuestion: {question}\n")
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"#{rank}  score={score:.3f}  source={chunk.metadata['source']}")
        print(f"    {chunk.page_content[:150]}...\n")