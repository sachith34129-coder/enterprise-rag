"""
Build the vector database from the documents.
Run once, and again whenever the documents change:   python ingest.py
"""

from rag.chunker import split_documents
from rag.embeddings import get_embedding_model
from rag.loader import load_documents
from rag.vector_store import build_vector_store


def main():
    documents = load_documents("data/handbook")
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)
    build_vector_store(chunks, get_embedding_model())


if __name__ == "__main__":
    main()