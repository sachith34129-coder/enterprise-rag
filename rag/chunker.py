"""
Chunker module
--------------
Splits large Documents into small, overlapping chunks so that we can
search for (and send to the LLM) only the relevant pieces.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    """Split Documents into chunks of about `chunk_size` characters."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


if __name__ == "__main__":
    from rag.loader import load_documents

    documents = load_documents("data/handbook")
    print()
    chunks = split_documents(documents)

    # 1. What does a chunk look like?
    print("\n--- Sample chunk ---")
    print(chunks[0].metadata)
    print(chunks[0].page_content)

    # 2. Where did our test question's answer end up?
    print("\n--- Chunks that mention 'cooldown' ---")
    for chunk in chunks:
        if "cooldown" in chunk.page_content.lower():
            print(f"\n[{chunk.metadata['source']}] ({len(chunk.page_content)} characters)")
            print(chunk.page_content)

    # 3. How does chunk size change the number of chunks?
    print("\n--- Comparing chunk sizes ---")
    for size in [250, 500, 1000, 1500]:
        split_documents(documents, chunk_size=size, chunk_overlap=50)