"""
Loader module
-------------
Reads every Markdown (.md) file in a folder and turns each one into a
Document. A Document = the text + metadata (which file it came from).
"""

from pathlib import Path

from langchain_core.documents import Document


def load_documents(folder: str) -> list[Document]:
    """Load every .md file in `folder` as a Document."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    documents = []
    for file_path in sorted(folder_path.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        doc = Document(page_content=text, metadata={"source": file_path.name})
        documents.append(doc)
        print(f"Loaded: {file_path.name} ({len(text)} characters)")

    return documents


if __name__ == "__main__":
    documents = load_documents("data/handbook")

    print(f"\nTotal documents: {len(documents)}")

    first = documents[0]
    print("\n--- First document ---")
    print("Metadata:", first.metadata)
    print("Preview :", first.page_content[:300])