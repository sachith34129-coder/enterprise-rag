"""
Prompts module
--------------
The instructions we send to the LLM, kept separate from the code
so they're easy to read and tweak.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a helpful assistant that answers employee questions using the company handbook.

Rules:
- Answer ONLY using the context provided. Do not use outside knowledge.
- If the answer is not in the context, reply exactly: "I couldn't find that in the handbook."
- Keep answers short, clear, and in full sentences.
- Answer in the same language as the question."""

USER_PROMPT = """Context:
{context}

Question: {question}"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
])


def format_context(chunks: list[Document]) -> str:
    """Join chunks into one block of text, each labelled with its source file."""
    return "\n\n---\n\n".join(
        f"[Source: {chunk.metadata['source']}]\n{chunk.page_content}"
        for chunk in chunks
    )


if __name__ == "__main__":
    sample_chunks = [
        Document(page_content="In between each cycle, we spend two weeks cooling down.",
                 metadata={"source": "how-we-work.md"}),
        Document(page_content="All teams operate on the same 6-week cadence.",
                 metadata={"source": "how-we-work.md"}),
    ]

    prompt_text = RAG_PROMPT.format(
        context=format_context(sample_chunks),
        question="How long is the cooldown?",
    )
    print(prompt_text)