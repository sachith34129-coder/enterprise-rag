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

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Write a short passage (3-4 sentences) that answers the question as if it "
               "came from a company employee handbook. If you don't know the facts, make up "
               "plausible details. Write only the passage, nothing else."),
    ("human", "{question}"),
])

FUSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You generate alternative search queries. Rewrite the user's question in "
               "{num_queries} different ways, using different words and phrasings but keeping "
               "the same meaning. Return one query per line, with no numbering, bullets, "
               "or extra text."),
    ("human", "{question}"),
])

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You classify employee questions for a company handbook assistant.
Reply with exactly ONE of these category names and nothing else:

EXACT   - mentions a specific term, acronym, name or code (e.g. "What is EOS?", "When are Heartbeats due?")
FACTUAL - a clear, specific question about a policy (e.g. "How long is the sabbatical?")
VAGUE   - casual, short or unclear wording (e.g. "gym stuff?", "what if I quit")
BROAD   - wide or multi-part, needs info from several places (e.g. "What benefits do employees get?")
CHAT    - greetings or small talk that needs no documents (e.g. "hi", "thanks!")"""),
    ("human", "{question}"),
])

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant for the company handbook. Reply briefly, "
               "and invite the user to ask about company policies."),
    ("human", "{question}"),
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