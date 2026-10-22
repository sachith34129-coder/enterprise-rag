"""
LLM module
----------
Loads a chat model (Llama 3.3 70B on Groq) through LangChain.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from the .env file


def get_llm(model: str = "openai/gpt-oss-120b", temperature: float = 0.1, max_tokens: int = 500) -> ChatGroq:
    """Create a Groq chat model."""
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

    return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)


if __name__ == "__main__":
    llm = get_llm()

    # 1. Basic call
    print("--- Basic call ---")
    response = llm.invoke("In one sentence, what is Retrieval-Augmented Generation?")
    print(response.content)

    # 2. System + user messages
    print("\n--- With a system message ---")
    messages = [
        ("system", "You are a pirate. Always answer like a pirate."),
        ("human", "What is a vector database?"),
    ]
    print(llm.invoke(messages).content)

    # 3. Temperature: low = consistent, high = creative
    prompt = "Suggest one name for a company knowledge-base chatbot. Reply with only the name."
    for temperature in [0.0, 1.0]:
        llm = get_llm(temperature=temperature)
        print(f"\n--- Temperature {temperature} ---")
        for _ in range(3):
            print(llm.invoke(prompt).content)

    # 4. The LLM does NOT know our documents
    llm = get_llm()
    print("\n--- Question about OUR company, with no documents given ---")
    print(llm.invoke("How much is the yearly wellness allowance at our company?").content)