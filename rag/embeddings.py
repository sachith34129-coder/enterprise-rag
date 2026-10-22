"""
Embeddings module
-----------------
Loads a model that converts text into vectors (lists of numbers).
Texts with similar meaning get vectors that point in similar directions.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Load a sentence-transformers embedding model."""
    print(f"Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)


if __name__ == "__main__":
    import numpy as np

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """1.0 = same meaning, around 0 = unrelated."""
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    model = get_embedding_model()

    # 1. What does an embedding look like?
    vector = model.embed_query("How long is the cooldown period?")
    print(f"\nVector length : {len(vector)}")
    print(f"First 5 values: {vector[:5]}")

    # 2. Compare a question with three sentences
    question = "How long is the cooldown period?"
    sentences = [
        "In between each cycle, we spend two weeks cooling down.",    # same meaning, different words
        "Employees receive a yearly wellness allowance.",             # different topic, same domain
        "The recipe needs two cups of flour.",                        # completely unrelated
    ]

    question_vector = model.embed_query(question)
    sentence_vectors = model.embed_documents(sentences)

    print(f"\nQuestion: {question}\n")
    for sentence, sentence_vector in zip(sentences, sentence_vectors):
        score = cosine_similarity(question_vector, sentence_vector)
        print(f"{score:.3f}  {sentence}")