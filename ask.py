import faiss              # Vector similarity search engine
import pickle             # Used to load stored text chunks (metadata)
import numpy as np        # Numerical operations (embeddings)
import requests
from sentence_transformers import SentenceTransformer  # Embedding model

# Load embedding model
# IMPORTANT:
# Must be the exact same model used to build the FAISS index.
# Otherwise embedding dimensions won't match.
model = SentenceTransformer("BAAI/bge-small-en")

# Load FAISS index (vector database)
# The pre-built similarity search index
index = faiss.read_index("index.faiss")

# Load metadata (original text chunks)
# FAISS only stores vectors, not text.
# Load the original chunks separately.
with open("metadata.pkl", "rb") as f:
    chunks = pickle.load(f)


def search(query, k=3, threshold=0.76):
    """
    Performs semantic similarity search.

    Parameters:
    - query: user question
    - k: number of chunks to retrieve
    - threshold: minimum cosine similarity required

    Returns:
    - None if query is out-of-domain
    - List of top-k relevant chunks otherwise
    """

    # Encode the user query into embedding vector
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True  # Required for cosine similarity
    )

    # Search FAISS index
    # D = similarity scores (cosine similarity because of IndexFlatIP)
    # I = indices of top matching chunks
    D, I = index.search(query_embedding.astype("float32"), k)

    print("Similarities:", D[0])

    # Keep only chunks above threshold
    valid_chunks = [
        chunks[i] for score, i in zip(D[0], I[0])
        if score >= threshold
    ]

    # Guardrail:
    # If similarity is too low → reject question as out-of-scope
    if not valid_chunks:
        return None

    # Return the actual text chunks corresponding to the top indices
    return [chunks[i] for i in I[0]]


def ask_mistral(context, question):
    """
    Sends retrieved context + question to local Mistral model.
    Uses Ollama REST API.
    """

    # Prompt engineering:
    # We strictly constrain model behavior to context-only answers.
    prompt = f"""
You are a WKF Karate regulations assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say:
"This question is outside WKF regulations."
- Do NOT use outside knowledge.
- Do NOT guess.

Context:
{context}

Question:
{question}
"""

    # Send request to Ollama
    response = requests.post(
        "http://ollama.server:11434/api/generate",
        json={
            "model": "mistral:7b",
            "prompt": prompt,
            "stream": False
        }
    )

    # Check HTTP status
    if response.status_code != 200:
        return f"HTTP Error {response.status_code}: {response.text}"

    # Parse JSON safely
    try:
        data = response.json()
    except Exception as e:
        return f"Invalid JSON response: {response.text}"

    # Safely extract model response
    return data.get("response", "No response field in API output.")

def validate_response(answer, context, min_overlap_ratio=0.35):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    overlap = answer_words.intersection(context_words)
    ratio = len(overlap) / max(len(answer_words), 1)

    if ratio < min_overlap_ratio:
        return False

    return True

if __name__ == "__main__":
    question = input("Ask a question: ")

    # Step 1: Retrieve relevant chunks
    retrieved_chunks = search(question)

    # If similarity threshold fails → reject
    if retrieved_chunks is None:
        print("This assistant only answers WKF regulation questions.")
    else:
        # Step 2: Build context from retrieved chunks
        context = "\n\n".join(retrieved_chunks)

         # Context size limiter (prevents model truncation)
        MAX_CONTEXT_CHARS = 8000
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]

        print(question)
        # Step 3: Send to LLM
        answer = ask_mistral(context, question)

        answer = ask_mistral(context, question)

        if not validate_response(answer, context):
            print("⚠️ The generated answer was rejected (possible hallucination).")
        else:
            print("\n--- ANSWER ---\n")
            print(answer)