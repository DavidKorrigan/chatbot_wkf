import re
import sys

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100):
    """
    The next script split text into chunks of ~300 characters/words:
    - Splits text into chunks.
    - Overlap: how many chars to overlap between chunks (helps retrieval context).
    """
    tokens = text.split()
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        start += chunk_size - overlap

    return chunks

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python chunk.py <text_file> <chunk_folder>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    # Adjust chunk size based on your embedding model’s token limit.
    # For RAG, often about ~300–500 tokens per chunk — but this depends on the model.
    chunking = 300
    # Overlap helps retrieval (context bleed between chunks).
    overlap_char = 50

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_text(content, chunk_size=chunking, overlap=overlap_char)
    print(f"✔️ Created {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        with open(f"{output_folder}/chunk_{i:03d}.txt", "w", encoding="utf-8") as f:
            f.write(chunk)
