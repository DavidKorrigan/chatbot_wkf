import os
import faiss        # Vector similarity search engine (Facebook AI)
import pickle       # For saving Python objects (our chunks metadata)
import numpy as np  # Numerical operations (embeddings are numpy arrays)
from sentence_transformers import SentenceTransformer # Embedding model

"""
This script create:
- index.faiss --> vector similarity index
- metadata.pkl --> original text chunks
"""


# Directory containing text chunks
CHUNK_DIR = "data/chunk"

# Output paths
INDEX_PATH = "index.faiss"
METADATA_PATH = "metadata.pkl"

print("Loading embedding model...")
# Small, efficient embedding model
# Good trade-off between speed and semantic quality
# This model works well locally and supports cosine similarity.
# We will normalize embeddings for proper similarity scoring.
model = SentenceTransformer("BAAI/bge-small-en")

print("Loading chunks...")

chunks = []

# Read every .txt file from the chunk directory
for filename in sorted(os.listdir(CHUNK_DIR)):
    if filename.endswith(".txt"):
        with open(os.path.join(CHUNK_DIR, filename), "r", encoding="utf-8") as f:
            chunks.append(f.read())

print(f"✔️ Loaded {len(chunks)} chunks")

print("Generating embeddings...")
embeddings = model.encode(
    chunks,
    convert_to_numpy=True,      # Return numpy array (needed for FAISS)
    normalize_embeddings=True,  # IMPORTANT: makes cosine similarity work
    show_progress_bar=True
)

print("Building FAISS index...")
dimension = embeddings.shape[1]
# Using Inner Product (IP) index
# Because embeddings are normalized:
#   Inner Product == Cosine Similarity
index = faiss.IndexFlatIP(dimension)
# FAISS requires float32
index.add(embeddings.astype("float32"))

print("Saving index...")
faiss.write_index(index, INDEX_PATH)

print("Saving metadata...")
with open(METADATA_PATH, "wb") as f:
    pickle.dump(chunks, f)

print("✔️ Embedding + indexing completed successfully.")