import os
import faiss
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# Config
NGRAM_RANGE = range(3, 8)  # 3 to 7 tokens
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024  # batch embeddings
INDEX_FILE = "reference.faiss"

# Model
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2", device=DEVICE)

def tokenize(text):
    return text.strip().split()

def extract_ngrams(tokens, ngram_range=NGRAM_RANGE):
    for n in ngram_range:
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i:i+n]).lower()

def build_faiss_index(input_files):
    all_ngrams = []
    
    # Step 1: Collect ngrams
    for file_path in input_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Collecting ngrams from {os.path.basename(file_path)}"):
                tokens = tokenize(line)
                all_ngrams.extend(extract_ngrams(tokens))
    
    # Deduplicate to save memory
    all_ngrams = list(set(all_ngrams))
    print(f"Total unique ngrams: {len(all_ngrams)}")
    
    # Step 2: Build FAISS index
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)  # cosine similarity (after normalization)
    
    # Step 3: Encode in batches
    for i in tqdm(range(0, len(all_ngrams), BATCH_SIZE), desc="Embedding ngrams"):
        batch = all_ngrams[i:i+BATCH_SIZE]
        emb = model.encode(batch, convert_to_numpy=True, batch_size=BATCH_SIZE, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(emb)
        index.add(emb)
    
    # Save FAISS + ngrams dictionary
    faiss.write_index(index, INDEX_FILE)
    np.save("reference_ngrams.npy", np.array(all_ngrams))
    print(f"✅ Saved FAISS index ({INDEX_FILE}) and ngram list (reference_ngrams.npy)")

    return index, all_ngrams


if __name__ == "__main__":
    languages = ['en','es','ru','ca','nl','de']
    raw_dir = "raw_txt"
    input_files = [os.path.join(raw_dir, f"{lang}_part_1.txt") for lang in languages if os.path.exists(os.path.join(raw_dir, f"{lang}_part_1.txt"))]
    build_faiss_index(input_files)