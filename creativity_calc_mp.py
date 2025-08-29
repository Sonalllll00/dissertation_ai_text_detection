import os
import glob
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
import torch
from sentence_transformers import SentenceTransformer
from bloom_filter2 import BloomFilter
from multiprocessing import Pool, cpu_count
import nltk
import spacy

# ==============================
# CONFIG
# ==============================
CSV_PATH = "multitude_with_features.csv"
OUTPUT_PATH = "multitude_with_creativity_features.csv"
FAISS_INDEX_PATH = "reference.faiss"
BLOOM_DIR = "bloom_shards"
NGRAM_RANGE = range(3, 8)       # 3 to 7
SEMANTIC_THRESHOLD = 0.75
BATCH_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANGUAGE_FILTER = ['en','es','ru','ca','nl','de']

# ==============================
# LOAD MODELS & INDEX
# ==============================
print("Loading embedding model + FAISS index...")
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2", device=DEVICE)
# ref_index = faiss.read_index(FAISS_INDEX_PATH)

def safe_load_bloom(path):
    with open(path, "rb") as f:
        bf = pickle.load(f)

    # ensure backend exists (sometimes missing after unpickle)
    if not hasattr(bf, "backend"):
        # reconstruct the backend
        bf.backend = bytearray(bf.num_bits // 8 + 1)

    return bf

# Mapping ISO code → NLTK full language name
NLTK_LANG_MAP = {
    'en': 'english',
    'es': 'spanish',
    'ru': 'russian',
    'nl': 'dutch',
    'ca': 'catalan',  # Not supported by NLTK punkt — will fall back to split()
    # 'cs': 'czech',    # Not supported — fallback
    'de': 'german'}

_spacy_models = {}

def load_spacy_model(lang):
    if lang not in _spacy_models:
        spacy_map = {
            # 'en': 'en_core_web_sm',
            'es': 'es_core_news_sm',
            'ru': 'ru_core_news_sm',
            'nl': 'nl_core_news_sm',
            'ca': 'ca_core_news_sm',
            'de': 'de_core_news_sm',
            'pt': 'pt_core_news_sm',
            'ar': 'ar_core_news_sm',
            'uk': 'uk_core_news_sm',
            # 'zh': 'zh_core_web_sm',
        }
        model_name = spacy_map.get(lang)
        if model_name:
            try:
                _spacy_models[lang] = spacy.load(model_name)
            except Exception as e:
                print(f"Failed loading spaCy model for {lang}: {e}")
                _spacy_models[lang] = None
        else:
            _spacy_models[lang] = None
    return _spacy_models.get(lang)

def tokenize(text, language="en"):
   
    if language == "en":
        return nltk.word_tokenize(text)
   
    nlp = load_spacy_model(language)
    if nlp:
        try:
            doc = nlp(text)
            return [token.text for token in doc if not token.is_space]
        except Exception as e:
            print(f"spaCy tokenization failed for {language}: {e}")
     # Fallback to NLTK if supported
    nltk_lang = NLTK_LANG_MAP.get(language)
    if nltk_lang is not None:  # crude check for availability
        try:
            return nltk.word_tokenize(text, language=nltk_lang)
        except LookupError:
            nltk.download("punkt")
            return nltk.word_tokenize(text, language=nltk_lang)

    # Final fallback — naive split
    return text.split()



ref_indices = {}
for lang in LANGUAGE_FILTER:
    faiss_path = os.path.join("reference_indices", f"{lang}_index.faiss")
    if os.path.exists(faiss_path):
        ref_indices[lang] = faiss.read_index(faiss_path)
    else:
        print(f"⚠️ No FAISS index found for {lang}, skipping...")
# ==============================
# BLOOM SHARD MANAGEMENT
# =============================='


class BloomWrapper:
    """
    Wraps either a real BloomFilter or a dict so we can do 'ngram in shard'
    """
    def __init__(self, obj):
        if isinstance(obj, BloomFilter):
            self._type = "bloom"
            self.bf = obj
            # patch backend if missing
            if not hasattr(self.bf, "backend"):
                self.bf.backend = bytearray(self.bf.num_bits // 8 + 1)
        elif isinstance(obj, dict):
            self._type = "dict"
            self.bf = obj
        else:
            raise ValueError(f"Unsupported shard object type: {type(obj)}")

    def __contains__(self, item):
        if self._type == "bloom":
            return item in self.bf
        else:
            return item in self.bf

import pickle
def load_bloom_shards(lang):
    """Load all Bloom filter shards for a given language."""
    paths = sorted(glob.glob(os.path.join(BLOOM_DIR, f"{lang}_shard_*.bf")))
    shards = []
    for p in paths:
        try:
            obj = pickle.load(open(p, "rb"))
            shards.append(BloomWrapper(obj))
        except Exception as e:
            print(f"⚠️ Failed to load {p}: {e}")
    return shards

bloom_shards = {lang: load_bloom_shards(lang) for lang in LANGUAGE_FILTER}

# ==============================
# HELPERS
# ==============================
def generate_ngrams(text, ngram_range=NGRAM_RANGE):
    # tokens = text.split()
    tokens = tokenize(text, lang)
    out = []
    for n in ngram_range:
        for i in range(len(tokens)-n+1):
            out.append((" ".join(tokens[i:i+n]).lower(), i, i+n))
    return out

def check_bloom(ngram, lang):
    for shard in bloom_shards.get(lang, []):
        if ngram in shard:
            return True
    return False

def embed_batch(ngrams):
    emb = model.encode(ngrams, convert_to_numpy=True, device=DEVICE, batch_size=BATCH_SIZE)
    faiss.normalize_L2(emb)
    return emb.astype("float32")

# ==============================
# MAIN PROCESSING PER ROW
# ==============================
def process_row(row):
    text = str(row['text'])
    lang = row['language']
    if lang not in LANGUAGE_FILTER:
        row['creativity_score'] = None
        row['spans'] = []
        return row

    # tokens = text.split()
    tokens = tokenize(text, lang)
    
    ngrams = generate_ngrams(text)
     # Prepare separate accumulators
    verbatim_weight = 0.0
    semantic_weight = 0.0

    # Separate verbatim vs survivors
    verbatim_flags = [check_bloom(ng, lang) for ng,_,_ in ngrams]
    survivors = [ng for (ng,_,flag) in [(n[0], n[1:], v) for n,v in zip(ngrams, verbatim_flags)] if not flag]

   
    if survivors and lang in ref_indices:
        embeddings = []
        for i in range(0, len(survivors), BATCH_SIZE):
            batch = survivors[i:i+BATCH_SIZE]
            batch_emb = model.encode(batch, convert_to_numpy=True, device=DEVICE).astype("float32")
            faiss.normalize_L2(batch_emb)
            embeddings.append(batch_emb)
        embeddings = np.vstack(embeddings)

        # language-specific index
        D, _ = ref_indices[lang].search(embeddings, 1)
        semantic_scores = list(D[:,0])
    else:
        semantic_scores = []

    # Assign spans
    spans = []
    sem_idx = 0
    for idx, (ng, i, j) in enumerate(ngrams):
        if verbatim_flags[idx]:
            spans.append({"start_index":i, "end_index":j, "span_text":ng, "type":"verbatim", "weight":1.0})
        else:
            if sem_idx < len(semantic_scores):
                score = float(semantic_scores[sem_idx])
                if score > SEMANTIC_THRESHOLD:
                    spans.append({"start_index":i, "end_index":j, "span_text":ng, "type":"semantic", "weight":score})
                sem_idx += 1

    # Merge overlapping spans
    spans.sort(key=lambda x: x["start_index"])
    merged = []
    for s in spans:
        if merged and s["start_index"] < merged[-1]["end_index"]:
            merged[-1]["weight"] = max(merged[-1]["weight"], s["weight"])
        else:
            merged.append(s)

        # After merging spans, compute separate coverages
    verbatim_weight = sum(s["weight"] for s in merged if s["type"] == "verbatim")
    semantic_weight = sum(s["weight"] for s in merged if s["type"] == "semantic")
    
    # Creativity score
    total_tokens = max(1, len(tokens))
    verbatim_coverage = verbatim_weight / total_tokens
    semantic_coverage = semantic_weight / total_tokens
    # creativity_score = sum(s["weight"] for s in merged) / total_tokens
    creativity_score = (verbatim_weight + semantic_weight) / total_tokens

    row['verbatim_coverage'] = verbatim_coverage
    row['semantic_coverage'] = semantic_coverage
    row['creativity_score'] = creativity_score
    
    ci = 1 - creativity_score
    row['creativity_index'] = ci
    row['spans'] = merged
    return row

# ==============================
# MAIN PIPELINE
# ==============================
def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df['language'].isin(LANGUAGE_FILTER)]
    rows = df.to_dict(orient='records')

    processed = []
    # with Pool(processes=6) as pool:
    for result in tqdm(rows, total=len(rows), desc="Processing rows"):
        processed.append(process_row(result))

    out = pd.DataFrame(processed)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Done! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
