import multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)
# mp.set_start_method('spawn', force=True)

import pandas as pd
import numpy as np
import jieba
import spacy
import nltk
from pandarallel import pandarallel  # <-- parallel processing
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import textstat
# import csv



#  Initialize pandarallel with 6 workers
pandarallel.initialize(progress_bar=True, nb_workers=6)

d = pd.read_csv("multitude.csv", engine='python', quoting=0,quotechar='"')
tqdm.pandas()  # progress bar

df = d.copy()

# df = df[df["split"]== "train"].reset_index(drop=True)

# Load spacy models for relevant languages
SPACY_MODELS = {
            'de': 'de_core_news_sm',
            'en': 'en_core_web_sm',
            'uk': 'uk_core_news_sm',    # Ukrainian
            'es': 'es_core_news_sm',
            'nl': 'nl_core_news_sm',
            'ca': 'ca_core_news_sm',    # Catalan
            'ru': 'ru_core_news_sm',
            'pt': 'pt_core_news_sm',
            'ar': 'ar_core_news_sm',
            'zh': 'zh_core_web_sm',
}
spacy_cache = {}
def get_spacy(lang):
    if lang in spacy_cache:
        return spacy_cache[lang]
    try:
        if lang in SPACY_MODELS:
            spacy_cache[lang] = spacy.load(SPACY_MODELS[lang])
        else:
            spacy_cache[lang] = None
    except Exception:
        spacy_cache[lang] = None
    return spacy_cache[lang]

def tokenize(text, lang):
    if lang == "zh":
        return list(jieba.cut(text))
    nlp = get_spacy(lang)
    if nlp:
        doc = nlp(text)
        return [token.text for token in doc if not token.is_space]
    try:
        return nltk.word_tokenize(text, language=lang)
    except:
        return text.split()

def split_sentences(text, lang):
    if lang == "zh":
        return [s for s in text.split("。") if s.strip()]
    nlp = get_spacy(lang)
    if nlp:
        return [sent.text for sent in nlp(text).sents]
    try:
        return nltk.sent_tokenize(text, language=lang)
    except:
        return [s for s in text.split(".") if s.strip()]

sentence_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

def semantic_coherence(emb):
    if len(emb) < 2:
        return 1.0
    # emb = sentence_model.encode(sentences)
    sims = [np.dot(emb[i], emb[i+1]) / (np.linalg.norm(emb[i]) * np.linalg.norm(emb[i+1]))
            for i in range(len(emb)-1)]
    return float(np.mean(sims))


df['sentences'] = df.progress_apply(lambda row: split_sentences(row['text'], row['language']), axis=1)
# This is CUDA heavy, so do serially before parallel processing
df['embeddings'] = df['sentences'].progress_apply(lambda sentences: sentence_model.encode(sentences) if len(sentences) > 1 else np.array([1.0]))



def feature_row(row):
    text, lang = row['text'], row['language']
    tokens = tokenize(text, lang)
    sentences = split_sentences(text, lang)
    unique_words = len(set(tokens))
    sent_count = len(sentences)
    avg_sent_length = row['length']/sent_count if sent_count > 0 else np.nan
    # For English only
    try:
        # import textstat
        flesch = textstat.flesch_reading_ease(text) if lang == 'en' else np.nan
    except:
        flesch = np.nan
    # coherence = semantic_coherence(sentences)
       # Use precomputed embeddings
    coherence = semantic_coherence(row['embeddings'])
    return pd.Series({
        'unique_word_count': unique_words,
        'type_token_ratio': unique_words/row['length'] if row['length'] > 0 else 0,
        'sentence_count': sent_count,
        'avg_sentence_length': avg_sent_length,
        'flesch_reading_ease': flesch,
        'semantic_coherence': coherence,
    })

# feature_df = df.progress_apply(feature_row, axis=1)
# 🚀 Parallel processing across 6 CPUs
feature_df = df.parallel_apply(feature_row, axis=1)
df = pd.concat([df, feature_df], axis=1)
df.to_csv("multitude_with_features.csv", index=False)