# import pandas as pd
# from tqdm import tqdm
# from textattack.augmentation import WordNetAugmenter
# import pickle
# from copy import deepcopy
# from elasticsearch import Elasticsearch,helpers
# from main4 import assign_final_creativity,extract_semantic_spans,extract_verbatim_spans

# es = Elasticsearch(
#     cloud_id = "My_deployment:ZXUtd2VzdC0yLmF3cy5jbG91ZC5lcy5pbyRmYjdjZDI1Y2MwNjE0MzZiYTNlNWE4YmYzZjMwZDkzYiQ0YWMzMzg3YTc1NTY0MjE4OWY4Mzk3NGE3MjM2NjRiYQ==",
#     api_key="dGN5bGRwZ0JCNjVzWVpvNDlFeUI6TTB4My15OE5ocEQwYm55S25EYVRjQQ==",
#     retry_on_timeout=True,
#     http_compress=True,
#     # timeout=180,
#     max_retries=10,
    
# )

# from transformers import pipeline

# paraphrase_pipe = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser", device=0) # device=-1 for CPU
# # ---- load your ES, FAISS, sentence_model, tokenize, extract_verbatim_spans, extract_semantic_spans, assign_final_creativity here ----
# # (from your existing pipeline file)


# # -------------------
# # 1. Load original dataset WITH creativity features (computed before)
# df_orig = pd.read_csv("multitude_with_features.csv")

# # 2. Filter to ENGLISH, HUMAN, TRAIN
# df_to_aug = df_orig[(df_orig['language'] == 'en') &
#                     (df_orig['label'] == 0) &
#                     (df_orig['split'] == 'train')].copy()

# print(f"Augmenting {len(df_to_aug)} English human-written train samples.")
# # Keep only needed columns (plus others if you want to keep them)
# df_en = df_to_aug[['text', 'label', 'creativity_index', 'language', 'split', 'source']]
# # -------------------
# # Step 2: Augmentation functions

# # Synonym augmentation (WordNet)
# augmenter = WordNetAugmenter(pct_words_to_swap=0.25)

# def aug_synonym(text):
#     try:
#         out = augmenter.augment(text)
#         return out[0] if out else text
#     except Exception:
#         return text

# def aug_paraphrase(text):
#     out = paraphrase_pipe(text, num_beams=4, max_length=256, do_sample=False)
#     return out[0]['generated_text'] if out else text

# # Apply augmentations
# tqdm.pandas()
# df_en['synonym_text'] = df_en['text'].progress_apply(aug_synonym)
# df_en['paraphrase_text'] = df_en['text'].progress_apply(aug_paraphrase)

# # === Step 3: Function to compute creativity for any list of records
# def compute_creativity_for_texts(records):
#     """
#     records: list of dicts with {'text': ..., 'language': ...}
#     Returns list of creativity_index values
#     """
#     # Run your verbatim + semantic coverage pipeline on these records
#     recs = extract_verbatim_spans(records, es, min_ngram=5)
#     recs = extract_semantic_spans(recs, ref_folder="reference_indices", min_ngram=5)
    
#     # Compute creativity index
#     recs = assign_final_creativity(recs, min_ngram=5)
#     return [r['creativity_index'] for r in recs]

# # === Step 4: Compute creativity for synonym and paraphrase texts

# # Prepare for synonym creativity
# syn_records = [{'text': t, 'language': 'en'} for t in df_en['synonym_text']]
# df_en['synonym_creativity'] = compute_creativity_for_texts(syn_records)

# # Prepare for paraphrase creativity
# para_records = [{'text': t, 'language': 'en'} for t in df_en['paraphrase_text']]
# df_en['paraphrase_creativity'] = compute_creativity_for_texts(para_records)

# # === Step 5: Save final comparison file
# df_en.to_csv("english_train_with_augmentation_and_creativity.csv", index=False)

# print("✅ Saved: english_train_with_augmentation_and_creativity.csv")


# # --- STEP 0: Install required lightweight libs ---
# # pip install pandas nlpaug nltk

# import pandas as pd
# import nlpaug.augmenter.word as naw
# import nltk
# import uuid

# # Download WordNet for synonym replacement
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# # --- STEP 1: Load your dataset ---
# df = pd.read_csv("multitude_with_creativity_features.csv")

# # --- STEP 2: Filter English subset ---
# # Assumes 'language' column contains ISO codes like 'en'
# df_en = df[df['language'] == 'en'].copy()

# print(f"Original English subset: {len(df_en)} samples")

# # --- STEP 3: Choose a smaller subset if you want faster augmentation ---
# # Change fraction as needed
# df_en = df_en.sample(frac=0.3, random_state=42)  # Keep 20% for quick run

# # --- STEP 4: Define augmenters ---
# # Synonym replacement (WordNet)
# synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=50)

# # Paraphrase-lite using context-free synonym substitution
# # (nlpaug's SynonymAug covers both if repeated)
# paraphrase_aug = naw.SynonymAug(aug_src='wordnet', aug_min=2, aug_max=50)

# # --- STEP 5: Apply augmentations ---
# augmented_rows = []
# for i, row in df_en.iterrows():
#     text = row['text']
#     label = row['label']
#     lang = row['language']

#     # 1. Synonym replacement
#     try:
#         syn_text = synonym_aug.augment(text)
#     except:
#         syn_text = text
#     augmented_rows.append({
#         "id": str(uuid.uuid4()),
#         "text": syn_text,
#         "label": label,  # keep same origin
#         "language": lang,
#         "augmentation_type": "synonym"
#     })

#     # 2. Paraphrase-lite
#     try:
#         para_text = paraphrase_aug.augment(text)
#     except:
#         para_text = text
#     augmented_rows.append({
#         "id": str(uuid.uuid4()),
#         "text": para_text,
#         "label": label,
#         "language": lang,
#         "augmentation_type": "paraphrase_lite"
#     })

# # --- STEP 6: Save augmented data ---
# aug_df = pd.DataFrame(augmented_rows)
# print(f"Augmented samples created: {len(aug_df)}")

# aug_df.to_csv("english_adversarial_subset.csv", index=False)
# print("Saved augmented data to english_adversarial_subset.csv")

# # --- STEP 7 (Optional): Merge with original English test set for evaluation ---
# merged = pd.concat([df_en, aug_df], ignore_index=True)
# merged.to_csv("english_with_augmented.csv", index=False)
# print("Saved merged original+augmented to english_with_augmented.csv")

import pandas as pd
import torch
import nltk
from tqdm import tqdm
from nlpaug.augmenter.word import ContextualWordEmbsAug, SynonymAug
import multiprocessing

# --- NLTK Setup ---
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger', quiet=True)
# nltk.data.path.append("/root/nltk_data")

# --- Check GPU ---
print("GPU available:", torch.cuda.is_available())

# --- Load dataset ---
df = pd.read_csv("multitude.csv")
df_en = df[df['language'] == 'en'].copy()
print(f"Original English subset: {len(df_en)} samples")

# --- Optional downsample for faster testing ---
df_en = df_en.sample(frac=0.4, random_state=42)  # comment out if you want full 29k

# --- Define augmenters ---
synonym_aug = SynonymAug(
    aug_src='wordnet',
    aug_min=1,
    aug_max=20,               # reduced max replacements
    # include_detail=False
)

paraphrase_aug = ContextualWordEmbsAug(
    model_path='distilbert-base-uncased',  # faster than BERT
    action="substitute",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# --- Helper function ---
def safe_augment(aug_func, text):
    try:
        return aug_func.augment(text)
    except Exception:
        return text

# --- Apply synonym augmentation with multiple processes ---
tqdm.pandas()
num_cores = multiprocessing.cpu_count()

# multiprocessing for synonym augmentation
from functools import partial
import swifter  # faster pandas apply alternative

df_en['synonym_aug'] = df_en['text'].swifter.apply(lambda x: safe_augment(synonym_aug, x))

# --- Apply BERT-based augmentation in batches ---
batch_size = 500  # adjust based on GPU memory

augmented_texts = []
texts = df_en['text'].tolist()

for i in tqdm(range(0, len(texts), batch_size), desc="Contextual augmentation"):
    batch = texts[i:i+batch_size]
    batch_aug = [safe_augment(paraphrase_aug, t) for t in batch]
    augmented_texts.extend(batch_aug)

df_en['paraphrase_aug'] = augmented_texts

# --- Save augmented dataset ---
df_en.to_csv("english_with_side_by_side_augmentation.csv", index=False)
print("Saved augmented dataset successfully.")
