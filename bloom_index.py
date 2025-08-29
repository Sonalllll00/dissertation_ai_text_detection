import os
import math
import mmap
import glob
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
from tqdm import tqdm
from bloom_filter2 import BloomFilter

# =========================
# CONFIGURATION
# =========================
RAW_DIR = "raw_txt"                    # Folder where raw text files like {lang}_part_1.txt are stored
LANGUAGES = ['en', 'es', 'ru', 'ca', 'nl', 'de']  # Languages to process
OUT_DIR = "bloom_shards"               # Folder where output Bloom filter shards (*.bf files) are saved
LINES_PER_SHARD = 300000               # Number of text lines per shard (controls shard size)
AVG_NGRAMS_PER_LINE = 30               # Rough estimate: how many 3–7 token n-grams per line
BLOOM_ERROR_RATE = 0.001               # Bloom filter false positive rate (0.1%)
MAX_WORKERS = 8                        # How many processes to use in parallel

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# UTILITIES
# =========================
def count_lines_fast(filepath: str) -> int:
    """
    Count the number of lines in a file very quickly without decoding text.
    Uses memory-mapping (mmap) to read bytes directly.
    """
    with open(filepath, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return sum(1 for _ in iter(mm.readline, b""))


def shard_plan(total_lines: int, lines_per_shard: int) -> List[Tuple[int, int, int]]:
    """
    Given the total number of lines and desired lines per shard,
    compute the shard plan.

    Returns a list of tuples: (shard_id, start_line, end_line)
    where start_line is inclusive and end_line is exclusive.
    """
    shards = []
    n_shards = max(1, math.ceil(total_lines / lines_per_shard))
    for sid in range(n_shards):
        start = sid * lines_per_shard
        end = min((sid + 1) * lines_per_shard, total_lines)
        shards.append((sid, start, end))
    return shards


def generate_ngrams_from_line(line: str, n_min: int = 3, n_max: int = 7):
    """
    Convert a line of text into all possible n-grams (between n_min and n_max tokens).
    Tokens are lowercased. Empty lines are skipped.
    Example: "I love NLP" with n_min=2 → ["i love", "love nlp"]
    """
    line = line.strip()
    if not line:
        return
    tokens = line.split()
    if not tokens:
        return
    lower = [t.lower() for t in tokens]
    for n in range(n_min, n_max + 1):
        for i in range(0, len(lower) - n + 1):
            yield " ".join(lower[i:i + n])


def build_one_shard(lang: str, shard_id: int, start_line: int, end_line: int) -> str:
    """
    Build a single shard Bloom filter for a given language.
    
    - Reads lines from start_line to end_line in the language text file.
    - Extracts all n-grams.
    - Adds them to a Bloom filter sized for the shard.
    - Saves the Bloom filter safely (atomic write: write to .tmp then rename).
    """
    src_path = os.path.join(RAW_DIR, f"{lang}_part_1.txt")
    shard_path = os.path.join(OUT_DIR, f"{lang}_shard_{shard_id:04d}.bf")
    tmp_path = shard_path + ".tmp"

    # Skip conditions
    if not os.path.exists(src_path):
        return f"[{lang}][shard {shard_id}] SKIPPED (missing file)"
    if os.path.exists(shard_path):
        return f"[{lang}][shard {shard_id}] SKIPPED (exists)"

    lines_in_shard = end_line - start_line
    if lines_in_shard <= 0:
        return f"[{lang}][shard {shard_id}] SKIPPED (empty shard)"

    # Estimate Bloom filter size based on line count
    est_ngrams = max(1, lines_in_shard * AVG_NGRAMS_PER_LINE)
    bf = BloomFilter(max_elements=est_ngrams, error_rate=BLOOM_ERROR_RATE)

    # Read the source file and populate the Bloom filter
    with open(src_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Fast-forward to shard start
        for _ in range(start_line):
            if not f.readline():
                break

        # Process lines with progress bar
        with tqdm(total=lines_in_shard,
                  desc=f"[{lang} shard {shard_id}]",
                  position=shard_id % MAX_WORKERS,
                  leave=False,
                  ncols=80) as pbar:
            count = 0
            while count < lines_in_shard:
                line = f.readline()
                if not line:
                    break
                for ng in generate_ngrams_from_line(line, 3, 7):
                    bf.add(ng)
                count += 1
                pbar.update(1)

    # Save Bloom filter safely (write to temporary file, then rename)
    with open(tmp_path, "wb") as out_f:
        pickle.dump(bf, out_f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, shard_path)

    return f"[{lang}][shard {shard_id}] DONE ({lines_in_shard} lines) -> {os.path.basename(shard_path)}"


def existing_shards(lang: str):
    """
    Return list of existing shard files for a given language.
    """
    return sorted(glob.glob(os.path.join(OUT_DIR, f"{lang}_shard_*.bf")))


def build_language_shards(lang: str):
    """
    Build all shards for one language.
    
    - Counts lines in raw file.
    - Plans shards.
    - Skips shards already built (checkpointing).
    - Runs missing shards in parallel using ProcessPoolExecutor.
    """
    src_path = os.path.join(RAW_DIR, f"{lang}_part_1.txt")
    if not os.path.exists(src_path):
        print(f"[{lang}] MISSING {src_path}, skipping.")
        return

    total_lines = count_lines_fast(src_path)
    plan = shard_plan(total_lines, LINES_PER_SHARD)

    # Figure out which shards already exist
    done_ids = set()
    for p in existing_shards(lang):
        base = os.path.basename(p)
        try:
            sid = int(base.split("_shard_")[1].split(".bf")[0])
            done_ids.add(sid)
        except Exception:
            pass
    todo = [(sid, s, e) for (sid, s, e) in plan if sid not in done_ids]

    print(f"[{lang}] total lines: {total_lines:,} | shards: {len(plan)} "
          f"(todo: {len(todo)}, done: {len(done_ids)})")

    if not todo:
        print(f"[{lang}] All shards already exist. ✅")
        return

    # Run shard builds in parallel
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        tasks = [ex.submit(build_one_shard, lang, sid, s, e) for (sid, s, e) in todo]
        for fut in tqdm(as_completed(tasks), total=len(tasks),
                        desc=f"[{lang}] Building shards", ncols=100):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append(f"[{lang}] ERROR: {e}")

    # Print all results in sorted order (shard IDs)
    for msg in sorted(results):
        print(msg)

    print(f"[{lang}] Done. Shards in: {OUT_DIR}")


# =========================
# MAIN PIPELINE
# =========================
def main():
    """
    Main driver:
    - Iterates over all languages.
    - Builds Bloom filter shards for each.
    """
    print(f"Workers: {MAX_WORKERS} | Lines/shard: {LINES_PER_SHARD} | FP: {BLOOM_ERROR_RATE}")
    for lang in LANGUAGES:
        build_language_shards(lang)
    print("✅ All languages processed.")


if __name__ == "__main__":
    main()
